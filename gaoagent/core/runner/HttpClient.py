from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterable, Iterator


@dataclass(frozen=True)
class HttpResponse:
    """
    统一封装 HTTP 调用结果。

    该结构用于上层 Runner 以一致的方式读取返回信息（状态码、错误原因、JSON、原始文本）。

    字段说明：
    - ok: 是否请求成功（通常表示拿到了 2xx 响应并且成功读取响应体）
    - status: HTTP 状态码；无法获得时为 None（例如网络异常）
    - reason: 失败原因的简要描述（异常类型/HTTP reason 等）
    - json: 若响应体可解析为 JSON，则为解析后的对象（通常是 dict）
    - text: 原始响应体文本；对于流式响应会是合成后的“最终 JSON 文本”
    """
    ok: bool
    status: int | None = None
    reason: str | None = None
    json: dict[str, Any] | None = None
    text: str | None = None


class OpenAICompatibleHttpClient:
    def __init__(self, *, base_url: str, api_key: str, timeout_s: int = 60) -> None:
        """
        创建一个兼容 OpenAI Chat Completions 接口的 HTTP 客户端。

        该客户端基于 urllib 实现，不依赖第三方库，面向“OpenAI API 兼容服务”的典型部署。

        参数：
        - base_url: 兼容服务的根地址。允许传入：
          - https://example.com
          - https://example.com/v1
          - https://example.com/v1/chat/completions
          客户端会自动拼出最终的 /v1/chat/completions 路径。
        - api_key: Bearer Token（将以 Authorization: Bearer ... 发送）
        - timeout_s: urllib 超时秒数（连接/读取的整体超时）
        """
        self._base_url = base_url
        self._api_key = api_key
        self._timeout_s = timeout_s

    def build_chat_completions_url(self) -> str:
        """
        由 base_url 生成最终 Chat Completions 请求地址。

        规则：
        - base_url 若以 /chat/completions 结尾：直接使用
        - base_url 若以 /v1 结尾：追加 /chat/completions
        - 其他情况：追加 /v1/chat/completions

        返回：
        - 规范化后的完整 URL（不以 / 结尾）
        """
        clean = self._base_url.strip().rstrip("/")
        if clean.endswith("/chat/completions"):
            return clean
        if clean.endswith("/v1"):
            return f"{clean}/chat/completions"
        return f"{clean}/v1/chat/completions"

    def _iter_sse_data_events(self, chunks: Iterable[bytes]) -> Iterator[str]:
        """
        将 SSE（text/event-stream）按“事件”粒度解析为 data 字符串。

        输入：
        - chunks: 逐行读取到的 bytes（通常来自 response.readline 的迭代器）

        行格式约定（兼容 OpenAI 风格 SSE）：
        - 空行（\\n 或 \\r\\n）表示一个事件结束
        - 以 "data:" 开头的行表示该事件的 data 片段

        输出：
        - 每个事件合并后的 data（可能是 JSON 字符串或 "[DONE]"）
        - 同一事件内若出现多行 data，会用 "\\n" 拼接并 strip
        """
        buf: list[str] = []
        for chunk in chunks:
            line = chunk.decode("utf-8", errors="replace")
            if line in ("\n", "\r\n"):
                if buf:
                    yield "\n".join(buf).strip()
                    buf = []
                continue
            if line.startswith("data:"):
                buf.append(line[len("data:") :].strip())
        if buf:
            yield "\n".join(buf).strip()

    def _stream_chat_completions_and_print(self, response: Any) -> tuple[dict[str, Any], str]:
        """
        读取 OpenAI 风格流式响应（SSE），实时打印内容，并合成最终响应结构。

        行为：
        - 从 response.readline 持续读取 SSE data 事件
        - 对每个事件尝试解析 JSON，提取 choices[0].delta
        - 若 delta.content 有增量文本：立即写入 stdout（实现“边生成边展示”）
        - 若 delta.tool_calls 有增量：按 index 聚合，拼接 function.arguments（常为分片 JSON 字符串）
        - 收到 "[DONE]" 结束

        返回：
        - final_payload: 兼容 chat.completion 的最终 dict（将 delta 合并为 message）
        - raw_text: final_payload 的 JSON 字符串形式（便于沿用非流式的返回处理方式）
        """
        content_parts: list[str] = []
        tool_calls: dict[int, dict[str, Any]] = {}
        first_event: dict[str, Any] | None = None
        finish_reason: str | None = None
        printed = False

        for data in self._iter_sse_data_events(iter(response.readline, b"")):
            if not data:
                continue
            if data == "[DONE]":
                break
            try:
                event = json.loads(data)
            except Exception:
                continue
            if first_event is None and isinstance(event, dict):
                first_event = event

            choices = event.get("choices") if isinstance(event, dict) else None
            if not isinstance(choices, list) or not choices:
                continue
            choice0 = choices[0] if isinstance(choices[0], dict) else {}
            delta = choice0.get("delta") if isinstance(choice0, dict) else None
            if isinstance(choice0, dict) and isinstance(choice0.get("finish_reason"), str):
                finish_reason = choice0.get("finish_reason")

            if isinstance(delta, dict):
                c = delta.get("content")
                if isinstance(c, str) and c:
                    content_parts.append(c)
                    sys.stdout.write(c)
                    sys.stdout.flush()
                    printed = True

                d_tool_calls = delta.get("tool_calls")
                if isinstance(d_tool_calls, list):
                    for item in d_tool_calls:
                        if not isinstance(item, dict):
                            continue
                        idx = item.get("index")
                        if not isinstance(idx, int):
                            idx = 0
                        cur = tool_calls.get(idx)
                        if cur is None:
                            cur = {"id": None, "type": "function", "function": {"name": None, "arguments": ""}}
                            tool_calls[idx] = cur
                        if isinstance(item.get("id"), str):
                            cur["id"] = item.get("id")
                        if isinstance(item.get("type"), str):
                            cur["type"] = item.get("type")
                        fn = item.get("function")
                        if isinstance(fn, dict):
                            if isinstance(fn.get("name"), str):
                                cur["function"]["name"] = fn.get("name")
                            if isinstance(fn.get("arguments"), str):
                                cur["function"]["arguments"] = str(cur["function"].get("arguments") or "") + fn.get("arguments")

        if printed:
            sys.stdout.write("\n")
            sys.stdout.flush()

        content = "".join(content_parts)
        message: dict[str, Any] = {"role": "assistant"}
        if content:
            message["content"] = content
        if tool_calls:
            message["tool_calls"] = [tool_calls[i] for i in sorted(tool_calls.keys())]

        model = ""
        if isinstance(first_event, dict) and isinstance(first_event.get("model"), str):
            model = first_event.get("model") or ""

        final_payload: dict[str, Any] = {
            "id": (first_event or {}).get("id") if isinstance(first_event, dict) else None,
            "object": "chat.completion",
            "model": model,
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        }
        return final_payload, json.dumps(final_payload, ensure_ascii=False)

    def post_json(self, url: str, payload: dict[str, Any]) -> HttpResponse:
        """
        向指定 URL 发送 JSON POST 请求，并返回统一封装的 HttpResponse。

        约定：
        - 默认启用 stream=True（若 payload 未指定 stream）
        - 请求头：
          - Content-Type: application/json
          - Accept: text/event-stream（优先请求流式返回）
          - Authorization: Bearer <api_key>

        响应处理：
        - 如果 Content-Type 包含 text/event-stream：按 SSE 流读取并实时打印内容，同时合成最终 JSON 结构返回
        - 否则：一次性读取 body，尽力解析 JSON；无论是否可解析，都会保留 text

        异常处理：
        - HTTPError：ok=False，status 为 HTTP code，text 为错误响应体（若可读取）
        - 其他异常：ok=False，status=None，reason 为 "异常类型: 异常信息"
        """
        req_payload = dict(payload)
        req_payload.setdefault("stream", True)
        sys.stdout.write(json.dumps(req_payload, ensure_ascii=False) + "\n")
        sys.stdout.flush()
        req_bytes = json.dumps(req_payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            url=url,
            data=req_bytes,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
                "Authorization": f"Bearer {self._api_key}",
            },
        )

        try:
            with urllib.request.urlopen(request, timeout=self._timeout_s) as response:
                ct = ""
                try:
                    ct = str(getattr(response, "headers", {}).get("Content-Type") or "")
                except Exception:
                    ct = ""
                if "text/event-stream" in ct:
                    j, raw = self._stream_chat_completions_and_print(response)
                    return HttpResponse(ok=True, status=getattr(response, "status", 200), reason=None, json=j, text=raw)

                raw = response.read().decode("utf-8")
                try:
                    j = json.loads(raw)
                except Exception:
                    j = None
                return HttpResponse(ok=True, status=getattr(response, "status", 200), reason=None, json=j, text=raw)
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8")
            except Exception:
                body = ""
            return HttpResponse(ok=False, status=int(getattr(e, "code", 0) or 0), reason=str(getattr(e, "reason", "")), json=None, text=body)
        except Exception as e:
            return HttpResponse(ok=False, status=None, reason=f"{type(e).__name__}: {e}", json=None, text=None)
