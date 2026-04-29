from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable

from gaoagent.core.runner.Console import Console
from gaoagent.core.runner.RunLogger import get_current_run_logger
from gaoagent.core.runner.Utils import redact, safe_json_dumps, summarize


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
    - text: 原始响应体文本；对于流式响应会是合成后的"最终 JSON 文本"
    """

    ok: bool
    status: int | None = None
    reason: str | None = None
    json: dict[str, Any] | None = None
    text: str | None = None


# 流式输出回调类型：(chunk_type, content) -> None
# chunk_type: "content" | "reasoning" | "tool_call_start" | "tool_call_args"
StreamCallback = Callable[[str, str], None]


class OpenAICompatibleHttpClient:
    """OpenAICompatibleHttpClient 类。
    
    职责:
    - 封装 OpenAI Chat Completions 接口的 HTTP 调用.
    
    """
    def __init__(self, *, base_url: str, api_key: str, timeout_s: int = 240) -> None:
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

    def _build_chat_completions_url(self) -> str:
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

    def post_chat_completions(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any | None = None,
        step: int | None = None,
        stream_callback: StreamCallback | None = None,
    ) -> HttpResponse:
        """post_chat_completions 方法。
        
        用途:
        - 调用 OpenAI Chat Completions 接口，获取模型回复.
        
        参数:
        - model: 模型名称.
        - messages: 输入的消息列表.
        - tools: 工具列表.
        - tool_choice: 工具选择.
        - stream_callback: 流式输出回调函数，接收 (chunk_type, content) 参数.
          chunk_type: "content" | "reasoning" | "tool_call_start" | "tool_call_args"
        
        返回:
        - HttpResponse: 返回模型回复的 HttpResponse 对象.
        """
        url = self._build_chat_completions_url()
        body_obj: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        if tools is not None:
            body_obj["tools"] = tools
        if tool_choice is not None:
            body_obj["tool_choice"] = tool_choice
        body_bytes = json.dumps(body_obj, ensure_ascii=False).encode("utf-8")
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        if isinstance(self._api_key, str) and self._api_key.strip():
            headers["Authorization"] = f"Bearer {self._api_key}"

        req = urllib.request.Request(url=url, data=body_bytes, headers=headers, method="POST")
        Console.debug(
            safe_json_dumps(
                {
                    "event": "llm_http_request",
                    "step": step,
                    "url": url,
                    "model": model,
                    "message_count": len(messages) if isinstance(messages, list) else 0,
                    "tool_count": len(tools) if isinstance(tools, list) else 0,
                    "stream": True,
                }
            )
        )

        run_logger = get_current_run_logger()
        if run_logger is not None:
            run_logger.log_event(
                "http_request",
                {
                    "url": url,
                    "method": "POST",
                    "headers": redact(headers),
                    "body": redact(body_obj),
                },
                step=step,
            )

        try:
            with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:
                status = getattr(resp, "status", None) or resp.getcode()
                content_type = resp.headers.get("Content-Type") or ""
                Console.debug(
                    safe_json_dumps(
                        {
                            "event": "llm_http_response_head",
                            "step": step,
                            "status": int(status) if isinstance(status, int) else status,
                            "content_type": content_type,
                        }
                    )
                )

                if "text/event-stream" in content_type.lower():
                    message: dict[str, Any] = {"role": "assistant"}
                    text_parts: list[str] = []
                    reasoning_parts: list[str] = []
                    finish_reason: str | None = None
                    tool_call_map: dict[int, dict[str, Any]] = {}
                    # 用于跟踪已输出的tool call，避免重复输出
                    tool_call_started: set[int] = set()

                    for raw_line in resp:
                        try:
                            line = raw_line.decode("utf-8", errors="replace").strip()
                        except Exception:
                            continue
                        if not line:
                            continue
                        if not line.startswith("data:"):
                            continue
                        data = line[len("data:") :].strip()
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                        except Exception:
                            continue
                        if not isinstance(chunk, dict):
                            continue
                        choices = chunk.get("choices")
                        if not isinstance(choices, list) or not choices:
                            continue
                        first = choices[0] if isinstance(choices[0], dict) else {}
                        if isinstance(first.get("finish_reason"), str):
                            finish_reason = first.get("finish_reason")
                        delta = first.get("delta")
                        if not isinstance(delta, dict):
                            delta = first.get("message") if isinstance(first.get("message"), dict) else {}

                        role = delta.get("role")
                        if isinstance(role, str) and role:
                            message["role"] = role

                        content = delta.get("content")
                        if isinstance(content, str) and content:
                            text_parts.append(content)
                            # 流式输出内容
                            if stream_callback:
                                stream_callback("content", content)

                        reasoning_content = delta.get("reasoning_content")
                        if isinstance(reasoning_content, str) and reasoning_content:
                            reasoning_parts.append(reasoning_content)
                            # 流式输出推理内容
                            if stream_callback:
                                stream_callback("reasoning", reasoning_content)

                        tool_calls_delta = delta.get("tool_calls")

                        if isinstance(tool_calls_delta, list) and tool_calls_delta:
                            for item in tool_calls_delta:
                                if not isinstance(item, dict):
                                    continue
                                idx_raw = item.get("index", 0)
                                idx = idx_raw if isinstance(idx_raw, int) else 0
                                entry = tool_call_map.get(idx)
                                if not isinstance(entry, dict):
                                    entry = {
                                        "type": "function",
                                        "function": {"name": None, "arguments": ""},
                                    }
                                    tool_call_map[idx] = entry

                                tool_id = item.get("id")
                                if isinstance(tool_id, str) and tool_id:
                                    entry["id"] = tool_id

                                fn = item.get("function")
                                if isinstance(fn, dict):
                                    name = fn.get("name")
                                    if isinstance(name, str) and name:
                                        entry["function"]["name"] = name
                                        # 流式输出工具调用开始
                                        if idx not in tool_call_started and stream_callback:
                                            tool_call_started.add(idx)
                                            stream_callback("tool_call_start", name)
                                    args = fn.get("arguments")
                                    if isinstance(args, str):
                                        entry["function"]["arguments"] += args
                                        # 流式输出工具调用参数
                                        if stream_callback:
                                            stream_callback("tool_call_args", args)
                                    elif args is not None:
                                        entry["function"]["arguments"] += safe_json_dumps(args)
                                        if stream_callback:
                                            stream_callback("tool_call_args", safe_json_dumps(args))

                    content_text = "".join(text_parts)
                    if content_text:
                        message["content"] = content_text
                    reasoning_text = "".join(reasoning_parts)
                    if reasoning_text:
                        message["reasoning_content"] = reasoning_text

                    if tool_call_map:
                        tool_calls: list[dict[str, Any]] = []
                        for i in sorted(tool_call_map.keys()):
                            call = tool_call_map[i]
                            if isinstance(call, dict):
                                tool_calls.append(call)
                        if tool_calls:
                            message["tool_calls"] = tool_calls

                    final_payload: dict[str, Any] = {
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "message": message,
                                "finish_reason": finish_reason,
                            }
                        ],
                    }
                    if run_logger is not None:
                        run_logger.log_event(
                            "http_response",
                            {
                                "ok": True,
                                "status": int(status) if isinstance(status, int) else None,
                                "reason": None,
                                "body": final_payload,
                            },
                            step=step,
                        )
                    Console.debug(
                        safe_json_dumps(
                            {
                                "event": "llm_http_response_summary",
                                "step": step,
                                "status": int(status) if isinstance(status, int) else None,
                                "finish_reason": finish_reason,
                                "has_tool_calls": bool(tool_call_map),
                                "reasoning_preview": summarize(reasoning_text, 240),
                                "content_preview": summarize(content_text, 240),
                            }
                        )
                    )
                    return HttpResponse(
                        ok=True,
                        status=int(status) if isinstance(status, int) else None,
                        json=final_payload,
                        text=safe_json_dumps(final_payload),
                    )

                raw = resp.read()
                text = raw.decode("utf-8", errors="replace")
                parsed = None
                try:
                    parsed = json.loads(text)
                except Exception:
                    parsed = None
                if run_logger is not None:
                    run_logger.log_event(
                        "http_response",
                        {
                            "ok": True,
                            "status": int(status) if isinstance(status, int) else None,
                            "reason": None,
                            "body": parsed if isinstance(parsed, dict) else text,
                        },
                        step=step,
                    )
                Console.debug(
                    safe_json_dumps(
                        {
                            "event": "llm_http_response_summary",
                            "step": step,
                            "status": int(status) if isinstance(status, int) else None,
                            "json_parsed": isinstance(parsed, dict),
                            "has_reasoning_content": (
                                isinstance(parsed, dict)
                                and isinstance(parsed.get("choices"), list)
                                and bool(parsed.get("choices"))
                                and isinstance(parsed["choices"][0], dict)
                                and isinstance(parsed["choices"][0].get("message"), dict)
                                and isinstance(
                                    parsed["choices"][0]["message"].get("reasoning_content"),
                                    str,
                                )
                            ),
                            "body_preview": summarize(parsed if isinstance(parsed, dict) else text, 240),
                        }
                    )
                )
                return HttpResponse(
                    ok=True,
                    status=int(status) if isinstance(status, int) else None,
                    json=parsed if isinstance(parsed, dict) else None,
                    text=text,
                )
        except urllib.error.HTTPError as e:
            try:
                raw = e.read()
                text = raw.decode("utf-8", errors="replace")
            except Exception:
                text = str(e)
            parsed = None
            try:
                parsed = json.loads(text)
            except Exception:
                parsed = None
            if run_logger is not None:
                run_logger.log_event(
                    "http_response",
                    {
                        "ok": False,
                        "status": int(getattr(e, "code", 0)) if getattr(e, "code", None) is not None else None,
                        "reason": getattr(e, "reason", None) or str(e),
                        "body": parsed if isinstance(parsed, dict) else text,
                    },
                    step=step,
                )
            reason = getattr(e, "reason", None) or str(e)
            Console.debug(
                safe_json_dumps(
                    {
                        "event": "llm_http_error",
                        "step": step,
                        "status": int(getattr(e, "code", 0)) if getattr(e, "code", None) is not None else None,
                        "reason": str(reason),
                        "body_preview": summarize(parsed if isinstance(parsed, dict) else text, 300),
                    }
                )
            )
            Console.fatal(f"  模型接口请求失败：HTTP {getattr(e, 'code', 'unknown')}")
            Console.warn(f"   原因：{reason}")
            if getattr(e, 'code', 0) == 401:
                Console.warn("   提示：请检查 API Key 是否正确")
            elif getattr(e, 'code', 0) == 429:
                Console.warn("   提示：请求过于频繁，请稍后重试")
            elif getattr(e, 'code', 0) >= 500:
                Console.warn("   提示：模型服务端异常，请稍后重试")
            return HttpResponse(
                ok=False,
                status=int(getattr(e, "code", 0)) if getattr(e, "code", None) is not None else None,
                reason=reason,
                json=parsed if isinstance(parsed, dict) else None,
                text=text,
            )
        except Exception as e:
            if run_logger is not None:
                run_logger.log_event(
                    "http_response",
                    {
                        "ok": False,
                        "status": None,
                        "reason": str(e),
                        "body": None,
                    },
                    step=step,
                )
            Console.debug(
                safe_json_dumps(
                    {
                        "event": "llm_http_exception",
                        "step": step,
                        "error_type": type(e).__name__,
                        "reason": str(e),
                    }
                )
            )
            Console.fatal(f"  模型接口连接失败：{e}")
            Console.warn("   可能原因：")
            Console.warn("   1. 网络连接问题")
            Console.warn("   2. API 地址配置错误")
            Console.warn("   3. 模型服务未启动")
            return HttpResponse(ok=False, reason=str(e))
