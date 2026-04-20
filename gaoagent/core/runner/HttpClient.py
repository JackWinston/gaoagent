from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterable, Iterator


@dataclass(frozen=True)
class HttpResponse:
    ok: bool
    status: int | None = None
    reason: str | None = None
    json: dict[str, Any] | None = None
    text: str | None = None


class OpenAICompatibleHttpClient:
    def __init__(self, *, base_url: str, api_key: str, timeout_s: int = 60) -> None:
        self._base_url = base_url
        self._api_key = api_key
        self._timeout_s = timeout_s

    def build_chat_completions_url(self) -> str:
        clean = self._base_url.strip().rstrip("/")
        if clean.endswith("/chat/completions"):
            return clean
        if clean.endswith("/v1"):
            return f"{clean}/chat/completions"
        return f"{clean}/v1/chat/completions"

    def _iter_sse_data_events(self, chunks: Iterable[bytes]) -> Iterator[str]:
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
        req_payload = dict(payload)
        req_payload.setdefault("stream", True)
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
