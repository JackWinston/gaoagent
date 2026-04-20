from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


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

    def post_json(self, url: str, payload: dict[str, Any]) -> HttpResponse:
        req_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            url=url,
            data=req_bytes,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
        )

        try:
            with urllib.request.urlopen(request, timeout=self._timeout_s) as response:
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
