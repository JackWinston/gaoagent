from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Any


def now_ms() -> int:
    return int(time.time() * 1000)


def truncate_text(s: str, limit: int) -> str:
    if len(s) <= limit:
        return s
    return s[: max(0, limit - 1)] + "…"


def safe_json_dumps(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except Exception:
        return repr(value)


def summarize(value: Any, limit: int = 400) -> str:
    if value is None:
        return "null"
    if isinstance(value, (str, int, float, bool)):
        return truncate_text(str(value), limit)
    return truncate_text(safe_json_dumps(value), limit)


def redact(value: Any) -> Any:
    sensitive_keys = {"api_key", "apikey", "key", "token", "secret", "password", "authorization"}
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            if isinstance(k, str) and k.lower() in sensitive_keys:
                out[k] = "***"
            else:
                out[k] = redact(v)
        return out
    if isinstance(value, list):
        return [redact(x) for x in value]
    return value


def normalize_exception(e: BaseException) -> dict[str, Any]:
    return {
        "type": type(e).__name__,
        "message": str(e),
        "traceback": "".join(traceback.format_exception(type(e), e, e.__traceback__)),
    }


def find_project_root(start: Path | None = None) -> Path:
    p = (start or Path.cwd()).resolve()
    for cur in (p, *p.parents):
        if (cur / "pyproject.toml").exists():
            return cur
    return p
