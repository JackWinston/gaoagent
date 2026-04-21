from __future__ import annotations

import os
import threading
from contextvars import ContextVar
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from gaoagent.core.runner.Utils import find_project_root


class RunLogger:
    def __init__(self, file_path: Path) -> None:
        self.file_path = file_path
        self._lock = threading.Lock()

    def log_event(self, event_type: str, payload: Any, *, step: int | None = None) -> None:
        record: dict[str, Any] = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "event": event_type,
        }
        if step is not None:
            record["step"] = step
        record["data"] = _to_jsonable(payload)
        line = _safe_json_dumps(record)
        with self._lock:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            with self.file_path.open("a", encoding="utf-8") as f:
                f.write(line)
                f.write("\n\n")


_CURRENT_RUN_LOGGER: ContextVar[RunLogger | None] = ContextVar("gaoagent_run_logger", default=None)


def get_current_run_logger() -> RunLogger | None:
    return _CURRENT_RUN_LOGGER.get()


def set_current_run_logger(logger: RunLogger | None):
    return _CURRENT_RUN_LOGGER.set(logger)


def reset_current_run_logger(token) -> None:
    _CURRENT_RUN_LOGGER.reset(token)


def create_run_logger() -> RunLogger:
    project_root = find_project_root()
    logs_dir = project_root / ".gaoagent" / "logs"
    ts = datetime.now().strftime("%Y-%m-%d,%H:%M:%S")
    if os.name == "nt":
        ts = ts.replace(":", "-")
    file_path = logs_dir / f"{ts}.log"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.touch(exist_ok=True)
    return RunLogger(file_path)


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        try:
            return {k: _to_jsonable(v) for k, v in asdict(value).items()}
        except Exception:
            return repr(value)
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            key = str(k)
            out[key] = _to_jsonable(v)
        return out
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(x) for x in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _safe_json_dumps(value: Any) -> str:
    import json

    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except Exception:
        return repr(value)

