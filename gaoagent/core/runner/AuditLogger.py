from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gaoagent.core.runner.Utils import now_ms


class AuditLogger:
    def __init__(self, audit_path: Path) -> None:
        self._path = audit_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fallback_path: Path | None = None

    @property
    def path(self) -> Path:
        return self._path

    def write(self, record: dict[str, Any]) -> None:
        payload = dict(record)
        payload["ts_ms"] = payload.get("ts_ms") or now_ms()
        line = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        try:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
            return
        except OSError:
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with self._path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
                return
            except OSError:
                pass

        if self._fallback_path is None:
            fallback_dir = Path.cwd() / ".gaoagent"
            fallback_dir.mkdir(parents=True, exist_ok=True)
            self._fallback_path = fallback_dir / "audit.jsonl"

        try:
            with self._fallback_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            return


def default_audit_path() -> Path:
    config_dir = Path.home() / ".gaoagent"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir / "audit.jsonl"
