from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from gaoagent.core.runner.Utils import safe_json_dumps


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    tool_call_id: str | None = None


ToolHandler = Callable[[Any, dict[str, Any]], Any]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolHandler] = {}

    def register(self, name: str, handler: ToolHandler) -> None:
        if not name or not isinstance(name, str):
            raise ValueError("tool name must be non-empty str")
        self._tools[name] = handler

    def call(self, ctx: Any, call: ToolCall) -> str:
        handler = self._tools.get(call.name)
        if handler is None:
            raise KeyError(f"tool not found: {call.name}")
        raw = handler(ctx, call.arguments)
        try:
            setattr(ctx, "last_observation_raw", raw)
        except Exception:
            pass
        content = raw if isinstance(raw, str) else safe_json_dumps(raw)
        return content

    def list_names(self) -> list[str]:
        return sorted(self._tools.keys())


def default_tool_registry() -> ToolRegistry:
    tools = ToolRegistry()

    def _list_dir(_ctx: Any, args: dict[str, Any]) -> dict[str, Any]:
        path = args.get("path", ".")
        if not isinstance(path, str) or not path.strip():
            return {
                "ok": False,
                "error": {
                    "type": "ValueError",
                    "message": "path must be non-empty str",
                },
            }
        try:
            p = Path(path).expanduser()
            if not p.is_absolute():
                p = (Path.cwd() / p).resolve()
            if not p.exists():
                return {
                    "success": False,
                    "error": {"type": "FileNotFoundError", "message": "path not found"},
                    "path": str(p),
                }
            if not p.is_dir():
                return {
                    "success": False,
                    "error": {
                        "type": "NotADirectoryError",
                        "message": "path is not a directory",
                    },
                    "path": str(p),
                }
            items: list[dict[str, Any]] = []
            for child in p.iterdir():
                try:
                    st = child.stat()
                    size = int(st.st_size)
                except Exception:
                    size = None
                items.append(
                    {
                        "name": child.name,
                        "path": str(child),
                        "is_dir": child.is_dir(),
                        "size": size,
                    }
                )
            items.sort(
                key=lambda x: (
                    not bool(x.get("is_dir")),
                    str(x.get("name") or "").lower(),
                )
            )
            return {"success": True, "path": str(p), "items": items}
        except Exception as e:
            return {
                "success": False,
                "error": {"type": type(e).__name__, "message": str(e)},
                "path": str(path),
            }

    def _read_file(_ctx: Any, args: dict[str, Any]) -> dict[str, Any]:
        path = args.get("path")
        encoding = args.get("encoding") or "utf-8"
        if not isinstance(path, str) or not path.strip():
            return {
                "success": False,
                "error": {
                    "type": "ValueError",
                    "message": "path must be non-empty str",
                },
            }
        if not isinstance(encoding, str) or not encoding.strip():
            return {
                "success": False,
                "error": {
                    "type": "ValueError",
                    "message": "encoding must be non-empty str",
                },
            }
        try:
            p = Path(path).expanduser()
            if not p.is_absolute():
                p = (Path.cwd() / p).resolve()
            content = p.read_text(encoding=encoding)
            return {
                "success": True,
                "path": str(p),
                "encoding": encoding,
                "content": content,
            }
        except Exception as e:
            return {
                "success": False,
                "error": {"type": type(e).__name__, "message": str(e)},
                "path": str(path),
            }

    def _ask_user(_ctx: Any, args: dict[str, Any]) -> str:
        prompt = args.get("prompt")
        default = args.get("default", None)
        choices = args.get("choices", None)
        if not isinstance(prompt, str) or not prompt.strip():
            return safe_json_dumps(
                {
                    "success": False,
                    "error": {
                        "type": "ValueError",
                        "message": "prompt must be non-empty str",
                    },
                }
            )
        if default is not None and not isinstance(default, str):
            default = str(default)
        if choices is not None and not isinstance(choices, list):
            return safe_json_dumps(
                {
                    "success": False,
                    "error": {
                        "type": "ValueError",
                        "message": "choices must be list[str]",
                    },
                }
            )
        if isinstance(choices, list) and any(not isinstance(x, str) for x in choices):
            return safe_json_dumps(
                {
                    "success": False,
                    "error": {
                        "type": "ValueError",
                        "message": "choices must be list[str]",
                    },
                }
            )
        try:
            typ = (
                click.Choice(choices) if isinstance(choices, list) and choices else str
            )
            click.echo(prompt.strip())
            answer = click.prompt("", default=default, type=typ, prompt_suffix="")
            return str(answer)
        except Exception as e:
            return safe_json_dumps(
                {
                    "success": False,
                    "error": {"type": type(e).__name__, "message": str(e)},
                    "prompt": str(prompt),
                }
            )

    def _write_file(_ctx: Any, args: dict[str, Any]) -> dict[str, Any]:
        path = args.get("path")
        content = args.get("content", "")
        encoding = args.get("encoding") or "utf-8"
        mkdirs = args.get("mkdirs", True)
        append = args.get("append", False)
        if not isinstance(path, str) or not path.strip():
            return {
                "success": False,
                "error": {
                    "type": "ValueError",
                    "message": "path must be non-empty str",
                },
            }
        if not isinstance(encoding, str) or not encoding.strip():
            return {
                "success": False,
                "error": {
                    "type": "ValueError",
                    "message": "encoding must be non-empty str",
                },
            }
        if not isinstance(mkdirs, bool):
            return {
                "success": False,
                "error": {"type": "ValueError", "message": "mkdirs must be bool"},
            }
        if not isinstance(append, bool):
            return {
                "success": False,
                "error": {"type": "ValueError", "message": "append must be bool"},
            }
        try:
            p = Path(path).expanduser()
            if not p.is_absolute():
                p = (Path.cwd() / p).resolve()
            if mkdirs:
                p.parent.mkdir(parents=True, exist_ok=True)
            text = content if isinstance(content, str) else str(content)
            mode = "a" if append else "w"
            with p.open(mode, encoding=encoding) as f:
                n = f.write(text)
            return {
                "success": True,
                "path": str(p),
                "encoding": encoding,
                "written_chars": n,
                "append": append,
            }
        except Exception as e:
            return {
                "success": False,
                "error": {"type": type(e).__name__, "message": str(e)},
                "path": str(path),
            }

    tools.register("list_dir", _list_dir)
    tools.register("read_file", _read_file)
    tools.register("ask_user", _ask_user)
    tools.register("write_file", _write_file)
    return tools
