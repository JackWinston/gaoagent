from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


ToolHandler = Callable[[Any, dict[str, Any]], Any]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolHandler] = {}

    def register(self, name: str, handler: ToolHandler) -> None:
        if not name or not isinstance(name, str):
            raise ValueError("tool name must be non-empty str")
        self._tools[name] = handler

    def call(self, ctx: Any, call: ToolCall) -> Any:
        handler = self._tools.get(call.name)
        if handler is None:
            raise KeyError(f"tool not found: {call.name}")
        return handler(ctx, call.arguments)

    def list_names(self) -> list[str]:
        return sorted(self._tools.keys())


def default_tool_registry() -> ToolRegistry:
    tools = ToolRegistry()

    def _echo(_ctx: Any, args: dict[str, Any]) -> dict[str, Any]:
        return {"echo": args}

    tools.register("echo", _echo)
    return tools
