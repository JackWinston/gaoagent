from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from gaoagent.core.CoreHandlers import CoreHandlers
from gaoagent.rag.RagHandlers import RagHandlers
from gaoagent.skills.SkillsHandlers import SkillsHandlers
from gaoagent.mcp.MCPHandlers import MCPHandlers

@dataclass(frozen=True)
class Route:
    factory: Callable[[], object]
    method_name: str

    def dispatch(self, **kwargs: object) -> None:
        target = self.factory()
        method = getattr(target, self.method_name)
        method(**kwargs)


ROUTES: dict[str, Route] = {
    "init": Route(factory=CoreHandlers, method_name="init"),
    "config": Route(factory=CoreHandlers, method_name="config"),
    "chat": Route(factory=CoreHandlers, method_name="chat"),
    "task": Route(factory=CoreHandlers, method_name="task"),

    "mcp.list": Route(factory=MCPHandlers, method_name="list"),
    "mcp.add": Route(factory=MCPHandlers, method_name="add"),
    "mcp.remove": Route(factory=MCPHandlers, method_name="remove"),
    "mcp.enable": Route(factory=MCPHandlers, method_name="enable"),
    "mcp.disable": Route(factory=MCPHandlers, method_name="disable"),
    "mcp.test": Route(factory=MCPHandlers, method_name="test"),

    "skills.list": Route(factory=SkillsHandlers, method_name="list"),
    "skills.install": Route(factory=SkillsHandlers, method_name="install"),
    "skills.uninstall": Route(factory=SkillsHandlers, method_name="uninstall"),

    "rag.list": Route(factory=RagHandlers, method_name="list"),
    "rag.add": Route(factory=RagHandlers, method_name="add"),
    "rag.remove": Route(factory=RagHandlers, method_name="remove"),
    "rag.search": Route(factory=RagHandlers, method_name="search"),
}


def dispatch(action: str, **kwargs: object) -> None:
    route = ROUTES.get(action)
    if route is None:
        raise KeyError(f"Unknown action: {action}")
    route.dispatch(**kwargs)
