from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from gaoagent.core.core_handlers import CoreHandlers
from gaoagent.rag.rag_handlers import RagHandlers
from gaoagent.skills.skills_handlers import SkillsHandlers
from gaoagent.mcp.mcp_handlers import MCPHandlers
from gaoagent.api.api_handlers import ApiHandlers
from gaoagent.agent.agent_handlers import AgentHandlers

@dataclass(frozen=True)
class Route:
    """路由描述对象。

    作用:
    - 将一个字符串 action 映射到具体处理器类与方法名。
    - 通过 `factory` 延迟创建处理器实例，避免模块导入时就初始化全部 Handler。
    """
    factory: Callable[[], object]
    method_name: str

    def dispatch(self, **kwargs: object) -> None:
        """执行当前路由。

        行为:
        - 调用 `factory()` 创建目标 Handler。
        - 通过 `method_name` 反射获取方法并透传 `kwargs` 执行。
        """
        target = self.factory()
        method = getattr(target, self.method_name)
        method(**kwargs)


ROUTES: dict[str, Route] = {
    "init": Route(factory=CoreHandlers, method_name="init"),
    "config": Route(factory=CoreHandlers, method_name="config"),
    "chat": Route(factory=CoreHandlers, method_name="chat"),
    "task": Route(factory=CoreHandlers, method_name="task"),
    "refresh": Route(factory=CoreHandlers, method_name="refresh"),

    "mcp.list": Route(factory=MCPHandlers, method_name="list"),
    "mcp.add": Route(factory=MCPHandlers, method_name="add"),
    "mcp.remove": Route(factory=MCPHandlers, method_name="remove"),
    "mcp.enable": Route(factory=MCPHandlers, method_name="enable"),
    "mcp.disable": Route(factory=MCPHandlers, method_name="disable"),
    "mcp.test": Route(factory=MCPHandlers, method_name="test"),

    "skills.list": Route(factory=SkillsHandlers, method_name="list"),
    "skills.add": Route(factory=SkillsHandlers, method_name="install"),
    "skills.remove": Route(factory=SkillsHandlers, method_name="uninstall"),

    "rag.list": Route(factory=RagHandlers, method_name="list"),
    "rag.add": Route(factory=RagHandlers, method_name="add"),
    "rag.update": Route(factory=RagHandlers, method_name="update"),
    "rag.remove": Route(factory=RagHandlers, method_name="remove"),
    "rag.search": Route(factory=RagHandlers, method_name="search"),


    "api.list": Route(factory=ApiHandlers, method_name="list"),
    "api.add": Route(factory=ApiHandlers, method_name="add"),
    "api.remove": Route(factory=ApiHandlers, method_name="remove"),
    "api.edit": Route(factory=ApiHandlers, method_name="edit"),
    "api.default": Route(factory=ApiHandlers, method_name="default"),

    "agent.list": Route(factory=AgentHandlers, method_name="list_agents"),
    "agent.add": Route(factory=AgentHandlers, method_name="add_agent"),
    "agent.remove": Route(factory=AgentHandlers, method_name="remove_agent"),
    "agent.serve": Route(factory=AgentHandlers, method_name="register_agent"),
}


def dispatch(action: str, **kwargs: object) -> None:
    """按 action 分发到对应业务处理器。

    参数:
    - `action`: 路由键，例如 `mcp.list`、`rag.add`。
    - `kwargs`: 透传给目标 Handler 方法的参数。

    异常:
    - 未注册 action 时抛出 `KeyError`，由上层 CLI/调用方决定如何展示错误。
    """
    route = ROUTES.get(action)
    if route is None:
        raise KeyError(f"Unknown action: {action}")
    route.dispatch(**kwargs)
