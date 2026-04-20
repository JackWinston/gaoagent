from __future__ import annotations

from typing import Any, Callable

from gaoagent.core.runner.AuditLogger import AuditLogger
from gaoagent.core.runner.BaseRunner import BaseRunner, Decision, RunnerConfig, RunnerContext
from gaoagent.core.runner.Tooling import ToolCall, ToolRegistry

Planner = Callable[[RunnerContext], list[dict[str, Any]]]


class PlanRunner(BaseRunner):
    def __init__(
        self,
        *,
        tools: ToolRegistry | None = None,
        audit: AuditLogger | None = None,
        config: RunnerConfig | None = None,
        planner: Planner | None = None,
    ) -> None:
        super().__init__(mode="plan", tools=tools, audit=audit, config=config)
        self._planner = planner or self._default_planner

    def decide(self, ctx: RunnerContext) -> Decision:
        if ctx.plan is None:
            plan = self._planner(ctx)
            ctx.plan = plan
            ctx.memory["plan"] = plan
            ctx.memory.setdefault("plan_index", 0)
            return Decision(
                kind="internal",
                internal={"name": "planner", "input": {"question": ctx.question}, "output": {"plan_len": len(plan)}},
            )

        idx = int(ctx.memory.get("plan_index", 0))
        if idx >= len(ctx.plan):
            return Decision(kind="final", final="计划执行完成")

        item = ctx.plan[idx]
        ctx.memory["plan_index"] = idx + 1
        if not isinstance(item, dict):
            return Decision(kind="final", final=f"计划项格式错误：{repr(item)}")

        t = item.get("type")
        if t == "final":
            return Decision(kind="final", final=str(item.get("content") or ""))
        if t == "tool":
            name = str(item.get("name") or "")
            args = item.get("arguments") or {}
            if not isinstance(args, dict):
                args = {"value": args}
            return Decision(kind="tool", tool_call=ToolCall(name=name, arguments=args))

        return Decision(kind="internal", internal={"name": "plan_item", "input": item, "output": {"skipped": True}})

    def _default_planner(self, _ctx: RunnerContext) -> list[dict[str, Any]]:
        return [
            {
                "type": "final",
                "content": "PlanRunner 未配置 planner（LLM/规则）。请注入 planner(ctx)->plan 后再运行。",
            }
        ]
