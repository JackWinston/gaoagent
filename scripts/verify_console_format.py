from __future__ import annotations

from gaoagent.core.runner.BaseRunner import BaseRunner, Decision, RunnerConfig, RunnerContext
from gaoagent.core.runner.Tooling import ToolCall, default_tool_registry


class DemoRunner(BaseRunner):
    def __init__(self) -> None:
        super().__init__(mode="react", tools=default_tool_registry(), config=RunnerConfig(max_steps=5, console=True))

    def decide(self, ctx: RunnerContext) -> Decision:
        if ctx.step == 1:
            return Decision(kind="thought", internal={"name": "thought", "output": "我需要先思考一下"})
        if ctx.step == 2:
            return Decision(kind="tool", tool_call=ToolCall(name="list_dir", arguments={"path": "."}))
        return Decision(kind="final", final="完成")


if __name__ == "__main__":
    DemoRunner().run("demo", shared_memory={})
