from __future__ import annotations

import click

from gaoagent.core.runner.BaseRunner import RunnerConfig
from gaoagent.core.runner.ReActRunner import ReActRunner
from gaoagent.core.runner.Tooling import ToolRegistry, default_tool_registry


class TaskRunner:
    def __init__(
        self,
        *,
        tools: ToolRegistry | None = None,
        config: RunnerConfig | None = None,
    ) -> None:
        self._cfg = config or RunnerConfig()
        self._tools = tools or default_tool_registry()

    def run(self, question: str, mode: str) -> None:
        """
        执行任务并将结果输出到终端。

        说明：
        - 该方法是“命令式输出”，而非返回结构化结果，便于 CLI 使用；
        - 若要在程序内二次封装，建议直接调用各 Runner.run 获得 RunnerResult。
        """
        m = (mode or "react").strip().lower()
        if m not in ("plan", "react", "retry"):
            m = "react"

        if m == "plan":
            #result = PlanRunner(config=self._cfg, tools=self._tools).run(question, shared_memory=shared_memory)
             result = ReActRunner(config=self._cfg, tools=self._tools).run(question, shared_memory=shared_memory)
        elif m == "retry":
            # result = RetryRunner(config=self._cfg, tools=self._tools).run(question, shared_memory=shared_memory)
            result = ReActRunner(config=self._cfg, tools=self._tools).run(question, shared_memory=shared_memory)
        else:
            result = ReActRunner(config=self._cfg, tools=self._tools).run(question, shared_memory=shared_memory)

        if result.ok:
            if result.final:
                if not self._cfg.console:
                    click.echo(result.final)
            return

        err = result.error or {"type": "RuntimeError", "message": "unknown error"}
        click.echo(f"任务失败：{err.get('type')}: {err.get('message')}")
