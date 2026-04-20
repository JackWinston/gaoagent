from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import click

from gaoagent.core.runner.BaseRunner import RunnerConfig
from gaoagent.core.runner.PlanRunner import PlanRunner
from gaoagent.core.runner.ReActRunner import ReActRunner
from gaoagent.core.runner.RetryRunner import RetryRunner
from gaoagent.core.runner.Tooling import ToolRegistry, default_tool_registry
from gaoagent.core.runner.Utils import find_project_root, now_ms, truncate_text


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

        shared_memory: dict[str, Any] = {}
        task_ts = now_ms()
        root = find_project_root(Path.cwd())
        log_dir = root / ".gaoagent" / "log"
        log_dir.mkdir(parents=True, exist_ok=True)
        ts_text = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = log_dir / f"{ts_text}.log"
        if log_path.exists():
            i = 1
            while i < 1000:
                candidate = log_dir / f"{ts_text}_{i:03d}.log"
                if not candidate.exists():
                    log_path = candidate
                    break
                i += 1
        shared_memory["netlog_path"] = str(log_path)
        try:
            with log_path.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "ts_ms": task_ts,
                            "event": "task_start",
                            "mode": m,
                            "question": truncate_text(str(question or ""), 4000),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        except OSError:
            pass

        if m == "plan":
            result = PlanRunner(config=self._cfg, tools=self._tools).run(question, shared_memory=shared_memory)
        elif m == "retry":
            result = RetryRunner(config=self._cfg, tools=self._tools).run(question, shared_memory=shared_memory)
        else:
            result = ReActRunner(config=self._cfg, tools=self._tools).run(question, shared_memory=shared_memory)

        if result.ok:
            if result.final:
                if not self._cfg.console:
                    click.echo(result.final)
            return

        err = result.error or {"type": "RuntimeError", "message": "unknown error"}
        click.echo(f"任务失败：{err.get('type')}: {err.get('message')}")
