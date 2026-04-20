from __future__ import annotations

from typing import Any, Callable

import click

from gaoagent.core.runner.AuditLogger import AuditLogger, default_audit_path
from gaoagent.core.runner.BaseRunner import RunnerConfig, RunnerContext, RunnerResult
from gaoagent.core.runner.ReActRunner import ReActRunner
from gaoagent.core.runner.Tooling import ToolRegistry, default_tool_registry
from gaoagent.core.runner.Utils import normalize_exception, redact, summarize

Reflector = Callable[[RunnerContext, dict[str, Any]], dict[str, Any]]


class RetryRunner:
    def __init__(
        self,
        *,
        inner_factory: Callable[[AuditLogger, RunnerConfig, ToolRegistry], Any] | None = None,
        tools: ToolRegistry | None = None,
        audit: AuditLogger | None = None,
        config: RunnerConfig | None = None,
        reflector: Reflector | None = None,
    ) -> None:
        self._cfg = config or RunnerConfig()
        self._tools = tools or default_tool_registry()
        audit_path = self._cfg.audit_path or default_audit_path()
        self._audit = audit or AuditLogger(audit_path)
        self._inner_factory = inner_factory or (lambda a, c, t: ReActRunner(audit=a, config=c, tools=t))
        self._reflector = reflector or self._default_reflector

    def run(self, question: str, shared_memory: dict[str, Any] | None = None) -> RunnerResult:
        memory = shared_memory or {}
        for attempt in range(0, self._cfg.max_retries + 1):
            ctx = RunnerContext(
                question=str(question).strip(),
                mode="retry",
                memory=memory,
                attempt=attempt,
                step=attempt + 1,
            )
            try:
                self._audit.write(
                    {
                        "mode": "retry",
                        "step": ctx.step,
                        "attempt": ctx.attempt,
                        "tool": "attempt",
                        "input_summary": summarize({"question": ctx.question}),
                        "output_summary": "",
                        "elapsed_ms": 0,
                        "status": "ok",
                        "error": None,
                    }
                )
                inner = self._inner_factory(self._audit, self._cfg, self._tools)
                result: RunnerResult = inner.run(question, shared_memory=memory)
                if self._cfg.console:
                    click.echo(f"[retry] attempt={attempt} ok={result.ok}")
                if result.ok:
                    return result

                err = result.error or {"type": "RuntimeError", "message": "unknown error"}
                reflection = self._reflector(ctx, err)
                memory.setdefault("retry_reflections", []).append(reflection)
                self._audit.write(
                    {
                        "mode": "retry",
                        "step": ctx.step,
                        "attempt": ctx.attempt,
                        "tool": "reflect",
                        "input_summary": summarize(redact(err)),
                        "output_summary": summarize(reflection),
                        "elapsed_ms": 0,
                        "status": "ok",
                        "error": None,
                    }
                )
            except Exception as e:
                err = normalize_exception(e)
                self._audit.write(
                    {
                        "mode": "retry",
                        "step": ctx.step,
                        "attempt": ctx.attempt,
                        "tool": "error",
                        "input_summary": "",
                        "output_summary": "",
                        "elapsed_ms": 0,
                        "status": "error",
                        "error": err,
                    }
                )
                return RunnerResult(ok=False, error=err)

        return RunnerResult(ok=False, error={"type": "RuntimeError", "message": f"max_retries reached: {self._cfg.max_retries}"})

    def _default_reflector(self, ctx: RunnerContext, err: dict[str, Any]) -> dict[str, Any]:
        return {"attempt": ctx.attempt, "error_type": err.get("type"), "strategy": "no-op"}

