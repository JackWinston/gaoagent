from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import click

from gaoagent.core.runner.AuditLogger import AuditLogger, default_audit_path
from gaoagent.core.runner.Tooling import ToolCall, ToolRegistry, default_tool_registry
from gaoagent.core.runner.Utils import normalize_exception, redact, safe_json_dumps, summarize, truncate_text

Mode = Literal["plan", "react", "retry"]


@dataclass
class RunnerConfig:
    max_steps: int = 32
    max_retries: int = 2
    audit_path: Path | None = None
    console: bool = True


@dataclass
class RunnerContext:
    question: str
    mode: Mode
    plan: list[dict[str, Any]] | None = None
    memory: dict[str, Any] = field(default_factory=dict)
    last_observation: Any | None = None
    last_observation_raw: Any | None = None
    last_error: dict[str, Any] | None = None
    step: int = 0
    attempt: int = 0


@dataclass(frozen=True)
class RunnerResult:
    ok: bool
    final: str | None = None
    error: dict[str, Any] | None = None


@dataclass(frozen=True)
class Decision:
    kind: Literal["tool", "final", "thought", "stop"]
    tool_call: ToolCall | None = None
    final: str | None = None
    internal: dict[str, Any] | None = None


class BaseRunner:
    def __init__(
        self,
        *,
        mode: Mode,
        tools: ToolRegistry | None = None,
        audit: AuditLogger | None = None,
        config: RunnerConfig | None = None,
    ) -> None:
        self._mode: Mode = mode
        self._tools = tools or default_tool_registry()
        cfg = config or RunnerConfig()
        audit_path = cfg.audit_path or default_audit_path()
        self._audit = audit or AuditLogger(audit_path)
        self._cfg = cfg

    def decide(self, ctx: RunnerContext) -> Decision:
        raise NotImplementedError()

    def run(self, question: str, shared_memory: dict[str, Any] | None = None) -> RunnerResult:
        if question is None or not str(question).strip():
            return RunnerResult(ok=False, error={"type": "ValueError", "message": "question is empty"})

        ctx = RunnerContext(question=str(question).strip(), mode=self._mode, memory=shared_memory if shared_memory is not None else {})
        for step in range(1, self._cfg.max_steps + 1):
            ctx.step = step
            t0 = time.perf_counter()
            tool_name = ""
            input_summary = ""
            output_summary = ""
            err: dict[str, Any] | None = None
            status: Literal["ok", "error", "stop"] = "ok"
            try:
                decision = self.decide(ctx)
                if decision.kind == "stop":
                    status = "stop"
                    tool_name = "stop"
                    output_summary = "stopped"
                    self._write_step(ctx, tool_name, input_summary, output_summary, status, err, t0)
                    return RunnerResult(ok=True, final=None)

                if decision.kind == "final":
                    status = "stop"
                    tool_name = "final"
                    input_summary = summarize({"question": ctx.question})
                    output_summary = summarize(decision.final)
                    self._write_step(ctx, tool_name, input_summary, output_summary, status, err, t0)
                    return RunnerResult(ok=True, final=decision.final)

                if decision.kind == "internal":
                    tool_name = (decision.internal or {}).get("name") or "internal"
                    input_summary = summarize(redact((decision.internal or {}).get("input")))
                    output_summary = summarize((decision.internal or {}).get("output"))
                    self._write_step(ctx, tool_name, input_summary, output_summary, status, err, t0)
                    continue
                
                if decision.kind == "thought":
                    tool_name = (decision.internal or {}).get("name") or "thought"
                    input_summary = summarize(redact((decision.internal or {}).get("input")))
                    output_summary = summarize((decision.internal or {}).get("output"))
                    self._write_step(ctx, tool_name, input_summary, output_summary, status, err, t0)
                    continue

                if decision.kind != "tool" or decision.tool_call is None:
                    status = "error"
                    tool_name = "invalid_decision"
                    err = {"type": "RuntimeError", "message": "invalid decision"}
                    self._write_step(ctx, tool_name, input_summary, output_summary, status, err, t0)
                    return RunnerResult(ok=False, error=err)

                tool_name = decision.tool_call.name
                input_summary = summarize(redact(decision.tool_call.arguments))
                raw_out = self._tools.call(ctx, decision.tool_call)
                ctx.last_observation = raw_out
                memory_messages = ctx.memory.get("messages") if isinstance(ctx.memory, dict) else None
                if isinstance(memory_messages, list):
                    tool_msg: dict[str, Any] = {"role": "user"}
                    if isinstance(decision.tool_call.tool_call_id, str) and decision.tool_call.tool_call_id.strip():
                        tool_msg["tool_call_id"] = decision.tool_call.tool_call_id
                    else:
                        tool_msg["name"] = decision.tool_call.name
            
                    tool_msg["content"] = f'{ "type": "observation", "content": "{raw_out}" }'
                    tool_msg["type"] = "observation"
                    memory_messages.append(tool_msg)

                output_summary = summarize(raw_out)
                self._write_step(ctx, tool_name, input_summary, output_summary, status, err, t0)
            except Exception as e:
                status = "error"
                err = normalize_exception(e)
                ctx.last_error = err
                if not tool_name:
                    tool_name = "error"
                self._write_step(ctx, tool_name, input_summary, output_summary, status, err, t0)
                return RunnerResult(ok=False, error=err)

        err = {"type": "RuntimeError", "message": f"max_steps reached: {self._cfg.max_steps}"}
        ctx.last_error = err
        self._write_step(ctx, "max_steps", "", "", "error", err, time.perf_counter())
        return RunnerResult(ok=False, error=err)

    def _write_step(
        self,
        ctx: RunnerContext,
        tool: str,
        input_summary: str,
        output_summary: str,
        status: Literal["ok", "error", "stop"],
        err: dict[str, Any] | None,
        t0: float,
    ) -> None:
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        record = {
            "mode": ctx.mode,
            "step": ctx.step,
            "attempt": ctx.attempt,
            "tool": tool,
            "input_summary": input_summary,
            "output_summary": output_summary,
            "elapsed_ms": elapsed_ms,
            "status": status,
            "error": err,
        }
        self._audit.write(record)
        if self._cfg.console:
            click.echo(f"[{ctx.mode}] step={ctx.step} tool={tool} status={status} ms={elapsed_ms}")
            if status == "error" and err:
                click.echo(truncate_text(err.get("message", ""), 400))
