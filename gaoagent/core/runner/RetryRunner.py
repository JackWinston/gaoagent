from __future__ import annotations

import json
from typing import Any, Callable

import click

from gaoagent.core.runner.ApiConfig import default_api_config_path, load_api_config, select_api_and_model
from gaoagent.core.runner.AuditLogger import AuditLogger, default_audit_path
from gaoagent.core.runner.BaseRunner import RunnerConfig, RunnerContext, RunnerResult
from gaoagent.core.runner.FunctionCallProtocol import http_error_to_final
from gaoagent.core.runner.HttpClient import OpenAICompatibleHttpClient
from gaoagent.core.runner.PromptBuilder import build_messages
from gaoagent.core.runner.ReActRunner import ReActRunner
from gaoagent.core.runner.Tooling import ToolRegistry, default_tool_registry
from gaoagent.core.runner.Utils import normalize_exception, redact, safe_json_dumps, summarize, truncate_text

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
        memory = shared_memory if shared_memory is not None else {}
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
                if result.ok:
                    return result

                err = result.error or {"type": "RuntimeError", "message": "unknown error"}
                reflection = self._reflector(ctx, err)
                patch = reflection.get("memory_patch") if isinstance(reflection, dict) else None
                if isinstance(patch, dict):
                    for k, v in patch.items():
                        memory[k] = v
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
        def extract_message_text(payload: dict[str, Any]) -> str:
            choices = payload.get("choices")
            if not isinstance(choices, list) or not choices:
                return ""
            first = choices[0] if isinstance(choices[0], dict) else {}
            message = first.get("message") if isinstance(first, dict) else None
            if not isinstance(message, dict):
                return ""
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                texts: list[str] = []
                for item in content:
                    if isinstance(item, dict):
                        text = item.get("text")
                        if isinstance(text, str):
                            texts.append(text)
                return "\n".join(texts)
            return ""

        def strip_code_fence(text: str) -> str:
            s = (text or "").strip()
            if s.startswith("```"):
                lines = s.splitlines()
                if len(lines) >= 2 and lines[0].startswith("```"):
                    i = 1
                    while i < len(lines) and lines[i].strip() == "":
                        i += 1
                    if i < len(lines) and lines[-1].strip() == "```":
                        return "\n".join(lines[i:-1]).strip()
            return s

        base = {"attempt": ctx.attempt, "error_type": err.get("type"), "strategy": "no-op"}
        try:
            ctx.last_error = err
            tool_names = self._tools.list_names()
            messages = build_messages(ctx, tool_names=tool_names, mode="retry")

            config_payload = load_api_config(default_api_config_path())
            selection = select_api_and_model(config_payload, ctx.memory)
            netlog_path = None
            if isinstance(ctx.memory, dict):
                netlog_path = ctx.memory.get("netlog_path")
            client = OpenAICompatibleHttpClient(
                base_url=selection.base_url,
                api_key=selection.api_key,
                timeout_s=60,
                network_log_path=netlog_path,
            )
            url = client.build_chat_completions_url()
            req_payload: dict[str, Any] = {"model": selection.model, "messages": messages, "temperature": 0.2}
            resp = client.post_json(url, req_payload)
            if not resp.ok:
                body = resp.text or ""
                if resp.status is not None:
                    final = http_error_to_final(resp.status, resp.reason or "", body)
                    base["note"] = str(final.get("content") or "")
                    return base
                base["note"] = f"LLM 请求失败：{resp.reason}"
                return base

            if resp.json is None:
                base["note"] = f"LLM 返回非 JSON：{truncate_text(resp.text or '', 500)}"
                return base

            raw_text = extract_message_text(resp.json)
            if not raw_text.strip():
                base["note"] = f"LLM 未返回可解析内容：{summarize(resp.json)}"
                return base

            raw_text = strip_code_fence(raw_text)
            try:
                parsed = json.loads(raw_text)
            except Exception:
                base["note"] = f"LLM 反思结果不是合法 JSON：{truncate_text(raw_text, 800)}"
                return base

            if isinstance(parsed, dict) and isinstance(parsed.get("reflection"), dict):
                parsed = parsed.get("reflection")

            if not isinstance(parsed, dict):
                base["note"] = f"LLM 反思结果必须是对象：{summarize(parsed)}"
                return base

            strategy = parsed.get("strategy")
            if isinstance(strategy, str) and strategy.strip():
                base["strategy"] = strategy.strip()
            else:
                base["strategy"] = "adjust-and-retry"

            memory_patch = parsed.get("memory_patch")
            if memory_patch is None:
                return base
            if not isinstance(memory_patch, dict):
                base["note"] = f"LLM memory_patch 必须是对象：{summarize(memory_patch)}"
                return base

            safe_patch: dict[str, Any] = {}
            for k, v in memory_patch.items():
                if not isinstance(k, str) or not k.strip():
                    continue
                safe_patch[k] = v
            if safe_patch:
                base["memory_patch"] = safe_patch

            note = parsed.get("note")
            if isinstance(note, str) and note.strip():
                base["note"] = note.strip()
            else:
                base["note"] = safe_json_dumps({"selected_api": getattr(selection, "api_name", ""), "selected_model": getattr(selection, "model", "")})
            return base
        except FileNotFoundError as e:
            base["note"] = f"未找到 API 配置文件：{e}"
            return base
        except Exception as e:
            base["note"] = f"{type(e).__name__}: {e}"
            return base
