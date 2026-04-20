from __future__ import annotations

from typing import Any, Callable

from gaoagent.core.runner.ApiConfig import default_api_config_path, load_api_config, select_api_and_model
from gaoagent.core.runner.AuditLogger import AuditLogger
from gaoagent.core.runner.BaseRunner import BaseRunner, Decision, RunnerConfig, RunnerContext
from gaoagent.core.runner.FunctionCallProtocol import (
    build_function_specs,
    http_error_to_final,
    map_chat_completion_to_protocol,
    protocol_to_decision,
)
from gaoagent.core.runner.HttpClient import OpenAICompatibleHttpClient
from gaoagent.core.runner.PromptBuilder import build_messages
from gaoagent.core.runner.Tooling import ToolRegistry
from gaoagent.core.runner.Utils import truncate_text

Policy = Callable[[RunnerContext], Decision]


class ReActRunner(BaseRunner):
    def __init__(
        self,
        *,
        tools: ToolRegistry | None = None,
        audit: AuditLogger | None = None,
        config: RunnerConfig | None = None,
        policy: Policy | None = None,
    ) -> None:
        super().__init__(mode="react", tools=tools, audit=audit, config=config)
        self._policy = policy or self._default_policy

    def decide(self, ctx: RunnerContext) -> Decision:
        return self._policy(ctx)

    def _default_policy(self, ctx: RunnerContext) -> Decision:
        tool_names = self._tools.list_names()
        messages = build_messages(ctx, tool_names=tool_names)
        tools = build_function_specs(tool_names)
        llm_raw = self._call_llm_function_call(ctx=ctx, messages=messages, tools=tools)
        if llm_raw is None:
            return Decision(
                kind="final",
                final="ReActRunner 的 LLM 调用尚未实现。请实现 _call_llm_function_call 后再运行。",
            )
        return protocol_to_decision(ctx.step, set(tool_names), llm_raw)

    def _call_llm_function_call(
        self,
        *,
        ctx: RunnerContext,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        try:
            config_payload = load_api_config(default_api_config_path())
            selection = select_api_and_model(config_payload, ctx.memory)
        except FileNotFoundError as e:
            return {"type": "final", "content": f"未找到 API 配置文件：{e}"}
        except KeyError as e:
            return {"type": "final", "content": str(e)}
        except Exception as e:
            return {"type": "final", "content": f"读取/选择 API 配置失败：{e}"}

        client = OpenAICompatibleHttpClient(base_url=selection.base_url, api_key=selection.api_key, timeout_s=60)
        url = client.build_chat_completions_url()
        req_payload: dict[str, Any] = {
            "model": selection.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": 0.2,
        }
        resp = client.post_json(url, req_payload)
        if not resp.ok:
            body = resp.text or ""
            if resp.status is not None:
                return http_error_to_final(resp.status, resp.reason or "", body)
            return {"type": "final", "content": f"LLM 请求失败：{resp.reason}"}

        if resp.json is None:
            return {"type": "final", "content": f"LLM 返回非 JSON：{truncate_text(resp.text or '', 500)}"}

        return map_chat_completion_to_protocol(resp.json)
