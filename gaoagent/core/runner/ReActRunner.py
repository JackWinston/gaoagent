from __future__ import annotations

from typing import Any

from gaoagent.core.runner.BaseRunner import (
    BaseRunner,
    RunnerConfig,
    RunnerContext,
    RunResult,
    StepResult,
)

from gaoagent.core.runner.HttpClient import OpenAICompatibleHttpClient
from gaoagent.core.runner.Tooling import ToolCall, ToolRegistry, default_tool_registry
from gaoagent.core.runner.Utils import safe_json_dumps, parse_llm_response
from gaoagent.core.runner.PromptBuilder import build_system_prompt
from gaoagent.core.runner.FunctionCallProtocol import build_function_specs


class ReActRunner(BaseRunner):
    def __init__(
        self,
        *,
        config: RunnerConfig | None = None,
        tools: ToolRegistry | None = None,
    ) -> None:
        cfg = RunnerConfig(
            max_steps=(config.max_steps if config else 32),
            tools=(tools or (config.tools if config else None) or default_tool_registry()),
        )
        super().__init__(
            mode="react",
            runner_config=cfg,
        )

    def decide(self, ctx: RunnerContext) -> StepResult:
        return self._callLLM(ctx)

    def run(self, question: str, shared_memory: dict[str, Any] | None = None) -> RunResult:
        if question is None or not str(question).strip():
            return RunResult(success=False, error="Invalid question")

        self.runner_context = RunnerContext(step=0, history=[])

        # 添加系统提示词
        tool_names = (
            self.runner_config.tools.list_names() if self.runner_config.tools else []
        )
        self.runner_context.history.append(
            {
                "role": "system",
                "content": build_system_prompt(
                    mode=self.mode, tool_names=tool_names
                ),
            }
        )
        # 添加用户的提问
        self.runner_context.history.append({"role": "user", "content": question})

        for step in range(1, self.runner_config.max_steps + 1):
            # 更新上下文中的 step 信息
            self.runner_context.step = step

            now_step = self.decide(self.runner_context)

            if now_step.decision == "function_call":
                calls = now_step.function_call or []
                normalized_calls: list[dict] = []
                for idx, call in enumerate(calls):
                    call_obj = call if isinstance(call, dict) else {}
                    call_id_raw = call_obj.get("tool_call_id")
                    call_id = (
                        call_id_raw
                        if isinstance(call_id_raw, str) and call_id_raw.strip()
                        else f"call_{step}_{idx}"
                    )
                    fn_name = call_obj.get("name")
                    fn_args = call_obj.get("arguments", {})
                    if not isinstance(fn_args, dict):
                        fn_args = {}
                    normalized_calls.append(
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": fn_name if isinstance(fn_name, str) else "",
                                "arguments": safe_json_dumps(fn_args),
                            },
                            "_runtime_name": fn_name,
                            "_runtime_arguments": call_obj.get("arguments", {}),
                        }
                    )

                self.runner_context.history.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": call["id"],
                                "type": call["type"],
                                "function": call["function"],
                            }
                            for call in normalized_calls
                        ],
                    }
                )

                if not self.runner_config.tools:
                    return RunResult(success=False, error="No tool registry configured")

                for call in normalized_calls:
                    name = call.get("_runtime_name")
                    arguments = call.get("_runtime_arguments", {})

                    if not isinstance(name, str) or not name.strip():
                        observation = safe_json_dumps(
                            {
                                "success": False,
                                "error": {
                                    "type": "ValueError",
                                    "message": "tool name must be non-empty str",
                                },
                            }
                        )
                    elif not isinstance(arguments, dict):
                        observation = safe_json_dumps(
                            {
                                "success": False,
                                "error": {
                                    "type": "ValueError",
                                    "message": "tool arguments must be object",
                                },
                            }
                        )
                    else:
                        try:
                            observation = self.runner_config.tools.call(
                                self.runner_context, ToolCall(name=name, arguments=arguments)
                            )
                        except Exception as e:
                            observation = safe_json_dumps(
                                {
                                    "success": False,
                                    "error": {"type": type(e).__name__, "message": str(e)},
                                }
                            )

                    self.runner_context.history.append(
                        {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "content": (
                                observation
                                if isinstance(observation, str)
                                else safe_json_dumps(observation)
                            ),
                        }
                    )
                continue
            if now_step.decision == "thought":
                protocol = (
                    now_step.raw.get("protocol")
                    if isinstance(now_step.raw, dict)
                    else None
                )
                if isinstance(protocol, dict):
                    content = protocol.get("content")
                    if content is None:
                        content = protocol.get("output")
                    assistant_content = str(content) if content is not None else ""
                else:
                    assistant_content = now_step.content or ""
                self.runner_context.history.append(
                    {"role": "assistant", "content": assistant_content}
                )
                continue
            if now_step.decision == "final":
               
                protocol = now_step.raw.get("protocol") if isinstance(now_step.raw, dict) else None
                if isinstance(protocol, dict):
                    assistant_content = safe_json_dumps(protocol)
                else:
                    assistant_content = safe_json_dumps(
                        {"type": "final", "content": now_step.content or ""}
                    )
                self.runner_context.history.append(
                    {"role": "assistant", "content": assistant_content}
                )
                return RunResult(success=True, final_result=now_step.content)

        return RunResult(success=False, error="Max steps reached")

    def _callLLM(self, ctx: RunnerContext) -> StepResult:
        if not self.request_base_info:
            return StepResult(decision="final", content="No valid API configuration")

        client = OpenAICompatibleHttpClient(
            base_url=self.request_base_info.baseurl,
            api_key=self.request_base_info.api_key,
        )
        tool_names = (
            self.runner_config.tools.list_names() if self.runner_config.tools else []
        )
        tools = build_function_specs(tool_names) if tool_names else None
        tool_choice = "auto" if tools else None
        response = client.post_chat_completions(
            model=self.request_base_info.modules,
            messages=ctx.history,
            tools=tools,
            tool_choice=tool_choice,
        )
        return parse_llm_response(response)
