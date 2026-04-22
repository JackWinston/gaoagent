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
from gaoagent.core.runner.Utils import (
    load_mcp_servers_raw,
    load_mcp_tools_cache,
    parse_llm_response,
    safe_json_dumps,
    write_mcp_tools_cache_for_current_scope,
)
from gaoagent.core.runner.PromptBuilder import build_system_prompt
from gaoagent.core.runner.FunctionCallProtocol import build_function_specs
from gaoagent.core.runner.RunLogger import get_current_run_logger
from gaoagent.mcp.MCPClientCompat import MCPStdioClientSync, build_mcp_tools_cache_payload


class ReActRunner(BaseRunner):
    @staticmethod
    def _enabled_mcp_servers(mcp_servers_raw: dict[str, Any] | None) -> dict[str, Any]:
        enabled: dict[str, Any] = {}
        if not isinstance(mcp_servers_raw, dict):
            return enabled
        for server_name, body in mcp_servers_raw.items():
            if not isinstance(server_name, str) or not isinstance(body, dict):
                continue
            if body.get("disabled") is True:
                continue
            enabled[server_name] = body
        return enabled

    @staticmethod
    def _filter_exported_map_for_servers(
        exported_map: dict[str, Any] | None,
        mcp_servers: dict[str, Any],
    ) -> dict[str, Any]:
        filtered: dict[str, Any] = {}
        if not isinstance(exported_map, dict):
            return filtered
        valid_servers = set(mcp_servers.keys())
        for exported_name, meta in exported_map.items():
            if not isinstance(exported_name, str) or not isinstance(meta, dict):
                continue
            server = meta.get("server")
            tool = meta.get("tool")
            if not isinstance(server, str) or server not in valid_servers:
                continue
            if not isinstance(tool, str) or not tool.strip():
                continue
            filtered[exported_name] = meta
        return filtered

    def __init__(
        self,
        *,
        config: RunnerConfig | None = None,
        tools: ToolRegistry | None = None,
    ) -> None:
        cfg = RunnerConfig(
            max_steps=(config.max_steps if config else 32),
            tools=(tools or (config.tools if config else None) or default_tool_registry()),
            llm_invalid_retry=(config.llm_invalid_retry if config else 2),
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

        # MCP 工具发现策略：
        # 1) 优先读取配置阶段写入的工具缓存（最快、最稳定）。
        # 2) 若缓存缺失但存在 mcpServers，则在运行时临时拉取一次工具清单作为兜底。
        #    这样即便用户忘记重新执行 config，也能在本次运行中使用 MCP 工具。
        mcp_servers_all = load_mcp_servers_raw()
        mcp_servers_raw = self._enabled_mcp_servers(mcp_servers_all)
        mcp_cache = load_mcp_tools_cache() or {}
        cached_exported_map = (
            mcp_cache.get("exported_map")
            if isinstance(mcp_cache.get("exported_map"), dict)
            else {}
        )
        mcp_exported_map = self._filter_exported_map_for_servers(cached_exported_map, mcp_servers_raw)
        mcp_discovery_errors: dict[str, str] = {}
        configured_servers = set(mcp_servers_raw.keys())
        mapped_servers = {
            meta.get("server")
            for meta in mcp_exported_map.values()
            if isinstance(meta, dict) and isinstance(meta.get("server"), str)
        }
        need_discovery = bool(configured_servers) and (
            (not mcp_exported_map) or (mapped_servers != configured_servers)
        )
        if need_discovery and isinstance(mcp_servers_raw, dict) and mcp_servers_raw:
            try:
                payload = build_mcp_tools_cache_payload(
                    mcp_servers_raw,
                    connect_and_list_tools=lambda name, body: MCPStdioClientSync.from_config(
                        server_name=name,
                        config=body,
                    ).list_tools(),
                    generated_at="runtime",
                )
                mcp_exported_map = (
                    payload.get("exported_map")
                    if isinstance(payload.get("exported_map"), dict)
                    else {}
                )
                mcp_exported_map = self._filter_exported_map_for_servers(mcp_exported_map, mcp_servers_raw)
                servers_payload = (
                    payload.get("servers")
                    if isinstance(payload.get("servers"), dict)
                    else {}
                )
                if isinstance(servers_payload, dict):
                    for server_name, server_body in servers_payload.items():
                        if isinstance(server_name, str) and isinstance(server_body, dict):
                            error = server_body.get("error")
                            if isinstance(error, str) and error.strip():
                                mcp_discovery_errors[server_name] = error
                if mcp_exported_map:
                    write_mcp_tools_cache_for_current_scope(payload)
            except Exception:
                mcp_exported_map = {}
                mcp_discovery_errors["_runtime"] = "build_mcp_tools_cache_payload failed"

        self._mcp_servers_raw = mcp_servers_raw
        self._mcp_exported_map = mcp_exported_map

        # 明确失败：有 MCP 配置但没有任何 MCP 工具时，不再静默降级为“仅本地工具”。
        if isinstance(mcp_servers_raw, dict) and mcp_servers_raw and not mcp_exported_map:
            run_logger = get_current_run_logger()
            reason_payload = {
                "configured_servers": sorted(list(mcp_servers_raw.keys())),
                "discovery_errors": mcp_discovery_errors,
            }
            if run_logger is not None:
                run_logger.log_event(
                    "mcp_tools_unavailable",
                    reason_payload,
                    step=0,
                )
            return RunResult(
                success=False,
                error=(
                    "MCP 已配置但未加载到任何工具，请先修复 MCP 服务可用性。"
                    f" details={safe_json_dumps(reason_payload)}"
                ),
            )

        # 添加系统提示词
        tool_names = (self.runner_config.tools.list_names() if self.runner_config.tools else [])
        # 将 MCP 导出工具名加入可调用工具清单，避免与内置工具重名。
        if isinstance(mcp_exported_map, dict) and mcp_exported_map:
            tool_names = list(tool_names) + sorted([str(x) for x in mcp_exported_map.keys()])
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

            run_logger = get_current_run_logger()
            if run_logger is not None:
                run_logger.log_event("step_result", now_step, step=step)

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
                            # 路由顺序：
                            # 1) 先走内置 ToolRegistry（本地工具）
                            # 2) 再走 MCP 导出工具映射（远程/stdio 工具）
                            # 3) 两者都找不到则返回 Unknown tool
                            if self.runner_config.tools and name in self.runner_config.tools.list_names():
                                observation = self.runner_config.tools.call(
                                    self.runner_context, ToolCall(name=name, arguments=arguments)
                                )
                            elif isinstance(mcp_exported_map, dict) and name in mcp_exported_map:
                                mcp_meta = mcp_exported_map.get(name) or {}
                                server_name = mcp_meta.get("server")
                                tool_name = mcp_meta.get("tool")
                                server_cfg = (
                                    mcp_servers_raw.get(server_name)
                                    if isinstance(server_name, str) and isinstance(mcp_servers_raw, dict)
                                    else None
                                )
                                if not isinstance(server_name, str) or not isinstance(tool_name, str) or not isinstance(server_cfg, dict):
                                    observation = {
                                        "success": False,
                                        "error": {
                                            "type": "ValueError",
                                            "message": f"MCP tool 映射无效：name={name}",
                                        },
                                    }
                                else:
                                    # MCP 调用结果统一封装为 success/result 结构，便于模型侧稳定解析。
                                    observation = MCPStdioClientSync.from_config(
                                        server_name=server_name,
                                        config=server_cfg,
                                    ).call_tool(tool_name=tool_name, arguments=arguments)
                                    observation = {
                                        "success": True,
                                        "server": server_name,
                                        "tool": tool_name,
                                        "result": observation,
                                    }
                            else:
                                observation = {
                                    "success": False,
                                    "error": {
                                        "type": "ValueError",
                                        "message": f"Unknown tool: {name}",
                                    },
                                }
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
        # 只使用 run() 阶段已确定的 MCP 工具映射，确保“模型可见工具集”
        # 与“执行期可路由工具集”完全一致，避免出现 Unknown tool 偏差。
        mcp_exported_map = getattr(self, "_mcp_exported_map", None)
        if not isinstance(mcp_exported_map, dict):
            mcp_exported_map = {}
        all_tool_names = list(tool_names)
        if isinstance(mcp_exported_map, dict) and mcp_exported_map:
            all_tool_names += sorted([str(x) for x in mcp_exported_map.keys()])

        tools = (
            build_function_specs(all_tool_names, mcp_exported_map=mcp_exported_map)
            if all_tool_names
            else None
        )
        tool_choice = "auto" if tools else None
        max_attempts = max(1, 1 + int(getattr(self.runner_config, "llm_invalid_retry", 0) or 0))
        attempt = 0
        last_step: StepResult | None = None

        while attempt < max_attempts:
            attempt += 1
            response = client.post_chat_completions(
                model=self.request_base_info.modules,
                messages=ctx.history,
                tools=tools,
                tool_choice=tool_choice,
                step=ctx.step,
            )
            step_result = parse_llm_response(response)
            last_step = step_result

            should_retry = step_result.decision == "retry"
            if not should_retry:
                return step_result

            if attempt >= max_attempts:
                break

            run_logger = get_current_run_logger()
            if run_logger is not None:
                run_logger.log_event(
                    "llm_invalid_response_retry",
                    {
                        "attempt": attempt,
                        "max_attempts": max_attempts,
                        "content": step_result.content,
                        "raw": step_result.raw,
                    },
                    step=ctx.step,
                )

        raw = dict(last_step.raw) if last_step is not None and isinstance(last_step.raw, dict) else {}
        raw["_retry"] = {"attempts": max_attempts, "reason": "invalid_llm_output"}
        last_content = (last_step.content or "") if last_step is not None else ""
        return StepResult(
            decision="final",
            content=(
                "LLM 输出为空或不符合协议，已自动重试仍失败。"
                f" attempts={max_attempts}"
                + (f"\nlast={last_content}" if last_content else "")
            ),
            raw=raw,
        )
