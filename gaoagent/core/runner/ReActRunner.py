from __future__ import annotations

from typing import Any

from gaoagent.core.runner.BaseRunner import (
    BaseRunner,
    RunnerConfig,
    RunnerContext,
    RunResult,
    StepResult,
)

from gaoagent.core.runner.Console import Console
from gaoagent.core.runner.HttpClient import OpenAICompatibleHttpClient
from gaoagent.core.runner.Tooling import ToolCall, ToolRegistry, default_tool_registry
from gaoagent.core.runner.Utils import (
    build_multimodal_content,
    is_image_file,
    load_mcp_servers_raw,
    load_mcp_tools_cache,
    parse_llm_response,
    safe_json_dumps,
    summarize,
    write_mcp_tools_cache_for_current_scope,
)
from gaoagent.core.runner.PromptBuilder import build_system_prompt
from gaoagent.core.runner.FunctionCallProtocol import build_function_specs
from gaoagent.core.runner.RunLogger import get_current_run_logger
from gaoagent.mcp.MCPClientCompat import MCPStdioClientSync, build_mcp_tools_cache_payload


class ReActRunner(BaseRunner):
    """ReActRunner 类。
    
    职责:
    - 封装 ReAct 模式的业务能力与状态。
    - 提供 ReActRunner 语义下的方法集合，供上层流程协调调用。
    
    继承关系:
    - 基类: BaseRunner
    """
    @staticmethod
    def _enabled_mcp_servers(mcp_servers_raw: dict[str, Any] | None) -> dict[str, Any]:
        """_enabled_mcp_servers 方法。
        
        用途:
        - 从原始 MCP 服务器配置中筛选出已启用的服务器。
        - 筛选出的服务器配置将被用于 ReAct 模式的运行。
        
        参数:
        - mcp_servers_raw: 输入参数，用于控制该方法的处理行为。
        
        返回:
        - dict[str, Any]: 已启用的 MCP 服务器配置字典。
        """
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
        """_filter_exported_map_for_servers 方法。
        
        - 从导出的工具映射中筛选出与已启用的 MCP 服务器相关的工具。
        - 筛选出的工具映射将被用于 ReAct 模式的运行。
        
        参数:
        - exported_map: 输入参数，用于控制该方法的处理行为。
        - mcp_servers: 输入参数，用于控制该方法的处理行为。
        
        返回:
        - dict[str, Any]: 筛选出的工具映射字典。
        """
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

    @staticmethod
    def _print_live_llm_request(step: int, attempt: int, max_attempts: int) -> None:
        """在终端输出本轮 LLM 请求进度。"""
        Console.interaction(
            f"第{step}步: 正在请求数据..."
        )

    @staticmethod
    def _print_live_llm_response(
        step: int,
        decision: str,
        content: str | None = None,
        reasoning_content: str | None = None,
    ) -> None:
        """在终端输出本轮 LLM 返回摘要。"""
        if isinstance(reasoning_content, str) and reasoning_content.strip():
            Console.info(
                f"第{step}步: 推理内容 : {reasoning_content.strip()}"
            )
        if decision == "tool_calls":
            Console.info(f"第{step}步: 收到响应 : 准备调工具")
            return
        if decision == "thought":
            Console.info(
                f"第{step}步: 思考中 : {summarize(content or '', 220)}"
            )
            return
        if decision == "final":
            Console.info(
                f"第{step}步: 答案 {summarize(content or '', 220)}"
            )
            return
        if decision == "retry":
            Console.warn(
                f"第{step}步: 收到响应 : 这次返回有点问题，马上再试一次：{summarize(content or '', 220)}"
            )
            return
        Console.info(
            f"第{step}步: 收到响应 : {decision} {summarize(content or '', 220)}"
        )

    @staticmethod
    def _print_live_tool_call(step: int, tool_name: str, arguments: dict[str, Any]) -> None:
        """在终端输出工具调用请求。"""
        args_preview = summarize(safe_json_dumps(arguments), 200)
        Console.interaction(
            f"第{step}步: 收到响应 : 准备调工具 {tool_name} | 参数={args_preview}"
        )

    @staticmethod
    def _print_live_tool_result(step: int, tool_name: str, observation: Any) -> None:
        """在终端输出工具调用结果摘要。"""
        # 多模态格式：提取文本部分用于展示，避免输出大量 base64 数据
        if isinstance(observation, list):
            text_parts = [p.get("text", "") for p in observation if isinstance(p, dict) and p.get("type") == "text"]
            preview = " ".join(text_parts) if text_parts else "[多模态内容]"
            Console.info(f"第{step}步: 工具跑完了 {tool_name} => {summarize(preview, 240)}")
        else:
            Console.info(
                f"第{step}步: 工具跑完了 {tool_name} => {summarize(observation, 240)}"
            )

    def __init__(
        self,
        *,
        config: RunnerConfig | None = None,
        tools: ToolRegistry | None = None,
    ) -> None:
        """__init__ 方法。
        
        用途:
        - 初始化 ReActRunner 实例，设置 ReAct 模式的运行配置。
        
        参数:
        - config: 输入参数，用于控制 ReActRunner 实例的运行配置。
        - tools: 输入参数，用于控制 ReActRunner 实例运行的工具注册。
        
        返回:
        - None: 构造函数仅完成实例初始化，不返回业务结果。
        """
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
        """decide 方法。
        
        用途:
        - 执行 ReAct 模式的决策逻辑。
        
        参数:
        - ctx: 输入参数，用于控制 ReActRunner 实例的运行上下文。
        
        返回:
        - StepResult: 返回当前步骤的结果。
        """
        return self._callLLM(ctx)

    def run(self, question: str, id: str | None = None, shared_memory: dict[str, Any] | None = None, images: str | None = None) -> RunResult:
        """执行一次完整的 ReAct 推理回合（主控循环）。

        这个方法是 Runner 的“编排入口”，负责把一次用户问题从“输入文本”
        推进到“最终答案”或“明确失败”，并在中间驱动 LLM 与工具多轮交互。

        方法职责（业务视角）:
        - 组装本轮会话上下文（system prompt + user question）。
        - 决定本轮可用工具集合（内置工具 + MCP 导出工具）。
        - 在 step 循环中执行 ReAct 协议：
          - 让 LLM 决策（`final` / `thought` / `tool_calls` / `retry`）。
          - 若是工具调用，则执行工具并将 observation 回填到历史。
          - 若是最终答案，则结束并返回。
        - 在关键异常场景下快速失败，避免静默降级造成“看似成功、实际不可用”。

        参数:
        - question: 用户输入问题。必须是非空字符串；空值会被直接拒绝。
        - id: 传入的历史会话ID，若有则导入并在此基础上继续。
        - shared_memory: 预留参数，当前实现未消费（用于未来跨轮共享记忆扩展）。目前,本轮的记忆是存在在 `RunnerContext` 中的,并未做本地持久化 , 后续版本可能会考虑添加本地持久化功能,并把 `shared_memory` 作为参数传递给 `decide()` 方法。

        返回:
        - RunResult:
          - `success=True`: LLM 产出 `final` 决策并返回最终文本。
          - `success=False`: 输入非法、MCP 工具不可用、工具注册缺失、或超过最大步数。

        核心流程（实现细节）:
        1) 参数校验与上下文重置
           - 拒绝空问题。
           - 新建 `RunnerContext(step=0, history=[])`，确保每次 `run()` 是独立回合。

        2) MCP 工具发现与可用性判定
           - 优先读取缓存（快路径）：`load_mcp_tools_cache()`。
           - 仅保留“已启用服务器”对应的工具映射，防止脏配置混入。
           - 当缓存缺失或覆盖不完整时，运行期临时拉取工具清单兜底。
           - 记录发现错误信息；若“配置了 MCP 但无任何可用 MCP 工具”，直接失败返回，
             不降级为“仅本地工具”模式（这是明确的业务保护策略）。

        3) 构建对话初始历史
           - 动态生成 system prompt，注入当前可见工具名集合。
           - 追加 user 问题消息。

        4) 进入逐步推理循环（1..max_steps）
           - 调用 `decide()`（内部即 `_callLLM()`）获取当前 StepResult。
           - 按决策类型分支：
            - `tool_calls`:
               - 规范化 tool call（补齐 call id、校验参数类型）。
               - 先写入 assistant 的 `tool_calls` 消息，再逐个执行工具。
               - 工具路由优先级：本地 ToolRegistry > MCP 导出工具 > Unknown tool 错误。
               - 工具执行结果统一写入 `role=tool` 消息，供下一轮 LLM 消费。
               - 分支结束后 `continue`，进入下一 step。
             - `thought`:
               - 将模型中间思考文本（协议字段或退化 content）回填 history。
               - `continue` 进入下一 step。
             - `final`:
               - 把最终协议信息写入 history（便于审计/追踪）。
               - 立即返回成功结果。

        5) 兜底失败
           - 若达到 `max_steps` 仍未产出 `final`，返回 `Max steps reached`。

        关键设计点:
        - “模型可见工具集”和“执行期可路由工具集”保持一致，减少 `Unknown tool` 偏差。
        - 工具 observation 统一 JSON 化，提升协议稳定性和可观测性。
        - 通过 run_logger 记录 step 结果与关键异常，便于线上排障。
        """
        if question is None or not str(question).strip():
            Console.fatal("这个任务没法跑：问题内容是空的。")
            return RunResult(success=False, error="Invalid question")

        self.runner_context = RunnerContext(step=0, history=[])
        Console.debug(
            safe_json_dumps(
                {
                    "event": "runner_start",
                    "mode": self.mode,
                    "max_steps": self.runner_config.max_steps,
                    "question_preview": (str(question).strip()[:200]),
                }
            )
        )

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
            Console.fatal("MCP 配好了，但一个能用的工具都没连上。先把 MCP 服务修好再试。")
            return RunResult(
                success=False,
                error=(
                    "MCP 已配置但未加载到任何工具，请先修复 MCP 服务可用性。"
                    f" details={safe_json_dumps(reason_payload)}"
                ),
            )

        from gaoagent.core.runner.Utils import load_history, save_history

        # 添加系统提示词
        tool_names = (self.runner_config.tools.list_names() if self.runner_config.tools else [])
        # 将 MCP 导出工具名加入可调用工具清单，避免与内置工具重名。
        if isinstance(mcp_exported_map, dict) and mcp_exported_map:
            tool_names = list(tool_names) + sorted([str(x) for x in mcp_exported_map.keys()])

        system_message = {
            "role": "system",
            "content": build_system_prompt(
                mode=self.mode, tool_names=tool_names
            ),
        }

        loaded_history = load_history(id) if id else None
        if loaded_history:
            if loaded_history and loaded_history[0].get("role") == "system":
                loaded_history[0] = system_message
            else:
                loaded_history.insert(0, system_message)
            self.runner_context.history = loaded_history
        else:
            self.runner_context.history.append(system_message)

        # 添加用户的提问
        # 解析图片路径
        image_paths: list[str] = []
        if images:
            paths = [p.strip() for p in images.split(",") if p.strip()]
            for p in paths:
                if is_image_file(p):
                    image_paths.append(p)

        # 构建多模态内容
        user_content = build_multimodal_content(question, image_paths)
        self.runner_context.history.append({"role": "user", "content": user_content})

        for step in range(1, self.runner_config.max_steps + 1):
            # 更新上下文中的 step 信息
            self.runner_context.step = step
            Console.debug(
                safe_json_dumps(
                    {
                        "event": "step_start",
                        "step": step,
                        "history_size": len(self.runner_context.history),
                    }
                )
            )

            now_step = self.decide(self.runner_context)
            Console.debug(
                safe_json_dumps(
                    {
                        "event": "step_decision",
                        "step": step,
                        "decision": now_step.decision,
                        "content_preview": (now_step.content[:200] if isinstance(now_step.content, str) else ""),
                    }
                )
            )

            run_logger = get_current_run_logger()
            if run_logger is not None:
                run_logger.log_event("step_result", now_step, step=step)

            payload = now_step.raw.get("payload") if isinstance(now_step.raw, dict) else None
            first_choice = (
                payload.get("choices")[0]
                if isinstance(payload, dict)
                and isinstance(payload.get("choices"), list)
                and payload.get("choices")
                and isinstance(payload.get("choices")[0], dict)
                else {}
            )
            payload_message = (
                first_choice.get("message") if isinstance(first_choice.get("message"), dict) else {}
            )
            reasoning_content = payload_message.get("reasoning_content")

            if now_step.decision == "tool_calls":
                calls = now_step.tool_calls or []
                Console.debug(
                    safe_json_dumps(
                        {
                            "event": "step_tool_calls_enter",
                            "step": step,
                            "call_count": len(calls),
                        }
                    )
                )
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

                assistant_message: dict[str, Any] = {
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
                if isinstance(reasoning_content, str) and reasoning_content:
                    assistant_message["reasoning_content"] = reasoning_content
                self.runner_context.history.append(assistant_message)

                if not self.runner_config.tools:
                    Console.fatal("工具系统没准备好：没找到可用的 ToolRegistry。")
                    if id:
                        save_history(id, self.runner_context.history)
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
                        self._print_live_tool_call(step, name, arguments)
                        try:
                            # 路由顺序：
                            # 1) 先走内置 ToolRegistry（本地工具）
                            # 2) 再走 MCP 导出工具映射（远程/stdio 工具）
                            # 3) 两者都找不到则返回 Unknown tool
                            if self.runner_config.tools and name in self.runner_config.tools.list_names():
                                Console.debug(
                                    safe_json_dumps(
                                        {
                                            "event": "tool_call_local",
                                            "step": step,
                                            "tool": name,
                                        }
                                    )
                                )
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
                                    Console.fatal(f"MCP 工具映射不完整，调用不了：{name}")
                                    observation = {
                                        "success": False,
                                        "error": {
                                            "type": "ValueError",
                                            "message": f"MCP tool 映射无效：name={name}",
                                        },
                                    }
                                else:
                                    Console.debug(
                                        safe_json_dumps(
                                            {
                                                "event": "tool_call_mcp",
                                                "step": step,
                                                "tool": name,
                                                "server": server_name,
                                                "remote_tool": tool_name,
                                            }
                                        )
                                    )
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
                                Console.fatal(f"模型想调用不存在的工具：{name}")
                                observation = {
                                    "success": False,
                                    "error": {
                                        "type": "ValueError",
                                        "message": f"Unknown tool: {name}",
                                    },
                                }
                        except Exception as e:
                            Console.fatal(f"工具调用失败了：{name}，错误：{e}")
                            observation = {
                                "success": False,
                                "error": {"type": type(e).__name__, "message": str(e)},
                            }

                    # 处理多模态格式的 observation（如 ask_user 工具返回的图片）
                    if isinstance(observation, list):
                        # 多模态格式：直接使用 list 作为 content
                        tool_content = observation
                    elif isinstance(observation, str):
                        # 纯文本格式
                        tool_content = observation
                    else:
                        # dict 或其他格式：转换为 JSON 字符串
                        tool_content = safe_json_dumps(observation)
                    
                    self.runner_context.history.append(
                        {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "content": tool_content,
                        }
                    )
                    # 多模态格式：提取文本部分用于日志预览
                    if isinstance(observation, list):
                        text_parts = [p.get("text", "") for p in observation if isinstance(p, dict) and p.get("type") == "text"]
                        observation_preview = summarize(" ".join(text_parts) if text_parts else "[多模态内容]", 180)
                    else:
                        observation_preview = summarize(observation, 180)
                    
                    Console.debug(
                        safe_json_dumps(
                            {
                                "event": "tool_observation_written",
                                "step": step,
                                "tool": name if isinstance(name, str) else "",
                                "observation_preview": observation_preview,
                            }
                        )
                    )
                    self._print_live_tool_result(
                        step,
                        name if isinstance(name, str) else "",
                        observation,
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
                assistant_message = {"role": "assistant", "content": assistant_content}
                if isinstance(reasoning_content, str) and reasoning_content:
                    assistant_message["reasoning_content"] = reasoning_content
                self.runner_context.history.append(assistant_message)
                Console.info(
                    f"第{step}步: 思考中 : {summarize(assistant_content, 240)}"
                )
                Console.debug(
                    safe_json_dumps(
                        {
                            "event": "step_thought_written",
                            "step": step,
                            "preview": summarize(assistant_content, 180),
                        }
                    )
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
                assistant_message = {"role": "assistant", "content": assistant_content}
                if isinstance(reasoning_content, str) and reasoning_content:
                    assistant_message["reasoning_content"] = reasoning_content
                self.runner_context.history.append(assistant_message)
                Console.debug(
                    safe_json_dumps(
                        {
                            "event": "runner_final",
                            "step": step,
                            "final_preview": summarize(now_step.content or "", 240),
                        }
                    )
                )
                if id:
                    save_history(id, self.runner_context.history)
                return RunResult(success=True, final_result=now_step.content)

        Console.fatal("任务跑到最大步数了，还没拿到最终结果。")
        if id:
            save_history(id, self.runner_context.history)
        return RunResult(success=False, error="Max steps reached")

    def _callLLM(self, ctx: RunnerContext) -> StepResult:
        """执行单步 LLM 决策调用，并对“非法协议输出”做有限重试。

        这个方法是 ReAct 每一步的“模型决策器”：
        输入当前 `ctx.history`，输出一个标准化 `StepResult`，交给 `run()`
        决定下一步是继续思考、调用工具、还是结束。

        方法职责（业务视角）:
        - 创建 OpenAI 兼容客户端并发起 chat completion 请求。
        - 把“当前可用工具”转换为 function calling 规格传给模型。
        - 解析模型响应为统一协议（`parse_llm_response`）。
        - 当模型输出不符合协议时，按配置进行有限次数自动重试。
        - 重试仍失败时，返回可解释的 `final` 失败说明，而不是抛异常中断主流程。

        参数:
        - ctx: 当前回合上下文，至少包含：
          - `history`: 截止当前 step 的完整消息历史。
          - `step`: 当前 step 序号（用于日志与可观测性）。

        返回:
        - StepResult:
          - 正常路径：返回解析后的 `thought` / `tool_calls` / `final`。
          - 异常协议路径：若持续 `retry` 到上限，返回 `decision="final"` 的失败说明。

        具体流程:
        1) 配置前置校验
           - 若缺少 API 基础配置（baseurl/api_key/model），立即返回 `final` 错误文本，
             避免进入无效网络调用。

        2) 构建请求上下文
           - 收集内置工具名，并合并 `run()` 阶段确定的 MCP 导出工具名。
           - 用 `build_function_specs()` 生成 function calling schema。
           - 仅当存在工具时启用 `tool_choice="auto"`，无工具则传 `None`。

        3) 发送请求并解析响应
           - 调用 `post_chat_completions()`，携带 model/messages/tools/step。
           - 用 `parse_llm_response()` 归一化响应，得到 `StepResult`。

        4) 非法输出重试
           - 当 `decision=="retry"` 视为“模型输出不合法/不可执行”。
           - 在 `llm_invalid_retry` 配置范围内重试，并记录日志事件
             `llm_invalid_response_retry`（attempt、raw、content）。
           - 一旦拿到非 `retry` 结果立即返回。

        5) 重试耗尽兜底
           - 组装 `_retry` 元信息写入 `raw`，并返回 `decision="final"`，
             内容明确包含“已重试仍失败 + 次数 + 最后一次输出摘要”。

        关键设计点:
        - 不把解析异常直接上抛，而是转换成协议内可消费结果，让外层流程保持稳定。
        - 与 `run()` 共用同一份 MCP 工具可见集，保证“可见即可调、可调即可见”。
        """
        if not self.request_base_info:
            Console.fatal("没拿到可用的 API 配置，模型请求发不出去。")
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
            build_function_specs(all_tool_names, mcp_exported_map=mcp_exported_map, tool_registry=self.runner_config.tools)
            if all_tool_names
            else None
        )
        tool_choice = "auto" if tools else None
        max_attempts = max(1, 1 + int(getattr(self.runner_config, "llm_invalid_retry", 0) or 0))
        attempt = 0
        last_step: StepResult | None = None

        while attempt < max_attempts:
            attempt += 1
            self._print_live_llm_request(ctx.step, attempt, max_attempts)
            Console.debug(
                safe_json_dumps(
                    {
                        "event": "llm_attempt",
                        "step": ctx.step,
                        "attempt": attempt,
                        "max_attempts": max_attempts,
                        "history_size": len(ctx.history),
                    }
                )
            )
            response = client.post_chat_completions(
                model=self.request_base_info.modules,
                messages=ctx.history,
                tools=tools,
                tool_choice=tool_choice,
                step=ctx.step,
            )
            step_result = parse_llm_response(response)
            last_step = step_result
            payload = step_result.raw.get("payload") if isinstance(step_result.raw, dict) else None
            first_choice = (
                payload.get("choices")[0]
                if isinstance(payload, dict)
                and isinstance(payload.get("choices"), list)
                and payload.get("choices")
                and isinstance(payload.get("choices")[0], dict)
                else {}
            )
            payload_message = (
                first_choice.get("message") if isinstance(first_choice.get("message"), dict) else {}
            )
            reasoning_content = payload_message.get("reasoning_content")
            self._print_live_llm_response(
                ctx.step,
                step_result.decision,
                step_result.content,
                reasoning_content if isinstance(reasoning_content, str) else None,
            )
            Console.debug(
                safe_json_dumps(
                    {
                        "event": "llm_parsed_result",
                        "step": ctx.step,
                        "attempt": attempt,
                        "decision": step_result.decision,
                        "content_preview": summarize(step_result.content, 200),
                    }
                )
            )

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
        Console.fatal(
            f"模型连续返回了不可执行内容，重试 {max_attempts} 次还是没成功。"
        )
        return StepResult(
            decision="final",
            content=(
                "LLM 输出为空或不符合协议，已自动重试仍失败。"
                f" attempts={max_attempts}"
                + (f"\nlast={last_content}" if last_content else "")
            ),
            raw=raw,
        )
