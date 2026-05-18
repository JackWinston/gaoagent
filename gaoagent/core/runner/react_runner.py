from __future__ import annotations

from typing import Any

from gaoagent.core.runner.base_runner import (
    BaseRunner,
    RequestBaseInfo,
    RunnerConfig,
    RunnerContext,
    RunResult,
    StepResult,
)

from gaoagent.core.runner.console import Console
from gaoagent.core.runner.http_client import OpenAICompatibleHttpClient, StreamCallback
from gaoagent.core.runner.tooling import ToolCall, ToolRegistry, default_tool_registry
from gaoagent.core.runner.utils import (
    build_multimodal_content,
    is_image_file,
    load_mcp_servers_raw,
    load_mcp_tools_cache,
    parse_llm_response,
    safe_json_dumps,
    summarize,
    write_mcp_tools_cache_for_current_scope,
)
from gaoagent.core.runner.prompt_builder import build_system_prompt
from gaoagent.core.runner.function_call_protocol import build_function_specs
from gaoagent.core.runner.run_logger import get_current_run_logger
from gaoagent.mcp.mcp_client_compat import MCPStdioClientSync, build_mcp_tools_cache_payload


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
        stream_printed: bool = False,
    ) -> None:
        """在终端输出本轮 LLM 返回摘要。
        
        参数:
        - step: 当前步骤号
        - decision: 决策类型
        - content: 内容
        - reasoning_content: 推理内容
        - stream_printed: 是否已通过流式输出打印过内容
        """
        # 如果已流式输出，只打印简短决策摘要
        if stream_printed:
            if decision == "tool_calls":
                Console.info(f"  第{step}步: 决策 → 调用工具")
            elif decision == "final":
                Console.info(f"  第{step}步: 决策 → 返回结果")
            elif decision == "thought":
                Console.info(f"  第{step}步: 决策 → 继续思考")
            elif decision == "retry":
                Console.warn(f"  第{step}步: 决策 → 重试")
            return
        
        # 未流式输出时，打印详细信息
        if isinstance(reasoning_content, str) and reasoning_content.strip():
            Console.info(
                f"第{step}步: 推理内容 : {reasoning_content.strip()}"
            )
        if decision == "tool_calls":
            Console.info(f"第{step}步: 收到响应 : 准备调工具")
            return
        if decision == "thought":
            Console.info(f"第{step}步: 思考中 :")
            Console.output_llm_result(content or '')
            return
        if decision == "final":
            Console.info(f"第{step}步: 答案 :")
            Console.output_llm_result(content or '')
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
    def _print_live_tool_call(step: int, tool_name: str, arguments: dict[str, Any], stream_printed: bool = False) -> None:
        """在终端输出工具调用请求。
        
        参数:
        - step: 当前步骤号
        - tool_name: 工具名称
        - arguments: 工具参数
        - stream_printed: 是否已通过流式输出打印过
        """
        if stream_printed:
            # 流式输出已打印工具名，这里只打印参数摘要
            args_preview = summarize(safe_json_dumps(arguments), 200)
            Console.info(f"  ️  第{step}步: 参数 → {args_preview}")
        else:
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

    @staticmethod
    def _create_stream_callback(step: int) -> StreamCallback:
        """创建流式输出回调函数。
        
        用途:
        - 为当前步骤创建流式输出回调，实时打印 LLM 响应内容。
        
        参数:
        - step: 当前步骤号。
        
        返回:
        - StreamCallback: 回调函数，接收 (chunk_type, content) 参数。
        """
        state = {
            "is_reasoning": False,
            "is_content": False,
            "is_tool_call": False,
            "has_output": False,
            "buffer": "",
        }
        
        def flush_buffer():
            """刷新缓冲区。"""
            if state["buffer"]:
                Console.stream_weak(state["buffer"])
                state["buffer"] = ""
        
        def callback(chunk_type: str, content: str) -> None:
            if chunk_type == "reasoning":
                if not state["is_reasoning"]:
                    flush_buffer()
                    state["is_reasoning"] = True
                    state["is_content"] = False
                    state["is_tool_call"] = False
                    Console.weak(f"\n    第{step}步 推理过程 ▸")
                state["buffer"] += content
                # 按行刷新，保持输出整洁
                while "\n" in state["buffer"]:
                    line, state["buffer"] = state["buffer"].split("\n", 1)
                    Console.stream_weak(line + "\n")
                state["has_output"] = True
            elif chunk_type == "content":
                if state["is_reasoning"] or not state["is_content"]:
                    flush_buffer()
                    state["is_reasoning"] = False
                    state["is_content"] = True
                    state["is_tool_call"] = False
                    Console.weak(f"\n    第{step}步 回复内容 ▸")
                state["buffer"] += content
                # 按行刷新
                while "\n" in state["buffer"]:
                    line, state["buffer"] = state["buffer"].split("\n", 1)
                    Console.stream_weak(line + "\n")
                state["has_output"] = True
            elif chunk_type == "tool_call_start":
                flush_buffer()
                state["is_reasoning"] = False
                state["is_content"] = False
                if not state["is_tool_call"]:
                    state["is_tool_call"] = True
                Console.weak(f"\n  ️  第{step}步 调用工具 ▸ {content}")
                state["has_output"] = True
            elif chunk_type == "tool_call_args":
                if state["is_tool_call"]:
                    state["buffer"] += content
                    # JSON参数按块输出
                    if len(state["buffer"]) > 80:
                        Console.stream_weak(state["buffer"])
                        state["buffer"] = ""
        
        def reset():
            """重置状态，用于重试场景。"""
            flush_buffer()
            state["is_reasoning"] = False
            state["is_content"] = False
            state["is_tool_call"] = False
            state["has_output"] = False
            state["buffer"] = ""
        
        callback.reset = reset
        callback.has_output = lambda: state["has_output"]
        callback.flush = flush_buffer
        
        return callback

    def __init__(
        self,
        *,
        config: RunnerConfig | None = None,
        tools: ToolRegistry | None = None,
        request_base_info: RequestBaseInfo | None = None,
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
            scene=(config.scene if config else "default"),
            disable_function_call=(config.disable_function_call if config else False),
            disable_mcp=(config.disable_mcp if config else False),
            disable_skill=(config.disable_skill if config else False),
            disable_rag=(config.disable_rag if config else False),
        )
        super().__init__(
            mode="react",
            runner_config=cfg,
            request_base_info=request_base_info,
        )
        self._current_stream_callback: StreamCallback | None = None
        self._project_overview_refresh_records: list[dict[str, Any]] = []

    def _is_init_project_overview_scene(self) -> bool:
        """判断当前是否处于初始化项目概览场景。"""
        return str(getattr(self.runner_config, "scene", "default") or "default") == "init_project_overview"

    def _is_default_scene(self) -> bool:
        """判断当前是否为默认普通任务场景。"""
        return str(getattr(self.runner_config, "scene", "default") or "default") == "default"

    def get_project_overview_refresh_records(self) -> list[dict[str, Any]]:
        """返回当前回合记录到的项目概览刷新触发记录。"""
        from gaoagent.core.project_overview_tool import ProjectOverviewTool

        return ProjectOverviewTool.clone_refresh_records(self._project_overview_refresh_records)

    def _record_project_overview_refresh_candidate(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        raw_observation: Any,
    ) -> None:
        """记录满足条件的文件新增/删除事件，为后续刷新 `project.md` 提供依据。"""
        from gaoagent.core.project_overview_tool import ProjectOverviewTool

        record = ProjectOverviewTool.build_refresh_record_from_tool_call(
            tool_name,
            arguments,
            raw_observation,
        )
        if record is not None:
            self._project_overview_refresh_records.append(record)

    def _enabled_local_tool_names(self) -> list[str]:
        """返回当前配置下允许暴露与执行的本地工具名。"""
        if self.runner_config.disable_function_call or not self.runner_config.tools:
            return []
        names = list(self.runner_config.tools.list_names())
        if self.runner_config.disable_rag:
            names = [name for name in names if name != "rag_search"]
        return names

    def decide(self, ctx: RunnerContext) -> StepResult:
        """decide 方法。
        
        用途:
        - 执行 ReAct 模式的决策逻辑。
        
        参数:
        - ctx: 输入参数，用于控制 ReActRunner 实例的运行上下文。
        
        返回:
        - StepResult: 返回当前步骤的结果。
        """
        return self._call_llm(ctx)

    # --------------- MCP 工具发现 ---------------

    def _discover_mcp_tools(self) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
        """发现并加载 MCP 工具。
        
        职责:
        - 优先读取缓存（快路径）
        - 当缓存缺失或覆盖不完整时，运行期临时拉取工具清单兜底
        - 记录发现错误信息
        
        返回:
        - tuple: (mcp_servers_raw, mcp_exported_map, mcp_discovery_errors)
        """
        if self.runner_config.disable_mcp or self.runner_config.disable_function_call:
            return {}, {}, {}

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
        return mcp_servers_raw, mcp_exported_map, mcp_discovery_errors

    def _check_mcp_availability(
        self,
        mcp_servers_raw: dict[str, Any],
        mcp_exported_map: dict[str, Any],
        mcp_discovery_errors: dict[str, str],
    ) -> RunResult | None:
        """检查 MCP 工具可用性，若不可用则返回失败结果。
        
        参数:
        - mcp_servers_raw: MCP 服务器配置
        - mcp_exported_map: MCP 导出工具映射
        - mcp_discovery_errors: 发现错误信息
        
        返回:
        - RunResult: 失败结果（若 MCP 不可用）
        - None: MCP 可用或未配置
        """
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
            Console.fatal("  MCP 服务连接失败：已配置 MCP 但无法加载任何工具。")
            Console.warn("   请检查：")
            Console.warn("   1. MCP 服务是否已启动")
            Console.warn("   2. 配置是否正确（运行 `gaoagent mcp test` 测试连通性）")
            return RunResult(
                success=False,
                error=(
                    "MCP 已配置但未加载到任何工具，请先修复 MCP 服务可用性。"
                    f" details={safe_json_dumps(reason_payload)}"
                ),
            )
        return None

    # --------------- 对话历史初始化 ---------------

    def _init_conversation_history(
        self,
        question: str,
        id: str | None,
        images: str | None,
        mcp_exported_map: dict[str, Any],
    ) -> None:
        """构建对话初始历史。
        
        职责:
        - 动态生成 system prompt，注入当前可见工具名集合
        - 加载历史记录（若有）
        - 追加 user 问题消息（支持多模态）
        
        参数:
        - question: 用户输入问题
        - id: 历史会话 ID
        - images: 图片路径（逗号分隔）
        - mcp_exported_map: MCP 导出工具映射
        """
        from gaoagent.core.runner.utils import load_history

        # 添加系统提示词
        tool_names = self._enabled_local_tool_names()
        # 将 MCP 导出工具名加入可调用工具清单，避免与内置工具重名。
        if not self.runner_config.disable_function_call and isinstance(mcp_exported_map, dict) and mcp_exported_map:
            tool_names = list(tool_names) + sorted([str(x) for x in mcp_exported_map.keys()])

        system_prompt = build_system_prompt(
            mode=self.mode,
            tool_names=tool_names,
            scene=str(getattr(self.runner_config, "scene", "default") or "default"),
            allow_function_call=not self.runner_config.disable_function_call,
            enable_rag=not self.runner_config.disable_rag,
            enable_skill=not self.runner_config.disable_skill,
        )

        system_message = {
            "role": "system",
            "content": system_prompt,
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

    # --------------- 步骤处理 ---------------

    def _extract_reasoning_content(self, now_step: StepResult) -> str | None:
        """从步骤结果中提取推理内容。
        
        参数:
        - now_step: 步骤结果
        
        返回:
        - str | None: 推理内容（若有）
        """
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
        return payload_message.get("reasoning_content")

    def _handle_tool_calls(
        self,
        step: int,
        now_step: StepResult,
        reasoning_content: str | None,
        id: str | None,
        mcp_servers_raw: dict[str, Any],
        mcp_exported_map: dict[str, Any],
    ) -> RunResult | None:
        """处理 tool_calls 决策。
        
        职责:
        - 规范化 tool call（补齐 call id、校验参数类型）
        - 写入 assistant 的 tool_calls 消息
        - 逐个执行工具（本地 ToolRegistry > MCP 导出工具 > Unknown tool）
        - 将工具执行结果写入 role=tool 消息
        
        参数:
        - step: 当前步骤号
        - now_step: 步骤结果
        - reasoning_content: 推理内容
        - id: 历史会话 ID
        - mcp_servers_raw: MCP 服务器配置
        - mcp_exported_map: MCP 导出工具映射
        
        返回:
        - RunResult: 失败结果（若工具系统异常）
        - None: 正常处理完成
        """
        from gaoagent.core.runner.utils import save_history

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
            Console.fatal("  工具系统异常：未找到可用的工具注册表。")
            Console.warn("   请尝试重新初始化项目：`gaoagent init`")
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
                stream_printed = (
                    self._current_stream_callback.has_output()
                    if self._current_stream_callback
                    else False
                )
                self._print_live_tool_call(step, name, arguments, stream_printed=stream_printed)
                try:
                    # 路由顺序：
                    # 1) 先走内置 ToolRegistry（本地工具）
                    # 2) 再走 MCP 导出工具映射（远程/stdio 工具）
                    # 3) 两者都找不到则返回 Unknown tool
                    enabled_local_tools = set(self._enabled_local_tool_names())
                    if self.runner_config.tools and name in enabled_local_tools:
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
                        if name in {"write_file", "delete_file"}:
                            self._record_project_overview_refresh_candidate(
                                name,
                                arguments,
                                getattr(self.runner_context, "last_observation_raw", None),
                            )
                    elif (
                        not self.runner_config.disable_function_call
                        and isinstance(mcp_exported_map, dict)
                        and name in mcp_exported_map
                    ):
                        mcp_meta = mcp_exported_map.get(name) or {}
                        server_name = mcp_meta.get("server")
                        tool_name = mcp_meta.get("tool")
                        server_cfg = (
                            mcp_servers_raw.get(server_name)
                            if isinstance(server_name, str) and isinstance(mcp_servers_raw, dict)
                            else None
                        )
                        if not isinstance(server_name, str) or not isinstance(tool_name, str) or not isinstance(server_cfg, dict):
                            Console.fatal(f"  MCP 工具配置异常：{name} 的映射信息不完整")
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
                        Console.fatal(f"  未知工具：模型尝试调用不存在的工具 `{name}`")
                        Console.warn(f"   可用工具：{', '.join(self.runner_config.tools.list_names()) if self.runner_config.tools else '无'}")
                        observation = {
                            "success": False,
                            "error": {
                                "type": "ValueError",
                                "message": f"Unknown tool: {name}",
                            },
                        }
                except Exception as e:
                    Console.fatal(f"  工具执行失败：{name}")
                    Console.warn(f"   错误详情：{e}")
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
        return None

    def _handle_thought(
        self,
        step: int,
        now_step: StepResult,
        reasoning_content: str | None,
    ) -> None:
        """处理 thought 决策。
        
        职责:
        - 将模型中间思考文本回填 history
        - 打印思考摘要
        
        参数:
        - step: 当前步骤号
        - now_step: 步骤结果
        - reasoning_content: 推理内容
        """
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
        # 如果已流式输出，只打印简短摘要
        stream_printed = (
            self._current_stream_callback.has_output()
            if self._current_stream_callback
            else False
        )
        if stream_printed:
            Console.info(f"  第{step}步: 决策 → 继续思考")
        else:
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

    def _handle_final(
        self,
        step: int,
        now_step: StepResult,
        reasoning_content: str | None,
        id: str | None,
    ) -> RunResult:
        """处理 final 决策。
        
        职责:
        - 把最终协议信息写入 history（便于审计/追踪）
        - 保存历史记录（若有 id）
        - 返回成功结果
        
        参数:
        - step: 当前步骤号
        - now_step: 步骤结果
        - reasoning_content: 推理内容
        - id: 历史会话 ID
        
        返回:
        - RunResult: 成功结果
        """
        from gaoagent.core.runner.utils import save_history

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
        self._refresh_project_overview_after_task()
        return RunResult(success=True, final_result=now_step.content)

    def _refresh_project_overview_after_task(self) -> None:
        """普通任务完成后，尝试刷新当前项目的 `project.md`。"""
        if not self._is_default_scene():
            return

        try:
            from gaoagent.core.project_overview_tool import ProjectOverviewTool

            if not ProjectOverviewTool.should_refresh_from_records(self._project_overview_refresh_records):
                return
            ProjectOverviewTool().refresh_current_project_overview_if_exists(
                refresh_records=self.get_project_overview_refresh_records()
            )
        except Exception as exc:
            Console.warn(f"项目概览刷新失败：{exc}")

    # --------------- 主控循环 ---------------

    def run(self, question: str, id: str | None = None, shared_memory: dict[str, Any] | None = None, images: str | None = None) -> RunResult:
        """执行一次完整的 ReAct 推理回合（主控循环）。

        这个方法是 Runner 的"编排入口"，负责把一次用户问题从"输入文本"
        推进到"最终答案"或"明确失败"，并在中间驱动 LLM 与工具多轮交互。

        方法职责（业务视角）:
        - 组装本轮会话上下文（system prompt + user question）。
        - 决定本轮可用工具集合（内置工具 + MCP 导出工具）。
        - 在 step 循环中执行 ReAct 协议：
          - 让 LLM 决策（`final` / `thought` / `tool_calls` / `retry`）。
          - 若是工具调用，则执行工具并将 observation 回填到历史。
          - 若是最终答案，则结束并返回。
        - 在关键异常场景下快速失败，避免静默降级造成"看似成功、实际不可用"。

        参数:
        - question: 用户输入问题。必须是非空字符串；空值会被直接拒绝。
        - id: 传入的历史会话ID，若有则导入并在此基础上继续。
        - shared_memory: 预留参数，当前实现未消费（用于未来跨轮共享记忆扩展）。目前,本轮的记忆是存在在 `RunnerContext` 中的,并未做本地持久化 , 后续版本可能会考虑添加本地持久化功能,并把 `shared_memory` 作为参数传递给 `decide()` 方法。

        返回:
        - RunResult:
          - `success=True`: LLM 产出 `final` 决策并返回最终文本。
          - `success=False`: 输入非法、MCP 工具不可用、工具注册缺失、或超过最大步数。
        """
        if question is None or not str(question).strip():
            Console.fatal("  任务无法执行：问题内容为空，请输入有效的任务描述。")
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

        # MCP 工具发现
        mcp_servers_raw, mcp_exported_map, mcp_discovery_errors = self._discover_mcp_tools()

        # 检查 MCP 可用性
        mcp_check_result = self._check_mcp_availability(mcp_servers_raw, mcp_exported_map, mcp_discovery_errors)
        if mcp_check_result is not None:
            return mcp_check_result

        # 构建对话初始历史
        self._init_conversation_history(question, id, images, mcp_exported_map)

        # 逐步推理循环
        from gaoagent.core.runner.utils import save_history

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

            reasoning_content = self._extract_reasoning_content(now_step)

            if now_step.decision == "tool_calls":
                error_result = self._handle_tool_calls(
                    step, now_step, reasoning_content, id, mcp_servers_raw, mcp_exported_map
                )
                if error_result is not None:
                    return error_result
                continue
            if now_step.decision == "thought":
                self._handle_thought(step, now_step, reasoning_content)
                continue
            if now_step.decision == "final":
                return self._handle_final(step, now_step, reasoning_content, id)

        Console.fatal(f"  任务执行超限：已达到最大步数 ({self.runner_config.max_steps})，仍未获得最终结果。")
        Console.warn("   可能原因：任务过于复杂或模型陷入循环")
        Console.warn("   建议：尝试简化任务描述，或使用 `--mode plan` 进行任务拆解")
        if id:
            save_history(id, self.runner_context.history)
        return RunResult(success=False, error="Max steps reached")

    def _call_llm(self, ctx: RunnerContext) -> StepResult:
        """执行单步 LLM 决策调用，并对"非法协议输出"做有限重试。

        这个方法是 ReAct 每一步的"模型决策器"：
        输入当前 `ctx.history`，输出一个标准化 `StepResult`，交给 `run()`
        决定下一步是继续思考、调用工具、还是结束。

        方法职责（业务视角）:
        - 创建 OpenAI 兼容客户端并发起 chat completion 请求。
        - 把"当前可用工具"转换为 function calling 规格传给模型。
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
           - 当 `decision=="retry"` 视为"模型输出不合法/不可执行"。
           - 在 `llm_invalid_retry` 配置范围内重试，并记录日志事件
             `llm_invalid_response_retry`（attempt、raw、content）。
           - 一旦拿到非 `retry` 结果立即返回。

        5) 重试耗尽兜底
           - 组装 `_retry` 元信息写入 `raw`，并返回 `decision="final"`，
             内容明确包含"已重试仍失败 + 次数 + 最后一次输出摘要"。

        关键设计点:
        - 不把解析异常直接上抛，而是转换成协议内可消费结果，让外层流程保持稳定。
        - 与 `run()` 共用同一份 MCP 工具可见集，保证"可见即可调、可调即可见"。
        """
        if not self.request_base_info:
            Console.fatal("  API 配置缺失：无法获取模型接口配置。")
            Console.warn("   请运行 `gaoagent api add` 添加 API 配置")
            return StepResult(decision="final", content="No valid API configuration")

        client = OpenAICompatibleHttpClient(
            base_url=self.request_base_info.baseurl,
            api_key=self.request_base_info.api_key,
        )
        tool_names = self._enabled_local_tool_names()
        # 只使用 run() 阶段已确定的 MCP 工具映射，确保"模型可见工具集"
        # 与"执行期可路由工具集"完全一致，避免出现 Unknown tool 偏差。
        mcp_exported_map = getattr(self, "_mcp_exported_map", None)
        if not isinstance(mcp_exported_map, dict):
            mcp_exported_map = {}
        all_tool_names = list(tool_names)
        if not self.runner_config.disable_function_call and isinstance(mcp_exported_map, dict) and mcp_exported_map:
            all_tool_names += sorted([str(x) for x in mcp_exported_map.keys()])

        tools = (
            build_function_specs(all_tool_names, mcp_exported_map=mcp_exported_map, tool_registry=self.runner_config.tools)
            if all_tool_names and not self.runner_config.disable_function_call
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
            # 创建流式输出回调（重试时重置状态）
            if attempt == 1:
                stream_callback = self._create_stream_callback(ctx.step)
            else:
                stream_callback.reset()
            
            # 保存到实例变量，供run方法使用
            self._current_stream_callback = stream_callback
            
            response = client.post_chat_completions(
                model=self.request_base_info.modules,
                messages=ctx.history,
                tools=tools,
                tool_choice=tool_choice,
                step=ctx.step,
                stream_callback=stream_callback,
            )
            # 刷新流式输出缓冲区
            stream_callback.flush()
            if stream_callback.has_output():
                Console.info("")
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
                stream_printed=stream_callback.has_output(),
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
        Console.fatal(f"  模型响应异常：连续 {max_attempts} 次返回无法执行的内容。")
        Console.warn("   可能原因：模型不支持当前工具调用协议")
        Console.warn("   建议：检查模型是否支持 Function Calling，或更换模型重试")
        return StepResult(
            decision="final",
            content=(
                "LLM 输出为空或不符合协议，已自动重试仍失败。"
                f" attempts={max_attempts}"
                + (f"\nlast={last_content}" if last_content else "")
            ),
            raw=raw,
        )
