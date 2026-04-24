from __future__ import annotations

from gaoagent.core.runner.Console import Console

from gaoagent.core.runner.BaseRunner import RunnerConfig
from gaoagent.core.runner.ReActRunner import ReActRunner
from gaoagent.core.runner.Tooling import ToolRegistry, default_tool_registry
from gaoagent.core.runner.RunLogger import (
    create_run_logger,
    reset_current_run_logger,
    set_current_run_logger,
)


class TaskRunner:
    """任务执行编排器（CLI `task` 子命令的执行入口）。

    定位:
    - 位于 `CoreHandlers.task()` 与具体 Runner（如 `ReActRunner`）之间。
    - 负责把“命令参数”转换成“可执行运行上下文”，并统一处理运行日志生命周期。

    核心职责:
    - 管理默认配置与工具注册表注入。
    - 规范化运行模式参数（`plan/react/retry`）。
    - 创建并设置当前运行日志上下文，确保每次任务都有独立追踪链路。
    - 调用底层 Runner 执行任务，并将结果以 CLI 友好形式输出到终端。

    边界:
    - 不维护多轮会话记忆，不处理 step 级决策细节。
    - 不实现具体推理协议，推理逻辑由 Runner 内部负责。
    """
    def __init__(
        self,
        *,
        tools: ToolRegistry | None = None,
        config: RunnerConfig | None = None,
    ) -> None:
        """初始化 TaskRunner 依赖项。

        参数:
        - `tools`: 可选工具注册表；为空时使用 `default_tool_registry()`。
        - `config`: 可选 Runner 配置；为空时使用默认 `RunnerConfig()`。

        说明:
        - 该构造函数仅做依赖装配，不触发任何任务执行。
        """
        self._cfg = config or RunnerConfig()
        self._tools = tools or default_tool_registry()

    def run(self, question: str, mode: str) -> None:
        """执行一次任务并将最终结果输出到终端。

        流程:
        1. 创建 run logger，并将其设为当前上下文 logger。
        2. 规范化 `mode`（去空白、转小写、非法值回退到 `react`）。
        3. 根据模式实例化 Runner 并执行 `run(question)`。
           - 目前只实现了 React 模式
           - 当前实现中 `plan/react/retry` 三种模式均路由到 `ReActRunner`。
        4. 在 `finally` 中恢复 logger 上下文，避免污染后续任务。
        5. 成功时输出 `final_result`；失败时输出错误信息。

        参数:
        - `question`: 用户任务描述。
        - `mode`: 期望运行模式字符串。

        返回:
        - `None`。该方法面向 CLI，采用命令式输出而非返回结构化对象。
        """
        run_logger = create_run_logger()
        token = set_current_run_logger(run_logger)
        try:
            m = (mode or "react").strip().lower()
            if m not in ("plan", "react", "retry"):
                m = "react"

            if m == "plan":
                result = ReActRunner(config=self._cfg, tools=self._tools).run(question)
            elif m == "retry":
                result = ReActRunner(config=self._cfg, tools=self._tools).run(question)
            else:
                result = ReActRunner(config=self._cfg, tools=self._tools).run(question)
        finally:
            reset_current_run_logger(token)

        if result.success:
            if result.final_result:
                Console.info(result.final_result)
            return

        Console.error(f"任务失败：{result.error or 'unknown error'}")
