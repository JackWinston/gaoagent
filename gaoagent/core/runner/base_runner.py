from __future__ import annotations


from typing import Any, Literal
from dataclasses import dataclass, field
from gaoagent.core.runner.tooling import ToolRegistry
from gaoagent.core.runner.utils import load_request_base_info

Mode = Literal["plan", "react", "retry"]
Decision = Literal["final", "tool_calls", "thought", "retry"]


@dataclass
class RunResult:
    """
    整个任务的运行结果.

    param success: 任务是否成功完成. 如果为 False, 则 final_result 通常会包含错误信息.
    param final_result: 任务的最终结果. 只有当 success=True 时才有意义.
    param error: 任务执行过程中发生的错误信息. 只有当 success=False 时才有意义.
    """

    success: bool
    final_result: str | None = None
    error: str | None = None


@dataclass
class RunnerContext:
    """
    每一步决策的上下文信息.

    param step: 当前是第几步了, 从 1 开始计数.
    param history: 到目前为止的对话历史, 包括用户和助手的消息. 每条消息是一个 dict, 至少包含 "role" 和 "content" 两个字段.
    """

    step: int
    history: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class RunnerConfig:
    """
    Runner 的配置项.

    param max_steps: 最多允许多少步决策. 超过这个步数后, Runner 就会停止并返回失败.
    param tools: 可用的工具列表. Runner 会根据需要调用这些工具来辅助决策和完成任务.
    param llm_invalid_retry: 当 LLM 返回空/不符合协议的响应时, 允许额外重试次数.
    """

    max_steps: int = 32
    tools: ToolRegistry | None = None
    llm_invalid_retry: int = 2


@dataclass
class StepResult:
    """
    每一步决策的结果.

    param decision: 决策的内容, 包括是调用工具还是返回最终结果等信息.
    param tool_calls: 如果决策是调用工具, 则这个字段包含要调用的工具的信息. 否则为 None.
    param content: 决策的文本内容, 例如助手的回复或者思考的内容等. 只有当 decision="thought" 或者 decision="final" 时才有意义.
    param raw: 决策的原始信息, 返回的内容.

    """

    decision: Decision
    tool_calls: list[dict[str, Any]] | None = None
    content: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class RequestBaseInfo:
    """
    请求基础信息, 包括请求的 baseurl、api_key、默认的 headers 等等.
    这些信息通常在 Runner 初始化时提供, 用于后续调用 LLM 接口时使用.

    param baseurl: LLM 接口的 baseurl, 例如 "https://api.openai.com/v1/chat/completions".
    param api_key: 调用 LLM 接口所需的 API key.
    param context_window: LLM 的上下文窗口大小, 用于控制发送给 LLM 的历史对话的长度.
    param timeout_seconds: 调用 LLM 接口的超时时间, 单位为秒.
    param headers: 调用 LLM 接口时需要使用的 HTTP headers, 例如 {"Authorization": "Bearer sk-xxxx"}.
    """

    baseurl: str
    api_key: str
    modules: str
    context_window: int = 4096
    timeout_seconds: int = 30
    headers: dict[str, str] = field(default_factory=dict)


class BaseRunner:

    """BaseRunner 类。
    这是所有 Runner 类的基类, 提供了基本的运行逻辑和状态管理.
    """
    def __init__(
        self,
        *,
        mode: Mode,
        runner_config: RunnerConfig,
        request_base_info: RequestBaseInfo | None = None,
    ) -> None:
        self.mode: Mode = mode
        self.runner_config = runner_config
        self.runner_context = RunnerContext(step=0, history=[])
        self.request_base_info = request_base_info or _get_default_request_base_info()

    def decide(self, ctx: RunnerContext) -> StepResult:
        """
        根据当前的上下文信息做出决策.

        param ctx: 当前的上下文信息, 包括当前是第几步了, 以及到目前为止的对话历史等.
        return: 决策的内容, 包括是调用工具还是返回最终结果等信息.
        """
        raise NotImplementedError("BaseRunner 是一个抽象类, 请实现 decide 方法")

    def run(self, question: str, id: str | None = None) -> RunResult:
        """run 方法。
        
        - 这个类一般是一个循环, 用于处理用户的问题并生成回复.
        
        参数:
        - question: 用户的初始问题
        - id: 会话ID，用于导入和保存历史记录
        
        返回:
        - RunResult: 返回任务的运行结果, 包括是否成功完成, 最终结果, 错误信息等.
        """
        raise NotImplementedError("BaseRunner 是一个抽象类, 请实现 run 方法")


def _get_default_request_base_info() -> RequestBaseInfo | None:
    # 从本地环境变量或者配置文件中获取默认的 RequestBaseInfo
    """
    _get_default_request_base_info 函数。
    用途:
    - 加载默认的请求基础信息, 用于初始化 Runner 时使用.
    
    """
    return load_request_base_info()
