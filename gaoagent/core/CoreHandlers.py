from gaoagent.core.runner.Console import Console
from gaoagent.core.CoreConfigDefault import CoreConfigDefault
from gaoagent.core.CoreInit import CoreInit
from gaoagent.core.TaskRunner import TaskRunner


class CoreHandlers:
    """CLI 核心命令处理器（Core 层入口聚合）。

    该类在架构中的定位:
    - 位于 CLI 命令与核心业务模块之间，承担“命令分发 + 参数透传”职责。
    - 自身不承载复杂业务逻辑，主要负责把用户动作路由到对应模块。
    - 通过统一入口降低命令层耦合，便于后续扩展新命令子域。

    设计意图:
    - `CoreHandlers` 作为“薄控制层（thin handler）”，将初始化、配置、任务执行等
      不同能力聚合到一个稳定的调用面。
    - 各方法尽量保持无状态、短路径调用，避免在 Handler 层堆积业务规则，
      把真正规则下沉到 `CoreInit`、`CoreConfigDefault`、`TaskRunner` 等专职组件。

    使用方式:
    - 通常由 CLI 命令注册层实例化后调用，如 `core init` / `core config` / `core task`。
    - 本类方法以“副作用执行”为主（打印、写配置、触发任务），而不是返回富结果对象。

    边界说明:
    - 不负责模型推理细节，不直接管理 Runner 的 step/history。
    - 不负责持久化协议定义，仅触发对应模块执行。
    """
    def init(self) -> None:
       """执行系统初始化流程。

       作用:
       - 触发 `CoreInit().init()`，完成项目/全局运行所需的基础初始化动作
         （如目录、默认配置、运行前置条件等，具体由 `CoreInit` 实现定义）。

       调用链:
       - `CoreHandlers.init()` -> `CoreInit.init()`

       参数:
       - 无。

       返回:
       - `None`。该方法通过副作用完成工作，不返回业务数据。
       """
       CoreInit().init()

    def config(self) -> None:
       """生成或更新默认配置。

       作用:
       - 触发 `CoreConfigDefault().config()`，用于落地默认配置模板或修复缺失配置项。
       - 该入口常用于首次使用时准备环境，或在配置异常时执行恢复。

       调用链:
       - `CoreHandlers.config()` -> `CoreConfigDefault.config()`

       参数:
       - 无。

       返回:
       - `None`。执行结果通过副作用体现（写入配置、终端输出等）。
       """
       CoreConfigDefault().config()

    def chat(
        self,
        new: bool = False,
        prompt: str | None = None,
        api: str | None = None,
        model: str | None = None,
        context_size: int | None = None,
    ) -> None:
        """聊天命令入口（当前为占位实现）。

        当前行为:
        - 仅将传入参数打印到终端，便于联调 CLI 参数解析是否正确。
        - 尚未接入真实对话会话管理与 Runner 编排。

        参数语义:
        - `new`: 是否开启新会话；`True` 通常表示丢弃旧会话上下文。
        - `prompt`: 本次输入的用户问题/提示词。
        - `api`: 目标 API 标识或端点别名（具体解释由上层约定）。
        - `model`: 指定模型名称。
        - `context_size`: 期望上下文窗口大小。

        返回:
        - `None`。仅做控制台输出，不返回结构化对话结果。

        后续扩展建议:
        - 可将此方法对齐 `task()` 的执行路径，复用 `TaskRunner` 或会话态 Runner。
        """
        Console.echo(f"CoreHandlers.chat , new={new}, prompt={prompt}, api={api}, model={model}, context_size={context_size}")

    def task(self,question:str,mode:str) -> None:
        """任务执行入口：将问题交给 TaskRunner 执行并输出结果。

        作用:
        - 作为 CLI `task` 子命令的薄路由层，将参数原样交给 `TaskRunner`。
        - 由 `TaskRunner` 负责模式选择、Runner 实例化、日志上下文管理与结果输出。

        参数:
        - `question`: 用户任务描述，最终作为 Runner 的输入问题。
        - `mode`: 运行模式字符串（如 `react` / `plan` / `retry`）。

        调用链:
        - `CoreHandlers.task()` -> `TaskRunner.run(question, mode)` ->
          `ReActRunner.run(question)`（当前三个模式均落到 ReActRunner）。

        返回:
        - `None`。任务结果通过终端输出反馈给用户。
        """
        TaskRunner().run(question,mode)
