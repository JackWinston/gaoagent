from __future__ import annotations

from gaoagent.core.runner.Console import Console
from gaoagent.core.ChatRunner import ChatRunner
from gaoagent.core.CoreConfigDefault import CoreConfigDefault
from gaoagent.core.CoreInit import CoreInit
from gaoagent.core.TaskRunner import TaskRunner


class CoreHandlers:
    """CLI 核心命令处理器（Core 层入口聚合）。

    该类在架构中的定位:
    - 位于 CLI 命令与核心业务模块之间，承担"命令分发 + 参数透传"职责。
    - 自身不承载复杂业务逻辑，主要负责把用户动作路由到对应模块。
    - 通过统一入口降低命令层耦合，便于后续扩展新命令子域。

    设计意图:
    - `CoreHandlers` 作为"薄控制层（thin handler）"，将初始化、配置、任务执行、
      聊天等不同能力聚合到一个稳定的调用面。
    - 各方法尽量保持无状态、短路径调用，避免在 Handler 层堆积业务规则，
      把真正规则下沉到 `CoreInit`、`CoreConfigDefault`、`TaskRunner`、`ChatRunner` 等专职组件。

    使用方式:
    - 通常由 CLI 命令注册层实例化后调用，如 `core init` / `core config` / `core task`。
    - 本类方法以"副作用执行"为主（打印、写配置、触发任务），而不是返回富结果对象。

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
        images: str | None = None,
    ) -> None:
        """聊天命令入口：将参数交给 ChatRunner 执行。

        调用链:
        - `CoreHandlers.chat()` -> `ChatRunner().run(new, prompt, api, model, context_size, images)`

        用法:
        - `gaoagent chat`：进入持续交互式聊天，输入 exit 退出。
        - `gaoagent chat --new`：丢弃历史上下文，开启全新会话。
        - `gaoagent chat --prompt "你好"`：单次发送一条消息并输出结果。
        - `gaoagent chat --api openai --model gpt-4.1`：指定 API 和模型。
        - `gaoagent chat --context-size 20`：限制上下文窗口为最近 20 条消息。
        - `gaoagent chat --prompt "描述图片" --images image.png`：发送带图片的消息。

        参数:
        - `new`: 是否重置上下文并开始新会话。为 True 时清空已有历史。
        - `prompt`: 单次模式的用户输入；传入后发送一条消息即退出。
          未传时进入持续交互式聊天循环。
        - `api`: 指定已保存的 API 提供方名称；为空时使用默认 API。
        - `model`: 指定模型名称；为空时使用默认模型。
        - `context_size`: 上下文消息窗口大小（消息条数）。
          超出时自动裁剪最早的非 system 消息。
        - `images`: 图片路径列表（逗号分隔），用于多模态输入。

        返回:
        - `None`。结果通过终端输出反馈给用户。
        """
        ChatRunner().run(
            new=new,
            prompt=prompt,
            api=api,
            model=model,
            context_size=context_size,
            images=images,
        )

    def task(self,question:str,mode:str,id:str|None=None,images:str|None=None) -> None:
        """任务执行入口：将问题交给 TaskRunner 执行并输出结果。

        作用:
        - 作为 CLI `task` 子命令的薄路由层，将参数原样交给 `TaskRunner`。
        - 由 `TaskRunner` 负责模式选择、Runner 实例化、日志上下文管理与结果输出。

        参数:
        - `question`: 用户任务描述，最终作为 Runner 的输入问题。
        - `mode`: 运行模式字符串（如 `react` / `plan` / `retry`）。
        - `id`: 会话ID，用于历史记录。
        - `images`: 图片路径列表（逗号分隔），用于多模态输入。

        调用链:
        - `CoreHandlers.task()` -> `TaskRunner.run(question, mode, id, images)` ->
          `ReActRunner.run(question, id, images)`（当前三个模式均落到 ReActRunner）。

        返回:
        - `None`。任务结果通过终端输出反馈给用户。
        """
        TaskRunner().run(question,mode,id=id,images=images)
