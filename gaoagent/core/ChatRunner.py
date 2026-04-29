from __future__ import annotations

from typing import Any

from gaoagent.core.runner.Console import Console
from gaoagent.core.runner.Utils import build_multimodal_content, is_image_file


def _trim_messages(messages: list[dict[str, Any]], context_size: int | None) -> list[dict[str, Any]]:
    """将消息列表裁剪到 context_size 条以内，始终保留 system 消息。

    用途:
    - 当对话轮次超出上下文窗口限制时，丢弃最早的消息以控制发送给模型的历史长度。

    参数:
    - messages: 完整消息列表。
    - context_size: 期望保留的最大消息条数；为空或 <= 0 时不裁剪。

    返回:
    - 裁剪后的消息列表。
    """
    if not context_size or context_size <= 0:
        return messages
    if len(messages) <= context_size:
        return messages
    has_system = bool(messages) and messages[0].get("role") == "system"
    if has_system:
        body = messages[1:]
        keep = max(0, context_size - 1)
        return [messages[0]] + body[-keep:]
    return messages[-context_size:]


def _extract_content(response_json: dict[str, Any]) -> str | None:
    """从 OpenAI 兼容响应中提取助手回复文本。

    参数:
    - response_json: Chat Completions 接口返回的 JSON 对象。

    返回:
    - 助手回复文本；无法提取时返回 None。
    """
    choices = response_json.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0] if isinstance(choices[0], dict) else {}
    message = first.get("message") if isinstance(first.get("message"), dict) else {}
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content
    return None


_SESSION_ID = "chat"


class ChatRunner:
    """聊天执行器（CLI `chat` 子命令的执行入口）。

    定位:
    - 位于 `CoreHandlers.chat()` 与底层 LLM HTTP 客户端之间。
    - 负责聊天会话的完整生命周期：历史加载/保存、消息收发、上下文裁剪。

    与 TaskRunner 的区别:
    - 不使用 ReAct 推理循环，不调用工具，仅做纯对话。
    - 支持持续交互式聊天循环（输入 exit 退出）。
    - 对话历史固定存储在项目级 `.gaoagent/history/chat.json`。

    核心职责:
    - 从配置中加载 API 信息（支持按名称覆盖默认值）。
    - 管理对话历史的持久化（每次交互后立即写盘）。
    - 支持 `--context-size` 裁剪窗口，始终保留 system 消息。
    - 提供单次模式（`--prompt`）和持续交互模式两种入口。
    """

    def run(
        self,
        new: bool = False,
        prompt: str | None = None,
        api: str | None = None,
        model: str | None = None,
        context_size: int | None = None,
        images: str | None = None,
    ) -> None:
        """聊天命令入口：与 LLM 进行多轮对话。

        用法:
        - `gaoagent chat`：进入持续交互式聊天，输入 exit 退出。
        - `gaoagent chat --new`：丢弃历史上下文，开启全新会话。
        - `gaoagent chat --prompt "你好"`：单次发送一条消息并输出结果。
        - `gaoagent chat --api openai --model gpt-4.1`：指定 API 和模型。
        - `gaoagent chat --context-size 20`：限制上下文窗口为最近 20 条消息。

        参数:
        - `new`: 是否重置上下文并开始新会话。为 True 时清空已有历史。
        - `prompt`: 单次模式的用户输入；传入后发送一条消息即退出。
          未传时进入持续交互式聊天循环。
        - `api`: 指定已保存的 API 提供方名称；为空时使用默认 API。
        - `model`: 指定模型名称；为空时使用默认模型。
        - `context_size`: 上下文消息窗口大小（消息条数）。
          超出时自动裁剪最早的非 system 消息。

        行为:
        1) 加载 API 配置（支持 `--api` / `--model` 覆盖默认值）。
        2) 从项目级 `.gaoagent/history/chat.json` 加载对话历史。
           - 若 `--new` 为 True，则忽略已有历史，从空会话开始。
        3) 若传入 `--prompt`，单次发送消息并输出回复后退出。
        4) 否则进入交互式循环：
           - 提示用户输入，输入 exit 或按 Ctrl+C 退出。
           - 将用户消息追加到历史，调用 LLM 接口获取回复。
           - 输出回复并追加到历史，每次交互后持久化历史。

        历史持久化:
        - 对话历史固定存储在项目级 `.gaoagent/history/chat.json`。
        - 每轮交互结束后立即写入磁盘，确保断电/中断不丢失。

        返回:
        - `None`。结果通过终端输出反馈给用户。
        """
        from gaoagent.core.runner.HttpClient import OpenAICompatibleHttpClient
        from gaoagent.core.runner.Utils import load_history, load_request_base_info

        base_info = load_request_base_info(api_name=api, model_name=model)
        if not base_info:
            Console.fatal("没找到可用的 API 配置，请先运行 `gaoagent api add` 或 `gaoagent config`。")
            return

        client = OpenAICompatibleHttpClient(
            base_url=base_info.baseurl,
            api_key=base_info.api_key,
        )

        if new:
            history: list[dict[str, Any]] = []
        else:
            history = load_history(_SESSION_ID) or []

        system_text = "You are a helpful assistant."
        if history and history[0].get("role") == "system":
            history[0]["content"] = system_text
        else:
            history.insert(0, {"role": "system", "content": system_text})

        if prompt is not None:
            self._single(client, base_info.modules, history, prompt, context_size, images)
            return

        self._loop(client, base_info.modules, history, context_size, images)

    def _single(
        self,
        client: Any,
        model: str,
        history: list[dict[str, Any]],
        prompt: str,
        context_size: int | None,
        images: str | None = None,
    ) -> None:
        """单次聊天模式：发送一条消息，输出回复后退出。

        参数:
        - client: OpenAI 兼容 HTTP 客户端。
        - model: 模型名称。
        - history: 当前对话历史（会被就地追加）。
        - prompt: 用户输入文本。
        - context_size: 上下文消息窗口大小。
        - images: 图片路径列表（逗号分隔），用于多模态输入。
        """
        from gaoagent.core.runner.Utils import save_history

        text = prompt.strip()
        if not text:
            Console.warn("输入内容为空，跳过。")
            return

        # 解析图片路径
        image_paths: list[str] = []
        if images:
            paths = [p.strip() for p in images.split(",") if p.strip()]
            for p in paths:
                if is_image_file(p):
                    image_paths.append(p)

        # 构建多模态内容
        user_content = build_multimodal_content(text, image_paths)
        history.append({"role": "user", "content": user_content})
        Console.interaction("正在请求数据...")
        response = client.post_chat_completions(
            model=model,
            messages=_trim_messages(history, context_size),
        )

        if response.ok and response.json:
            content = _extract_content(response.json)
            if content:
                Console.info(content)
                history.append({"role": "assistant", "content": content})
            else:
                Console.warn("模型返回了空内容。")
        else:
            Console.fatal(f"请求失败：{response.reason or 'unknown error'}")

        save_history(_SESSION_ID, history)

    def _loop(
        self,
        client: Any,
        model: str,
        history: list[dict[str, Any]],
        context_size: int | None,
        images: str | None = None,
    ) -> None:
        """持续交互式聊天循环。

        参数:
        - client: OpenAI 兼容 HTTP 客户端。
        - model: 模型名称。
        - history: 当前对话历史（会被就地追加）。
        - context_size: 上下文消息窗口大小。
        - images: 图片路径列表（逗号分隔），用于多模态输入。
        """
        from gaoagent.core.runner.Utils import save_history

        Console.info("进入聊天模式（输入 exit 退出）")

        # 如果有图片，在循环开始前解析一次
        image_paths: list[str] = []
        if images:
            paths = [p.strip() for p in images.split(",") if p.strip()]
            for p in paths:
                if is_image_file(p):
                    image_paths.append(p)

        while True:
            try:
                user_input = Console.prompt("你")
            except (KeyboardInterrupt, EOFError):
                Console.info("\n已退出聊天。")
                save_history(_SESSION_ID, history)
                break

            text = (user_input or "").strip()
            if not text:
                continue
            if text.lower() == "exit":
                Console.info("已退出聊天。")
                save_history(_SESSION_ID, history)
                break

            # 构建多模态内容（每轮对话都传递图片）
            user_content = build_multimodal_content(text, image_paths)
            history.append({"role": "user", "content": user_content})

            Console.interaction("正在请求数据...")
            response = client.post_chat_completions(
                model=model,
                messages=_trim_messages(history, context_size),
            )

            if response.ok and response.json:
                content = _extract_content(response.json)
                if content:
                    Console.info(content)
                    history.append({"role": "assistant", "content": content})
                else:
                    Console.warn("模型返回了空内容。")
            else:
                Console.fatal(f"请求失败：{response.reason or 'unknown error'}")

            save_history(_SESSION_ID, history)
