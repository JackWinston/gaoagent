from __future__ import annotations

import json
from typing import Any

from rich.console import Console as RichConsole
from rich.text import Text
from rich.theme import Theme

from gaoagent.core.runner.Utils import global_config_dir

# 自定义主题
_theme = Theme({
    "info": "default",
    "warning": "yellow",
    "error": "bold red",
    "fatal": "bold red",
    "weak": "dim",
    "interaction": "blue",
    "debug": "dim cyan",
    "step": "bold cyan",
    "tool": "green",
    "reasoning": "dim italic",
    "content": "default",
})

# 全局 Rich Console 实例
_rich = RichConsole(theme=_theme, highlight=False)


class Console:
    """统一控制台输出入口（基于 Rich）。

    架构定位:
    - 集中管理 CLI 输出，避免在各模块中直接散落调用 print。
    - 基于 Rich 库提供丰富的终端输出效果。
    - 为后续统一样式、重定向输出或测试替身注入提供稳定抽象层。

    设计意图:
    - 底层委托给 Rich Console，保持接口一致性。
    - 保留原有方法签名，业务代码无需改动。
    """

    @staticmethod
    def echo(message: Any = "", *, err: bool = False, nl: bool = True) -> None:
        """输出一条控制台消息。"""
        if err:
            _rich.stderr = True
            _rich.print(message, end="\n" if nl else "")
            _rich.stderr = False
        else:
            _rich.print(message, end="\n" if nl else "")

    @staticmethod
    def print(message: Any = "", *, err: bool = False, nl: bool = True) -> None:
        """兼容历史 print 风格的输出别名。"""
        Console.echo(message=message, err=err, nl=nl)

    @staticmethod
    def info(message: Any) -> None:
        """输出普通信息。"""
        _rich.print(message)

    @staticmethod
    def error(message: Any) -> None:
        """输出错误信息（红色，标准错误流）。"""
        _rich.print(f"[error]{message}[/error]", stderr=True)

    @staticmethod
    def weak(message: Any) -> None:
        """输出弱提示信息（比 info 更浅）。"""
        _rich.print(f"[weak]{message}[/weak]")

    @staticmethod
    def interaction(message: Any) -> None:
        """输出交互提示信息（蓝色）。"""
        _rich.print(f"[interaction]{message}[/interaction]")

    @staticmethod
    def warn(message: Any) -> None:
        """输出警告信息（黄色）。"""
        _rich.print(f"[warning]{message}[/warning]")

    @staticmethod
    def fatal(message: Any) -> None:
        """输出导致程序终止的错误信息（红色）。"""
        _rich.print(f"[fatal]{message}[/fatal]", stderr=True)

    @staticmethod
    def debug(message: Any) -> None:
        """按全局开关输出 debug 信息。"""
        if not _is_debug_enabled():
            return
        _rich.print(f"[debug]{message}[/debug]")

    @staticmethod
    def interaction_text(message: Any) -> str:
        """构造蓝色交互提示文本（供 prompt/confirm 等输入场景复用）。"""
        return f"[interaction]{message}[/interaction]"

    @staticmethod
    def prompt(text: Any, *args, **kwargs):
        """交互输入：提示文案使用蓝色。"""
        from rich.prompt import Prompt
        return Prompt.ask(f"[interaction]{text}[/interaction]", *args, **kwargs)

    @staticmethod
    def confirm(text: Any, *args, **kwargs):
        """交互确认：提示文案使用蓝色。"""
        from rich.prompt import Confirm
        return Confirm.ask(f"[interaction]{text}[/interaction]", *args, **kwargs)

    # ========== 流式输出专用方法 ==========

    @staticmethod
    def stream_weak(text: str) -> None:
        """流式输出弱提示（不换行，用于实时输出）。"""
        _rich.print(f"[weak]{text}[/weak]", end="")

    @staticmethod
    def stream_info(text: str) -> None:
        """流式输出普通信息（不换行，用于实时输出）。"""
        _rich.print(text, end="")

    @staticmethod
    def stream_flush() -> None:
        """刷新流式输出缓冲区。"""
        _rich.file.flush()

    # ========== Rich 特色方法 ==========

    @staticmethod
    def rule(title: str = "", style: str = "dim") -> None:
        """输出分隔线。"""
        _rich.rule(title, style=style)

    @staticmethod
    def status(message: str, spinner: str = "dots") -> Any:
        """返回状态上下文管理器（带 spinner）。"""
        return _rich.status(message, spinner=spinner)

    @staticmethod
    def progress() -> Any:
        """返回进度条上下文管理器。"""
        from rich.progress import Progress
        return Progress()

    @staticmethod
    def table(title: str | None = None) -> Any:
        """返回表格对象。"""
        from rich.table import Table
        return Table(title=title, show_header=True, header_style="bold")

    @staticmethod
    def panel(message: Any, title: str | None = None, border_style: str = "blue") -> None:
        """输出面板。"""
        from rich.panel import Panel
        _rich.print(Panel(str(message), title=title, border_style=border_style))

    @staticmethod
    def markdown(content: str) -> None:
        """渲染 Markdown 内容。"""
        from rich.markdown import Markdown
        _rich.print(Markdown(content))

    @staticmethod
    def syntax(code: str, language: str = "python", theme: str = "monokai") -> None:
        """输出语法高亮代码。"""
        from rich.syntax import Syntax
        _rich.print(Syntax(code, language, theme=theme))


def _is_debug_enabled() -> bool:
    """读取全局配置目录 env.json 的 debug 字段。"""
    env_file = global_config_dir() / "env.json"
    if not env_file.exists() or not env_file.is_file():
        return False
    try:
        payload = json.loads(env_file.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(payload, dict):
        return False
    raw = payload.get("debug", False)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    if isinstance(raw, str):
        return raw.strip().lower() in {"1", "true", "yes", "on", "y"}
    return False
