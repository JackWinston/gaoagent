from __future__ import annotations

import json
from typing import Any

import click

from gaoagent.core.runner.Utils import global_config_dir


class Console:
    """统一控制台输出入口。

    架构定位:
    - 集中管理 CLI 输出，避免在各模块中直接散落调用 `click.echo` 与内置 `print`。
    - 为后续统一样式、重定向输出或测试替身注入提供稳定抽象层。

    设计意图:
    - 当前底层仍委托给 `click.echo`，保持既有 CLI 行为一致。
    - 保留 `print` 兼容别名，便于历史代码迁移并显式收口输出调用。
    """

    @staticmethod
    def echo(message: Any = "", *, err: bool = False, nl: bool = True) -> None:
        """输出一条控制台消息。"""
        click.echo(message, err=err, nl=nl)

    @staticmethod
    def print(message: Any = "", *, err: bool = False, nl: bool = True) -> None:
        """兼容历史 `print` 风格的输出别名。"""
        Console.echo(message=message, err=err, nl=nl)

    @staticmethod
    def info(message: Any) -> None:
        """输出普通信息。"""
        Console.echo(message)

    @staticmethod
    def error(message: Any) -> None:
        """输出错误信息（红色，标准错误流）。"""
        text = click.style(str(message), fg="red")
        click.echo(text, err=True)

    @staticmethod
    def weak(message: Any) -> None:
        """输出弱提示信息（比 info 更浅）。"""
        text = click.style(str(message), fg="bright_black", dim=True)
        click.echo(text)

    @staticmethod
    def interaction(message: Any) -> None:
        """输出交互提示信息（蓝色）。"""
        text = Console.interaction_text(message)
        click.echo(text)

    @staticmethod
    def warn(message: Any) -> None:
        """输出警告信息（黄色）。"""
        text = click.style(str(message), fg="yellow")
        click.echo(text)

    @staticmethod
    def fatal(message: Any) -> None:
        """输出导致程序终止的错误信息（红色）。"""
        text = click.style(str(message), fg="red")
        click.echo(text, err=True)

    @staticmethod
    def debug(message: Any) -> None:
        """按全局开关输出 debug 信息（颜色与 info 一致）。"""
        if not _is_debug_enabled():
            return
        Console.info(message)

    @staticmethod
    def interaction_text(message: Any) -> str:
        """构造蓝色交互提示文本（供 prompt/confirm 等输入场景复用）。"""
        return click.style(str(message), fg="blue")

    @staticmethod
    def prompt(text: Any, *args, **kwargs):
        """交互输入：提示文案使用蓝色。"""
        return click.prompt(Console.interaction_text(text), *args, **kwargs)

    @staticmethod
    def confirm(text: Any, *args, **kwargs):
        """交互确认：提示文案使用蓝色。"""
        return click.confirm(Console.interaction_text(text), *args, **kwargs)


def _is_debug_enabled() -> bool:
    """读取全局配置目录 `env.json` 的 `debug` 字段。"""
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
