from __future__ import annotations

from typing import Any

import click


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
        """输出错误信息到标准错误流。"""
        Console.echo(message, err=True)
