import click
from gaoagent.core.CoreConfig import CoreConfig


class CoreHandlers:
    def init(self) -> None:
        click.echo("CoreHandlers.init")

    def config(self) -> None:
       CoreConfig().config()

    def chat(
        self,
        new: bool = False,
        prompt: str | None = None,
        api: str | None = None,
        model: str | None = None,
        context_size: int | None = None,
    ) -> None:
        click.echo(f"CoreHandlers.chat , new={new}, prompt={prompt}, api={api}, model={model}, context_size={context_size}")

    def task(self) -> None:
        click.echo("CoreHandlers.task")
