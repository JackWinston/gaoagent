import click
from gaoagent.core.CoreConfigDefault import CoreConfigDefault
from gaoagent.core.CoreInit import CoreInit
from gaoagent.core.TaskRunner import TaskRunner


class CoreHandlers:
    def init(self) -> None:
       CoreInit().init()

    def config(self) -> None:
       CoreConfigDefault().config()

    def chat(
        self,
        new: bool = False,
        prompt: str | None = None,
        api: str | None = None,
        model: str | None = None,
        context_size: int | None = None,
    ) -> None:
        click.echo(f"CoreHandlers.chat , new={new}, prompt={prompt}, api={api}, model={model}, context_size={context_size}")

    def task(self,question:str,mode:str) -> None:
        TaskRunner().run(question,mode)
