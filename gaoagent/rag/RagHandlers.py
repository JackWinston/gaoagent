import click



class RagHandlers:
    def index(self) -> None:
        click.echo("RagHandlers.index")

    def query(self) -> None:
        click.echo("RagHandlers.query")

    def status(self) -> None:
        click.echo("RagHandlers.status")

    def clear(self) -> None:
        click.echo("RagHandlers.clear")