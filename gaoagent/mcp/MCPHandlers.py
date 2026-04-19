import click

class MCPHandlers:
    def list(self) -> None:
        click.echo("MCPHandlers.list")

    def add(self) -> None:
        click.echo("MCPHandlers.add")

    def remove(self) -> None:
        click.echo("MCPHandlers.remove")

    def enable(self) -> None:
        click.echo("MCPHandlers.enable")

    def disable(self) -> None:
        click.echo("MCPHandlers.disable")

    def test(self) -> None:
        click.echo("MCPHandlers.test")