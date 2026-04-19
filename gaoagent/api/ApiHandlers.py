import click


class ApiHandlers:
    def list(self) -> None:
        click.echo("ApiHandlers.list")

    def add(self) -> None:
        click.echo("ApiHandlers.add")

    def edit(self) -> None:
        click.echo("ApiHandlers.edit")

    def remove(self) -> None:
        click.echo("ApiHandlers.remove")