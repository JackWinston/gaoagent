import click

class SkillsHandlers:
    def list(self) -> None:
        click.echo("SkillsHandlers.list")

    def install(self) -> None:
        click.echo("SkillsHandlers.install")

    def uninstall(self) -> None:
        click.echo("SkillsHandlers.uninstall")

    def enable(self) -> None:
        click.echo("SkillsHandlers.enable")

    def disable(self) -> None:
        click.echo("SkillsHandlers.disable")
