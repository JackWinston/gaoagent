import click

from gaoagent import __version__
from gaoagent.router import dispatch


@click.group(
    context_settings={"help_option_names": ["-h", "--help"]},
    invoke_without_command=True,
)
@click.version_option(version=__version__, prog_name="gaoagent")
@click.option("--task", "task_question", type=str, help="创建并运行一个任务")
@click.option(
    "--mode",
    type=click.Choice(["plan", "react", "retry"], case_sensitive=False),
    default="react",
    show_default=True,
    help="运行模式（配合 --task 使用）",
)
@click.pass_context
def cli(ctx: click.Context, task_question: str | None, mode: str) -> None:
    """AI Agent CLI 工具"""
    q = (task_question or "").strip()
    if q:
        dispatch("task", question=q, mode=mode)
        return


@cli.command("init", help="在当前目录初始化工具")
def init_cmd() -> None:
    dispatch("init")


@cli.command("config", help="为项目配置默认信息")
def config_cmd() -> None:
    dispatch("config")


@cli.command("chat", help="开始聊天")
@click.option("--new", is_flag=True, help="重置上下文，重新开始聊天")
@click.option("--prompt", type=str, help="为聊天输入用户 prompt")
@click.option("--api", type=str, help="指定已经保存过的 API 厂家")
@click.option("--model", type=str, help="指定模型")
@click.option("--context-size", "--contextSize", "context_size", type=int, help="指定上下文长度")
def chat_cmd(
    new: bool,
    prompt: str | None,
    api: str | None,
    model: str | None,
    context_size: int | None,
) -> None:
    dispatch("chat", new=new, prompt=prompt, api=api, model=model, context_size=context_size)


@cli.command("task", help="创建一个任务")
@click.argument("question", required=False)
@click.option(
    "--mode",
    type=click.Choice(["plan", "react", "retry"], case_sensitive=False),
    default="react",
    show_default=True,
    help="运行模式",
)
def task_cmd(question: str | None, mode: str) -> None:
    """
    创建并运行一个任务。

    - question: 任务描述（可省略，省略时会进入交互式输入）
    - mode: 执行模式（plan/react/retry）
    """
    q = (question or "").strip()
    if not q:
        q = click.prompt("请输入任务", type=str).strip()
    dispatch("task", question=q, mode=mode)


@cli.group("mcp", help="MCP 相关命令")
def mcp_group() -> None:
    pass


@mcp_group.command("list", help="列出 MCP")
def mcp_list_cmd() -> None:
    dispatch("mcp.list")


@mcp_group.command("add", help="添加 MCP")
def mcp_add_cmd() -> None:
    dispatch("mcp.add")


@mcp_group.command("remove", help="移除 MCP")
@click.argument("name", required=False)
def mcp_remove_cmd(name: str | None) -> None:
    dispatch("mcp.remove", name=name)


@mcp_group.command("enable", help="启用 MCP")
@click.argument("name", required=False)
def mcp_enable_cmd(name: str | None) -> None:
    dispatch("mcp.enable", name=name)


@mcp_group.command("disable", help="禁用 MCP")
@click.argument("name", required=False)
def mcp_disable_cmd(name: str | None) -> None:
    dispatch("mcp.disable", name=name)


@mcp_group.command("test", help="测试 MCP")
@click.argument("name", required=False)
def mcp_test_cmd(name: str | None) -> None:
    dispatch("mcp.test", name=name)


@cli.group("skills", help="Skills 相关命令")
def skills_group() -> None:
    pass


@skills_group.command("list", help="列出 skills")
def skills_list_cmd() -> None:
    dispatch("skills.list")


@skills_group.command("install", help="安装 skill")
def skills_install_cmd() -> None:
    dispatch("skills.install")


@skills_group.command("uninstall", help="卸载 skill")
def skills_uninstall_cmd() -> None:
    dispatch("skills.uninstall")


@cli.group("rag", help="RAG 相关命令")
def rag_group() -> None:
    pass


@rag_group.command("index", help="构建索引")
def rag_index_cmd() -> None:
    dispatch("rag.index")


@rag_group.command("query", help="查询")
def rag_query_cmd() -> None:
    dispatch("rag.query")


@rag_group.command("status", help="查看状态")
def rag_status_cmd() -> None:
    dispatch("rag.status")


@rag_group.command("clear", help="清理索引")
def rag_clear_cmd() -> None:
    dispatch("rag.clear")


@cli.group("api", help="OpenAI API 相关命令")
def api_group() -> None:
    pass


@api_group.command("list", help="列出 API 配置")
def api_list_cmd() -> None:
    dispatch("api.list")


@api_group.command("add", help="添加 API 配置")
def api_add_cmd() -> None:
    dispatch("api.add")


@api_group.command("edit", help="编辑 API 配置")
def api_edit_cmd() -> None:
    dispatch("api.edit")


@api_group.command("remove", help="移除 API 配置")
def api_remove_cmd() -> None:
    dispatch("api.remove")


def main() -> None:
    cli(prog_name="gaoagent")
