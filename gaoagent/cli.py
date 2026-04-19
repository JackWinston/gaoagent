import click

from gaoagent import __version__
from gaoagent.router import dispatch


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(version=__version__, prog_name="gaoagent")
def cli() -> None:
    """AI Agent CLI 工具"""


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
def task_cmd() -> None:
    dispatch("task")


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
def mcp_remove_cmd() -> None:
    dispatch("mcp.remove")


@mcp_group.command("enable", help="启用 MCP")
def mcp_enable_cmd() -> None:
    dispatch("mcp.enable")


@mcp_group.command("disable", help="禁用 MCP")
def mcp_disable_cmd() -> None:
    dispatch("mcp.disable")


@mcp_group.command("test", help="测试 MCP")
def mcp_test_cmd() -> None:
    dispatch("mcp.test")


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


@skills_group.command("enable", help="启用 skill")
def skills_enable_cmd() -> None:
    dispatch("skills.enable")


@skills_group.command("disable", help="禁用 skill")
def skills_disable_cmd() -> None:
    dispatch("skills.disable")


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
