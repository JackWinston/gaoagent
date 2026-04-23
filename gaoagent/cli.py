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


@rag_group.command("list", help="列出知识库")
def rag_list_cmd() -> None:
    dispatch("rag.list")


@rag_group.command("add", help="新增知识库（可选自定义切片器）")
@click.argument("name", required=False)
@click.argument("chunker_py_file", required=False)
def rag_add_cmd(name: str | None, chunker_py_file: str | None) -> None:
    """
    新增知识库。

    用法：
    - gaoagent rag add [name] [chunker_py_file]

    参数：
    - name: 知识库名称（可省略，省略时进入交互输入）
    - chunker_py_file: 自定义切片器 Python 文件路径（可省略）

    自定义切片器方法签名（在 chunker_py_file 中定义）：
    - def chunk_document(*, kb_name, kb_dir, file_path, text, chunk_size, chunk_overlap): ...

    返回值要求：
    - list[str]，每个元素是一段切片文本；或
    - list[dict]，每项可包含：
      - id: str（可选）
      - document: str（必填）
      - metadata: dict（可选）
    """
    dispatch("rag.add", name=name, chunker_py_file=chunker_py_file)


@rag_group.command("remove", help="移除知识库")
@click.argument("name", required=False)
def rag_remove_cmd(name: str | None) -> None:
    dispatch("rag.remove", name=name)


@rag_group.command("search", help="在知识库中检索")
@click.argument("kb_name", required=True)
@click.argument("query", required=True)
@click.option("--top-k", type=int, default=5, help="返回结果数量")
def rag_search_cmd(kb_name: str, query: str, top_k: int) -> None:
    dispatch("rag.search", kb_name=kb_name, query=query, top_k=top_k)


def main() -> None:
    cli(prog_name="gaoagent")
