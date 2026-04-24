import click
from gaoagent.core.runner.Console import Console

from gaoagent import __version__
from gaoagent.router import dispatch


@click.group(
    context_settings={"help_option_names": ["-h", "--help"]},
    invoke_without_command=True,
    help=(
        "GaoAgent 命令行入口。\n"
        "支持两种模式：\n"
        "1) 子命令模式：gaoagent <command>\n"
        "2) 快捷任务：gaoagent --task \"任务描述\" --mode react|plan|retry\n"
        "示例：\n"
        "  gaoagent init\n"
        "  gaoagent task \"帮我写一个发布脚本\"\n"
        "  gaoagent --task \"分析这个仓库结构\" --mode plan"
    ),
)
@click.version_option(version=__version__, prog_name="gaoagent")
@click.option(
    "--task",
    "task_question",
    type=str,
    help=(
        "创建并运行一个任务（无需显式输入 task 子命令）。\n"
        "示例：gaoagent --task \"实现一个文件上传接口\" --mode react"
    ),
)
@click.option(
    "--mode",
    type=click.Choice(["plan", "react", "retry"], case_sensitive=False),
    default="react",
    show_default=True,
    help=(
        "运行模式（配合 --task 使用）：plan=先规划，react=边想边做，retry=重试。\n"
        "示例：gaoagent --task \"重构登录模块\" --mode plan"
    ),
)
@click.pass_context
def cli(ctx: click.Context, task_question: str | None, mode: str) -> None:
    """
    CLI 根命令入口。

    用法：
    - gaoagent [子命令] [参数]
    - gaoagent --task "任务描述" [--mode react|plan|retry]

    说明：
    - 支持 `--task` 快捷模式：无需显式输入 `task` 子命令即可直接执行任务。
    - 未传 `--task` 时，由 Click 按子命令路由到 `init/config/chat/task/mcp/skills/rag/api`。

    参数：
    - ctx: Click 上下文对象（用于命令分发上下文，不直接参与业务逻辑）。
    - task_question: 任务文本；非空时直接触发 `dispatch("task", ...)`。
    - mode: 任务运行模式，可选 `react/plan/retry`。
    """
    q = (task_question or "").strip()
    if q:
        dispatch("task", question=q, mode=mode)
        return


@cli.command(
    "init",
    help=(
        "在当前目录初始化 GaoAgent 项目配置。\n"
        "示例：gaoagent init"
    ),
)
def init_cmd() -> None:
    """
    初始化当前目录为 gaoagent 项目。

    用法：
    - gaoagent init

    行为：
    - 触发 `dispatch("init")`，由 `CoreHandlers.init()` 执行项目初始化流程。
    """
    dispatch("init")


@cli.command(
    "config",
    help=(
        "执行项目配置初始化/修复流程。\n"
        "示例：gaoagent config"
    ),
)
def config_cmd() -> None:
    """
    执行配置初始化/修复流程。

    用法：
    - gaoagent config

    行为：
    - 触发 `dispatch("config")`，用于补全或修复项目默认配置。
    """
    dispatch("config")


@cli.command(
    "chat",
    help=(
        "进入聊天模式，可指定 API、模型和上下文长度。\n"
        "示例：\n"
        "  gaoagent chat\n"
        "  gaoagent chat --prompt \"你好\"\n"
        "  gaoagent chat --api openai --model gpt-4.1"
    ),
)
@click.option(
    "--new",
    is_flag=True,
    help="重置上下文并开始新会话。示例：gaoagent chat --new",
)
@click.option(
    "--prompt",
    type=str,
    help="直接发送一条用户输入。示例：gaoagent chat --prompt \"帮我解释这段代码\"",
)
@click.option(
    "--api",
    type=str,
    help="指定已保存的 API 名称。示例：gaoagent chat --api openai",
)
@click.option(
    "--model",
    type=str,
    help="指定模型名称。示例：gaoagent chat --model gpt-4.1",
)
@click.option(
    "--context-size",
    "--contextSize",
    "context_size",
    type=int,
    help="指定上下文长度。示例：gaoagent chat --context-size 20",
)
def chat_cmd(
    new: bool,
    prompt: str | None,
    api: str | None,
    model: str | None,
    context_size: int | None,
) -> None:
    """
    聊天命令入口。

    用法：
    - gaoagent chat
    - gaoagent chat --new
    - gaoagent chat --prompt "你好"
    - gaoagent chat --api <api_name> --model <model_name> [--context-size 20]

    参数：
    - new: 是否重置上下文并开始新会话。
    - prompt: 可选用户输入；未传时进入交互式聊天输入。
    - api: 指定已保存的 API 提供方名称。
    - model: 指定模型名称。
    - context_size: 指定上下文窗口长度（消息条数）。

    行为：
    - 参数会透传给 `dispatch("chat", ...)`，最终由 `CoreHandlers.chat()` 处理。
    """
    dispatch("chat", new=new, prompt=prompt, api=api, model=model, context_size=context_size)


@cli.command(
    "task",
    help=(
        "创建并运行任务。\n"
        "示例：\n"
        "  gaoagent task \"生成接口文档\"\n"
        "  gaoagent task --mode plan \"先输出改造方案\""
    ),
)
@click.argument("question", required=False)
@click.option(
    "--mode",
    type=click.Choice(["plan", "react", "retry"], case_sensitive=False),
    default="react",
    show_default=True,
    help=(
        "任务执行模式：plan=先规划，react=边想边做，retry=重试。\n"
        "示例：gaoagent task --mode plan \"设计数据库表结构\""
    ),
)
def task_cmd(question: str | None, mode: str) -> None:
    """
    创建并运行一个任务。

    - question: 任务描述（可省略，省略时会进入交互式输入）
    - mode: 执行模式（plan/react/retry）目前只实现了 react 模式。

    用法：
    - gaoagent task "帮我重构这个模块"
    - gaoagent task --mode react "先做方案设计"
    - gaoagent task   （不传 question 时会提示输入）
    """
    q = (question or "").strip()
    if not q:
        q = Console.prompt("请输入任务", type=str).strip()
    dispatch("task", question=q, mode=mode)


@cli.group(
    "mcp",
    help=(
        "MCP 服务管理命令组：list/add/remove/enable/disable/test。\n"
        "示例：gaoagent mcp list"
    ),
)
def mcp_group() -> None:
    """
    MCP 子命令组入口。

    可用子命令：
    - list/add/remove/enable/disable/test
    """
    pass


@mcp_group.command(
    "list",
    help="列出 MCP 服务配置。示例：gaoagent mcp list",
)
def mcp_list_cmd() -> None:
    """
    列出 MCP 服务。

    用法：
    - gaoagent mcp list

    行为：
    - 触发 `dispatch("mcp.list")`，展示当前可用 MCP 配置。
    """
    dispatch("mcp.list")


@mcp_group.command(
    "add",
    help="添加 MCP 服务配置（交互式）。示例：gaoagent mcp add",
)
def mcp_add_cmd() -> None:
    """
    添加 MCP 服务配置。

    用法：
    - gaoagent mcp add

    行为：
    - 触发 `dispatch("mcp.add")`，在 Handler 中通过交互式流程新增配置。
    """
    dispatch("mcp.add")


@mcp_group.command(
    "remove",
    help="移除 MCP 服务。示例：gaoagent mcp remove my-mcp",
)
@click.argument("name", required=False)
def mcp_remove_cmd(name: str | None) -> None:
    """
    移除 MCP 服务。

    用法：
    - gaoagent mcp remove
    - gaoagent mcp remove <name>

    参数：
    - name: MCP 名称；可省略，省略时在 Handler 层交互选择。
    """
    dispatch("mcp.remove", name=name)


@mcp_group.command(
    "enable",
    help="启用 MCP 服务。示例：gaoagent mcp enable my-mcp",
)
@click.argument("name", required=False)
def mcp_enable_cmd(name: str | None) -> None:
    """
    启用 MCP 服务（设置 `disabled=false`）。

    用法：
    - gaoagent mcp enable
    - gaoagent mcp enable <name>

    参数：
    - name: MCP 名称；可省略，省略时在 Handler 层交互选择。
    """
    dispatch("mcp.enable", name=name)


@mcp_group.command(
    "disable",
    help="禁用 MCP 服务。示例：gaoagent mcp disable my-mcp",
)
@click.argument("name", required=False)
def mcp_disable_cmd(name: str | None) -> None:
    """
    禁用 MCP 服务（设置 `disabled=true`）。

    用法：
    - gaoagent mcp disable
    - gaoagent mcp disable <name>

    参数：
    - name: MCP 名称；可省略，省略时在 Handler 层交互选择。
    """
    dispatch("mcp.disable", name=name)


@mcp_group.command(
    "test",
    help="测试 MCP 服务连通性。示例：gaoagent mcp test my-mcp",
)
@click.argument("name", required=False)
def mcp_test_cmd(name: str | None) -> None:
    """
    测试 MCP 服务连通性和工具可用性。

    用法：
    - gaoagent mcp test
    - gaoagent mcp test <name>

    参数：
    - name: MCP 名称；可省略，省略时在 Handler 层交互选择。
    """
    dispatch("mcp.test", name=name)


@cli.group(
    "skills",
    help=(
        "Skills 管理命令组：list/add/remove。\n"
        "示例：gaoagent skills list"
    ),
)
def skills_group() -> None:
    """
    Skills 子命令组入口。

    可用子命令：
    - list/add/remove
    """
    pass


@skills_group.command(
    "list",
    help="列出当前作用域可见的 Skills。示例：gaoagent skills list",
)
def skills_list_cmd() -> None:
    """
    列出 Skills。

    用法：
    - gaoagent skills list

    行为：
    - 触发 `dispatch("skills.list")`，展示当前作用域可见的技能清单。
    """
    dispatch("skills.list")


@skills_group.command(
    "add",
    help="添加 Skill（交互式）。示例：gaoagent skills add",
)
def skills_add_cmd() -> None:
    """
    添加 Skill 到当前作用域。

    用法：
    - gaoagent skills add

    行为：
    - 触发 `dispatch("skills.add")`，在 Handler 层执行交互式添加流程。
    """
    dispatch("skills.add")


@skills_group.command(
    "remove",
    help="移除 Skill（交互式）。示例：gaoagent skills remove",
)
def skills_remove_cmd() -> None:
    """
    从当前作用域移除 Skill。

    用法：
    - gaoagent skills remove

    行为：
    - 触发 `dispatch("skills.remove")`，在 Handler 层执行交互式移除流程。
    """
    dispatch("skills.remove")


@cli.group(
    "rag",
    help=(
        "RAG 知识库命令组：list/add/update/remove/search。\n"
        "示例：gaoagent rag list"
    ),
)
def rag_group() -> None:
    """
    RAG 子命令组入口。

    可用子命令：
    - list/add/update/remove/search
    """
    pass


@rag_group.command(
    "list",
    help="列出知识库。示例：gaoagent rag list",
)
def rag_list_cmd() -> None:
    """
    列出知识库。

    用法：
    - gaoagent rag list

    行为：
    - 触发 `dispatch("rag.list")`，展示当前作用域知识库列表。
    """
    dispatch("rag.list")


@rag_group.command(
    "add",
    help=(
        "新增知识库（可选自定义切片器）。\n"
        "示例：gaoagent rag add mykb ./chunker.py"
    ),
)
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


@rag_group.command(
    "update",
    help=(
        "更新知识库（增量入库，可选自定义切片器）。\n"
        "示例：gaoagent rag update mykb ./chunker.py"
    ),
)
@click.argument("name", required=False)
@click.argument("chunker_py_file", required=False)
def rag_update_cmd(name: str | None, chunker_py_file: str | None) -> None:
    """
    更新知识库。

    用法：
    - gaoagent rag update [name] [chunker_py_file]

    参数：
    - name: 知识库名称（可省略，省略时进入交互选择）
    - chunker_py_file: 自定义切片器 Python 文件路径（可省略）
    """
    dispatch("rag.update", name=name, chunker_py_file=chunker_py_file)


@rag_group.command(
    "remove",
    help="移除知识库。示例：gaoagent rag remove mykb",
)
@click.argument("name", required=False)
def rag_remove_cmd(name: str | None) -> None:
    """
    移除知识库。

    用法：
    - gaoagent rag remove
    - gaoagent rag remove <name>

    参数：
    - name: 知识库名称；可省略，省略时由 Handler 交互选择。
    """
    dispatch("rag.remove", name=name)


@rag_group.command(
    "search",
    help="在知识库中检索。示例：gaoagent rag search mykb \"如何部署\" --top-k 5",
)
@click.argument("kb_name", required=True)
@click.argument("query", required=True)
@click.option(
    "--top-k",
    type=int,
    default=5,
    help="返回结果数量。示例：gaoagent rag search mykb \"向量检索\" --top-k 8",
)
def rag_search_cmd(kb_name: str, query: str, top_k: int) -> None:
    """
    在指定知识库中执行语义检索并输出结果。

    用法：
    - gaoagent rag search <kb_name> <query> [--top-k 5]

    参数：
    - kb_name: 目标知识库名称（必填）。
    - query: 检索问题或关键词（必填）。
    - top_k: 返回结果数量，默认 5。
    """
    dispatch("rag.search", kb_name=kb_name, query=query, top_k=top_k)


@cli.group(
    "api",
    help=(
        "API 配置命令组：list/add/remove/edit/default。\n"
        "示例：gaoagent api list"
    ),
)
def api_group() -> None:
    """
    API 配置子命令组入口。

    可用子命令：
    - list/add/remove/edit/default
    """
    pass


@api_group.command(
    "list",
    help="列出 API 配置。示例：gaoagent api list",
)
def api_list_cmd() -> None:
    """
    列出当前作用域 API 配置。

    用法：
    - gaoagent api list
    """
    dispatch("api.list")


@api_group.command(
    "add",
    help="新增 API 配置（交互式）。示例：gaoagent api add",
)
def api_add_cmd() -> None:
    """
    新增 API 配置。

    用法：
    - gaoagent api add

    行为：
    - 在 Handler 层进入交互式配置新增流程。
    """
    dispatch("api.add")


@api_group.command(
    "remove",
    help="删除 API 配置。示例：gaoagent api remove openai",
)
@click.argument("name", required=True)
def api_remove_cmd(name: str) -> None:
    """
    删除指定 API。

    用法：
    - gaoagent api remove <name>

    参数：
    - name: API 配置名称（必填）。
    """
    dispatch("api.remove", name=name)


@api_group.command(
    "edit",
    help="编辑 API 配置。示例：gaoagent api edit openai",
)
@click.argument("name", required=True)
def api_edit_cmd(name: str) -> None:
    """
    编辑指定 API。

    用法：
    - gaoagent api edit <name>

    参数：
    - name: API 配置名称（必填）。
    """
    dispatch("api.edit", name=name)


@api_group.command(
    "default",
    help="设置默认 API。示例：gaoagent api default openai",
)
@click.argument("name", required=True)
def api_default_cmd(name: str) -> None:
    """
    设置指定 API 为默认。

    用法：
    - gaoagent api default <name>

    参数：
    - name: API 配置名称（必填）。
    """
    dispatch("api.default", name=name)


def main() -> None:
    """
    程序入口函数。

    行为：
    - 以 `gaoagent` 作为程序名启动 Click CLI。
    """
    cli(prog_name="gaoagent")
