
class CoreConfig:
    """
    核心配置入口。

    在程序初始化时调用，用于引导用户完成默认配置的创建与管理。
    """

    def config(self) -> None:
        """
        配置流程（交互式）：

        1. 检查是否存在 ~/.gaoagent；不存在则创建。
        2. 检查是否已存在默认配置文件；存在则展示当前配置。
        3. 若不存在默认配置文件，引导用户完成配置创建：
           a. 添加 API 配置并命名（url、key、model、context_size）。
           b. 添加 MCP 配置并命名（写入 gao_client_mcp_setting.json）。
           c. 添加 Skills 配置并命名（创建 skills/，将用户添加的 skill.md 放入其中）。
           d. 添加 RAG 配置并命名（创建 rag/；每个知识库独立子目录；调用三方库切片并写入向量库）。
           e. 写入最终 config.json。
        """
        click.echo("CoreHandlers.config")

