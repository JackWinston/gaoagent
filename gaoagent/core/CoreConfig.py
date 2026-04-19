from pathlib import Path
from typing import Any

import click
import json


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
           a. 添加 API 配置并命名（url、key、model、每个model的context_windows）。
           b. 添加 MCP 配置并命名（写入 gao_client_mcp_setting.json）。
           c. 添加 Skills 配置并命名（创建 skills/，将用户添加的 skill.md 放入其中）。
           d. 添加 RAG 配置并命名（创建 rag/；每个知识库独立子目录；调用三方库切片并写入向量库）。
           e. 每完成一步,都写入最终 config.json。
        """
        self._ensure_config_dir()

        api_configs: list[dict[str, Any]] = []
        api_names: set[str] = set()
        while True:
            api_config = self._import_api_config()
            if api_config is None:
                break

            if api_config["name"] in api_names:
                click.echo("API 配置名称重复，请重新输入")
                continue

            api_names.add(api_config["name"])
            api_configs.append(api_config)
            click.echo(
                f"API 配置已采集：name={api_config['name']}, base_url={api_config['base_url']}, models={list(api_config['models'].keys())}"
            )

            if not click.confirm("继续添加一组 API 配置？", default=False):
                break

        click.echo(f"API 配置采集完成，共 {len(api_configs)} 组")

        mcp_configs: dict[str, Any] = {}
        while True:
            mcp_config = self._import_mcp_config()
            if mcp_config is None:
                break

            (mcp_name, mcp_body) = next(iter(mcp_config.items()))
            if mcp_name in mcp_configs:
                click.echo(f"MCP 配置已存在，将覆盖：{mcp_name}")
            mcp_configs[mcp_name] = mcp_body
            click.echo(f"MCP 配置已采集：{mcp_name}")

            if not click.confirm("继续添加一组 MCP 配置？", default=False):
                break

        click.echo(f"MCP 配置采集完成，共 {len(mcp_configs)} 组")

    def _ensure_config_dir(self) -> Path:
        """
        确保用户级配置目录存在。

        配置目录固定为 `~/.gaoagent`（即 `Path.home() / ".gaoagent"`）。
        """
        config_dir = Path.home() / ".gaoagent"
        if config_dir.exists() and not config_dir.is_dir():
            raise RuntimeError(f"配置路径已存在但不是目录：{config_dir}")

        config_dir.mkdir(parents=True, exist_ok=True)
        return config_dir

    def _import_api_config(self) -> dict[str, Any] | None:
        """
        引导用户输入 API 相关配置，并返回可序列化的 dict；若用户选择跳过则返回 None。

        字段：
        1. name: 该组配置的名字
        2. base_url: API 的 base url
        3. api_key: API key（仅用于认证；不会在终端回显）
        4. models: 模型集合（key 为模型名）
           - context_window: 上下文窗口大小
           - capabilities: 默认能力（vision/tools/reasoning）
           - aliases: 别名列表（用于命令行快捷选择）
        """
        if click.confirm("是否跳过 API 配置？", default=False):
            return None

        name = self._prompt_non_empty_str("请输入 API 配置名称")

        base_url = self._prompt_non_empty_str("请输入 API Base URL")

        api_key = self._prompt_non_empty_str("请输入 API Key", hide_input=True)

        models: dict[str, Any] = {}
        while True:
            model_id = self._prompt_non_empty_str("请输入模型名")
            if model_id in models:
                click.echo("模型名重复，请重新输入")
                continue

            context_window = self._prompt_positive_int(
                "请输入该模型 context window", default=8192, show_default=True
            )

            vision = click.confirm("该模型是否支持图片（vision）？", default=False)
            tools = click.confirm("该模型是否支持工具调用（tools）？", default=True)
            reasoning = click.confirm("该模型是否支持推理（reasoning）？", default=False)

            aliases_raw = click.prompt("请输入别名（逗号分隔，可留空）", type=str, default="", show_default=False).strip()
            aliases = [a.strip() for a in aliases_raw.split(",") if a.strip()] if aliases_raw else []

            models[model_id] = {
                "id": model_id,
                "context_window": context_window,
                "capabilities": {"vision": vision, "tools": tools, "reasoning": reasoning},
                "aliases": aliases,
            }

            if not click.confirm("继续添加模型？", default=False):
                break

        return {"name": name, "base_url": base_url, "api_key": api_key, "models": models}

    def _import_mcp_config(self) -> dict[str, Any] | None:
        """
        引导用户输入 MCP 相关配置，并返回可序列化的 dict；若用户选择跳过则返回 None。

        格式示例：
        {
          "bing-search": {
            "disabled": true,
            "timeout": 60,
            "type": "stdio",
            "command": "npx",
            "args": ["-y", "bing-cn-mcp"]
          }
        }
        """
        if click.confirm("是否跳过 MCP 配置？", default=False):
            return None

        while True:
            raw = click.prompt("请输入 MCP JSON对象", type=str).strip()
            try:
                value = json.loads(raw)
            except Exception:
                click.echo("格式错误：请输入合法的 JSON 对象")
                continue

            if not isinstance(value, dict):
                click.echo("格式错误：MCP 配置必须是 JSON 对象")
                continue

            try:
                self._validate_mcp_config(value)
            except Exception as e:
                click.echo(f"格式错误：{e}")
                continue

            return value

    def _prompt_non_empty_str(self, text: str, *, hide_input: bool = False) -> str:
        """
        获取非空字符串输入；为空则提示并重新输入。
        """
        while True:
            value = click.prompt(text, type=str, hide_input=hide_input).strip()
            if value:
                return value
            click.echo("输入不能为空，请重新输入")

    def _prompt_positive_int(self, text: str, *, default: int, show_default: bool = True) -> int:
        """
        获取正整数输入；非正整数则提示并重新输入。
        """
        while True:
            value = click.prompt(text, type=int, default=default, show_default=show_default)
            if value > 0:
                return value
            click.echo("请输入正整数")

    def _prompt_json_str_list(self, text: str) -> list[str]:
        """
        获取 JSON 字符串数组输入（例如 ["-y","bing-cn-mcp"]）；解析失败则提示并重新输入。
        """
        while True:
            raw = click.prompt(text, type=str).strip()
            try:
                value = json.loads(raw)
            except Exception:
                click.echo("格式错误：请输入合法的 JSON 数组")
                continue

            if not isinstance(value, list) or any((not isinstance(x, str)) for x in value):
                click.echo("格式错误：args 必须是字符串数组，例如 [\"-y\",\"bing-cn-mcp\"]")
                continue

            return value

    def _validate_mcp_config(self, config: dict[str, Any]) -> None:
        """
        校验 MCP 配置格式；不符合时抛出 ValueError。
        """
        if not isinstance(config, dict) or len(config) != 1:
            raise ValueError("MCP 配置必须是仅包含 1 个键的对象（键为 MCP 名称）")

        (name, body) = next(iter(config.items()))
        if not isinstance(name, str) or not name.strip():
            raise ValueError("MCP 名称必须是非空字符串")
        if not isinstance(body, dict):
            raise ValueError("MCP 配置内容必须是对象")

        required_keys = {"disabled", "timeout", "type", "command", "args"}
        missing = required_keys - set(body.keys())
        if missing:
            raise ValueError(f"MCP 配置缺少字段：{sorted(missing)}")

        if not isinstance(body["disabled"], bool):
            raise ValueError("MCP.disabled 必须为 bool")
        if not isinstance(body["timeout"], int) or body["timeout"] <= 0:
            raise ValueError("MCP.timeout 必须为正整数")
        if body["type"] not in ("stdio",):
            raise ValueError("MCP.type 目前仅支持 stdio")
        if not isinstance(body["command"], str) or not body["command"].strip():
            raise ValueError("MCP.command 必须为非空字符串")
        if not isinstance(body["args"], list) or any((not isinstance(x, str)) for x in body["args"]):
            raise ValueError("MCP.args 必须为字符串数组")
