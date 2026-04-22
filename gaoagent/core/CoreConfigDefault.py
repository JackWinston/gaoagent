from pathlib import Path
from typing import Any

import click
import json


class CoreConfigDefault:
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

        config_dir = self._ensure_config_dir()
        api_config_file = config_dir / "gao_client_api_config.json"
        mcp_config_file = config_dir / "gao_client_mcp_setting.json"

        apis: dict[str, Any] = {}
        existing_api_payload = self._read_json(api_config_file)
        if isinstance(existing_api_payload, dict) and isinstance(existing_api_payload.get("apis"), dict):
            apis = existing_api_payload["apis"]

        api_names: set[str] = set(apis.keys())
        new_api_count = 0
        while True:
            api_config = self._import_api_config()
            if api_config is None:
                break

            if api_config["name"] in api_names:
                click.echo("API 配置名称重复，请重新输入")
                continue

            api_names.add(api_config["name"])
            apis[api_config["name"]] = {
                "base_url": api_config["base_url"],
                "api_key": api_config["api_key"],
                "models": api_config["models"],
            }
            self._write_api_config(apis)
            new_api_count += 1
            click.echo(
                f"API 配置已采集：name={api_config['name']}, base_url={api_config['base_url']}, models={list(api_config['models'].keys())}"
            )

            if not click.confirm("继续添加一组 API 配置？", default=False):
                break

        click.echo(f"API 配置采集完成，本次新增 {new_api_count} 组")

        mcp_configs: dict[str, Any] = {}
        existing_mcp_payload = self._read_json(mcp_config_file)
        if isinstance(existing_mcp_payload, dict):
            if isinstance(existing_mcp_payload.get("mcpServers"), dict):
                mcp_configs = existing_mcp_payload["mcpServers"]

        new_mcp_count = 0
        while True:
            mcp_config = self._import_mcp_config()
            if mcp_config is None:
                break

            (mcp_name, mcp_body) = next(iter(mcp_config.items()))
            if mcp_name in mcp_configs:
                click.echo(f"MCP 配置已存在，将覆盖：{mcp_name}")
            mcp_configs[mcp_name] = mcp_body
            self._write_mcp_config(mcp_configs)
            new_mcp_count += 1
            click.echo(f"MCP 配置已采集：{mcp_name}")

            if not click.confirm("继续添加一组 MCP 配置？", default=False):
                break

        click.echo(f"MCP 配置采集完成，本次新增 {new_mcp_count} 组")

        isInitSkills = self._import_skills_config()

        if isInitSkills:
            skills_dir = Path.home() / ".gaoagent" / "skills"
            (skills, invalid_skills) = self._load_skills_metadata(skills_dir)
            click.echo(f"Skills 配置采集完成，共 {len(skills)} 个")
            for skill in skills:
                click.echo(f"- {skill['name']}: {skill['description']}")
            if invalid_skills:
                click.echo(f"以下 SKILL.md 格式不正确，共 {len(invalid_skills)} 个")
                for item in invalid_skills:
                    click.echo(f"- {item['path']}: {item['reason']}")

        self._import_rag_config()

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

    def _read_json(self, file_path: Path) -> Any | None:
        while True:
            if not file_path.exists():
                return None
            try:
                return json.loads(file_path.read_text(encoding="utf-8"))
            except Exception as e:
                click.echo(f"读取失败：{file_path}，{e}")
                if click.confirm("忽略该文件并继续？", default=True):
                    return None

    def _write_json(self, file_path: Path, payload: Any) -> None:
        tmp_file = file_path.with_name(f"{file_path.name}.tmp")
        tmp_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        tmp_file.replace(file_path)

    def _write_api_config(self, apis: dict[str, Any]) -> None:
        """
        覆盖写入 `~/.gaoagent/gao_client_api_config.json`。
        """
        config_dir = self._ensure_config_dir()
        config_file = config_dir / "gao_client_api_config.json"
        self._write_json(config_file, {"apis": apis})

    def _write_mcp_config(self, mcp_configs: dict[str, Any]) -> None:
        """
        覆盖写入 `~/.gaoagent/gao_client_mcp_setting.json`。
        """
        config_dir = self._ensure_config_dir()
        config_file = config_dir / "gao_client_mcp_setting.json"
        self._write_json(config_file, {"mcpServers": mcp_configs})

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

    def _import_skills_config(self) -> bool :
        """
        引导用户输入技能相关配置。
        """

        if click.confirm("是否跳过 Skills 配置？", default=False):
            return False        

        skills_dir = Path.home() / ".gaoagent" / "skills"
        skills_dir.mkdir(parents=True, exist_ok=True)

        click.echo(f"请将Skills对应的md文件复制到 {skills_dir}")
        return click.confirm("是否已经完成?", default=False)

    def _import_rag_config(self) -> bool :
        """
        引导用户输入 RAG 相关配置。
        """
        if click.confirm("是否跳过 RAG 配置？", default=False):
            return False        
        rag_dir = Path.home() / ".gaoagent" / "rag"
        rag_dir.mkdir(parents=True, exist_ok=True)
        # TODO: 比较复杂,暂时不实现
        return False
        
    def _load_skills_metadata(self, skills_dir: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
        skills: list[dict[str, str]] = []
        invalid_skills: list[dict[str, str]] = []
        if not skills_dir.exists() or not skills_dir.is_dir():
            return (skills, invalid_skills)

        for file_path in skills_dir.rglob("*"):
            if not file_path.is_file():
                continue
            if file_path.name.lower() != "skill.md":
                continue

            (meta, reason) = self._parse_skill_frontmatter(file_path)
            if meta is not None:
                skills.append(meta)
                continue
            if reason:
                invalid_skills.append({"path": str(file_path), "reason": reason})

        skills.sort(key=lambda x: x["name"])
        return (skills, invalid_skills)

    def _parse_skill_frontmatter(self, file_path: Path) -> tuple[dict[str, str] | None, str | None]:
        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception as e:
            return (None, f"读取失败：{e}")

        lines = content.splitlines()
        if not lines or lines[0].strip() != "---":
            return (None, "缺少 YAML frontmatter 起始标记 ---")

        i = 1
        frontmatter: list[str] = []
        while i < len(lines):
            if lines[i].strip() == "---":
                break
            frontmatter.append(lines[i])
            i += 1
        if i >= len(lines) or lines[i].strip() != "---":
            return (None, "缺少 YAML frontmatter 结束标记 ---")

        name: str | None = None
        description: str | None = None
        j = 0
        while j < len(frontmatter):
            line = frontmatter[j].rstrip("\n")
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                j += 1
                continue

            if stripped.startswith("name:"):
                name_val = stripped.split(":", 1)[1].strip()
                name = name_val.strip('"').strip("'")
                j += 1
                continue

            if stripped.startswith("description:"):
                desc_val = stripped.split(":", 1)[1].strip()
                if desc_val in (">", ">-", "|", "|-"):
                    j += 1
                    block: list[str] = []
                    while j < len(frontmatter):
                        nxt = frontmatter[j]
                        if nxt.strip() and not nxt.startswith((" ", "\t")):
                            break
                        block.append(nxt.lstrip())
                        j += 1
                    description = " ".join(" ".join(block).split()).strip()
                    continue

                description = desc_val.strip('"').strip("'")
                j += 1
                continue

            j += 1

        if not name or not description:
            missing: list[str] = []
            if not name:
                missing.append("name")
            if not description:
                missing.append("description")
            return (None, f"缺少字段 {', '.join(missing)}")

        return ({"name": name, "description": description}, None)

    
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
