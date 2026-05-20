from __future__ import annotations

from pathlib import Path
from typing import Any

import click

from gaoagent.core.core_config_default import CoreConfigDefault
from gaoagent.core.handler_utils import prompt_multi_select
from gaoagent.core.runner.console import Console
from gaoagent.rag.rag_store_path import is_internal_rag_store_dir_name


class InitConfigTool:
    """初始化阶段的配置加载、交互选择与基础文件维护工具。"""

    def _load_global_apis(self, config_default: CoreConfigDefault, api_file: Path) -> dict[str, Any]:
        """加载全局 API 配置；必要时进入交互采集。"""
        apis: dict[str, Any] = {}
        payload = config_default._read_json(api_file)
        if isinstance(payload, dict) and isinstance(payload.get("apis"), dict):
            apis = payload["apis"]

        if apis:
            return apis

        if api_file.exists():
            Console.info(f"全局 API 配置文件存在但内容为空：{api_file}")

        Console.info("未找到可用的全局 API 配置，将进入采集流程")

        api_names: set[str] = set()
        while True:
            api_config = config_default._import_api_config()
            if api_config is None:
                break

            if api_config["name"] in api_names:
                Console.info("API 配置名称重复，请重新输入")
                continue

            api_names.add(api_config["name"])
            apis[api_config["name"]] = {
                "base_url": api_config["base_url"],
                "api_key": api_config["api_key"],
                "models": api_config["models"],
            }
            config_default._write_api_config(apis)
            Console.info(
                f"API 配置已采集：name={api_config['name']}, base_url={api_config['base_url']}, models={list(api_config['models'].keys())}"
            )

            if not Console.confirm("继续添加一组 API 配置？", default=False):
                break

        return apis

    def _prompt_default_api_and_model(self, apis: dict[str, Any]) -> tuple[str, str]:
        """交互选择项目默认 API 与默认模型。"""
        api_names = sorted([str(x) for x in apis.keys()])
        default_api = api_names[0]
        if len(api_names) > 1:
            default_api = Console.prompt(
                "请选择默认 API", type=click.Choice(api_names, case_sensitive=False), default=default_api, show_default=True
            )
        else:
            Console.info(f"默认 API：{default_api}")

        models_raw = apis.get(default_api, {}).get("models", {})
        if not isinstance(models_raw, dict) or not models_raw:
            raise RuntimeError(f"API {default_api} 未配置任何模型")

        model_names = sorted([str(x) for x in models_raw.keys()])
        default_model = model_names[0]
        if len(model_names) > 1:
            default_model = Console.prompt(
                "请选择默认模型",
                type=click.Choice(model_names, case_sensitive=False),
                default=default_model,
                show_default=True,
            )
        else:
            Console.info(f"默认模型：{default_model}")

        return (default_api, default_model)

    def _load_global_mcp_configs(self, config_default: CoreConfigDefault, mcp_file: Path) -> dict[str, Any]:
        """读取全局 MCP 配置并返回 `mcpServers` 映射。"""
        payload = config_default._read_json(mcp_file)
        if isinstance(payload, dict) and isinstance(payload.get("mcpServers"), dict):
            return payload["mcpServers"]
        return {}

    def _load_global_a2a_configs(self, config_default: CoreConfigDefault, a2a_file: Path) -> dict[str, Any]:
        """读取全局 A2A Agent 配置并返回 `agents` 映射。"""
        payload = config_default._read_json(a2a_file)
        if isinstance(payload, dict) and isinstance(payload.get("agents"), dict):
            return payload["agents"]
        return {}

    def _prompt_select_a2a(self, a2a_configs: dict[str, Any]) -> dict[str, Any]:
        """交互选择要写入项目配置的 A2A Agent 子集。"""
        if not a2a_configs:
            Console.info("未检测到任何全局 A2A Agent 配置（将写入空配置）")
            return {}

        names = sorted([str(x) for x in a2a_configs.keys()])
        Console.info("已加载的 A2A Agent：")
        for i, name in enumerate(names, start=1):
            Console.info(f"{i}. {name}")

        selected = prompt_multi_select("请选择 A2A Agent（输入序号或名称，逗号分隔；回车跳过；输入 all 选择全部）", names)
        if not selected:
            return {}

        return {name: a2a_configs[name] for name in selected}

    def _prompt_select_mcp(self, mcp_configs: dict[str, Any]) -> dict[str, Any]:
        """交互选择要写入项目配置的 MCP 服务子集。"""
        if not mcp_configs:
            Console.info("未检测到任何全局 MCP 配置（将写入空配置）")
            return {}

        names = sorted([str(x) for x in mcp_configs.keys()])
        Console.info("已加载的 MCP 服务：")
        for i, name in enumerate(names, start=1):
            Console.info(f"{i}. {name}")

        selected = prompt_multi_select("请选择 MCP（输入序号或名称，逗号分隔；回车跳过；输入 all 选择全部）", names)
        if not selected:
            return {}

        return {name: mcp_configs[name] for name in selected}

    def _prompt_select_skills(self, config_default: CoreConfigDefault, skills_dir: Path) -> list[dict[str, Any]]:
        """读取全局 Skill 元数据并让用户选择要安装到项目的 Skill。"""
        if not skills_dir.exists() or not skills_dir.is_dir():
            Console.info(f"未检测到全局 Skills 目录：{skills_dir}（已跳过）")
            return []

        (skills, invalid_skills) = config_default._load_skills_metadata(skills_dir)
        if invalid_skills:
            Console.info(f"以下 SKILL.md 不符合规范，共 {len(invalid_skills)} 个：")
            for item in invalid_skills:
                Console.info(f"- {item['path']}: {item['reason']}")

        if not skills:
            Console.info("未加载到任何 Skill（已跳过）")
            return []

        skills.sort(key=lambda x: x["name"])
        Console.info("已加载的 Skills：")
        for i, item in enumerate(skills, start=1):
            Console.info(f"{i}. {item['name']}: {item['description']}")

        names = [item["name"] for item in skills]
        selected_names = prompt_multi_select("请选择 Skill（输入序号或名称，逗号分隔；回车跳过；输入 all 选择全部）", names)
        if not selected_names:
            return []

        selected_set = set(selected_names)
        return [item for item in skills if item["name"] in selected_set]

    def _prompt_select_rag(self, rag_dir: Path) -> list[str]:
        """交互选择要导入项目的全局 RAG 知识库名称。"""
        if not rag_dir.exists() or not rag_dir.is_dir():
            Console.info(f"未检测到全局 RAG 目录：{rag_dir}（已跳过）")
            return []

        names = sorted([p.name for p in rag_dir.iterdir() if p.is_dir() and not is_internal_rag_store_dir_name(p.name)])
        if not names:
            Console.info("未加载到任何全局 RAG 知识库（已跳过）")
            return []

        Console.info("已加载的 RAG 知识库：")
        for i, name in enumerate(names, start=1):
            Console.info(f"{i}. {name}")

        return prompt_multi_select(
            "请选择 RAG 知识库（输入序号或名称，逗号分隔；回车跳过；输入 all 选择全部）",
            names,
        )

    def _ensure_gitignore_contains(self, project_root: Path, entry: str) -> None:
        """确保项目 `.gitignore` 包含指定条目（如 `.gaoagent/`）。"""
        gitignore = project_root / ".gitignore"
        if not gitignore.exists() or not gitignore.is_file():
            return None

        try:
            content = gitignore.read_text(encoding="utf-8")
        except Exception as e:
            Console.error(f"读取 .gitignore 失败：{e}")
            return None

        lines = content.splitlines()
        existing = [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]
        candidates = {".gaoagent", ".gaoagent/", "/.gaoagent", "/.gaoagent/"}
        if any(line in candidates for line in existing):
            return None

        suffix = "" if (not content) or content.endswith(("\n", "\r\n")) else "\n"
        gitignore.write_text(f"{content}{suffix}{entry}\n", encoding="utf-8")
        Console.info("已将 .gaoagent 写入 .gitignore")
        return None
