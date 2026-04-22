
from pathlib import Path
from typing import Any

import click
import json
import shutil
import datetime

from gaoagent.core.CoreConfigDefault import CoreConfigDefault
from gaoagent.mcp.MCPClientCompat import MCPStdioClientSync, build_mcp_tools_cache_payload


class CoreInit:
    def init(self) -> None:
        """
        初始化核心组件。

        1,查看是否已经全局配置过.如果没有,则提示用户先运行gaoagent config. 返回.
        2,查看全局 gao_client_api_config.json 文件是否存在. 不存在,则运行 _import_api_config 的逻辑,并写入全局 gao_client_api_config.json文件.如果没有加载任何一个api,则init失败,返回.
        3,在当前目录创建.gaoagent目录
        4,询问用户选择默认的api和模型(). 如果用户没有选择,则默认选择第一个api.写入.gaoagent目录 gao_client_api_config.json 配置文件.
        5,展示已经加载的mcp服务,并提示用户选择mcp服务.用户可跳过.将选择好的mcp服务写入.gaoagent目录 gao_client_mcp_setting.json 配置文件.
        6,展示已经加载的Skill,并提示用户选择Skill.用户可跳过.将选择好的Skill复制进.gaoagent下面的skills目录
        7,显示已经导入的RAG,并提示用户选择.用户可跳过.将选择好的RAG复制进.gaoagent下面的rag目录(暂时不实现)
        8,如果当前项目目录下有 .gitignore 文件,则将.gaoagent目录添加到 .gitignore 文件中.
        9,初始化完成.
        10,创建项目索引,加速全局搜索
        """
        global_dir = Path.home() / ".gaoagent"
        if not global_dir.exists():
            click.echo(f"未检测到全局配置目录：{global_dir}")
            click.echo("请先运行：gaoagent config")
            return None

        config_default = CoreConfigDefault()
        global_api_file = global_dir / "gao_client_api_config.json"
        global_mcp_file = global_dir / "gao_client_mcp_setting.json"
        global_skills_dir = global_dir / "skills"

        apis = self._load_global_apis(config_default, global_api_file)
        if not apis:
            click.echo("未加载到任何 API 配置，初始化失败")
            return None

        project_dir = Path.cwd() / ".gaoagent"
        project_dir.mkdir(parents=True, exist_ok=True)

        (default_api, default_model) = self._prompt_default_api_and_model(apis)
        project_api_file = project_dir / "gao_client_api_config.json"
        self._write_json(project_api_file, {"default_api": default_api, "default_model": default_model, "apis": apis})
        click.echo(f"已写入项目 API 配置：{project_api_file}")

        project_mcp_file = project_dir / "gao_client_mcp_setting.json"
        mcp_configs = self._load_global_mcp_configs(config_default, global_mcp_file)
        selected_mcp = self._prompt_select_mcp(mcp_configs)
        self._write_json(project_mcp_file, {"mcpServers": selected_mcp})
        click.echo(f"已写入项目 MCP 配置：{project_mcp_file}")
        project_mcp_cache_file = project_dir / "gao_client_mcp_tools_cache.json"
        if selected_mcp:
            try:
                cache_payload = build_mcp_tools_cache_payload(
                    selected_mcp,
                    connect_and_list_tools=lambda name, body: MCPStdioClientSync.from_config(
                        server_name=name,
                        config=body,
                    ).list_tools(),
                    generated_at=datetime.datetime.now(datetime.UTC).isoformat(),
                )
                self._write_json(project_mcp_cache_file, cache_payload)
                tool_count = len((cache_payload.get("exported_map") or {}).keys())
                click.echo(f"已写入项目 MCP 工具缓存：{project_mcp_cache_file}（{tool_count} 个工具）")
            except Exception as e:
                click.echo(f"项目 MCP 工具缓存生成失败：{e}")

        selected_skills = self._prompt_select_skills(config_default, global_skills_dir)
        if selected_skills:
            project_skills_dir = project_dir / "skills"
            project_skills_dir.mkdir(parents=True, exist_ok=True)
            installed_count = 0
            for item in selected_skills:
                dst = project_skills_dir / item["name"]
                self._copy_dir(item["src_dir"], dst)
                installed_count += 1
            click.echo(f"已复制 Skills：{installed_count} 个 → {project_skills_dir}")
        else:
            click.echo("未选择任何 Skill（已跳过）")

        click.echo("RAG 初始化暂不实现（已跳过）")

        self._ensure_gitignore_contains(Path.cwd(), ".gaoagent/")

        click.echo("初始化完成")

        # TODO: 创建搜索索引
        return None

    def _load_global_apis(self, config_default: CoreConfigDefault, api_file: Path) -> dict[str, Any]:
        apis: dict[str, Any] = {}
        payload = config_default._read_json(api_file)
        if isinstance(payload, dict) and isinstance(payload.get("apis"), dict):
            apis = payload["apis"]

        if apis:
            return apis

        if api_file.exists():
            click.echo(f"全局 API 配置文件存在但内容为空：{api_file}")

        click.echo("未找到可用的全局 API 配置，将进入采集流程")

        api_names: set[str] = set()
        while True:
            api_config = config_default._import_api_config()
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
            config_default._write_api_config(apis)
            click.echo(
                f"API 配置已采集：name={api_config['name']}, base_url={api_config['base_url']}, models={list(api_config['models'].keys())}"
            )

            if not click.confirm("继续添加一组 API 配置？", default=False):
                break

        return apis

    def _prompt_default_api_and_model(self, apis: dict[str, Any]) -> tuple[str, str]:
        api_names = sorted([str(x) for x in apis.keys()])
        default_api = api_names[0]
        if len(api_names) > 1:
            default_api = click.prompt(
                "请选择默认 API", type=click.Choice(api_names, case_sensitive=False), default=default_api, show_default=True
            )
        else:
            click.echo(f"默认 API：{default_api}")

        models_raw = apis.get(default_api, {}).get("models", {})
        if not isinstance(models_raw, dict) or not models_raw:
            raise RuntimeError(f"API {default_api} 未配置任何模型")

        model_names = sorted([str(x) for x in models_raw.keys()])
        default_model = model_names[0]
        if len(model_names) > 1:
            default_model = click.prompt(
                "请选择默认模型",
                type=click.Choice(model_names, case_sensitive=False),
                default=default_model,
                show_default=True,
            )
        else:
            click.echo(f"默认模型：{default_model}")

        return (default_api, default_model)

    def _load_global_mcp_configs(self, config_default: CoreConfigDefault, mcp_file: Path) -> dict[str, Any]:
        payload = config_default._read_json(mcp_file)
        if isinstance(payload, dict):
            if isinstance(payload.get("mcpServers"), dict):
                return payload["mcpServers"]
        return {}

    def _prompt_select_mcp(self, mcp_configs: dict[str, Any]) -> dict[str, Any]:
        if not mcp_configs:
            click.echo("未检测到任何全局 MCP 配置（将写入空配置）")
            return {}

        names = sorted([str(x) for x in mcp_configs.keys()])
        click.echo("已加载的 MCP 服务：")
        for i, name in enumerate(names, start=1):
            click.echo(f"{i}. {name}")

        selected = self._prompt_multi_select("请选择 MCP（输入序号或名称，逗号分隔；回车跳过；输入 all 选择全部）", names)
        if not selected:
            return {}

        return {name: mcp_configs[name] for name in selected}

    def _prompt_select_skills(self, config_default: CoreConfigDefault, skills_dir: Path) -> list[dict[str, Any]]:
        if not skills_dir.exists() or not skills_dir.is_dir():
            click.echo(f"未检测到全局 Skills 目录：{skills_dir}（已跳过）")
            return []

        skills: list[dict[str, Any]] = []
        for file_path in skills_dir.rglob("*"):
            if not file_path.is_file():
                continue
            if file_path.name.lower() != "skill.md":
                continue

            (meta, _) = config_default._parse_skill_frontmatter(file_path)
            if meta is None:
                continue
            meta_name = meta["name"]
            skills.append({"name": meta_name, "description": meta["description"], "src_dir": file_path.parent})

        if not skills:
            click.echo("未加载到任何 Skill（已跳过）")
            return []

        skills.sort(key=lambda x: x["name"])
        click.echo("已加载的 Skills：")
        for i, item in enumerate(skills, start=1):
            click.echo(f"{i}. {item['name']}: {item['description']}")

        names = [item["name"] for item in skills]
        selected_names = self._prompt_multi_select(
            "请选择 Skill（输入序号或名称，逗号分隔；回车跳过；输入 all 选择全部）", names
        )
        if not selected_names:
            return []

        selected_set = set(selected_names)
        return [item for item in skills if item["name"] in selected_set]

    def _prompt_multi_select(self, prompt: str, options: list[str]) -> list[str]:
        if not options:
            return []

        while True:
            raw = click.prompt(prompt, type=str, default="", show_default=False).strip()
            if raw == "":
                return []

            lowered = raw.lower()
            if lowered in ("all", "*"):
                return options

            parts = [p.strip() for p in raw.split(",") if p.strip()]
            selected: list[str] = []
            ok = True
            for part in parts:
                if part.isdigit():
                    idx = int(part)
                    if idx < 1 or idx > len(options):
                        ok = False
                        break
                    selected.append(options[idx - 1])
                    continue

                match = None
                for opt in options:
                    if opt.lower() == part.lower():
                        match = opt
                        break
                if match is None:
                    ok = False
                    break
                selected.append(match)

            if ok:
                deduped: list[str] = []
                seen: set[str] = set()
                for item in selected:
                    if item in seen:
                        continue
                    seen.add(item)
                    deduped.append(item)
                return deduped

            click.echo("输入不合法，请重新输入")

    def _write_json(self, file_path: Path, payload: Any) -> None:
        tmp_file = file_path.with_name(f"{file_path.name}.tmp")
        tmp_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        tmp_file.replace(file_path)

    def _copy_dir(self, src: Path, dst: Path) -> None:
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

    def _ensure_gitignore_contains(self, project_root: Path, entry: str) -> None:
        gitignore = project_root / ".gitignore"
        if not gitignore.exists() or not gitignore.is_file():
            return None

        try:
            content = gitignore.read_text(encoding="utf-8")
        except Exception:
            return None

        lines = content.splitlines()
        existing = [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]
        candidates = {".gaoagent", ".gaoagent/", "/.gaoagent", "/.gaoagent/"}
        if any(line in candidates for line in existing):
            return None

        suffix = "" if (not content) or content.endswith(("\n", "\r\n")) else "\n"
        gitignore.write_text(f"{content}{suffix}{entry}\n", encoding="utf-8")
        click.echo("已将 .gaoagent 写入 .gitignore")
        return None
