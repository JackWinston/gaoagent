
from pathlib import Path
from typing import Any

import click
from gaoagent.core.runner.console import Console
import json
import datetime
from gaoagent.core.runner.utils import PROJECTS_REGISTRY_FILENAME
from gaoagent.core.handler_utils import (
    write_json,
    copy_dir,
    rewrite_index_meta_store_dir,
    prompt_multi_select,
)

from gaoagent.core.core_config_default import CoreConfigDefault
from gaoagent.mcp.mcp_client_compat import MCPStdioClientSync, build_mcp_tools_cache_payload
from gaoagent.rag.rag_store_path import (
    resolve_chroma_store_dir,
    resolve_index_meta_file,
    is_internal_rag_store_dir_name,
)


class CoreInit:
    """项目初始化编排器（`gaoagent init` 的核心实现）。

    该类负责把“全局配置”落地为“当前项目可运行配置”，并在初始化阶段完成
    API/MCP/Skills/RAG 等资源的选择与复制。整体上属于交互式初始化流程的控制层。

    核心职责:
    - 校验全局环境是否已准备好（如 `~/.gaoagent`、全局 API 配置）。
    - 在项目目录创建 `.gaoagent` 并写入项目级配置。
    - 交互式选择默认 API 与模型、MCP 服务、Skill、RAG 知识库。
    - 维护项目注册表（记录已初始化项目根目录），并清理失效条目。
    - 维护 `.gitignore`，避免项目级私有配置被误提交。

    设计边界:
    - 仅负责“初始化期编排”，不负责任务执行期的推理逻辑。
    - 不负责 LLM 调用细节；只负责准备运行所需配置与资源。
    """

    def init(self) -> None:
        """执行当前项目的一次完整初始化流程。

        流程概览:
        1. 校验全局目录 `~/.gaoagent` 是否存在；不存在则提示先执行 `gaoagent config`。
        2. 禁止在用户 Home 根目录执行初始化，避免误把家目录当项目目录。
        3. 清理项目注册表中的无效路径，保证全局索引干净。
        4. 加载全局 API 配置；若缺失则进入交互采集；仍为空则初始化失败。
        5. 创建项目 `.gaoagent` 目录并写入项目 API 配置（默认 API/模型 + apis）。
        6. 读取全局 MCP 配置，交互选择后写入项目 MCP 配置。
        7. 若选择了 MCP，尝试预生成并写入项目 MCP 工具缓存（加速后续运行）。
        8. 交互选择全局 Skills，并复制到项目 `.gaoagent/skills`。
        9. 交互选择全局 RAG 知识库，并复制到项目 `.gaoagent/rag`；
           同步复制对应 Chroma 存储目录并修正 `index_meta.json` 的 `store_dir`。
        10. 若存在 `.gitignore`，确保包含 `.gaoagent/`。
        11. 注册当前项目到全局初始化项目列表，最后输出“初始化完成”。

        返回:
        - `None`。该方法通过文件系统副作用与终端输出体现结果。
        """
        global_dir = Path.home() / ".gaoagent"
        if not global_dir.exists():
            Console.info(f"未检测到全局配置目录：{global_dir}")
            Console.info("请先运行：gaoagent config")
            return None
        current_root = Path.cwd().resolve()
        if current_root == Path.home().resolve():
            Console.info("禁止在 ~/ 目录执行 gaoagent init，请进入具体项目目录后重试")
            return None

        registry_file = self._project_registry_file(global_dir)
        self._cleanup_project_registry(registry_file)

        config_default = CoreConfigDefault()
        global_api_file = global_dir / "gao_client_api_config.json"
        global_mcp_file = global_dir / "gao_client_mcp_setting.json"
        global_a2a_file = global_dir / "gao_client_a2a_setting.json"
        global_skills_dir = global_dir / "skills"
        global_rag_dir = global_dir / "rag"

        apis = self._load_global_apis(config_default, global_api_file)
        if not apis:
            Console.info("未加载到任何 API 配置，初始化失败")
            return None

        project_dir = current_root / ".gaoagent"
        project_dir.mkdir(parents=True, exist_ok=True)

        (default_api, default_model) = self._prompt_default_api_and_model(apis)
        project_api_file = project_dir / "gao_client_api_config.json"
        write_json(project_api_file, {"default_api": default_api, "default_model": default_model, "apis": apis})
        Console.info(f"已写入项目 API 配置：{project_api_file}")

        project_mcp_file = project_dir / "gao_client_mcp_setting.json"
        mcp_configs = self._load_global_mcp_configs(config_default, global_mcp_file)
        selected_mcp = self._prompt_select_mcp(mcp_configs)
        write_json(project_mcp_file, {"mcpServers": selected_mcp})
        Console.info(f"已写入项目 MCP 配置：{project_mcp_file}")
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
                write_json(project_mcp_cache_file, cache_payload)
                tool_count = len((cache_payload.get("exported_map") or {}).keys())
                Console.info(f"已写入项目 MCP 工具缓存：{project_mcp_cache_file}（{tool_count} 个工具）")
            except Exception as e:
                Console.info(f"项目 MCP 工具缓存生成失败：{e}")

        selected_skills = self._prompt_select_skills(config_default, global_skills_dir)
        if selected_skills:
            project_skills_dir = project_dir / "skills"
            project_skills_dir.mkdir(parents=True, exist_ok=True)
            installed_count = 0
            for item in selected_skills:
                dst = project_skills_dir / item["name"]
                copy_dir(item["src_dir"], dst)
                installed_count += 1
            Console.info(f"已复制 Skills：{installed_count} 个 → {project_skills_dir}")
        else:
            Console.info("未选择任何 Skill（已跳过）")

        selected_rag = self._prompt_select_rag(global_rag_dir)
        if selected_rag:
            project_rag_dir = project_dir / "rag"
            project_rag_dir.mkdir(parents=True, exist_ok=True)
            imported_count = 0
            for kb_name in selected_rag:
                src = global_rag_dir / kb_name
                dst = project_rag_dir / kb_name
                copy_dir(src, dst)
                src_store_dir = resolve_chroma_store_dir(kb_dir=src, kb_name=kb_name)
                dst_store_dir = resolve_chroma_store_dir(kb_dir=dst, kb_name=kb_name)
                if src_store_dir.exists() and src_store_dir.is_dir():
                    copy_dir(src_store_dir, dst_store_dir)
                    rewrite_index_meta_store_dir(kb_dir=dst, kb_name=kb_name)
                imported_count += 1
            Console.info(f"已复制 RAG 知识库：{imported_count} 个 → {project_rag_dir}")
        else:
            Console.info("未选择任何 RAG 知识库（已跳过）")

        project_a2a_file = project_dir / "gao_client_a2a_setting.json"
        a2a_configs = self._load_global_a2a_configs(config_default, global_a2a_file)
        selected_a2a = self._prompt_select_a2a(a2a_configs)
        if selected_a2a:
            write_json(project_a2a_file, {"agents": selected_a2a})
            Console.info(f"已写入项目 A2A Agent 配置：{project_a2a_file}")
        else:
            Console.info("未选择任何 A2A Agent（已跳过）")

        self._ensure_gitignore_contains(current_root, ".gaoagent/")
        self._register_project_root(registry_file, current_root)

        Console.info("初始化完成")

        # TODO: 创建搜索索引
        return None

    def _load_global_apis(self, config_default: CoreConfigDefault, api_file: Path) -> dict[str, Any]:
        """加载全局 API 配置；必要时进入交互采集。

        行为:
        - 优先读取 `api_file` 内的 `apis` 字段。
        - 若无可用配置，则循环调用 `config_default._import_api_config()` 采集。
        - 对 API 名称做去重校验，采集一条即落盘一次，避免中途退出丢数据。

        返回:
        - `dict[str, Any]`：可直接写入项目配置的 API 配置映射。
        """
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
        """交互选择项目默认 API 与默认模型。

        规则:
        - API 与模型列表均按名称排序，首项作为默认值。
        - 当候选项超过 1 个时，弹出 `click.prompt` 让用户选择。
        - 若选中 API 没有任何模型，抛出 `RuntimeError` 阻止写入非法配置。

        返回:
        - `(default_api, default_model)` 二元组。
        """
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
        """读取全局 MCP 配置并返回 `mcpServers` 映射。

        返回:
        - `dict[str, Any]`：服务名到配置体的映射；无配置时返回空字典。
        """
        payload = config_default._read_json(mcp_file)
        if isinstance(payload, dict):
            if isinstance(payload.get("mcpServers"), dict):
                return payload["mcpServers"]
        return {}

    def _load_global_a2a_configs(self, config_default: CoreConfigDefault, a2a_file: Path) -> dict[str, Any]:
        """读取全局 A2A Agent 配置并返回 `agents` 映射。"""
        payload = config_default._read_json(a2a_file)
        if isinstance(payload, dict):
            if isinstance(payload.get("agents"), dict):
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
        """交互选择要写入项目配置的 MCP 服务子集。

        交互方式:
        - 先展示可选服务列表，再调用 `_prompt_multi_select()` 支持序号/名称/all 输入。

        返回:
        - 仅包含用户选中项的 MCP 配置字典；跳过时返回空字典。
        """
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
        """读取全局 Skill 元数据并让用户选择要安装到项目的 Skill。

        处理细节:
        - 调用 `CoreConfigDefault._load_skills_metadata()` 解析 `SKILL.md`。
        - 对不合规 Skill 给出提示，但不阻断其余合法 Skill 的选择流程。
        - 使用 `_prompt_multi_select()` 进行多选，最后返回选中 Skill 元数据。

        返回:
        - `list[dict[str, Any]]`：每项包含 Skill 名称、描述、源目录等信息。
        """
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
        selected_names = prompt_multi_select(
            "请选择 Skill（输入序号或名称，逗号分隔；回车跳过；输入 all 选择全部）", names
        )
        if not selected_names:
            return []

        selected_set = set(selected_names)
        return [item for item in skills if item["name"] in selected_set]

    def _prompt_select_rag(self, rag_dir: Path) -> list[str]:
        """交互选择要导入项目的全局 RAG 知识库名称。

        规则:
        - 仅展示目录型知识库，且过滤内部管理目录（如 `.chrome_store`）。
        - 支持输入序号/名称/all；回车表示跳过。

        返回:
        - `list[str]`：选中的知识库名称列表。
        """
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
        """确保项目 `.gitignore` 包含指定条目（如 `.gaoagent/`）。

        规则:
        - 若 `.gitignore` 不存在或不可读，静默返回。
        - 若已存在等价条目（含带 `/` 或前缀 `/` 形式），不重复写入。
        """
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
        Console.info("已将 .gaoagent 写入 .gitignore")
        return None

    def _project_registry_file(self, global_dir: Path) -> Path:
        """返回“已初始化项目注册表”文件路径。"""
        return global_dir / PROJECTS_REGISTRY_FILENAME

    def _cleanup_project_registry(self, registry_file: Path) -> list[Path]:
        """清理注册表中的失效项目路径并返回保留结果。

        判定有效条件:
        - 项目根目录存在且为目录。
        - 项目下 `.gaoagent` 目录存在。
        """
        existing = self._load_project_registry(registry_file)
        valid: list[Path] = []
        for root in existing:
            config_dir = root / ".gaoagent"
            if root.exists() and root.is_dir() and config_dir.exists() and config_dir.is_dir():
                valid.append(root)
        self._write_project_registry(registry_file, valid)
        return valid

    def _register_project_root(self, registry_file: Path, project_root: Path) -> None:
        """将当前项目根目录写入注册表（若不存在则追加）。"""
        roots = self._cleanup_project_registry(registry_file)
        normalized = project_root.resolve()
        if normalized not in roots:
            roots.append(normalized)
            self._write_project_registry(registry_file, roots)

    def _load_project_registry(self, registry_file: Path) -> list[Path]:
        """读取并解析项目注册表文件，返回去重后的项目路径列表。

        容错行为:
        - 文件不存在、读取失败、单行路径非法时均跳过，不抛异常中断初始化。
        """
        if not registry_file.exists() or not registry_file.is_file():
            return []
        try:
            lines = registry_file.read_text(encoding="utf-8").splitlines()
        except Exception:
            return []
        paths: list[Path] = []
        seen: set[str] = set()
        for line in lines:
            raw = line.strip()
            if not raw:
                continue
            try:
                p = Path(raw).expanduser().resolve()
            except Exception:
                continue
            key = str(p)
            if key in seen:
                continue
            seen.add(key)
            paths.append(p)
        return paths

    def _write_project_registry(self, registry_file: Path, roots: list[Path]) -> None:
        """将项目路径列表去重后写入注册表文件。"""
        registry_file.parent.mkdir(parents=True, exist_ok=True)
        deduped: list[str] = []
        seen: set[str] = set()
        for root in roots:
            key = str(root.resolve())
            if key in seen:
                continue
            seen.add(key)
            deduped.append(key)
        content = "\n".join(deduped)
        if content:
            content += "\n"
        registry_file.write_text(content, encoding="utf-8")
