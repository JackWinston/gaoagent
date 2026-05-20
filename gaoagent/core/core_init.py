
from pathlib import Path

import datetime
from typing import Any

from gaoagent.core.runner.console import Console
from gaoagent.core.handler_utils import (
    write_json,
    copy_dir,
    rewrite_index_meta_store_dir,
)

from gaoagent.core.core_config_default import CoreConfigDefault
from gaoagent.core.init_config_tool import InitConfigTool
from gaoagent.core.init_registry_tool import InitRegistryTool
from gaoagent.core.project_overview_tool import ProjectOverviewTool
from gaoagent.mcp.mcp_client_compat import MCPClientSync, build_mcp_tools_cache_payload
from gaoagent.rag.rag_store_path import (
    resolve_chroma_store_dir,
)


class CoreInit(InitConfigTool, ProjectOverviewTool, InitRegistryTool):
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
        11. 调用 LLM 分析当前初始化目标项目，生成项目概览并写入 `.gaoagent/project.md`。
        12. 注册当前项目到全局初始化项目列表，最后输出“初始化完成”。

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
                    connect_and_list_tools=lambda name, body: MCPClientSync.from_config(
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
        project_overview_file = project_dir / "project.md"
        self._write_project_overview(current_root, project_overview_file)
        Console.info(f"已写入项目概览：{project_overview_file}")
        self._register_project_root(registry_file, current_root)

        Console.info("初始化完成")

        # TODO: 创建搜索索引
        return None
