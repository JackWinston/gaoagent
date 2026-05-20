from __future__ import annotations

import ast
from pathlib import Path
import re
import tomllib
from typing import Any

from gaoagent.core.runner.console import Console


class ProjectOverviewTool:
    """项目概览生成与目录分析工具。"""

    _PROJECT_OVERVIEW_REFRESH_MIN_BYTES = 2048
    _PROJECT_OVERVIEW_MAX_DEPTH = 6
    _PROJECT_OVERVIEW_MAX_ENTRIES = 400
    _PROJECT_OVERVIEW_MAX_REFRESH_RECORD_CONTEXT_ITEMS = 8
    _PROJECT_OVERVIEW_MAX_REFRESH_RECORD_CONTEXT_CHARS = 1200
    _PROJECT_OVERVIEW_IGNORED_NAMES = {
        ".gaoagent",
        ".git",
        ".hg",
        ".idea",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".svn",
        ".tox",
        ".venv",
        ".vscode",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
        "venv",
    }
    _PROJECT_OVERVIEW_IGNORED_SUFFIXES = {".pyc", ".pyo"}

    @classmethod
    def build_refresh_record_from_tool_call(
        cls,
        tool_name: str,
        arguments: dict[str, Any],
        raw_observation: Any,
    ) -> dict[str, Any] | None:
        """根据工具调用结果生成一次有效的项目概览刷新记录。"""
        if not isinstance(raw_observation, dict) or raw_observation.get("success") is not True:
            return None
        path = raw_observation.get("path")
        if not isinstance(path, str) or not path.strip():
            return None
        if tool_name == "write_file":
            if raw_observation.get("file_created") is not True:
                return None
            written_chars = raw_observation.get("written_chars")
            if not isinstance(written_chars, int) or written_chars <= cls._PROJECT_OVERVIEW_REFRESH_MIN_BYTES:
                return None
            content = arguments.get("content", "")
            if not isinstance(content, str):
                content = str(content)
            return {
                "event": "create",
                "path": path,
                "content": content,
                "append": bool(raw_observation.get("append", arguments.get("append", False))),
                "written_chars": written_chars,
            }
        if tool_name == "delete_file":
            deleted_bytes = raw_observation.get("deleted_bytes")
            if not isinstance(deleted_bytes, int) or deleted_bytes <= cls._PROJECT_OVERVIEW_REFRESH_MIN_BYTES:
                return None
            return {
                "event": "delete",
                "path": path,
                "content": "",
                "append": False,
                "written_chars": deleted_bytes,
                "deleted_bytes": deleted_bytes,
            }
        return None

    @staticmethod
    def clone_refresh_records(records: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        """复制项目概览刷新记录列表。"""
        if not isinstance(records, list):
            return []
        return [dict(item) for item in records if isinstance(item, dict)]

    @classmethod
    def merge_refresh_records_from_runner(
        cls,
        current_records: list[dict[str, Any]] | None,
        runner: Any,
    ) -> list[dict[str, Any]]:
        """从子执行器中提取并合并项目概览刷新记录。"""
        merged = cls.clone_refresh_records(current_records)
        if not hasattr(runner, "get_project_overview_refresh_records"):
            return merged
        try:
            child_records = runner.get_project_overview_refresh_records()
        except Exception as e:
            Console.error(f"获取子执行器刷新记录失败：{e}")
            return merged
        merged.extend(cls.clone_refresh_records(child_records))
        return merged

    @classmethod
    def should_refresh_from_records(cls, records: list[dict[str, Any]] | None) -> bool:
        """判断当前记录集合是否足以触发一次项目概览刷新。"""
        return bool(cls.clone_refresh_records(records))

    def _write_project_overview(
        self,
        project_root: Path,
        output_file: Path,
        refresh_records: list[dict[str, Any]] | None = None,
    ) -> None:
        """为当前初始化目标项目生成概览并写入 `.gaoagent/project.md`。"""
        try:
            content = self._build_project_overview(project_root, refresh_records=refresh_records)
        except Exception as exc:
            content = self._build_fallback_project_overview(project_root, exc, refresh_records=refresh_records)

        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(content, encoding="utf-8")

    def refresh_current_project_overview_if_exists(
        self,
        refresh_records: list[dict[str, Any]] | None = None,
    ) -> bool:
        """若当前项目已存在 `.gaoagent/project.md`，则按最新上下文刷新它。"""
        from gaoagent.core.runner.utils import try_project_root_dir

        project_root = try_project_root_dir()
        if project_root is None:
            return False
        output_file = project_root / ".gaoagent" / "project.md"
        if not output_file.exists() or not output_file.is_file():
            return False
        self._write_project_overview(project_root, output_file, refresh_records=refresh_records)
        return True

    def rebuild_current_project_overview(
        self,
        refresh_records: list[dict[str, Any]] | None = None,
    ) -> bool:
        """强制重建当前项目的 `.gaoagent/project.md`。"""
        from gaoagent.core.runner.utils import try_project_root_dir

        project_root = try_project_root_dir()
        if project_root is None:
            return False
        output_file = project_root / ".gaoagent" / "project.md"
        self._write_project_overview(project_root, output_file, refresh_records=refresh_records)
        return True

    def _build_project_overview(
        self,
        project_root: Path,
        refresh_records: list[dict[str, Any]] | None = None,
    ) -> str:
        """调用 LLM 为当前初始化目标项目构建概览 Markdown。"""
        llm_context = self._build_llm_context(project_root, refresh_records=refresh_records)
        content = self._request_project_overview_from_llm(project_root, llm_context)
        return self._normalize_llm_markdown(content, project_root.name)

    def _build_fallback_project_overview(
        self,
        project_root: Path,
        exc: Exception,
        refresh_records: list[dict[str, Any]] | None = None,
    ) -> str:
        """当 LLM 生成失败时，回退为基于规则的项目概览。"""
        tree_text, entries, truncated = self._collect_project_entries(project_root)
        lines = [
            f"# 项目概览：{project_root.name}",
            "",
            "## 1. 项目的主要功能",
            "- LLM 生成失败，已回退为基础概览。",
        ]
        lines.extend(f"- {item}" for item in self._infer_main_features_from_tree(entries))
        lines.extend(
            [
                "",
                "## 2. 项目的技术架构",
                f"- 当前项目根目录：`{project_root}`。",
                f"- LLM 生成失败原因：{self._summarize_overview_error(exc)}",
            ]
        )
        lines.extend(f"- {item}" for item in self._collect_tech_architecture(project_root, entries))
        refresh_context = self._format_refresh_record_context(project_root, refresh_records)
        if refresh_context:
            lines.extend(
                [
                    "",
                    "## 附：最近刷新触发记录",
                    refresh_context,
                ]
            )
        lines.extend(
            [
                "",
                "## 3. 项目的文件树形结构",
                "- 说明：以下结构已忽略常见依赖、缓存、构建产物与 `.gaoagent/` 目录。",
                "```text",
                tree_text,
                "```",
                "",
                "## 4. 每个文件/文件夹的功能",
            ]
        )
        if truncated:
            lines.append("- 说明：项目条目过多，树形结构和功能说明只展示前 400 个可见条目。")
        lines.extend(
            f"- `{entry['relative_path']}{'/' if entry['is_dir'] else ''}`：{self._describe_path(project_root, entry['path'])}"
            for entry in entries
        )
        return "\n".join(lines).rstrip() + "\n"

    def _build_llm_context(
        self,
        project_root: Path,
        refresh_records: list[dict[str, Any]] | None = None,
    ) -> str:
        """构建发送给 LLM 的项目上下文。"""
        tree_text, entries, truncated = self._collect_project_entries(project_root)
        architecture = self._collect_tech_architecture(project_root, entries)
        entry_descriptions = [
            f"- `{entry['relative_path']}{'/' if entry['is_dir'] else ''}`：{self._describe_path(project_root, entry['path'])}"
            for entry in entries[:40]
        ]

        lines = [
            f"项目名称：{project_root.name}",
            f"项目根目录：{project_root}",
            "",
            "请基于以下目标项目信息，为执行 `gaoagent init` 的这个项目生成 `.gaoagent/project.md`。",
            "注意：这里分析的是当前被初始化的业务项目，不是 GaoAgent 自身源码仓库。",
            "你可以按需调用 `list_dir` 和 `read_file` 来进一步查看项目文件，不要编造未读取到的文件内容。",
            "",
            "### 技术架构线索",
        ]
        lines.extend(f"- {item}" for item in architecture)
        lines.extend(
            [
                "",
                "### 项目目录树",
                f"- 说明：已忽略常见依赖、缓存、构建产物与 `.gaoagent/` 目录。{' 条目过多时已截断。' if truncated else ''}",
                "```text",
                tree_text,
                "```",
                "",
                "### 关键路径说明",
            ]
        )
        lines.extend(entry_descriptions or ["- 暂未提取到可见文件条目。"])
        refresh_context = self._format_refresh_record_context(project_root, refresh_records)
        if refresh_context:
            lines.extend(
                [
                    "",
                    "### 最近刷新触发记录",
                    "以下内容来自本轮任务中触发项目概览刷新的文件新增/删除记录，可作为理解最新结构变化的高优先级上下文：",
                    refresh_context,
                ]
            )
        return "\n".join(lines).rstrip()

    def _request_project_overview_from_llm(self, project_root: Path, project_context: str) -> str:
        """通过 ReActRunner 调用 LLM 生成目标项目概览。"""
        from gaoagent.core.runner.base_runner import RunnerConfig
        from gaoagent.core.runner.react_runner import ReActRunner
        from gaoagent.core.runner.tooling import ToolRegistry, _list_dir, _read_file

        base_info = self._load_request_base_info_for_project(project_root)
        if base_info is None:
            raise RuntimeError("未找到可用的项目 API 配置，无法调用 LLM 生成项目概览")

        overview_tools = ToolRegistry()
        overview_tools.register("list_dir", _list_dir)
        overview_tools.register("read_file", _read_file)
        runner = ReActRunner(
            config=RunnerConfig(
                max_steps=8,
                tools=overview_tools,
                llm_invalid_retry=1,
                scene="init_project_overview",
                disable_skill=True,
                disable_rag=True,
                disable_mcp=True,
            ),
            tools=overview_tools,
            request_base_info=base_info,
        )
        prompt = (
            "你现在的任务是为执行 `gaoagent init` 的目标项目生成 "
            "`.gaoagent/project.md`。\n\n"
            "你可以使用 `list_dir` 和 `read_file` 浏览项目内容，但不要调用其他无关工具。\n"
            "请直接输出最终 Markdown，不要输出解释，不要输出 JSON，不要使用代码块包裹整个结果。\n"
            "必须包含以下 4 个二级标题，并保持标题文本完全一致：\n"
            "## 1. 项目的主要功能\n"
            "## 2. 项目的技术架构\n"
            "## 3. 项目的文件树形结构\n"
            "## 4. 每个文件/文件夹的功能\n\n"
            "要求：\n"
            "1. 内容必须严格基于提供的目标项目上下文，禁止编造不存在的目录、文件或能力。\n"
            "2. 第 1 节请优先根据目录树、关键路径说明以及你自行读取的关键文件归纳主要功能，不要套用固定目录名模板。\n"
            "3. 第 3 节保留树形结构代码块。\n"
            "4. 第 4 节优先描述关键文件和关键目录，不要求穷举所有文件。\n"
            "5. 仅在需要确认文件内容时再调用 `read_file`，避免无意义遍历。\n"
            "6. 输出语言为中文。\n\n"
            f"目标项目名称：`{project_root.name}`\n\n"
            f"{project_context}"
        )
        result = runner.run(prompt)
        if not result.success or not isinstance(result.final_result, str) or not result.final_result.strip():
            raise RuntimeError(result.error or "ReActRunner 未返回有效的项目概览内容")
        return result.final_result

    def _format_refresh_record_context(
        self,
        project_root: Path,
        refresh_records: list[dict[str, Any]] | None,
    ) -> str:
        """把最近的项目概览刷新触发记录整理为可注入 LLM 的文本块。"""
        if not refresh_records:
            return ""

        sections: list[str] = []
        for item in refresh_records[-self._PROJECT_OVERVIEW_MAX_REFRESH_RECORD_CONTEXT_ITEMS :]:
            if not isinstance(item, dict):
                continue
            raw_path = item.get("path")
            content = item.get("content")
            event = item.get("event")
            if not isinstance(raw_path, str) or not raw_path.strip():
                continue
            if not isinstance(event, str) or not event.strip():
                continue

            path_obj = Path(raw_path)
            try:
                display_path = path_obj.resolve().relative_to(project_root.resolve()).as_posix()
            except Exception as e:
                Console.debug(f"路径解析失败，回退原始路径：{raw_path}，{e}")
                display_path = raw_path
            size_chars = item.get("written_chars")
            if not isinstance(size_chars, int):
                size_chars = len(content) if isinstance(content, str) else 0
            sections.extend(
                [
                    f"#### `{display_path}`",
                    f"- 触发事件：{event}",
                    f"- 影响大小：{size_chars}",
                ]
            )
            if event == "create" and isinstance(content, str) and content.strip():
                snippet = content.replace("\r\n", "\n").strip()
                if len(snippet) > self._PROJECT_OVERVIEW_MAX_REFRESH_RECORD_CONTEXT_CHARS:
                    snippet = snippet[: self._PROJECT_OVERVIEW_MAX_REFRESH_RECORD_CONTEXT_CHARS - 4].rstrip() + "\n..."
                sections.extend(
                    [
                        "```text",
                        snippet,
                        "```",
                    ]
                )
        return "\n".join(sections).rstrip()

    def _load_request_base_info_for_project(self, project_root: Path) -> Any | None:
        """从目标项目自身的 `.gaoagent` 配置中解析 LLM 请求信息。"""
        from gaoagent.core.runner.base_runner import RequestBaseInfo
        import json

        config_path = project_root / ".gaoagent" / "gao_client_api_config.json"
        if not config_path.exists() or not config_path.is_file():
            return None

        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception as e:
            Console.error(f"读取项目 API 配置失败：{config_path}，{e}")
            return None

        if not isinstance(config, dict):
            return None

        apis = config.get("apis")
        if not isinstance(apis, dict) or not apis:
            return None

        default_api = config.get("default_api")
        if not isinstance(default_api, str) or default_api not in apis:
            default_api = next(iter(apis.keys()))

        api_config = apis.get(default_api)
        if not isinstance(api_config, dict):
            return None

        models = api_config.get("models")
        if not isinstance(models, dict) or not models:
            return None

        default_model = config.get("default_model")
        if not isinstance(default_model, str) or default_model not in models:
            default_model = next(iter(models.keys()))

        model_info = models.get(default_model)
        if not isinstance(model_info, dict):
            return None

        return RequestBaseInfo(
            baseurl=str(api_config.get("base_url", "")),
            api_key=str(api_config.get("api_key", "")),
            modules=default_model,
            context_window=int(model_info.get("context_window", 4096) or 4096),
        )

    def _infer_main_features_from_tree(self, entries: list[dict[str, Any]]) -> list[str]:
        """根据目录树和可见条目生成更通用的“主要功能”摘要。"""
        lines: list[str] = []
        visible_paths = {entry["relative_path"] for entry in entries}
        top_level_dirs = [
            entry["relative_path"]
            for entry in entries
            if entry["is_dir"] and "/" not in entry["relative_path"]
        ]
        top_level_files = [
            entry["relative_path"]
            for entry in entries
            if not entry["is_dir"] and "/" not in entry["relative_path"]
        ]

        if top_level_dirs:
            lines.append(
                "核心功能按顶层目录组织，主要围绕 "
                + "、".join(f"`{name}/`" for name in top_level_dirs[:6])
                + " 等模块展开。"
            )

        entrypoints = self._detect_entrypoints(entries)
        if entrypoints:
            lines.append("检测到 " + "、".join(f"`{item}`" for item in entrypoints) + " 等入口文件，说明项目包含明确的启动或调度流程。")

        if "tests" in visible_paths or any(path.startswith("tests/") for path in visible_paths):
            lines.append("包含测试相关目录或文件，说明项目包含功能验证或回归检查能力。")

        if "docs" in visible_paths or any(path.startswith("docs/") for path in visible_paths):
            lines.append("包含文档相关目录，说明项目提供使用说明、设计文档或协作资料。")

        markdown_files = [path for path in top_level_files if Path(path).suffix.lower() == ".md"]
        if markdown_files:
            lines.append("根目录包含 " + "、".join(f"`{item}`" for item in markdown_files[:3]) + " 等说明文件，可辅助理解项目目标和使用方式。")

        if not lines:
            if top_level_files:
                lines.append("当前项目的主要功能集中在根目录文件中，整体结构较为轻量。")
            else:
                lines.append("当前项目可见条目较少，建议结合目录树和关键文件进一步判断其主要功能。")

        return self._dedupe_preserve_order(lines, limit=6)

    def _summarize_overview_error(self, exc: Exception) -> str:
        """收敛写入项目文档的错误信息，避免泄露内部细节。"""
        error_type = type(exc).__name__
        if isinstance(exc, RuntimeError):
            return f"`{error_type}`：项目概览的 LLM 生成失败，已改用规则分析。"
        return f"`{error_type}`：项目概览构建过程出现异常，已改用规则分析。"

    def _normalize_llm_markdown(self, content: str, project_name: str) -> str:
        """规整 LLM 返回的 Markdown 内容。"""
        normalized = content.replace("\r\n", "\n").strip()
        normalized = re.sub(r"^```(?:markdown|md)?\s*", "", normalized)
        normalized = re.sub(r"\s*```$", "", normalized)
        if not normalized.startswith("#"):
            normalized = f"# 项目概览：{project_name}\n\n{normalized}"
        return normalized.rstrip() + "\n"

    def _collect_tech_architecture(self, project_root: Path, entries: list[dict[str, Any]]) -> list[str]:
        """提取“技术架构”摘要。"""
        lines: list[str] = []
        primary_language = self._detect_primary_language(entries)
        if primary_language:
            lines.append(f"主要语言：{primary_language}。")

        build_summary = self._detect_build_and_dependency_style(project_root)
        if build_summary:
            lines.append(build_summary)

        entrypoints = self._detect_entrypoints(entries)
        if entrypoints:
            lines.append("入口组织：检测到 " + "、".join(f"`{item}`" for item in entrypoints) + " 等启动或调度入口。")

        source_dirs = [
            entry["relative_path"]
            for entry in entries
            if entry["is_dir"]
            and "/" not in entry["relative_path"]
            and entry["relative_path"] not in {"tests", "docs", "scripts"}
        ]
        if source_dirs:
            lines.append("源码布局：核心目录包括 " + "、".join(f"`{name}/`" for name in source_dirs[:6]) + "。")

        if "tests" in {entry["relative_path"] for entry in entries}:
            lines.append("质量保障：存在 `tests/` 目录，用于单元测试和回归验证。")

        lines.append("项目级运行配置会写入 `.gaoagent/`，用于隔离 API、MCP、Skills、RAG 等私有配置。")
        return self._dedupe_preserve_order(lines, limit=8)

    def _collect_project_entries(self, project_root: Path) -> tuple[str, list[dict[str, Any]], bool]:
        """收集项目目录树与条目列表。"""
        tree_lines = [f"{project_root.name}/"]
        entries: list[dict[str, Any]] = []
        truncated = False

        def walk(current_dir: Path, prefix: str, depth: int) -> bool:
            nonlocal truncated
            children = self._list_visible_children(current_dir)
            if not children:
                return False

            if depth > self._PROJECT_OVERVIEW_MAX_DEPTH:
                tree_lines.append(f"{prefix}└── ...")
                truncated = True
                return False

            for index, child in enumerate(children):
                if len(entries) >= self._PROJECT_OVERVIEW_MAX_ENTRIES:
                    tree_lines.append(f"{prefix}{'└── ' if index == len(children) - 1 else '├── '}...")
                    truncated = True
                    return True

                is_last = index == len(children) - 1
                connector = "└── " if is_last else "├── "
                suffix = "/" if child.is_dir() else ""
                relative_path = child.relative_to(project_root).as_posix()
                tree_lines.append(f"{prefix}{connector}{child.name}{suffix}")
                entries.append(
                    {
                        "path": child,
                        "relative_path": relative_path,
                        "is_dir": child.is_dir(),
                    }
                )
                if child.is_dir():
                    child_prefix = prefix + ("    " if is_last else "│   ")
                    if walk(child, child_prefix, depth + 1):
                        return True
            return False

        walk(project_root, "", 1)
        return ("\n".join(tree_lines), entries, truncated)

    def _list_visible_children(self, dir_path: Path) -> list[Path]:
        """返回目录下用于项目概览的可见子项。"""
        children: list[Path] = []
        try:
            iterator = list(dir_path.iterdir())
        except Exception as e:
            Console.debug(f"遍历目录失败：{dir_path}，{e}")
            return []

        for child in iterator:
            if self._should_ignore_path(child):
                continue
            children.append(child)

        return sorted(children, key=lambda item: (not item.is_dir(), item.name.lower()))

    def _should_ignore_path(self, path: Path) -> bool:
        """判断当前路径是否应从项目概览中忽略。"""
        name = path.name
        if path.is_symlink():
            return True
        if name in self._PROJECT_OVERVIEW_IGNORED_NAMES:
            return True
        if path.suffix.lower() in self._PROJECT_OVERVIEW_IGNORED_SUFFIXES:
            return True
        if name.startswith(".") and name not in {".github", ".gitignore"}:
            return True
        return False

    def _detect_primary_language(self, entries: list[dict[str, Any]]) -> str | None:
        """根据源码后缀粗略推断项目主要语言。"""
        language_map = {
            ".py": "Python",
            ".ts": "TypeScript",
            ".tsx": "TypeScript/React",
            ".js": "JavaScript",
            ".jsx": "JavaScript/React",
            ".java": "Java",
            ".go": "Go",
            ".rs": "Rust",
            ".cs": "C#",
            ".cpp": "C++",
            ".c": "C",
            ".kt": "Kotlin",
            ".swift": "Swift",
            ".php": "PHP",
            ".rb": "Ruby",
        }
        counts: dict[str, int] = {}
        for entry in entries:
            if entry["is_dir"]:
                continue
            suffix = Path(entry["relative_path"]).suffix.lower()
            language = language_map.get(suffix)
            if language is None:
                continue
            counts[language] = counts.get(language, 0) + 1
        if not counts:
            return None
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        top_languages = [name for name, _ in ranked[:3]]
        return " / ".join(top_languages)

    def _detect_build_and_dependency_style(self, project_root: Path) -> str | None:
        """根据常见配置文件判断构建与依赖管理方式。"""
        pyproject_file = project_root / "pyproject.toml"
        if pyproject_file.exists():
            summary = "依赖与构建：使用 `pyproject.toml` 管理项目元数据与依赖"
            try:
                payload = tomllib.loads(pyproject_file.read_text(encoding="utf-8"))
            except Exception as e:
                Console.debug(f"解析 pyproject.toml 失败：{e}")
                payload = {}
            project_info = payload.get("project") if isinstance(payload, dict) else {}
            if isinstance(project_info, dict):
                requires_python = project_info.get("requires-python")
                if isinstance(requires_python, str) and requires_python.strip():
                    summary += f"，Python 版本要求为 `{requires_python.strip()}`"
            return summary + "。"

        if (project_root / "package.json").exists():
            return "依赖与构建：使用 `package.json` 管理前端或 Node.js 依赖与脚本。"
        if (project_root / "Cargo.toml").exists():
            return "依赖与构建：使用 `Cargo.toml` 管理 Rust crate 依赖与编译配置。"
        if (project_root / "go.mod").exists():
            return "依赖与构建：使用 `go.mod` 管理 Go Module 依赖。"
        if (project_root / "requirements.txt").exists():
            return "依赖与构建：使用 `requirements.txt` 管理 Python 依赖。"
        return None

    def _detect_entrypoints(self, entries: list[dict[str, Any]]) -> list[str]:
        """探测常见启动入口文件。"""
        candidates: list[str] = []
        preferred_names = {
            "__main__.py",
            "main.py",
            "app.py",
            "cli.py",
            "manage.py",
            "server.py",
            "index.js",
            "main.go",
        }
        for entry in entries:
            if entry["is_dir"]:
                continue
            relative_path = entry["relative_path"]
            parts = relative_path.split("/")
            if len(parts) > 3:
                continue
            if Path(relative_path).name in preferred_names:
                candidates.append(relative_path)
        return self._dedupe_preserve_order(candidates, limit=4)

    def _describe_path(self, project_root: Path, path: Path) -> str:
        """描述文件或目录的用途。"""
        if path.is_dir():
            return self._describe_directory(path)
        return self._describe_file(project_root, path)

    def _describe_directory(self, path: Path) -> str:
        """描述目录用途。"""
        child_count = len(self._list_visible_children(path))
        if path.name.startswith("."):
            return f"隐藏目录，当前包含 {child_count} 个可见子项，通常用于工具配置或元数据管理。"
        return f"目录，当前包含 {child_count} 个可见子项，用于归类相关代码、资源或配置。"

    def _describe_file(self, project_root: Path, path: Path) -> str:
        """描述文件用途。"""
        relative_path = path.relative_to(project_root).as_posix()

        if relative_path.startswith("tests/") and path.stem.startswith("test_"):
            target = path.stem.removeprefix("test_")
            return f"测试文件，用于验证 `{target}` 相关功能。"

        if path.suffix == ".py":
            docstring = self._extract_python_docstring(path)
            if docstring:
                return docstring

            stem = path.stem
            if stem == "__init__":
                return "Python 包初始化文件，用于声明包边界或导出公共接口。"
            if stem == "__main__":
                return "Python 模块执行入口，通常支持 `python -m` 方式启动。"
            if stem == "cli":
                return "命令行入口文件，用于定义可执行命令、参数与调度逻辑。"
            return "Python 源码文件，承载项目的具体实现逻辑。"

        if path.suffix.lower() == ".md":
            summary = self._extract_markdown_summary(path)
            if summary:
                return summary
            return "Markdown 文档文件，用于记录说明、设计或操作指南。"

        if path.suffix.lower() in {".json", ".toml", ".yaml", ".yml", ".ini"}:
            return "配置文件，用于声明结构化参数、环境配置或工具设置。"

        if path.suffix:
            return f"`{path.suffix}` 文件，用于承载项目相关内容或资源。"
        return "普通文件，用途需结合上下文进一步判断。"

    def _extract_python_docstring(self, path: Path) -> str | None:
        """提取 Python 文件的模块级 docstring 首句。"""
        text = self._safe_read_text(path)
        if not text:
            return None
        try:
            module = ast.parse(text)
        except SyntaxError:
            return None
        docstring = ast.get_docstring(module)
        return self._shorten_text(docstring)

    def _extract_markdown_summary(self, path: Path) -> str | None:
        """提取 Markdown 文件的标题或首段摘要。"""
        text = self._safe_read_text(path)
        if not text:
            return None
        heading: str | None = None
        paragraph: str | None = None
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("#") and heading is None:
                heading = self._normalize_text(re.sub(r"^#+\s*", "", line))
                continue
            if line.startswith(("[![", "![", "-", "*", "`")):
                continue
            paragraph = self._normalize_text(line)
            break
        if heading and paragraph:
            return self._shorten_text(f"{heading}，{paragraph}")
        return self._shorten_text(paragraph or heading)

    def _safe_read_text(self, path: Path) -> str | None:
        """安全读取文本文件。"""
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                return path.read_text(encoding="utf-8-sig")
            except Exception as e:
                Console.debug(f"读取文件失败（utf-8-sig 回退）：{path}，{e}")
                return None
        except Exception as e:
            Console.debug(f"读取文件失败：{path}，{e}")
            return None

    def _dedupe_preserve_order(self, items: list[str], limit: int) -> list[str]:
        """去重并保留原顺序。"""
        results: list[str] = []
        seen: set[str] = set()
        for item in items:
            normalized = self._normalize_text(item)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            results.append(normalized)
            if len(results) >= limit:
                break
        return results

    def _normalize_text(self, text: str | None) -> str:
        """收敛空白字符，避免概览内容过长。"""
        if not text:
            return ""
        return re.sub(r"\s+", " ", text).strip()

    def _shorten_text(self, text: str | None, limit: int = 120) -> str | None:
        """截断过长文本，避免文档冗长失控。"""
        normalized = self._normalize_text(text)
        if not normalized:
            return None
        if len(normalized) <= limit:
            return normalized
        return normalized[: limit - 3].rstrip() + "..."
