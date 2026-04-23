from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import click

from gaoagent.core.runner.Utils import scan_skills_metadata


class SkillsHandlers:
    """Skill 命令处理器（`gaoagent skills` 子命令入口）。

    定位:
    - 负责 Skill 的展示、安装、卸载等 CLI 交互流程。
    - 根据当前工作目录自动判定项目作用域或全局作用域。

    核心职责:
    - 列出可用 Skill 元数据（名称、描述）并提示不合规 Skill 文件。
    - 在项目中从全局 Skill 仓库批量安装 Skill。
    - 在项目中卸载已安装 Skill。

    边界:
    - 不负责 Skill 的运行时执行，仅负责文件级管理与元数据校验展示。
    - Skill 元数据解析依赖 `scan_skills_metadata()`。
    """
    _PROJECTS_REGISTRY_FILENAME = "inited_projects.txt"

    def list(self) -> None:
        """列出当前作用域（项目或全局）下的 Skill 列表。"""
        (scope, skills_dir) = self._resolve_scope_and_paths()
        if not skills_dir.exists() or not skills_dir.is_dir():
            click.echo(f"未检测到{scope} Skills 目录：{skills_dir}")
            return

        skills = self._get_skills_in_dir(skills_dir)
        
        if not skills:
            click.echo(f"{scope} 目录下无可用 Skill")
            return

        click.echo(f"{scope} Skills 列表：")
        for idx, skill in enumerate(skills, start=1):
            name = skill["name"]
            click.echo(f"{idx}. {name} - {skill.get('description', '')}")

    def install(self, name: str | None = None) -> None:
        """从全局 Skill 仓库安装 Skill 到当前项目。

        参数:
        - `name`: 指定要安装的 Skill 名称；为空时进入多选交互。

        安装流程:
        - 校验全局 Skill 目录与当前项目根目录是否存在。
        - 展示可安装 Skill 列表并解析用户选择。
        - 将选中 Skill 复制到 `项目/.gaoagent/skills`。
        - 目标已存在时跳过并提示，不覆盖已有内容。
        """
        global_dir = self._global_config_dir() / "skills"
        if not global_dir.exists() or not global_dir.is_dir():
            click.echo(f"未检测到全局 Skills 目录：{global_dir}")
            return
            
        project_root = self._detect_project_root()
        if not project_root:
            click.echo("未检测到项目根目录（当前不在已初始化的 gaoagent 项目中），无法安装 Skill。")
            return
            
        project_skills_dir = project_root / ".gaoagent" / "skills"
        global_skills = self._get_skills_in_dir(global_dir)
        
        if not global_skills:
            click.echo("全局目录暂无可用 Skill，请先配置全局 Skill。")
            return
            
        names = [s["name"] for s in global_skills]
        click.echo("可安装的全局 Skills：")
        for i, s in enumerate(global_skills, start=1):
            click.echo(f"{i}. {s['name']} - {s.get('description', '')}")
            
        if name and name in names:
            selected_names = [name]
        else:
            selected_names = self._prompt_multi_select("请选择要安装的 Skill（输入序号或名称，逗号分隔；回车跳过；输入 all 选择全部）", names)
        
        if not selected_names:
            return
            
        project_skills_dir.mkdir(parents=True, exist_ok=True)
        selected_set = set(selected_names)
        selected_items = [item for item in global_skills if item["name"] in selected_set]
        installed_count = 0
        for item in selected_items:
            skill_name = item["name"]
            src = Path(item["src_dir"])
            dst = project_skills_dir / skill_name
            if dst.exists():
                click.echo(f"Skill '{skill_name}' 已存在，跳过。")
                continue
            self._copy_dir(src, dst)
            installed_count += 1
            
        if installed_count > 0:
            click.echo(f"成功安装 {installed_count} 个 Skill 到项目目录：{project_skills_dir}")

    def uninstall(self, name: str | None = None) -> None:
        """卸载当前项目中的 Skill。

        参数:
        - `name`: 目标 Skill 名称；为空时交互选择。

        约束:
        - 仅允许在项目作用域执行卸载；全局作用域会直接拒绝。
        """
        (scope, skills_dir) = self._resolve_scope_and_paths()
        if scope == "全局":
            click.echo("只能在项目中卸载 Skill。")
            return
            
        skills = self._get_skills_in_dir(skills_dir)
        if not skills:
            click.echo(f"当前项目无已安装的 Skill。")
            return
            
        if name and any(s["name"] == name for s in skills):
            target = name
        else:
            target = self._prompt_skill_name(skills, action="卸载")
            
        if not target:
            return

        target_dir: Path | None = None
        for item in skills:
            if item["name"] == target:
                target_dir = Path(item["src_dir"])
                break
        if target_dir is None:
            target_dir = skills_dir / target

        if target_dir.exists():
            shutil.rmtree(target_dir)

        click.echo(f"已卸载 Skill：{target}")

    def _resolve_scope_and_paths(self) -> tuple[str, Path]:
        """解析当前操作作用域并返回对应 Skills 目录路径。"""
        project_root = self._detect_project_root()
        if project_root is not None:
            return (
                "项目",
                project_root / ".gaoagent" / "skills",
            )
        return (
            "全局",
            self._global_config_dir() / "skills",
        )

    def _global_config_dir(self) -> Path:
        """返回全局配置目录（`~/.gaoagent`）。"""
        return Path.home() / ".gaoagent"

    def _project_registry_file(self) -> Path:
        """返回项目注册表文件路径。"""
        return self._global_config_dir() / self._PROJECTS_REGISTRY_FILENAME

    def _detect_project_root(self) -> Path | None:
        """检测当前命令对应的项目根目录。

        判定规则:
        - 当前目录包含 `.gaoagent` 时直接命中。
        - 否则从注册表中匹配当前路径的最深父目录。
        """
        cwd = Path.cwd().resolve()
        if (cwd / ".gaoagent").is_dir():
            return cwd

        candidates: list[Path] = []
        for root in self._load_project_registry_paths():
            config_dir = root / ".gaoagent"
            if not (root.exists() and root.is_dir() and config_dir.exists() and config_dir.is_dir()):
                continue
            if root == cwd or root in cwd.parents:
                candidates.append(root)

        if not candidates:
            return None
        candidates.sort(key=lambda p: len(p.parts), reverse=True)
        return candidates[0]

    def _load_project_registry_paths(self) -> list[Path]:
        """读取项目注册表并返回去重后的路径列表。"""
        registry_file = self._project_registry_file()
        if not registry_file.exists() or not registry_file.is_file():
            return []
        try:
            lines = registry_file.read_text(encoding="utf-8").splitlines()
        except Exception:
            return []

        roots: list[Path] = []
        seen: set[str] = set()
        for line in lines:
            raw = line.strip()
            if not raw:
                continue
            try:
                root = Path(raw).expanduser().resolve()
            except Exception:
                continue
            key = str(root)
            if key in seen:
                continue
            seen.add(key)
            roots.append(root)
        return roots

    def _get_skills_in_dir(self, skills_dir: Path) -> list[dict[str, Any]]:
        """扫描目录并返回合法 Skill 元数据列表。"""
        if not skills_dir.exists() or not skills_dir.is_dir():
            return []

        (skills, invalid_skills) = scan_skills_metadata(skills_dir)
        if invalid_skills:
            click.echo(f"发现不符合规范的 Skill 文件，共 {len(invalid_skills)} 个：")
            for item in invalid_skills:
                click.echo(f"- {item['path']}: {item['reason']}")
        return skills

    def _prompt_skill_name(self, skills: list[dict[str, Any]], *, action: str) -> str | None:
        """从候选 Skill 中交互选择一个目标名称。"""
        names = [s["name"] for s in skills]
        if not names:
            return None
            
        default_name = names[0]
        choice = click.prompt(
            f"请输入要{action}的 Skill 名称",
            type=click.Choice(names, case_sensitive=False),
            default=default_name,
            show_default=True,
        )
        return choice

    def _prompt_multi_select(self, prompt: str, options: list[str]) -> list[str]:
        """通用多选输入解析器（支持序号/名称/all/回车跳过）。"""
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

    def _copy_dir(self, src: Path, dst: Path) -> None:
        """目录覆盖复制：若目标已存在则先删除后复制。"""
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
