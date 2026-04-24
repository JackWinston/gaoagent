from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import click
from gaoagent.core.runner.Console import Console

from gaoagent.core.runner.Utils import (
    global_config_dir,
    scan_skills_metadata,
    try_project_root_dir,
)


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
    def list(self) -> None:
        """列出当前作用域（项目或全局）下的 Skill 列表。"""
        (scope, skills_dir) = self._resolve_scope_and_paths()
        if not skills_dir.exists() or not skills_dir.is_dir():
            skills_dir.mkdir(parents=True, exist_ok=True)
            Console.info(f"没找到{scope} Skills 目录，已自动创建.")

        skills = self._get_skills_in_dir(skills_dir)
        
        if not skills:
            Console.info(f"{scope} 目录下无可用 Skill")
            should_install = Console.confirm(
                "要不要现在从全局仓库安装 Skill？",
                default=True,
                show_default=True,
            )
            if should_install:
                self.install()
            return

        Console.info(f"{scope} Skills 列表：")
        for idx, skill in enumerate(skills, start=1):
            name = skill["name"]
            Console.info(f"{idx}. {name} - {skill.get('description', '')}")

    def install(self, name: str | None = None) -> None:
        """在当前作用域安装 Skill。

        参数:
        - `name`: 指定要安装的 Skill 名称；为空时进入多选交互。

        安装流程:
        - 校验全局 Skill 目录与当前项目根目录是否存在。
        - 展示可安装 Skill 列表并解析用户选择。
        - 将选中 Skill 复制到 `项目/.gaoagent/skills`。
        - 目标已存在时跳过并提示，不覆盖已有内容。
        """
        (scope, current_skills_dir) = self._resolve_scope_and_paths()
        global_skills_dir = (global_config_dir() / "skills").resolve()

        current_skills_dir.mkdir(parents=True, exist_ok=True)
        global_skills_dir.mkdir(parents=True, exist_ok=True)

        global_skills = self._get_skills_in_dir(global_skills_dir)
        
        if not global_skills:
            Console.info("全局目录里还没有可安装的 Skill。")
            return
            
        names = [s["name"] for s in global_skills]
        Console.info("可安装的全局 Skills：")
        for i, s in enumerate(global_skills, start=1):
            Console.info(f"{i}. {s['name']} - {s.get('description', '')}")
            
        if name and name in names:
            selected_names = [name]
        else:
            selected_names = self._prompt_multi_select("请选择要安装的 Skill（输入序号或名称，逗号分隔；回车跳过；输入 all 选择全部）", names)
        
        if not selected_names:
            return
            
        selected_set = set(selected_names)
        selected_items = [item for item in global_skills if item["name"] in selected_set]
        installed_count = 0
        installed_names: list[str] = []
        for item in selected_items:
            skill_name = item["name"]
            src = Path(item["src_dir"])
            dst = current_skills_dir / skill_name
            if dst.exists():
                Console.info(f"Skill '{skill_name}' 已存在，跳过。")
                continue
            self._copy_dir(src, dst)
            installed_count += 1
            installed_names.append(skill_name)
            
        if installed_count > 0:
            Console.info(
                f"在{scope}作用域安装了 {installed_count} 个 Skill：{', '.join(installed_names)}"
            )

    def uninstall(self, name: str | None = None) -> None:
        """卸载当前作用域中的 Skill。

        参数:
        - `name`: 目标 Skill 名称；为空时交互选择。

        约束:
        - 仅删除当前作用域目录，不影响另一个作用域。
        """
        (scope, skills_dir) = self._resolve_scope_and_paths()
            
        skills = self._get_skills_in_dir(skills_dir)
        if not skills:
            Console.info(f"{scope}作用域下还没有已安装的 Skill。")
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

        Console.info(f"已在{scope}作用域卸载 Skill：{target}")

    def _resolve_scope_and_paths(self) -> tuple[str, Path]:
        """解析当前操作作用域并返回对应 Skills 目录路径。"""
        project_root = try_project_root_dir()
        if project_root is not None:
            return ("项目", (project_root / ".gaoagent" / "skills").resolve())
        return ("全局", (global_config_dir() / "skills").resolve())

    def _get_skills_in_dir(self, skills_dir: Path) -> list[dict[str, Any]]:
        """扫描目录并返回合法 Skill 元数据列表。"""
        if not skills_dir.exists() or not skills_dir.is_dir():
            return []

        (skills, invalid_skills) = scan_skills_metadata(skills_dir)
        if invalid_skills:
            Console.info(f"发现不符合规范的 Skill 文件，共 {len(invalid_skills)} 个：")
            for item in invalid_skills:
                Console.info(f"- {item['path']}: {item['reason']}")
        return skills

    def _prompt_skill_name(self, skills: list[dict[str, Any]], *, action: str) -> str | None:
        """从候选 Skill 中交互选择一个目标名称。"""
        names = [s["name"] for s in skills]
        if not names:
            return None
            
        default_name = names[0]
        choice = Console.prompt(
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
            raw = Console.prompt(prompt, type=str, default="", show_default=False).strip()
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

            Console.info("输入不合法，请重新输入")

    def _copy_dir(self, src: Path, dst: Path) -> None:
        """目录覆盖复制：若目标已存在则先删除后复制。"""
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
