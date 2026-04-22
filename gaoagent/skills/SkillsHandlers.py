from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import click

from gaoagent.core.runner.Utils import parse_skill_frontmatter


class SkillsHandlers:
    _PROJECTS_REGISTRY_FILENAME = "inited_projects.txt"

    def list(self) -> None:
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
            selected = [name]
        else:
            selected = self._prompt_multi_select("请选择要安装的 Skill（输入序号或名称，逗号分隔；回车跳过；输入 all 选择全部）", names)
        
        if not selected:
            return
            
        project_skills_dir.mkdir(parents=True, exist_ok=True)
        installed_count = 0
        for skill_name in selected:
            src = global_dir / skill_name
            dst = project_skills_dir / skill_name
            if dst.exists():
                click.echo(f"Skill '{skill_name}' 已存在，跳过。")
                continue
            self._copy_dir(src, dst)
            installed_count += 1
            
        if installed_count > 0:
            click.echo(f"成功安装 {installed_count} 个 Skill 到项目目录：{project_skills_dir}")

    def uninstall(self, name: str | None = None) -> None:
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
            
        target_dir = skills_dir / target
        if target_dir.exists():
            shutil.rmtree(target_dir)

        click.echo(f"已卸载 Skill：{target}")

    def _resolve_scope_and_paths(self) -> tuple[str, Path]:
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
        return Path.home() / ".gaoagent"

    def _project_registry_file(self) -> Path:
        return self._global_config_dir() / self._PROJECTS_REGISTRY_FILENAME

    def _detect_project_root(self) -> Path | None:
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
        if not skills_dir.exists() or not skills_dir.is_dir():
            return []
            
        skills = []
        for file_path in skills_dir.rglob("*"):
            if not file_path.is_file() or file_path.name.lower() != "skill.md":
                continue
            meta = parse_skill_frontmatter(file_path)
            if meta:
                skills.append(meta)
                
        skills.sort(key=lambda x: x.get("name", ""))
        return skills

    def _prompt_skill_name(self, skills: list[dict[str, Any]], *, action: str) -> str | None:
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
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
