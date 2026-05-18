from __future__ import annotations

from pathlib import Path

from gaoagent.core.runner.utils import PROJECTS_REGISTRY_FILENAME


class InitRegistryTool:
    """初始化项目注册表相关工具。"""

    def _project_registry_file(self, global_dir: Path) -> Path:
        """返回“已初始化项目注册表”文件路径。"""
        return global_dir / PROJECTS_REGISTRY_FILENAME

    def _cleanup_project_registry(self, registry_file: Path) -> list[Path]:
        """清理注册表中的失效项目路径并返回保留结果。"""
        existing = self._load_project_registry(registry_file)
        valid: list[Path] = []
        for root in existing:
            config_dir = root / ".gaoagent"
            if root.exists() and root.is_dir() and config_dir.exists() and config_dir.is_dir():
                valid.append(root)
        self._write_project_registry(registry_file, valid)
        return valid

    def _register_project_root(self, registry_file: Path, project_root: Path) -> None:
        """将当前项目根目录写入注册表。"""
        roots = self._cleanup_project_registry(registry_file)
        normalized = project_root.resolve()
        if normalized not in roots:
            roots.append(normalized)
            self._write_project_registry(registry_file, roots)

    def _load_project_registry(self, registry_file: Path) -> list[Path]:
        """读取并解析项目注册表文件，返回去重后的项目路径列表。"""
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
