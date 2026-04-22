from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import click

from gaoagent.core.CoreConfigDefault import CoreConfigDefault
from gaoagent.mcp.MCPClientCompat import MCPStdioClientSync


class MCPHandlers:
    _PROJECTS_REGISTRY_FILENAME = "inited_projects.txt"
    _MCP_CONFIG_FILENAME = "gao_client_mcp_setting.json"

    def list(self) -> None:
        (scope, config_file) = self._resolve_scope_and_config_path()
        servers = self._load_mcp_servers(config_file)
        if not servers:
            click.echo(f"未检测到{scope} MCP 服务：{config_file}")
            return

        click.echo(f"{scope} MCP 服务列表：")
        for idx, name in enumerate(sorted(servers.keys()), start=1):
            body = servers.get(name)
            status = self._status_of(body)
            server_type = body.get("type") if isinstance(body, dict) else None
            suffix = f", type={server_type}" if isinstance(server_type, str) and server_type.strip() else ""
            click.echo(f"{idx}. {name} [{status}{suffix}]")

    def add(self) -> None:
        added = self._prompt_mcp_config()
        (name, body) = next(iter(added.items()))
        project_root = self._detect_project_root()
        global_file = self._global_config_dir() / self._MCP_CONFIG_FILENAME

        if project_root is not None:
            project_file = project_root / ".gaoagent" / self._MCP_CONFIG_FILENAME
            self._upsert_mcp_server(global_file, name, body)
            self._upsert_mcp_server(project_file, name, body)
            click.echo(f"已添加 MCP：{name}")
            click.echo(f"- 全局配置：{global_file}")
            click.echo(f"- 项目配置：{project_file}")
            return

        self._upsert_mcp_server(global_file, name, body)
        click.echo(f"已添加 MCP：{name}")
        click.echo(f"- 全局配置：{global_file}")

    def remove(self, name: str | None = None) -> None:
        (scope, config_file) = self._resolve_scope_and_config_path()
        servers = self._load_mcp_servers(config_file)
        if not servers:
            click.echo(f"未检测到{scope} MCP 配置：{config_file}")
            return

        target = self._resolve_target_name(servers, action="移除", name=name)
        if target is None:
            return

        del servers[target]
        self._write_mcp_servers(config_file, servers)
        click.echo(f"已从{scope}配置移除 MCP：{target}")

    def enable(self, name: str | None = None) -> None:
        self._set_disabled_flag(False, name=name)

    def disable(self, name: str | None = None) -> None:
        self._set_disabled_flag(True, name=name)

    def test(self, name: str | None = None) -> None:
        (scope, config_file) = self._resolve_scope_and_config_path()
        servers = self._load_mcp_servers(config_file)
        if not servers:
            click.echo(f"未检测到{scope} MCP 配置：{config_file}")
            return

        target = self._resolve_target_name(servers, action="测试", name=name)
        if target is None:
            return
        body = servers.get(target)
        if not isinstance(body, dict):
            click.echo(f"MCP 配置无效：{target}")
            return
        if body.get("disabled") is True:
            click.echo(f"提示：MCP `{target}` 当前状态为 disabled=true，仍将尝试连通性测试")

        start = time.perf_counter()
        try:
            tools = MCPStdioClientSync.from_config(server_name=target, config=body).list_tools()
            duration_ms = int((time.perf_counter() - start) * 1000)
            click.echo(f"测试通过：{target}（{duration_ms} ms）")
            click.echo(f"可用工具数：{len(tools)}")
            if tools:
                tool_names = [str(item.get('name')) for item in tools if isinstance(item, dict) and item.get("name")]
                if tool_names:
                    click.echo(f"工具：{', '.join(tool_names)}")
        except Exception as e:
            duration_ms = int((time.perf_counter() - start) * 1000)
            click.echo(f"测试失败：{target}（{duration_ms} ms）")
            click.echo(str(e))

    def _set_disabled_flag(self, disabled: bool, *, name: str | None = None) -> None:
        action = "disable" if disabled else "enable"
        (scope, config_file) = self._resolve_scope_and_config_path()
        servers = self._load_mcp_servers(config_file)
        if not servers:
            click.echo(f"未检测到{scope} MCP 配置：{config_file}")
            return

        target = self._resolve_target_name(servers, action=action, name=name)
        if target is None:
            return

        body = servers.get(target)
        if not isinstance(body, dict):
            click.echo(f"未找到 MCP：{target}")
            return

        body["disabled"] = disabled
        servers[target] = body
        self._write_mcp_servers(config_file, servers)
        click.echo(f"已更新 {scope} MCP：{target}, disabled={str(disabled).lower()}")

    def _resolve_scope_and_config_path(self) -> tuple[str, Path]:
        project_root = self._detect_project_root()
        if project_root is not None:
            return ("项目", project_root / ".gaoagent" / self._MCP_CONFIG_FILENAME)
        return ("全局", self._global_config_dir() / self._MCP_CONFIG_FILENAME)

    def _global_config_dir(self) -> Path:
        return Path.home() / ".gaoagent"

    def _project_registry_file(self) -> Path:
        return self._global_config_dir() / self._PROJECTS_REGISTRY_FILENAME

    def _detect_project_root(self) -> Path | None:
        """
        参考 Utils.project_root_dir() 的判定逻辑：
        1) cwd 直接包含 `.gaoagent` 视为项目根；
        2) 否则从全局已初始化项目清单中匹配最深父目录。
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

    def _read_json_file(self, file_path: Path) -> Any | None:
        if not file_path.exists() or not file_path.is_file():
            return None
        try:
            return json.loads(file_path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _load_mcp_servers(self, file_path: Path) -> dict[str, Any]:
        payload = self._read_json_file(file_path)
        if not isinstance(payload, dict):
            return {}
        servers = payload.get("mcpServers")
        return servers if isinstance(servers, dict) else {}

    def _write_mcp_servers(self, file_path: Path, servers: dict[str, Any]) -> None:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"mcpServers": servers}
        tmp_file = file_path.with_name(f"{file_path.name}.tmp")
        tmp_file.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        tmp_file.replace(file_path)

    def _upsert_mcp_server(self, file_path: Path, name: str, body: dict[str, Any]) -> None:
        servers = self._load_mcp_servers(file_path)
        servers[name] = body
        self._write_mcp_servers(file_path, servers)

    def _prompt_mcp_config(self) -> dict[str, Any]:
        validator = CoreConfigDefault()
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
                validator._validate_mcp_config(value)
            except Exception as e:
                click.echo(f"格式错误：{e}")
                continue
            return value

    def _prompt_server_name(self, servers: dict[str, Any], *, action: str) -> str:
        names = sorted([str(x) for x in servers.keys()])
        default_name = names[0] if names else ""
        if default_name:
            return click.prompt(
                f"请输入要{action}的 MCP 名称",
                type=click.Choice(names, case_sensitive=False),
                default=default_name,
                show_default=True,
            )
        return click.prompt(f"请输入要{action}的 MCP 名称", type=str).strip()

    def _resolve_target_name(self, servers: dict[str, Any], *, action: str, name: str | None) -> str | None:
        if not isinstance(name, str) or not name.strip():
            return self._prompt_server_name(servers, action=action)
        lowered = name.strip().lower()
        for candidate in servers.keys():
            if isinstance(candidate, str) and candidate.lower() == lowered:
                return candidate
        click.echo(f"未找到 MCP：{name}")
        return None

    def _status_of(self, body: Any) -> str:
        if not isinstance(body, dict):
            return "invalid"
        return "disabled" if body.get("disabled") is True else "enabled"
