from __future__ import annotations

# 本文件提供 MCP 的“同步兼容层”：
# - 底层 SDK 是 async 接口，但 Runner/Config 流程主要是同步调用。
# - 因此这里将 async connect/list/call 封装为可在同步代码中安全调用的接口。
# - 同时统一缓存结构，供 ReActRunner 把 MCP 工具暴露给 LLM。

import asyncio
import datetime
import hashlib
import json
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


def _config_dir() -> Path:
    """获取 MCP 工具缓存目录，支持 `GAOAGENT_CONFIG_DIR` 覆盖。"""
    override = os.environ.get("GAOAGENT_CONFIG_DIR")
    if override and override.strip():
        return Path(override).expanduser().resolve()
    return Path.home() / ".gaoagent"


def _sanitize_token(value: str) -> str:
    """将任意字符串规范化为 MCP 导出名可用 token。

    规则:
    - 仅保留字母、数字、下划线、短横线。
    - 连续非法字符折叠为单个 `_`，并去除首尾 `_`。
    - 结果为空时回退为 `"x"`。
    """
    s = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip())
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "x"


def export_tool_name(server_name: str, tool_name: str, *, max_len: int = 64) -> str:
    """
    生成导出给 LLM 的工具名。

    命名规则：`mcp__{server}__{tool}`
    作用：避免不同 MCP server 下同名 tool 冲突。
    """
    prefix = "mcp__"
    s = f"{prefix}{_sanitize_token(server_name)}__{_sanitize_token(tool_name)}"
    if len(s) <= max_len:
        return s
    # 超长时追加稳定哈希后缀，降低截断后的重名概率。
    suffix = f"__{hashlib.sha1(s.encode('utf-8')).hexdigest()[:8]}"
    keep = max(1, max_len - len(suffix))
    return f"{s[:keep]}{suffix}"


def _run_coro(coro: Any) -> Any:
    """
    在同步上下文中执行协程。

    - 若当前线程没有事件循环，直接 `asyncio.run`。
    - 若已有运行中的事件循环（常见于某些宿主环境），在新线程内执行，
      避免 `RuntimeError: asyncio.run() cannot be called from a running event loop`。
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is None:
        return asyncio.run(coro)

    result: Any = None
    error: BaseException | None = None

    def runner() -> None:
        """线程入口：在子线程中执行协程并捕获异常。"""
        nonlocal result, error
        try:
            result = asyncio.run(coro)
        except BaseException as e:
            error = e

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    t.join()
    if error is not None:
        raise error
    return result


@dataclass(frozen=True)
class MCPServerConfig:
    """MCP 服务配置的数据模型（宽松输入、严格产出）。

    用途:
    - 作为配置解析后的中间标准结构，统一字段类型，减少后续分支判断复杂度。
    - 支持 `stdio`、`sse`、`streamable_http` 三类 transport 共同字段。
    """
    type: str
    command: str = ""
    args: list[str] = field(default_factory=list)
    env: dict[str, str] | None = None
    url: str = ""
    headers: dict[str, str] | None = None
    timeout: int | None = None
    disabled: bool = False

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "MCPServerConfig":
        """从原始 JSON 配置解析为 `MCPServerConfig`。

        解析特性:
        - 对 `timeout` 做字符串/整数兼容解析。
        - 对 `args/env/headers` 做类型过滤，避免非字符串污染底层调用。
        - `disabled` 仅在值显式为 `True` 时生效。
        """
        timeout_raw = payload.get("timeout")
        timeout_value: int | None = None
        if isinstance(timeout_raw, int) and not isinstance(timeout_raw, bool):
            timeout_value = int(timeout_raw)
        elif isinstance(timeout_raw, str):
            t = timeout_raw.strip()
            if t and re.fullmatch(r"[+-]?\d+", t):
                timeout_value = int(t)

        args_value = [str(x) for x in (payload.get("args") or []) if isinstance(x, (str, int, float))]
        env_raw = payload.get("env")
        env_value: dict[str, str] | None = None
        if isinstance(env_raw, dict):
            env_value = {}
            for k, v in env_raw.items():
                if isinstance(k, str) and isinstance(v, str):
                    env_value[k] = v
        headers_raw = payload.get("headers")
        headers_value: dict[str, str] | None = None
        if isinstance(headers_raw, dict):
            headers_value = {}
            for k, v in headers_raw.items():
                if isinstance(k, str) and isinstance(v, str):
                    headers_value[k] = v

        return MCPServerConfig(
            type=str(payload.get("type") or ""),
            command=str(payload.get("command") or ""),
            args=args_value,
            env=env_value,
            url=str(payload.get("url") or ""),
            headers=headers_value,
            timeout=timeout_value,
            disabled=bool(payload.get("disabled") is True),
        )


class MCPStdioClientSync:
    """MCP 同步客户端兼容层（面向同步业务代码）。

    设计背景:
    - 官方 MCP SDK 以 async API 为主，而 Runner/CLI 以同步流程为主。
    - 本类将 connect/list/call 封装为同步接口，内部通过 `_run_coro()` 执行协程。

    支持传输:
    - `stdio`: 启动本地进程作为 MCP 服务端。
    - `sse`: 通过 SSE 连接远端服务。
    - `streamable_http`: 通过流式 HTTP 连接远端服务。
    """
    def __init__(
        self,
        *,
        server_name: str,
        transport_type: str,
        command: str,
        args: list[str],
        env: dict[str, str] | None,
        url: str,
        headers: dict[str, str] | None,
        timeout: int | None,
    ) -> None:
        """构造同步 MCP 客户端对象（仅保存配置，不发起连接）。"""
        self.server_name = server_name
        self.transport_type = transport_type
        self.command = command
        self.args = args
        self.env = env
        self.url = url
        self.headers = headers
        self.timeout = timeout

    @staticmethod
    def from_config(*, server_name: str, config: dict[str, Any]) -> "MCPStdioClientSync":
        """从配置字典构建客户端，并执行 transport 级前置校验。"""
        cfg = MCPServerConfig.from_dict(config)
        transport_type = cfg.type.strip().lower()
        if transport_type not in ("stdio", "sse", "streamable_http"):
            raise ValueError(f"MCP server `{server_name}` 的 type 不受支持：{cfg.type}")
        if transport_type == "stdio" and not cfg.command.strip():
            raise ValueError(f"MCP server `{server_name}` 缺少可执行 command")
        if transport_type in ("sse", "streamable_http") and not cfg.url.strip():
            raise ValueError(f"MCP server `{server_name}` 缺少 url")
        return MCPStdioClientSync(
            server_name=server_name,
            transport_type=transport_type,
            command=cfg.command,
            args=(cfg.args or []),
            env=cfg.env,
            url=cfg.url,
            headers=cfg.headers,
            timeout=cfg.timeout,
        )

    def list_tools(self) -> list[dict[str, Any]]:
        """
        连接 MCP server 并列出工具清单。

        返回结构与缓存结构对齐：name/description/inputSchema。
        """
        async def _inner() -> list[dict[str, Any]]:
            """异步内部实现：建立连接并拉取可用工具列表。"""
            (ClientSession, StdioServerParameters, stdio_client, sse_client, streamablehttp_client) = _import_sdk()
            if self.transport_type == "stdio":
                # 合并进程环境与用户配置，既支持额外变量，也不丢失 PATH 等基础变量。
                merged_env = dict(os.environ)
                if isinstance(self.env, dict):
                    merged_env.update(self.env)
                params = StdioServerParameters(command=self.command, args=self.args, env=merged_env)
                transport = stdio_client(params)
            elif self.transport_type == "sse":
                timeout_seconds = float(self.timeout) if isinstance(self.timeout, int) and self.timeout > 0 else 5.0
                transport = sse_client(self.url, headers=self.headers, timeout=timeout_seconds)
            elif self.transport_type == "streamable_http":
                timeout_seconds = float(self.timeout) if isinstance(self.timeout, int) and self.timeout > 0 else 30.0
                transport = streamablehttp_client(self.url, headers=self.headers, timeout=timeout_seconds)
            else:
                raise ValueError(f"Unsupported MCP transport type: {self.transport_type}")
            async with transport as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    tools = await session.list_tools()
                    return _extract_tools(tools)

        return _run_coro(_inner())

    def call_tool(self, *, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """调用指定 MCP 工具并归一化返回值。"""
        async def _inner() -> dict[str, Any]:
            """异步内部实现：连接服务并执行一次工具调用。"""
            (ClientSession, StdioServerParameters, stdio_client, sse_client, streamablehttp_client) = _import_sdk()
            if self.transport_type == "stdio":
                # 合并进程环境与用户配置，既支持额外变量，也不丢失 PATH 等基础变量。
                merged_env = dict(os.environ)
                if isinstance(self.env, dict):
                    merged_env.update(self.env)
                params = StdioServerParameters(command=self.command, args=self.args, env=merged_env)
                transport = stdio_client(params)
            elif self.transport_type == "sse":
                timeout_seconds = float(self.timeout) if isinstance(self.timeout, int) and self.timeout > 0 else 5.0
                transport = sse_client(self.url, headers=self.headers, timeout=timeout_seconds)
            elif self.transport_type == "streamable_http":
                timeout_seconds = float(self.timeout) if isinstance(self.timeout, int) and self.timeout > 0 else 30.0
                transport = streamablehttp_client(self.url, headers=self.headers, timeout=timeout_seconds)
            else:
                raise ValueError(f"Unsupported MCP transport type: {self.transport_type}")
            timeout_seconds = (
                int(self.timeout)
                if isinstance(self.timeout, int) and self.timeout > 0
                else None
            )
            async with transport as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    result = await session.call_tool(
                        name=tool_name,
                        arguments=arguments,
                        read_timeout_seconds=(
                            datetime.timedelta(seconds=timeout_seconds)
                            if timeout_seconds is not None
                            else None
                        ),
                    )
                    return _serialize_call_result(result)

        return _run_coro(_inner())


def _extract_tools(tools_result: Any) -> list[dict[str, Any]]:
    """从 SDK `list_tools` 返回对象中提取稳定工具结构。

    标准字段:
    - `name`
    - `description`
    - `inputSchema`
    """
    items: list[dict[str, Any]] = []
    raw_tools = getattr(tools_result, "tools", None)
    if isinstance(raw_tools, list):
        for t in raw_tools:
            name = getattr(t, "name", None)
            if not isinstance(name, str) or not name.strip():
                continue
            items.append(
                {
                    "name": name,
                    "description": getattr(t, "description", "") or "",
                    "inputSchema": (
                        getattr(t, "inputSchema", None)
                        or getattr(t, "input_schema", None)
                        or {"type": "object"}
                    ),
                }
            )
    return items


def _serialize_call_result(result: Any) -> dict[str, Any]:
    """
    把 MCP SDK 的返回对象转成稳定的 JSON 结构。

    优先提取 `content[]` 的文本元素；无法直接序列化时用 `repr` 兜底，
    防止 observation 写历史时失败。
    """
    if result is None:
        return {"content": []}
    content = getattr(result, "content", None)
    if isinstance(content, list):
        out_items: list[dict[str, Any]] = []
        for item in content:
            if isinstance(item, dict):
                out_items.append(item)
                continue
            text = getattr(item, "text", None)
            if isinstance(text, str):
                out_items.append({"type": "text", "text": text})
                continue
            out_items.append({"type": type(item).__name__, "value": repr(item)})
        return {"content": out_items}
    try:
        json.dumps(result)
        return {"content": result}
    except Exception as e:
        from gaoagent.core.runner.console import Console
        Console.debug(f"MCP 结果 JSON 序列化失败，回退 repr：{e}")
        return {"content": [{"type": type(result).__name__, "value": repr(result)}]}


def _import_sdk():
    # 仅使用官方 `mcp` SDK。
    """延迟导入官方 MCP SDK；缺失时抛出可读错误。"""
    try:
        from mcp import ClientSession, StdioServerParameters  # type: ignore
        from mcp.client.stdio import stdio_client  # type: ignore
        from mcp.client.sse import sse_client  # type: ignore
        from mcp.client.streamable_http import streamablehttp_client  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "未安装可用 MCP SDK：请安装 `mcp`"
        ) from e
    return (ClientSession, StdioServerParameters, stdio_client, sse_client, streamablehttp_client)


def build_mcp_tools_cache_payload(
    mcp_servers: dict[str, Any],
    *,
    connect_and_list_tools: Callable[[str, dict[str, Any]], list[dict[str, Any]]],
    generated_at: str,
) -> dict[str, Any]:
    """构建 MCP 工具缓存载荷（供配置期写盘与运行期查表）。

    输入:
    - `mcp_servers`: MCP 服务配置映射。
    - `connect_and_list_tools`: 由调用方提供的“连接并列工具”函数，便于测试与复用。
    - `generated_at`: 缓存生成时间戳字符串。

    输出:
    - `servers`: 每个服务的工具明细或错误信息，便于排障。
    - `exported_map`: 导出工具名到真实 server/tool/schema 的映射，供运行时路由。

    冲突处理:
    - 当导出名冲突时自动追加 `__2/__3/...` 后缀，保证映射键唯一。
    """
    servers: dict[str, Any] = {}
    exported_map: dict[str, Any] = {}

    for server_name, body in (mcp_servers or {}).items():
        if not isinstance(server_name, str) or not isinstance(body, dict):
            continue
        if body.get("disabled") is True:
            continue
        tools: list[dict[str, Any]] = []
        try:
            tools = connect_and_list_tools(server_name, body) or []
        except Exception as e:
            servers[server_name] = {"error": str(e), "tools": []}
            continue

        server_tools: list[dict[str, Any]] = []
        for t in tools:
            if not isinstance(t, dict):
                continue
            name = t.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            exported = export_tool_name(server_name, name)
            if exported in exported_map:
                existing = exported_map.get(exported) or {}
                if existing.get("server") != server_name or existing.get("tool") != name:
                    index = 2
                    while True:
                        suffix = f"__{index}"
                        keep = max(1, 64 - len(suffix))
                        candidate = f"{exported[:keep]}{suffix}"
                        occupied = exported_map.get(candidate)
                        if occupied is None:
                            exported = candidate
                            break
                        if occupied.get("server") == server_name and occupied.get("tool") == name:
                            exported = candidate
                            break
                        index += 1
            item = {
                "name": name,
                "description": t.get("description") or "",
                "inputSchema": t.get("inputSchema") or {"type": "object"},
                "exported_name": exported,
            }
            server_tools.append(item)
            exported_map[exported] = {
                "server": server_name,
                "tool": name,
                "description": item["description"],
                "parameters": item["inputSchema"],
            }

        servers[server_name] = {"tools": server_tools}

    return {
        "version": 1,
        "generated_at": generated_at,
        "servers": servers,
        "exported_map": exported_map,
    }


def write_mcp_tools_cache(payload: dict[str, Any]) -> None:
    """原子写入 MCP 工具缓存文件，避免中断时产生半写入文件。"""
    cfg_dir = _config_dir()
    cfg_dir.mkdir(parents=True, exist_ok=True)
    path = cfg_dir / "gao_client_mcp_tools_cache.json"
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def read_mcp_tools_cache() -> dict[str, Any] | None:
    """读取 MCP 工具缓存；格式不合法时返回 None。"""
    path = _config_dir() / "gao_client_mcp_tools_cache.json"
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        from gaoagent.core.runner.console import Console
        Console.error(f"读取 MCP 工具缓存失败：{path}，{e}")
        return None
    return value if isinstance(value, dict) else None
