from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from gaoagent.core.runner.HttpClient import HttpResponse
    from gaoagent.core.runner.BaseRunner import RequestBaseInfo, StepResult


def now_ms() -> int:
    return int(time.time() * 1000)


def truncate_text(s: str, limit: int) -> str:
    if len(s) <= limit:
        return s
    return s[: max(0, limit - 1)] + "…"


def safe_json_dumps(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except Exception:
        return repr(value)


def summarize(value: Any, limit: int = 400) -> str:
    if value is None:
        return "null"
    if isinstance(value, (str, int, float, bool)):
        return truncate_text(str(value), limit)
    return truncate_text(safe_json_dumps(value), limit)


def redact(value: Any) -> Any:
    sensitive_keys = {
        "api_key",
        "apikey",
        "key",
        "token",
        "secret",
        "password",
        "authorization",
    }
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            if isinstance(k, str) and k.lower() in sensitive_keys:
                out[k] = "***"
            else:
                out[k] = redact(v)
        return out
    if isinstance(value, list):
        return [redact(x) for x in value]
    return value


def normalize_exception(e: BaseException) -> dict[str, Any]:
    return {
        "type": type(e).__name__,
        "message": str(e),
        "traceback": "".join(traceback.format_exception(type(e), e, e.__traceback__)),
    }


def find_project_root(start: Path | None = None) -> Path:
    p = (start or Path.cwd()).resolve()
    for cur in (p, *p.parents):
        if (cur / "pyproject.toml").exists():
            return cur
    return p


def _config_dir() -> Path:
    """
    返回默认配置目录。

    约定：
    - 若设置环境变量 `GAOAGENT_CONFIG_DIR`，优先使用该目录（便于测试与沙箱运行）。
    - 否则使用用户目录 `~/.gaoagent`。
    """
    override_dir = _env_config_dir()
    if override_dir is not None:
        return override_dir
    return Path.home() / ".gaoagent"


def _env_config_dir() -> Path | None:
    """解析环境变量覆盖目录；未设置时返回 None。"""
    override = os.environ.get("GAOAGENT_CONFIG_DIR")
    if override and override.strip():
        return Path(override).expanduser().resolve()
    return None


def _project_config_dir() -> Path:
    """返回项目级配置目录 `<project_root>/.gaoagent`。"""
    return find_project_root() / ".gaoagent"


def _find_config_file(name: str) -> Path:
    """
    解析配置文件路径，按以下优先级查找：
    1. `GAOAGENT_CONFIG_DIR` 指向的目录（用于测试隔离与临时覆盖）；
    2. 项目级 `.gaoagent/`（便于仓库内固定配置）；
    3. 用户级 `~/.gaoagent/`（全局默认）。

    注意：该函数不保证路径一定存在，调用方需自行检查。
    """
    env_dir = _env_config_dir()
    if env_dir is not None:
        candidate = env_dir / name
        if candidate.exists():
            return candidate
    project_dir = _project_config_dir()
    candidate = project_dir / name
    if candidate.exists():
        return candidate
    return _config_dir() / name


def load_request_base_info() -> RequestBaseInfo | None:
    """加载请求基础信息, 包括 baseurl、api_key、默认 headers 等等"""
    from gaoagent.core.runner.BaseRunner import RequestBaseInfo

    path = _config_dir() / "gao_client_api_config.json"
    if not path.exists():
        return None
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    if not isinstance(config, dict):
        return None

    apis = config.get("apis", {})

    # ===================== 1. 处理 default_api 缺失 =====================
    default_api = config.get("default_api")
    if not default_api or default_api not in apis:
        # 兜底：取第一个 API
        if not apis:
            return None
        default_api = next(iter(apis.keys()))

    api_config = apis[default_api]
    api_key = api_config.get("api_key", "")
    base_url = api_config.get("base_url", "")
    models = api_config.get("models", {})

    # ===================== 2. 处理 default_model 缺失 =====================
    default_model = config.get("default_model")
    if not default_model or default_model not in models:
        # 兜底：取第一个模型
        if not models:
            return None
        default_model = next(iter(models.keys()))

    model_info = models[default_model]
    context_window = model_info.get("context_window", 4096)

    return RequestBaseInfo(
        baseurl=base_url,
        api_key=api_key,
        modules=default_model,
        context_window=context_window,
    )


def load_mcp() -> dict[str, Any]:
    """
    读取 MCP server 配置并输出给 Prompt 注入层使用。

    兼容两种历史格式：
    - 新格式：`{"mcpServers": {...}}`
    - 旧格式：`{"serverA": {...}, "serverB": {...}}`

    返回值仅保留可用服务（disabled != true），并做最小字段裁剪，
    避免把无关字段扩散到上层提示词。
    """
    path = _find_config_file("gao_client_mcp_setting.json")
    if not path.exists():
        return {"available": False, "servers": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"available": False, "servers": []}

    if not isinstance(payload, dict):
        return {"available": False, "servers": []}

    servers_payload: dict[str, Any] = {}
    if isinstance(payload.get("mcpServers"), dict):
        servers_payload = payload.get("mcpServers")
    else:
        servers_payload = payload

    servers: list[dict[str, Any]] = []
    for name, body in servers_payload.items():
        if not isinstance(name, str) or not isinstance(body, dict):
            continue
        if body.get("disabled") is True:
            continue
        servers.append(
            {
                "name": name,
                "timeout": body.get("timeout"),
                "type": body.get("type"),
                "command": body.get("command"),
                "args": body.get("args"),
            }
        )
    servers.sort(key=lambda x: x.get("name") or "")
    return {"available": True, "servers": servers}


def load_mcp_servers_raw() -> dict[str, Any]:
    """
    读取原始 MCP servers 配置（不裁剪字段）。

    用途：
    - ReActRunner 在真正调用 MCP 工具时，需要拿到完整 `command/args/timeout`。
    - 与 `load_mcp()` 不同，本函数面向运行时执行，不面向提示词展示。
    """
    path = _find_config_file("gao_client_mcp_setting.json")
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    if isinstance(payload.get("mcpServers"), dict):
        servers = payload.get("mcpServers")
        return servers if isinstance(servers, dict) else {}
    return payload


def load_mcp_tools_cache() -> dict[str, Any] | None:
    """
    读取 MCP 工具缓存。

    缓存由配置阶段生成，核心结构是：
    - `servers`: 各 server 的工具列表
    - `exported_map`: 导出工具名 -> (server/tool/schema) 映射
    """
    path = _find_config_file("gao_client_mcp_tools_cache.json")
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def write_mcp_tools_cache_for_current_scope(payload: dict[str, Any]) -> None:
    """
    将 MCP 工具缓存写回当前生效配置作用域。

    规则：
    - 若已存在 `gao_client_mcp_setting.json`，缓存写到同目录；
    - 否则写到默认配置目录。
    """
    settings_path = _find_config_file("gao_client_mcp_setting.json")
    cfg_dir = settings_path.parent if settings_path.exists() else _config_dir()
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cfg_dir / "gao_client_mcp_tools_cache.json"
    tmp_path = cache_path.with_name(f"{cache_path.name}.tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    tmp_path.replace(cache_path)


def load_skills() -> dict[str, Any]:
    """加载技能库配置"""
    skills_dir = _config_dir() / "skills"
    if not skills_dir.exists() or not skills_dir.is_dir():
        return {"available": False, "items": []}

    items: list[dict[str, Any]] = []
    for file_path in skills_dir.rglob("*"):
        if not file_path.is_file() or file_path.name.lower() != "skill.md":
            continue
        meta = parse_skill_frontmatter(file_path)
        if meta:
            meta["path"] = str(file_path)
            items.append(meta)

    items.sort(key=lambda x: x.get("name") or "")
    return {"available": True, "items": items}


def parse_skill_frontmatter(file_path: Path) -> dict[str, Any] | None:
    """解析Skill.md的前置元数据"""
    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception:
        return None

    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return None

    # 提取前置元数据
    frontmatter = []
    i = 1
    while i < len(lines):
        if lines[i].strip() == "---":
            break
        frontmatter.append(lines[i])
        i += 1
    else:
        return None

    name = description = None
    j = 0
    while j < len(frontmatter):
        line = frontmatter[j].strip()
        if not line or line.startswith("#"):
            j += 1
            continue

        if line.startswith("name:"):
            name = line.split(":", 1)[1].strip().strip("\"'")
        elif line.startswith("description:"):
            desc_val = line.split(":", 1)[1].strip()
            if desc_val in (">", ">-", "|", "|-"):
                j += 1
                block = []
                while j < len(frontmatter) and not frontmatter[j].strip():
                    block.append(frontmatter[j].lstrip())
                    j += 1
                description = " ".join(" ".join(block).split()).strip()
            else:
                description = desc_val.strip("\"'")
        j += 1

    return {"name": name, "description": description} if name and description else None


def load_rag() -> dict[str, Any]:
    """加载RAG索引配置"""
    rag_dir = _config_dir() / "rag"
    if not rag_dir.exists() or not rag_dir.is_dir():
        return {"available": False, "indexes": []}

    indexes = sorted([p.name for p in rag_dir.iterdir() if p.is_dir()])
    return {"available": True, "indexes": indexes}


def parse_llm_response(response: HttpResponse) -> StepResult:
    """
    将底层 HTTP 客户端返回的 LLM 响应规范化为 StepResult。

    该函数是 Runner 的“协议收口层”，职责是把不同形态的上游响应
    （HTTP 失败、空响应、tool_calls、thought/final 文本协议等）
    统一映射为 StepResult，供主循环直接消费。

    解析约定：
    - 网络/HTTP 失败：返回 decision="final"，content 为可读错误摘要。
    - 响应体非 JSON 对象：返回 decision="retry"，提示上层重试。
    - 协议 type="tool_calls"：返回 decision="function_call"。
    - 协议 type="thought"：返回 decision="thought"。
    - 协议 type="final"：返回 decision="final"。
    - 协议 type="retry"：返回 decision="retry"。
    - 协议 type in {"question","observation"}：向后兼容，映射为 thought。
    - 其他未知 type：返回 decision="retry" 说明协议错误（通常可重试）。

    参数:
    - response: HttpClient 返回的 HttpResponse。

    返回:
    - StepResult: 统一后的单步决策结果。
    """
    from gaoagent.core.runner.HttpClient import HttpResponse
    from gaoagent.core.runner.BaseRunner import StepResult

    if not isinstance(response, HttpResponse):
        return StepResult(
            decision="final",
            content=f"LLM 响应类型错误：{type(response).__name__}",
            raw={"response": repr(response)},
        )

    # HTTP 层失败（非 2xx 或网络异常）直接终止当前步，并返回可诊断信息。
    if not response.ok:
        status_text = f"status={response.status}" if response.status is not None else "status=null"
        reason_text = response.reason or "unknown"
        body = response.text or (safe_json_dumps(response.json) if response.json is not None else "")
        content = f"LLM 请求失败：{status_text}, reason={reason_text}"
        if body:
            # 错误体可能很长，避免污染终端与日志。
            content = f"{content}\n{truncate_text(body, 800)}"

        return StepResult(
            decision="final",
            content=content,
            raw={
                "http": {
                    "ok": response.ok,
                    "status": response.status,
                    "reason": response.reason,
                }
            },
        )

    # 优先使用已解析 JSON；若缺失则尝试从 text 二次反序列化。
    payload: dict[str, Any] | None = response.json if isinstance(response.json, dict) else None
    if payload is None and isinstance(response.text, str) and response.text.strip():
        try:
            parsed = json.loads(response.text)
        except Exception:
            parsed = None
        payload = parsed if isinstance(parsed, dict) else None

    # 上游返回空文本或非对象 JSON，视为可重试的临时协议异常。
    if payload is None:
        text = response.text or ""
        return StepResult(
            decision="retry",
            content=truncate_text(text, 800) if text else "LLM 返回为空或不是 JSON 对象",
            raw={"http": {"ok": True, "status": response.status}, "text": text},
        )

    from gaoagent.core.runner.FunctionCallProtocol import map_chat_completion_to_protocol

    # 统一协议映射：把 ChatCompletions payload 转成内部 protocol dict。
    protocol = map_chat_completion_to_protocol(payload)
    action_type = protocol.get("type") if isinstance(protocol, dict) else None

    if action_type == "tool_calls":
        # 新协议：支持多 tool call；每个调用保留 name/arguments/tool_call_id。
        calls: list[dict[str, Any]] = []
        protocol_calls = protocol.get("calls")
        if isinstance(protocol_calls, list):
            for item in protocol_calls:
                if not isinstance(item, dict):
                    continue
                call: dict[str, Any] = {
                    "name": item.get("name"),
                    "arguments": item.get("arguments", {}),
                }
                if isinstance(item.get("tool_call_id"), str):
                    call["tool_call_id"] = item.get("tool_call_id")
                calls.append(call)

        return StepResult(
            decision="function_call",
            function_call=calls,
            raw={"payload": payload, "protocol": protocol},
        )

    if action_type == "thought":
        # thought 允许 content 或 output，两者都缺失时回退为空字符串。
        content = protocol.get("content")
        if content is None:
            content = protocol.get("output")
        if content is None:
            content = ""
        return StepResult(
            decision="thought",
            content=str(content),
            raw={"payload": payload, "protocol": protocol},
        )

    if action_type == "final":
        # final 直接透传，交由上层作为本轮最终回答处理。
        content = protocol.get("content", "")
        return StepResult(
            decision="final",
            content=str(content),
            raw={"payload": payload, "protocol": protocol},
        )

    if action_type == "retry":
        content = protocol.get("content", "")
        return StepResult(
            decision="retry",
            content=str(content),
            raw={"payload": payload, "protocol": protocol},
        )

    if action_type in ("question", "observation"):
        # 历史兼容：旧类型在当前执行器中按 thought 处理，不直接中断流程。
        content = protocol.get("content", "")
        return StepResult(
            decision="thought",
            content=str(content),
            raw={"payload": payload, "protocol": protocol},
        )

    # 未识别协议类型：按可重试错误处理，避免直接终止任务。
    return StepResult(
        decision="retry",
        content=f"LLM 协议错误：未知 type={repr(action_type)}",
        raw={"payload": payload, "protocol": protocol},
    )
