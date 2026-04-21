from __future__ import annotations

import json
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
    """配置文件根目录"""
    return Path.home() / ".gaoagent"


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
    """加载MCP服务配置"""
    path = _config_dir() / "gao_client_mcp_setting.json"
    if not path.exists():
        return {"available": False, "servers": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"available": False, "servers": []}

    if not isinstance(payload, dict):
        return {"available": False, "servers": []}

    servers: list[dict[str, Any]] = []
    for name, body in payload.items():
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
    """解析LLM响应，提取决策内容"""
    from gaoagent.core.runner.HttpClient import HttpResponse
    from gaoagent.core.runner.BaseRunner import StepResult

    if not isinstance(response, HttpResponse):
        return StepResult(
            decision="final",
            content=f"LLM 响应类型错误：{type(response).__name__}",
            raw={"response": repr(response)},
        )

    if not response.ok:
        status_text = f"status={response.status}" if response.status is not None else "status=null"
        reason_text = response.reason or "unknown"
        body = response.text or (safe_json_dumps(response.json) if response.json is not None else "")
        content = f"LLM 请求失败：{status_text}, reason={reason_text}"
        if body:
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

    payload: dict[str, Any] | None = response.json if isinstance(response.json, dict) else None
    if payload is None and isinstance(response.text, str) and response.text.strip():
        try:
            parsed = json.loads(response.text)
        except Exception:
            parsed = None
        payload = parsed if isinstance(parsed, dict) else None

    if payload is None:
        text = response.text or ""
        return StepResult(
            decision="final",
            content=truncate_text(text, 800) if text else "LLM 返回为空或不是 JSON 对象",
            raw={"http": {"ok": True, "status": response.status}, "text": text},
        )

    from gaoagent.core.runner.FunctionCallProtocol import map_chat_completion_to_protocol

    protocol = map_chat_completion_to_protocol(payload)
    action_type = protocol.get("type") if isinstance(protocol, dict) else None

    if action_type == "tool_calls":
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

        # 兼容旧格式（单 tool call）
        if not calls:
            fallback_call: dict[str, Any] = {
                "name": protocol.get("name"),
                "arguments": protocol.get("arguments", {}),
            }
            if isinstance(protocol.get("tool_call_id"), str):
                fallback_call["tool_call_id"] = protocol.get("tool_call_id")
            calls.append(fallback_call)

        return StepResult(
            decision="function_call",
            function_call=calls,
            raw={"payload": payload, "protocol": protocol},
        )

    if action_type == "thought":
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
        content = protocol.get("content", "")
        return StepResult(
            decision="final",
            content=str(content),
            raw={"payload": payload, "protocol": protocol},
        )

    if action_type in ("question", "observation"):
        content = protocol.get("content", "")
        return StepResult(
            decision="thought",
            content=str(content),
            raw={"payload": payload, "protocol": protocol},
        )

    return StepResult(
        decision="final",
        content=f"LLM 协议错误：未知 type={repr(action_type)}",
        raw={"payload": payload, "protocol": protocol},
    )
