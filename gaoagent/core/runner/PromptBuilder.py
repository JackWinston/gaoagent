from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gaoagent.core.runner.Utils import safe_json_dumps


def build_messages(ctx: Any, *, tool_names: list[str], mode: str | None = None) -> list[dict[str, Any]]:
    memory = getattr(ctx, "memory", {}) or {}
    injections = collect_injections(memory)
    m = (mode or getattr(ctx, "mode", "react") or "react").strip().lower()
    if m == "plan":
        system_text = build_plan_system_text(tool_names=tool_names, injections=injections)
    elif m == "retry":
        system_text = build_retry_system_text(tool_names=tool_names, injections=injections)
    else:
        system_text = build_react_system_text(tool_names=tool_names, injections=injections)
    user_payload = {
        "question": getattr(ctx, "question", ""),
        "mode": m,
        "step": getattr(ctx, "step", 0),
        "last_observation": getattr(ctx, "last_observation", None),
        "last_error": getattr(ctx, "last_error", None),
        "injections": injections,
    }
    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": safe_json_dumps(user_payload)},
    ]


def build_react_system_text(*, tool_names: list[str], injections: dict[str, Any]) -> str:
    base = (
        "你是一个 ReAct Agent。你必须遵循 function-call 响应协议："
        "仅返回 JSON 对象，type 只能是 function_call/final/internal。"
    )
    resources = {
        "tools": tool_names,
        "mcp": injections.get("mcp"),
        "skills": injections.get("skills"),
        "rag": injections.get("rag"),
    }
    return base + "\n" + safe_json_dumps({"resources": resources})


def build_plan_system_text(*, tool_names: list[str], injections: dict[str, Any]) -> str:
    base = (
        "你是一个任务规划器。你的目标是把用户问题拆解为可执行的计划步骤。\n"
        "你必须只输出 JSON（不要输出任何解释文字或 Markdown）。\n"
        "输出格式必须是 JSON 数组，每个元素是对象，且必须满足以下之一：\n"
        '1) {"type":"tool","name":<tool_name>,"arguments":<object>}  调用一个已注册工具\n'
        '2) {"type":"final","content":<string>}  直接给出最终答案并结束\n'
        "注意：tool_name 必须来自 resources.tools；arguments 必须是对象（JSON object）。\n"
        "如果不需要任何工具，请直接输出一个 final。"
    )
    resources = {
        "tools": tool_names,
        "mcp": injections.get("mcp"),
        "skills": injections.get("skills"),
        "rag": injections.get("rag"),
    }
    return base + "\n" + safe_json_dumps({"resources": resources})


def build_retry_system_text(*, tool_names: list[str], injections: dict[str, Any]) -> str:
    base = (
        "你是一个 Retry 反思器（reflector）。你的目标是在一次尝试失败后，给出下一次重试的策略与可选的 memory_patch。\n"
        "你必须只输出 JSON（不要输出任何解释文字或 Markdown）。\n"
        "输出必须是 JSON 对象，建议字段：\n"
        '- strategy: string，重试策略（必填）\n'
        "- memory_patch: object，可选。将被合并到共享 memory，用于影响下一次尝试（例如 api_name/model）。\n"
        "- note: string，可选。简短说明（不影响程序执行）。\n"
        "注意：memory_patch 必须是对象（JSON object），不要包含敏感信息。"
    )
    resources = {
        "tools": tool_names,
        "mcp": injections.get("mcp"),
        "skills": injections.get("skills"),
        "rag": injections.get("rag"),
    }
    return base + "\n" + safe_json_dumps({"resources": resources})


def collect_injections(memory: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    out["mcp"] = memory.get("mcp") or _load_mcp()
    out["skills"] = memory.get("skills") or _load_skills()
    out["rag"] = memory.get("rag") or _load_rag()
    return out


def _config_dir() -> Path:
    return Path.home() / ".gaoagent"


def _load_mcp() -> dict[str, Any]:
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


def _load_skills() -> dict[str, Any]:
    skills_dir = _config_dir() / "skills"
    if not skills_dir.exists() or not skills_dir.is_dir():
        return {"available": False, "items": []}

    items: list[dict[str, Any]] = []
    for file_path in skills_dir.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.name.lower() != "skill.md":
            continue
        meta = _parse_skill_frontmatter(file_path)
        if meta is None:
            continue
        meta["path"] = str(file_path)
        items.append(meta)

    items.sort(key=lambda x: x.get("name") or "")
    return {"available": True, "items": items}


def _parse_skill_frontmatter(file_path: Path) -> dict[str, Any] | None:
    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception:
        return None

    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return None

    i = 1
    frontmatter: list[str] = []
    while i < len(lines):
        if lines[i].strip() == "---":
            break
        frontmatter.append(lines[i])
        i += 1
    if i >= len(lines) or lines[i].strip() != "---":
        return None

    name: str | None = None
    description: str | None = None
    j = 0
    while j < len(frontmatter):
        line = frontmatter[j].rstrip("\n")
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            j += 1
            continue

        if stripped.startswith("name:"):
            name_val = stripped.split(":", 1)[1].strip()
            name = name_val.strip('"').strip("'")
            j += 1
            continue

        if stripped.startswith("description:"):
            desc_val = stripped.split(":", 1)[1].strip()
            if desc_val in (">", ">-", "|", "|-"):
                j += 1
                block: list[str] = []
                while j < len(frontmatter):
                    nxt = frontmatter[j]
                    if nxt.strip() and not nxt.startswith((" ", "\t")):
                        break
                    block.append(nxt.lstrip())
                    j += 1
                description = " ".join(" ".join(block).split()).strip()
                continue

            description = desc_val.strip('"').strip("'")
            j += 1
            continue

        j += 1

    if not name or not description:
        return None
    return {"name": name, "description": description}


def _load_rag() -> dict[str, Any]:
    rag_dir = _config_dir() / "rag"
    if not rag_dir.exists() or not rag_dir.is_dir():
        return {"available": False, "indexes": []}

    indexes: list[str] = []
    for p in rag_dir.iterdir():
        if p.is_dir():
            indexes.append(p.name)
    indexes.sort()
    return {"available": True, "indexes": indexes}
