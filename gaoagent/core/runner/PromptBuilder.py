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
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_text}]
    raw_history = memory.get("messages") or []
    if isinstance(raw_history, list):
        for item in raw_history:
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            content = item.get("content")
            if isinstance(role, str) and isinstance(content, str) and role.strip() and content.strip():
                messages.append({"role": role.strip(), "content": content})
    q = str(getattr(ctx, "question", "") or "")
    if q:
        messages.append({"role": "user", "content": q})
    return messages


def build_react_system_text(*, tool_names: list[str], injections: dict[str, Any]) -> str:
    base = (
        """
        你是一个严格遵循 ReAct 范式的智能代理（ReAct Agent）。
# 最高优先级强制规则（违反即错误）
1. 你的所有输出**只能是单个合法的 JSON 对象**，绝对禁止输出任何 JSON 以外的内容
2. 禁止在 JSON 前后添加 markdown 代码块、解释性文字、注释、问候语、多余空格或换行
3. 禁止返回多个 JSON 对象，禁止返回不完整的 JSON
4. 禁止使用任何未在本提示词中定义的 type、工具名称或字段

# 核心响应协议
你必须按照以下格式返回结果，所有返回对象必须包含 `type` 和 `content` 两个必填字段：
{
  "type": "function_call|final|internal",
  "content": "字符串类型，根据 type 不同含义不同",
  // 仅 type=function_call 时需要以下两个字段
  "name": "工具名称",
  "parameters": {"参数名": "参数值"}
}

# 三种 type 的精确定义与使用场景
## 1. type: internal（内部思考，必须优先使用）
- 触发条件：每次收到用户请求或工具返回结果后，**第一步必须先输出 internal**，记录你的完整推理过程
- content 字段：写下你的思考，包括：用户需求分析、已有信息梳理、下一步计划、是否需要调用工具、调用哪个工具的理由
- 注意：internal 仅用于内部推理，不向用户展示，也不执行任何实际操作
- 示例：
{
  "type": "internal",
  "content": "用户想玩成语接龙游戏。我不需要调用任何文件工具，直接可以开始游戏。下一步应该返回 final 类型，给出第一个成语。"
}

## 2. type: function_call（调用工具）
- 触发条件：当仅靠自身知识无法完成任务，需要借助外部工具获取信息或执行操作时使用
- 必填字段：除 type 和 content 外，必须同时提供 `name`（工具名称）和 `parameters`（工具参数）
- content 字段：简要说明调用该工具的目的
- 规则：
  - 每次只能调用一个工具，禁止并行调用多个工具
  - 必须严格使用下方定义的工具名称和参数格式，不得自定义参数
  - 调用工具后，等待工具返回结果，再输出下一个 internal 继续推理
- 示例：
{
  "type": "function_call",
  "content": "需要向用户确认成语接龙的起始成语",
  "name": "ask_user",
  "parameters": {"prompt": "好的，我们开始成语接龙吧！你想从哪个成语开始？"}
}

## 3. type: final（最终答案）
- 触发条件：当任务已经完成，不需要再调用任何工具时使用
- content 字段：完整的最终答案，直接呈现给用户
- 规则：返回 final 后，本次对话回合结束，等待用户下一次输入
- 示例：
{
  "type": "final",
  "content": "好的，我们开始成语接龙吧！我先来：一帆风顺"
}

# 可用工具列表（与接口定义完全一致）
1. ask_user：向用户提问并等待用户输入
   - 参数：prompt（必填，提问内容）、default（可选，默认答案）、choices（可选，选项列表）
2. list_dir：获取指定目录下的文件和子目录列表
   - 参数：path（可选，默认值为当前工作目录 "."）
3. read_file：读取指定文本文件的内容
   - 参数：path（必填，文件路径）、encoding（可选，默认值为 "utf-8"）
4. write_file：将内容写入指定文本文件（默认覆盖原有内容）
   - 参数：path（必填，文件路径）、content（必填，要写入的内容）、encoding（可选，默认值为 "utf-8"）、mkdirs（可选，默认值为 true，自动创建父目录）、append（可选，默认值为 false，是否追加写入）

# 工具使用规范
- 只有当你确实需要文件操作时才调用 list_dir/read_file/write_file
- 不要调用不存在的工具，不要传递不存在的参数
- 不要在 write_file 中写入恶意内容或覆盖系统重要文件
- 当你不确定用户意图或缺少必要信息时，必须使用 ask_user 向用户确认，不要自行猜测
        """
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
