from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_messages(ctx: Any, tool_names: list[str], mode: str | None = None) -> list[dict[str, Any]]:
    """
    构建LLM请求上下文（标准格式：system + 历史对话 + 当前用户消息）
    完整支持ReAct多轮推理、上下文记忆、工具调用
    """

    # 统一模式处理
    m = (mode or getattr(ctx, "mode", "react") or "react").strip().lower()

    memory = getattr(ctx, "memory", None)
    if not isinstance(memory, dict):
        memory = {}

    messages_key = "messages" if m == "react" else f"messages_{m}"
    existing = memory.get(messages_key)
    if isinstance(existing, list) and existing and all(isinstance(x, dict) for x in existing):
        return existing

    injections = collect_injections(memory)

    # 生成对应模式的系统提示词
    if m == "plan":
        system_text = build_plan_system_text(tool_names=tool_names, injections=injections)
    elif m == "retry":
        system_text = build_retry_system_text(tool_names=tool_names, injections=injections)
    else:
        system_text = build_react_system_text(tool_names=tool_names, injections=injections)

    question = str(getattr(ctx, "question", "") or "")
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_text}]
    if question.strip():
        messages.append({"role": "user", "content": json.dumps({"type": "question", "content": question}, ensure_ascii=False)})

    memory[messages_key] = messages
    try:
        setattr(ctx, "memory", memory)
    except Exception:
        pass
    return messages



def build_react_system_text(*, tool_names: list[str], injections: dict[str, Any]) -> str:
    
    base_prompt = """
你是一个严格遵循 ReAct 范式的智能代理（ReAct Agent）。
    
【最高强制规则】
1. 仅输出 单个合法JSON对象，禁止输出任何文字、注释、代码块
2. 禁止返回不完整JSON/多个JSON，禁止使用未定义的type/工具
3. 输入输出固定格式：{"type":"question/thought/function_call/observation/final","content":"...","name":"","parameters":{}} ,其中 name 和 parameters 是可选的.
4. 输出格式中, content 字段表示思考的内容或者最终答案
5. 收到的 question 或者 observation 消息后,必须先返回 thought 类型的消息,包含思考的内容

【type 严格定义】
1. question：用户问题的提问
2. thought：你的思考过程
3. function_call：调用工具，必须携带 name + parameters
4. observation：调用工具或者环境返回的结果
5. final：任务完成，返回最终答案，结束对话

你需要解决用户的问题。为此，你需要将问题分解为多个步骤。
首先使用 thought 思考如何解决这个问题，然后使用 function_call 调用一个工具 。接着，你会收到 工具或者环境返回的结果 observation 。
持续这个思考和过程,直到你解决了用户的问题,返回 final 类型的消息,包含最终答案。

例子 1:
{"type":"question","content":"埃菲尔铁塔有多高？"}
{"type":"thought","content":"我需要找到埃菲尔铁塔的高度。可以使用搜索工具。"}
{"type":"function_call","name":"get_height","parameters":{"building_name":"埃菲尔铁塔"}}
{"type":"observation","content":"埃菲尔铁塔的高度约为330米（包含天线）。"}
{"type":"thought","content":"搜索结果显示了高度。我已经得到答案了。"}
{"type":"final_answer","content":"埃菲尔铁塔的高度约为330米"}

例子 2:

{"type":"question","content":"帮我找一个简单的番茄炒蛋食谱，并看看家里的冰箱里有没有西红柿。"}
{"type":"thought","content":"这个任务分两步。第一步，找到番茄炒蛋的食谱。第二步，检查冰箱里是否有西红柿。我先用 find_recipe 工具找食谱。"}
{"type":"function_call","name":"find_recipe","parameters":{"dish":"番茄炒蛋"}}
{"type":"observation","content":"简单的番茄炒蛋食谱：将2个鸡蛋打散，2个番茄切块。热油，先炒鸡蛋，盛出。再热油，炒番茄至软烂，加入鸡蛋，放盐调味即可。"}
{"type":"thought","content":"好的，我已经有食谱了。食谱需要西红柿。现在我需要用 check_fridge 工具看看冰箱里有没有西红柿。"}
{"type":"function_call","name":"check_fridge","parameters":{"item":"西红柿"}}
{"type":"observation","content":"冰箱检查结果：有3个西红柿。"}
{"type":"thought","content":"我找到了食谱，并且确认了冰箱里有西红柿。可以回答问题了。"}
{"type":"final_answer","content":"简单的番茄炒蛋食谱是：鸡蛋打散，番茄切块。先炒鸡蛋，再炒番茄，混合后加盐调味。冰箱里有3个西红柿"}




【可用工具】
ask_user：向用户提问 | list_dir：列出目录 | read_file：读取文件 | write_file：写入文件

【资源配置】
"""

    # 关键修复：内嵌JSON标准化，无语法错误
    resources = {
        "tools": tool_names,
        "mcp": injections.get("mcp"),
        "skills": injections.get("skills"),
        "rag": injections.get("rag")
    }
    return base_prompt + json.dumps(resources, ensure_ascii=False)


def build_plan_system_text(*, tool_names: list[str], injections: dict[str, Any]) -> str:
    """任务规划器系统提示词（标准化）"""
    base_prompt = """你是一个任务规划器。
【强制规则】
1. 仅输出JSON数组，禁止任何额外内容
2. 输出格式三选一：
   - 调用工具：{"type":"tool","name":"工具名","arguments":{}}
   - 最终答案：{"type":"final","content":"答案"}
【资源配置】"""

    resources = {
        "tools": tool_names,
        "mcp": injections.get("mcp"),
        "skills": injections.get("skills"),
        "rag": injections.get("rag")
    }
    return base_prompt + json.dumps(resources, ensure_ascii=False)


def build_retry_system_text(*, tool_names: list[str], injections: dict[str, Any]) -> str:
    """重试反思器系统提示词（标准化）"""
    base_prompt = """你是一个重试反思器。
【强制规则】
1. 仅输出JSON对象，禁止任何额外内容
2. 必填字段：strategy（重试策略）
3. 可选字段：memory_patch（记忆补丁）、note（备注）
【资源配置】"""

    resources = {
        "tools": tool_names,
        "mcp": injections.get("mcp"),
        "skills": injections.get("skills"),
        "rag": injections.get("rag")
    }
    return base_prompt + json.dumps(resources, ensure_ascii=False)


def collect_injections(memory: dict[str, Any]) -> dict[str, Any]:
    """收集资源注入配置（MCP/RAG/Skills）"""
    out: dict[str, Any] = {}
    out["mcp"] = memory.get("mcp") or _load_mcp()
    out["skills"] = memory.get("skills") or _load_skills()
    out["rag"] = memory.get("rag") or _load_rag()
    return out


def _config_dir() -> Path:
    """配置文件根目录"""
    return Path.home() / ".gaoagent"


def _load_mcp() -> dict[str, Any]:
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
        servers.append({
            "name": name,
            "timeout": body.get("timeout"),
            "type": body.get("type"),
            "command": body.get("command"),
            "args": body.get("args"),
        })
    servers.sort(key=lambda x: x.get("name") or "")
    return {"available": True, "servers": servers}


def _load_skills() -> dict[str, Any]:
    """加载技能库配置"""
    skills_dir = _config_dir() / "skills"
    if not skills_dir.exists() or not skills_dir.is_dir():
        return {"available": False, "items": []}

    items: list[dict[str, Any]] = []
    for file_path in skills_dir.rglob("*"):
        if not file_path.is_file() or file_path.name.lower() != "skill.md":
            continue
        meta = _parse_skill_frontmatter(file_path)
        if meta:
            meta["path"] = str(file_path)
            items.append(meta)

    items.sort(key=lambda x: x.get("name") or "")
    return {"available": True, "items": items}


def _parse_skill_frontmatter(file_path: Path) -> dict[str, Any] | None:
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
            name = line.split(":", 1)[1].strip().strip('"\'')
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
                description = desc_val.strip('"\'')
        j += 1

    return {"name": name, "description": description} if name and description else None


def _load_rag() -> dict[str, Any]:
    """加载RAG索引配置"""
    rag_dir = _config_dir() / "rag"
    if not rag_dir.exists() or not rag_dir.is_dir():
        return {"available": False, "indexes": []}

    indexes = sorted([p.name for p in rag_dir.iterdir() if p.is_dir()])
    return {"available": True, "indexes": indexes}
