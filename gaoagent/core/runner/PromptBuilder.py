from __future__ import annotations

import json
import os
import platform
from typing import Any
from gaoagent.core.runner.BaseRunner import Mode
from gaoagent.core.runner.Utils import load_mcp, load_rag, load_skills, project_root_dir


def build_system_prompt(tool_names: list[str], mode: Mode) -> str:
    """
    构建LLM请求上下文（标准格式：system + 历史对话 + 当前用户消息）
    完整支持ReAct多轮推理、上下文记忆、工具调用
    """

    injections = _collect_injections()

    # 生成对应模式的系统提示词
    if mode == "react":
        return build_react_system_text(tool_names=tool_names, injections=injections)
    elif mode == "plan":
        return build_react_system_text(tool_names=tool_names, injections=injections)
    elif mode == "retry":
        return build_react_system_text(tool_names=tool_names, injections=injections)


def build_react_system_text(
    *, tool_names: list[str] | None, injections: dict[str, Any]
) -> str:
    available_tools = tool_names or []
    tool_line = " | ".join(available_tools) if available_tools else "无可用工具"

    base_prompt = f"""

你是一个严格遵循 ReAct 范式的智能代理（ReAct Agent），**优先使用自身知识库解答问题，仅在必要时调用工具**，杜绝无意义、重复的工具调用，避免Token浪费。

【目标】
逐步解决用户问题：thought -> tool_calls -> tool_calls result -> thought ... -> final。

【输出协议（必须严格遵守）】
1. 需要调用工具时：
   - 使用 Chat Completions 的 tool_calls 机制发起函数调用。
2. 不调用工具时：
   - 仅输出单个 JSON 对象，且 type 只能是 thought 或 final。
   - thought 格式Json：{{"type":"thought","content":"你要输出的内容"}}
   - final 格式Json：{{"type":"final","content":"你要输出的内容"}}
3. 禁止输出 Markdown 代码块、额外解释文字、多余符号。
4. 收到tool_calls result后，必须先输出 thought格式Json，再决定是否继续调用工具或输出 final格式Json。**禁止连续调用工具**。
5. 不得编造工具结果；结论必须基于已有上下文或工具返回。
6. 任务完成或用户明确结束时，必须输出 final。

【核心决策准则（优先级从高到低）】
1. **优先自有知识作答**：若问题属于常识、通用知识、主观问答、简单推理，自身知识库可完整解答→直接输出 final，不调用任何工具。
2. **仅必要时调用工具**：仅当问题满足以下任一条件，才调用工具：
   - 问题涉及**实时数据、最新资讯、精准数值**；
   - 问题属于**专业领域未知知识、特定领域检索需求**；
   - 现有信息**完全不足以支撑有效回答**。
3. **工具调用限制**：单次对话最多调用1次工具，禁止重复、无意义搜索。
4. 无可用工具或工具无法获取有效信息：在 thought 中说明依据与限制，再输出 final。
5. 仅在任务完成时输出 final；final 需直接回答用户问题，简洁完整。

【Skill 使用规则（按需加载）】
当且仅当用户任务与某个 Skill 高度相关时，才按需读取对应的 SKILL.md 正文。
以下是 Skill 索引：{injections.get("skills")}。


【可用工具】
{tool_line}

【资源配置】
"""

    resources = {
        "system_info": injections.get("system_info"),
        "project_root": injections.get("project_root"),
        "rag": injections.get("rag"),
    }
    return base_prompt + json.dumps(resources, ensure_ascii=False)


def _build_system_info() -> dict[str, Any]:
    return {
        "os": os.name,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cwd": str(os.getcwd()),
    }


def _resolve_project_root_for_prompt() -> str:
    try:
        return str(project_root_dir())
    except Exception:
        return str(os.getcwd())


def _build_skill_index(raw_skills: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(raw_skills, dict):
        return {"available": False, "items": []}

    items = raw_skills.get("items")
    if not isinstance(items, list):
        return {"available": bool(raw_skills.get("available")), "items": []}

    indexed_items: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        indexed_items.append(
            {
                "name": item.get("name"),
                "description": item.get("description"),
                "path": item.get("path"),
            }
        )

    return {"available": bool(raw_skills.get("available")), "items": indexed_items}



def _collect_injections() -> dict[str, Any]:
    """收集资源注入配置（MCP/RAG/Skills）"""
    out: dict[str, Any] = {}
    out["mcp"] = load_mcp()
    out["skills"] = _build_skill_index(load_skills())
    out["rag"] = load_rag()
    out["system_info"] = _build_system_info()
    out["project_root"] = _resolve_project_root_for_prompt()
    return out
