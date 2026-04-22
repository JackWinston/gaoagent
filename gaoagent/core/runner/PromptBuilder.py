from __future__ import annotations

import json
from typing import Any
from gaoagent.core.runner.BaseRunner import Mode
from gaoagent.core.runner.Utils import load_mcp, load_rag, load_skills


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
你是一个严格遵循 ReAct 范式的智能代理（ReAct Agent）。

【目标】
逐步解决用户问题：Thought -> Action(工具调用) -> Observation -> ... -> Final。

【输出协议（必须遵守）】
1. 需要调用工具时：
   - 使用 Chat Completions 的 tool_calls 机制发起函数调用。
   - 不要在 content 中伪造“tool_calls 文本”。
2. 不调用工具时：
   - 仅输出单个 JSON 对象，且 type 只能是 thought 或 final。
   - thought 格式：{{"type":"thought","content":"..."}}
   - final 格式：{{"type":"final","content":"..."}} 
3. 禁止输出 Markdown 代码块、额外解释文字。
4. `question` 不是合法输出类型；需要向用户追问时，必须调用 ask_user 工具。
5. 工具返回 observation 后，必须先输出 thought，再决定是否继续调用工具或给 final。
6. 不得编造工具结果；结论必须基于已有上下文或工具返回。
7. 任务完成或用户明确结束时，必须输出 final。



【决策准则】
1. 信息不足且有可用工具：优先调用工具。
2. 无可用工具或工具无法获取更多信息：在 thought 中说明依据与限制，再给 final。
3. 仅在任务完成或用户明确结束时输出 final；final 要直接回答用户问题，简洁完整。

【可用工具】
{tool_line}

【资源配置】
"""

    resources = {
        "skills": injections.get("skills"),
        "rag": injections.get("rag"),
    }
    return base_prompt + json.dumps(resources, ensure_ascii=False)



def _collect_injections() -> dict[str, Any]:
    """收集资源注入配置（MCP/RAG/Skills）"""
    out: dict[str, Any] = {}
    out["mcp"] = load_mcp()
    out["skills"] = load_skills()
    out["rag"] = load_rag()
    return out
