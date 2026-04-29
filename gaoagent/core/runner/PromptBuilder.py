from __future__ import annotations

import time
import os
import platform
from gaoagent.core.runner.BaseRunner import Mode
from gaoagent.core.runner.Utils import load_skills, try_project_root_dir


def build_system_prompt(tool_names: list[str], mode: Mode) -> str:
    """
    构建LLM请求上下文（标准格式：system + 历史对话 + 当前用户消息）
    完整支持ReAct多轮推理、上下文记忆、工具调用
    """

    # 生成对应模式的系统提示词
    if mode == "react":
        return build_react_system_text(tool_names=tool_names)
    elif mode == "plan":
        return build_plan_system_text()
    elif mode == "retry":
        return build_react_system_text(tool_names=tool_names)


def build_plan_system_text() -> str:
    """
    构建 Plan 模式（任务规划与评估）的系统提示词。
    该模式下的模型扮演项目经理或调度者，只负责拆解步骤和评估结果，不直接执行工具。
    """
    base_prompt = f"""
你是一个高级任务规划与评估专家，负责管理和调度复杂的代码任务。
你的主要职责是：
1. 分析用户的复杂请求，将其拆解为多个明确、可顺序执行的子任务。
2. 评估子任务执行后的结果，判断总任务是否已经完成，或者是否需要根据新情况调整后续计划。

【核心原则】
- 拆解任务时：保持颗粒度适中。如果任务太大，请拆分成 3-5 个合理步骤；如果任务简单，1-2 步即可。
- 评估任务时：不要主观臆断。请严格根据历史子任务的“执行结果”判断状态。
- **你只能输出合法的 JSON 字符串**，绝对不要包含任何 Markdown 代码块（如 ```json ... ```），也不要包含任何开场白或解释性文字。

【系统信息】
OS : {os.name}
Platform : {platform.platform()}
Python Version : {platform.python_version()}
当前时间 : {time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime())}

【当前项目目录】
{ try_project_root_dir() or os.getcwd()}
"""
    return base_prompt

def build_react_system_text(*, tool_names: list[str] | None) -> str:
    """build_react_system_text 函数。

    用途:
    - 构建 ReAct 模式的系统提示词。

    参数:
    - tool_names: 工具名称列表.

    返回:
    - str: 系统提示词.
    """
    available_tools = tool_names or []

    from gaoagent.core.runner.Utils import load_rag

    rag_info = load_rag()
    rag_section : str = ""
    if rag_info is None:
        rag_section =  ""
    else :
        kb_list = rag_info.get("indexes", [])
        kb_str = ", ".join(kb_list)
        rag_section = (
        f"""
【RAG 检索与引用规则】
当前可用 RAG 知识库：{kb_str}
如果问题需要查询特定领域知识，请使用 `rag_search` 工具并指定 `kb_name`。
在 final 结论中，若基于 rag_search 的结果回答，请在相关内容后**附带来源引用**（如：`[来源: source_file]`），提高回答可信度。
"""
        if kb_list
        else ""
    )

    a2a_str = _getA2AStr()
    a2a_section : str = ""
    if a2a_str is None:
        a2a_section = ""
    else :
        a2a_section = (
        f"""
【A2A 智能体协作规则】
当前项目已连接以下远程 A2A 智能体。如果任务超出你的能力边界或需要协作，请通过 `a2a_call` 工具委派任务给对应的智能体：
{a2a_str}
"""
        if a2a_str
        else ""
    )

    skill_str = _getSkillStr()
    skill_section : str = ""
    if skill_str is None:
        skill_section = ""
    else :
        skill_section = (
        f"""
【Skill 使用规则】
当且仅当用户任务与某个 Skill 高度相关时，才按需读取对应的 SKILL.md 正文。
以下是 Skill 索引：
{skill_str}
"""
        if skill_str
        else ""
    )

    tools_section = (
        f"""
【可用工具】
{" | ".join(available_tools)}
"""
        if available_tools
        else ""
    )

    base_prompt = f"""

你是一个擅长解决代码问题的智能代理，**优先使用自身知识库解答问题，仅在必要时调用工具**。

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

{rag_section}

{a2a_section}

{skill_section}

{tools_section}

【系统信息】
OS : {os.name}
Platform : {platform.platform()}
Python Version : {platform.python_version()}
当前时间 : {time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime())}

【当前项目目录】
{ try_project_root_dir() or os.getcwd()}

"""
    return base_prompt


def _getSkillStr() -> str | None:
    """_getSkillStr 函数。

    用途:
    - 构建 Skill 索引字符串，用于在系统提示词中引用。

    返回:
    - str: Skill 索引字符串.
    """
    raw_skills = load_skills()
    if raw_skills is None:
        return None
    if not isinstance(raw_skills, dict):
        return None
    items = raw_skills.get("items")
    if not isinstance(items, list):
        return None
    result: list[str] = []
    index = 1
    for item in items:
        if not isinstance(item, dict):
            continue
        result.append(
            f"""
{index}. {item.get("name")}
  name : {item.get("name")}
  description : {item.get("description")}
  path : {item.get("path")}

     """
        )
        index += 1

    return "".join(result)

def _getA2AStr() -> str | None:
    """
    构建 A2A 远程 Agent 列表字符串，用于在系统提示词中引用。
    """
    from gaoagent.core.runner.Utils import load_a2a_agents
    agents = load_a2a_agents()
    if not agents:
        return None
    
    result: list[str] = []
    for idx, (name, info) in enumerate(sorted(agents.items()), 1):
        url = info.get("url", "未知地址")
        result.append(f"{idx}. {name} [地址: {url}]\n")
    return "".join(result)
