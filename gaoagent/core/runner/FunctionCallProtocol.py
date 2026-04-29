from __future__ import annotations

import json
from typing import Any

from gaoagent.core.runner.Utils import summarize, truncate_text


def build_function_specs(
    tool_names: list[str],
    *,
    mcp_exported_map: dict[str, Any] | None = None,
    tool_registry: Any | None = None,
) -> list[dict[str, Any]]:
    """build_function_specs 函数。
    
    用途:
    - 给模型提供函数调用的规格, 用于在运行时动态调用外部函数.
    
    参数:
    - tool_names: 本地定义的函数名称列表
    - mcp_exported_map: MCP函数映射表, 用于在运行时调用外部函数.
    - tool_registry: 可选的 ToolRegistry 实例，用于动态获取工具元数据.
    
    返回:
    - list[dict[str, Any]]: 返回模型需要的函数调用规格列表.
    """
    # 优先从 tool_registry 获取元数据，否则使用硬编码的 known 字典作为回退
    specs: list[dict[str, Any]] = []
    for name in tool_names:
        desc = None
        params = None
        
        # 1. 尝试从 tool_registry 获取元数据
        if tool_registry is not None:
            try:
                spec = tool_registry.get_spec(name)
                if spec is not None:
                    desc = spec.description
                    params = spec.parameters
            except Exception:
                pass
        
        # 2. 如果 tool_registry 没有提供，尝试从 mcp_exported_map 获取
        if desc is None or params is None:
            mcp_meta = (
                (mcp_exported_map or {}).get(name)
                if isinstance(mcp_exported_map, dict)
                else None
            )
            if isinstance(mcp_meta, dict):
                if desc is None:
                    desc = mcp_meta.get("description")
                if params is None:
                    params = mcp_meta.get("parameters")
        
        # 3. 最终回退：使用默认描述和空参数
        if desc is None:
            desc = f"Tool `{name}` registered in ToolRegistry"
        if params is None:
            params = {"type": "object", "properties": {}, "additionalProperties": True}
        
        specs.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": desc,
                    "parameters": params,
                },
            }
        )
    return specs


def map_chat_completion_to_protocol(payload: dict[str, Any]) -> dict[str, Any]:
    """
    将 Chat Completions 响应映射为内部协议对象。

    该函数负责“容错解析 + 协议归一化”，目标是尽量从模型返回中提取
    可执行动作（tool_calls）或可消费文本（thought/final），并在异常形态下
    返回 retry 信号，提示上层执行自动重试。

    处理顺序：
    - 校验顶层结构：choices -> message。
    - 优先解析 message.tool_calls（支持多调用）。
    - 再解析 message.content：
      - 字符串：尝试 JSON 对象、连续 JSON 对象序列、最后回退纯文本 final。
      - 列表：提取 text 字段并拼接为 final。
    - 若无 tool_calls 且无可用 content：返回 retry 兜底提示。

    参数:
    - payload: OpenAI 兼容接口返回的 JSON 对象。

    返回:
    - dict[str, Any]: 内部协议对象，常见 type 为 tool_calls / final / thought / retry。
    """

    def strip_code_fence(text: str) -> str:
        # 兼容模型把 JSON 包在 ```json ... ``` 中的输出形态。
        """strip_code_fence 函数。
        
        用途:
        - 移除模型输出中的代码围栏, 仅保留 JSON 内容.
        
        参数:
        - text: 模型输出的文本, 可能包含代码围栏.
        
        返回:
        - str: 返回移除代码围栏后的 JSON 内容.
        """
        s = (text or "").strip()
        if not s.startswith("```"):
            return s
        lines = s.splitlines()
        if len(lines) < 2:
            return s
        if not lines[0].startswith("```"):
            return s
        if lines[-1].strip() != "```":
            return s
        i = 1
        while i < len(lines) and not lines[i].strip():
            i += 1
        return "\n".join(lines[i:-1]).strip()

    def normalize_protocol_dict(data: dict[str, Any]) -> dict[str, Any]:
        # 对历史/别名字段做归一化，降低上层协议分支复杂度。
        """normalize_protocol_dict 函数。
        
        用途:
        - 对历史/别名字段做归一化， 用于在上层协议分支中统一处理.
        
        参数:
        - data: 输入的协议字典, 包含 type 字段.
        
        返回:
        - dict[str, Any]: 返回归一化后的协议字典.
        """
        out = dict(data)
        t = out.get("type")
        if t == "final_answer":
            out["type"] = "final"
        if "arguments" not in out and isinstance(out.get("parameters"), dict):
            out["arguments"] = out.get("parameters")
        return out

    def parse_json_object_sequence(text: str) -> list[dict[str, Any]]:
        # 支持连续对象输出：{"type":"thought"}{"type":"final"}。
        """parse_json_object_sequence 函数。
        
        用途:
        - 解析连续 JSON 对象序列, 支持 thought/final 类型.
        
        参数:
        - text: 输入的 JSON 对象序列文本, 可能包含多个对象.
        
        返回:
        - list[dict[str, Any]]: 返回解析后的对象列表.
        """
        decoder = json.JSONDecoder()
        idx = 0
        n = len(text)
        items: list[dict[str, Any]] = []
        while idx < n:
            while idx < n and text[idx].isspace():
                idx += 1
            if idx >= n:
                break
            try:
                obj, next_idx = decoder.raw_decode(text, idx)
            except Exception:
                break
            if isinstance(obj, dict):
                items.append(obj)
            idx = next_idx
        return items

    choices = payload.get("choices")
    # 顶层结构缺失时返回 retry，避免把暂时性异常直接当作最终答案。
    if not isinstance(choices, list) or not choices:
        return {"type": "retry", "content": f"LLM 响应缺少 choices：{summarize(payload)}"}

    first = choices[0] if isinstance(choices[0], dict) else {}
    message = first.get("message") if isinstance(first, dict) else None
    # 仅取第一条 choice；当前执行器按单步单动作消费。
    if not isinstance(message, dict):
        return {"type": "retry", "content": f"LLM 响应缺少 message：{summarize(first)}"}

    tool_calls = message.get("tool_calls")
    # 优先工具调用：一旦可解析到工具动作，就不再走文本协议分支。
    if isinstance(tool_calls, list) and tool_calls:
        calls: list[dict[str, Any]] = []
        for raw_call in tool_calls:
            if not isinstance(raw_call, dict):
                continue
            fn = raw_call.get("function")
            if not isinstance(fn, dict):
                continue
            call_item: dict[str, Any] = {
                "name": fn.get("name"),
                "arguments": parse_tool_arguments(fn.get("arguments")),
            }
            tool_call_id = raw_call.get("id")
            if isinstance(tool_call_id, str) and tool_call_id:
                call_item["tool_call_id"] = tool_call_id
            calls.append(call_item)

        if calls:
            return {"type": "tool_calls", "calls": calls}

    content = message.get("content")
    if isinstance(content, str):
        # 先清理 markdown code fence，再做 JSON 协议解析。
        raw_text = (strip_code_fence(content))
        try:
            parsed = json.loads(raw_text)
        except Exception:
            parsed = None
        # 单对象 JSON 协议：{"type":"thought|final|..."}。
        if isinstance(parsed, dict) and isinstance(parsed.get("type"), str):
            return normalize_protocol_dict(parsed)

        # 连续对象协议：取最后一个 typed 对象作为当前步动作。
        multi_objs = parse_json_object_sequence(raw_text)
        typed_objs = [
            normalize_protocol_dict(x)
            for x in multi_objs
            if isinstance(x.get("type"), str)
        ]
        if typed_objs:
            # 对于 "thought + question/observation/final" 连续输出，取最后一个动作作为当前步决策。
            return typed_objs[-1]
        # 既非协议 JSON，也不是可拆分对象序列，则按普通文本 final 处理。
        return {"type": "final", "content": raw_text}
    if isinstance(content, list):
        # 兼容多模态内容数组，尽力抽取 text 片段。
        texts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    texts.append(text)
        if texts:
            return {"type": "final", "content": "\n".join(texts)}

    # 兜底：无 tool_calls 且无可用文本，通常是上游流式输出异常或被截断。
    return {"type": "retry", "content": "LLM 未返回可执行 tool_call，也未返回文本结果"}


def parse_tool_arguments(raw: Any) -> dict[str, Any]:
    """parse_tool_arguments 函数。
    
    用途:
    - 解析工具调用参数, 支持 JSON 格式.
    
    参数:
    - raw: 输入的工具调用参数文本.
    
    返回:
    - dict[str, Any]: 返回解析后的参数字典.
    """
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
            return {"value": parsed}
        except Exception:
            return {"_raw": text}
    return {}
