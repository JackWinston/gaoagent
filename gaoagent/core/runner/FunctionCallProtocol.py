from __future__ import annotations

import json
from typing import Any

from gaoagent.core.runner.BaseRunner import Decision
from gaoagent.core.runner.Utils import summarize, truncate_text


def build_function_specs(
    tool_names: list[str],
    *,
    mcp_exported_map: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """build_function_specs 函数。
    
    用途:
    - 给模型提供函数调用的规格, 用于在运行时动态调用外部函数.
    
    参数:
    - tool_names: 本地定义的函数名称列表
    - mcp_exported_map: MCP函数映射表, 用于在运行时调用外部函数.
    
    返回:
    - list[dict[str, Any]]: 返回模型需要的函数调用规格列表.
    """
    known: dict[str, dict[str, Any]] = {
        "list_dir": {
            "description": "获取目录下的文件/子目录列表（默认列出当前工作目录）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "default": "."},
                },
                "additionalProperties": False,
            },
        },
        "read_file": {
            "description": "读取文本文件内容。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "encoding": {"type": "string", "default": "utf-8"},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
        "ask_user": {
            "description": (
                "向用户发起一次阻塞式提问并等待输入，返回用户原始回答。"
                "当任务需要多轮交互（如游戏、追问、确认）时必须调用本工具，"
                "不要用 assistant 文本模拟提问。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "default": {"type": "string"},
                    "choices": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["prompt"],
                "additionalProperties": False,
            },
        },
        "write_file": {
            "description": "写入文本文件内容（默认覆盖）。可自动创建父目录。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                    "encoding": {"type": "string", "default": "utf-8"},
                    "mkdirs": {"type": "boolean", "default": True},
                    "append": {"type": "boolean", "default": False},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        },
        "run_command": {
            "description": (
                "在本地执行控制台命令并返回输出。"
                "workdir 必须是当前工作目录或其子目录。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "workdir": {"type": "string"},
                    "command": {"type": "string"},
                },
                "required": ["workdir", "command"],
                "additionalProperties": False,
            },
        },
        "search_workspace": {
            "description": (
                "在当前项目内执行全文检索（基于 ripgrep），并遵循 .gitignore 过滤规则。"
                "该工具不会搜索项目目录之外的文件。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "检索关键词或正则表达式（默认按 regex 语义）"
                    },
                    "scope_path": {
                        "type": "string",
                        "description": "可选；在该目录或文件范围搜索（绝对路径，且必须位于当前项目内）"
                    },
                    "file_glob": {
                        "description": "可选文件过滤；支持字符串或字符串数组（例如 *.py 或 [\"*.py\", \"*.md\"]）",
                        "oneOf": [
                            {"type": "string"},
                            {"type": "array", "items": {"type": "string"}}
                        ]
                    },
                    "max_results": {
                        "type": "integer",
                        "default": 50,
                        "description": "最多返回的命中数（上限 500）"
                    },
                    "case_sensitive": {
                        "type": "boolean",
                        "default": False,
                        "description": "是否大小写敏感；false 时使用 smart-case"
                    },
                    "literal": {
                        "type": "boolean",
                        "default": False,
                        "description": "是否按字面量搜索（不使用正则）"
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
        "rag_search": {
            "description": (
                "在指定的 RAG 知识库中进行向量检索，获取与问题最相关的文档切片。"
                "当用户询问特定领域的知识或项目代码时，使用此工具获取上下文。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "kb_name": {
                        "type": "string",
                        "description": "知识库名称（如果不确定，可先不传或询问用户，或者默认使用最相关的）"
                    },
                    "query": {
                        "type": "string",
                        "description": "检索的查询语句，通常是用户的原问题或提取的关键词"
                    },
                    "top_k": {
                        "type": "integer",
                        "default": 5,
                        "description": "返回的最相关文档切片数量"
                    }
                },
                "required": ["kb_name", "query"],
                "additionalProperties": False,
            },
        },
    }
    specs: list[dict[str, Any]] = []
    for name in tool_names:
        meta = known.get(name)
        mcp_meta = (
            (mcp_exported_map or {}).get(name)
            if isinstance(mcp_exported_map, dict)
            else None
        )
        desc = (
            (meta or {}).get("description")
            or (mcp_meta.get("description") if isinstance(mcp_meta, dict) else None)
            or f"Tool `{name}` registered in ToolRegistry"
        )
        params = (
            (meta or {}).get("parameters")
            or (mcp_meta.get("parameters") if isinstance(mcp_meta, dict) else None)
            or {"type": "object", "properties": {}, "additionalProperties": True}
        )
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


def protocol_to_decision(step: int, tool_names: set[str], payload: Any) -> Decision:
    """protocol_to_decision 函数。
    
    用途:
    - 将协议字典转换为 Decision 对象, 用于在执行器中使用.
    
    参数:
    - step: 当前步骤编号.
    - tool_names: 工具名称集合.
    - payload: 输入的协议字典.
    
    返回:
    - Decision: 返回转换后的 Decision 对象.
    """
    if not isinstance(payload, dict):
        return Decision(kind="final", final=f"LLM 响应必须是对象，实际是：{type(payload).__name__}")

    action_type = payload.get("type")
    if action_type == "question":
        content = payload.get("content", "")
        return Decision(kind="thought", internal={"name": "question", "output": str(content)})

    if action_type == "observation":
        content = payload.get("content", "")
        return Decision(kind="thought", internal={"name": "observation", "output": str(content)})

    if action_type == "final":
        content = payload.get("content", "")
        return Decision(kind="final", final=str(content))

    if action_type == "thought":
        inner_name = payload.get("name") or "thought"
        content = payload.get("content", None)
        output = payload.get("output")
        if output is None and content is not None:
            output = str(content)
        return Decision(
            kind="thought",
            internal={
                "name": str(inner_name),
                "input": payload.get("input"),
                "output": output,
            },
        )

    return Decision(
        kind="final",
        final=(
            "协议错误：LLM 响应 type 必须是 question/thought/observation/final。"
            f" 当前为 {repr(action_type)}，step={step}"
        ),
    )


def http_error_to_final(status: int, reason: str, body: str) -> dict[str, Any]:
    """http_error_to_final 函数。
    
    用途:
    - 将 HTTP 错误转换为 final 动作.
    
    参数:
    - status: HTTP 状态码.
    - reason: 错误原因.
    - body: 输入的错误响应体.
    
    返回:
    - dict[str, Any]: 返回转换后的 final 动作字典.  
    """
    return {
        "type": "final",
        "content": f"LLM HTTPError: status={status}, reason={reason}, body={truncate_text(body, 500)}",
    }
