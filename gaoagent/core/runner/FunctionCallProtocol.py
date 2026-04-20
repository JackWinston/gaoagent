from __future__ import annotations

import json
from typing import Any

from gaoagent.core.runner.BaseRunner import Decision
from gaoagent.core.runner.Tooling import ToolCall
from gaoagent.core.runner.Utils import summarize, truncate_text


def build_function_specs(tool_names: list[str]) -> list[dict[str, Any]]:
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
            "description": "向用户提问并等待用户输入，返回用户的回答。",
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
    }
    specs: list[dict[str, Any]] = []
    for name in tool_names:
        meta = known.get(name)
        desc = (meta or {}).get("description") or f"Tool `{name}` registered in ToolRegistry"
        params = (meta or {}).get("parameters") or {"type": "object", "properties": {}, "additionalProperties": True}
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
    def strip_code_fence(text: str) -> str:
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

    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return {"type": "final", "content": f"LLM 响应缺少 choices：{summarize(payload)}"}

    first = choices[0] if isinstance(choices[0], dict) else {}
    message = first.get("message") if isinstance(first, dict) else None
    if not isinstance(message, dict):
        return {"type": "final", "content": f"LLM 响应缺少 message：{summarize(first)}"}

    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        first_call = tool_calls[0] if isinstance(tool_calls[0], dict) else {}
        tool_call_id = first_call.get("id") if isinstance(first_call.get("id"), str) else None
        fn = first_call.get("function") if isinstance(first_call, dict) else None
        if isinstance(fn, dict):
            name = fn.get("name")
            args = parse_tool_arguments(fn.get("arguments"))
            out: dict[str, Any] = {"type": "function_call", "name": name, "arguments": args}
            if tool_call_id:
                out["tool_call_id"] = tool_call_id
            return out

    content = message.get("content")
    if isinstance(content, str):
        raw_text = strip_code_fence(content)
        try:
            parsed = json.loads(raw_text)
        except Exception:
            parsed = None
        if isinstance(parsed, dict) and isinstance(parsed.get("type"), str):
            t = parsed.get("type")
            if t == "final_answer":
                parsed["type"] = "final"
            if "arguments" not in parsed and isinstance(parsed.get("parameters"), dict):
                parsed["arguments"] = parsed.get("parameters")
            return parsed
        return {"type": "final", "content": content}
    if isinstance(content, list):
        texts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    texts.append(text)
        if texts:
            return {"type": "final", "content": "\n".join(texts)}

    return {"type": "final", "content": "LLM 未返回可执行 tool_call，也未返回文本结果"}


def parse_tool_arguments(raw: Any) -> dict[str, Any]:
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
    if not isinstance(payload, dict):
        return Decision(kind="final", final=f"LLM 响应必须是对象，实际是：{type(payload).__name__}")

    action_type = payload.get("type")
    if action_type == "question":
        content = payload.get("content", "")
        return Decision(kind="thought", internal={"name": "question", "output": str(content)})

    if action_type == "observation":
        content = payload.get("content", "")
        return Decision(kind="thought", internal={"name": "observation", "output": str(content)})

    if action_type == "function_call":
        name = payload.get("name")
        args = payload.get("arguments", payload.get("parameters", {}))
        tool_call_id = payload.get("tool_call_id")
        if not isinstance(name, str) or not name.strip():
            return Decision(kind="final", final="协议错误：function_call.name 必须是非空字符串")
        if name not in tool_names:
            return Decision(kind="final", final=f"协议错误：未知工具 {name}")
        if not isinstance(args, dict):
            return Decision(kind="final", final="协议错误：function_call.parameters/arguments 必须是对象")
        if tool_call_id is not None and not isinstance(tool_call_id, str):
            tool_call_id = None
        return Decision(kind="tool", tool_call=ToolCall(name=name, arguments=args, tool_call_id=tool_call_id))

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
            "协议错误：LLM 响应 type 必须是 question/thought/function_call/observation/final。"
            f" 当前为 {repr(action_type)}，step={step}"
        ),
    )


def http_error_to_final(status: int, reason: str, body: str) -> dict[str, Any]:
    return {
        "type": "final",
        "content": f"LLM HTTPError: status={status}, reason={reason}, body={truncate_text(body, 500)}",
    }
