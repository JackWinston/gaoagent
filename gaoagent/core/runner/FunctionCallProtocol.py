from __future__ import annotations

import json
from typing import Any

from gaoagent.core.runner.BaseRunner import Decision
from gaoagent.core.runner.Tooling import ToolCall
from gaoagent.core.runner.Utils import summarize, truncate_text


def build_function_specs(tool_names: list[str]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for name in tool_names:
        specs.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": f"Tool `{name}` registered in ToolRegistry",
                    "parameters": {"type": "object", "additionalProperties": True},
                },
            }
        )
    return specs


def map_chat_completion_to_protocol(payload: dict[str, Any]) -> dict[str, Any]:
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
        fn = first_call.get("function") if isinstance(first_call, dict) else None
        if isinstance(fn, dict):
            name = fn.get("name")
            args = parse_tool_arguments(fn.get("arguments"))
            return {"type": "function_call", "name": name, "arguments": args}

    content = message.get("content")
    if isinstance(content, str):
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
    if action_type == "function_call":
        name = payload.get("name")
        args = payload.get("arguments", {})
        if not isinstance(name, str) or not name.strip():
            return Decision(kind="final", final="协议错误：function_call.name 必须是非空字符串")
        if name not in tool_names:
            return Decision(kind="final", final=f"协议错误：未知工具 {name}")
        if not isinstance(args, dict):
            return Decision(kind="final", final="协议错误：function_call.arguments 必须是对象")
        return Decision(kind="tool", tool_call=ToolCall(name=name, arguments=args))

    if action_type == "final":
        content = payload.get("content", "")
        return Decision(kind="final", final=str(content))

    if action_type == "internal":
        inner_name = payload.get("name") or "internal"
        return Decision(
            kind="internal",
            internal={
                "name": str(inner_name),
                "input": payload.get("input"),
                "output": payload.get("output"),
            },
        )

    return Decision(
        kind="final",
        final=(
            "协议错误：LLM 响应 type 必须是 function_call/final/internal。"
            f" 当前为 {repr(action_type)}，step={step}"
        ),
    )


def http_error_to_final(status: int, reason: str, body: str) -> dict[str, Any]:
    return {
        "type": "final",
        "content": f"LLM HTTPError: status={status}, reason={reason}, body={truncate_text(body, 500)}",
    }
