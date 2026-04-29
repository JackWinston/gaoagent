from __future__ import annotations

import unittest

from gaoagent.core.runner.FunctionCallProtocol import (
    build_function_specs,
    map_chat_completion_to_protocol
)


class TestBuildFunctionSpecs(unittest.TestCase):
    def test_known_tool(self) -> None:
        specs = build_function_specs(["list_dir"])
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0]["type"], "function")
        self.assertEqual(specs[0]["function"]["name"], "list_dir")
        self.assertIn("description", specs[0]["function"])

    def test_unknown_tool_gets_generic_schema(self) -> None:
        specs = build_function_specs(["custom_tool"])
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0]["function"]["name"], "custom_tool")
        self.assertIn("parameters", specs[0]["function"])

    def test_mcp_tool_uses_mcp_meta(self) -> None:
        mcp_map = {
            "mcp__srv__tool": {
                "description": "MCP tool desc",
                "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
            }
        }
        specs = build_function_specs(["mcp__srv__tool"], mcp_exported_map=mcp_map)
        self.assertEqual(specs[0]["function"]["description"], "MCP tool desc")

    def test_empty_tool_names(self) -> None:
        specs = build_function_specs([])
        self.assertEqual(specs, [])

    def test_multiple_tools(self) -> None:
        specs = build_function_specs(["list_dir", "read_file", "unknown"])
        self.assertEqual(len(specs), 3)


class TestMapChatCompletionToProtocol(unittest.TestCase):
    def test_tool_calls(self) -> None:
        payload = {
            "choices": [{
                "message": {
                    "tool_calls": [
                        {"id": "call_1", "function": {"name": "list_dir", "arguments": '{"path":"."}'}}
                    ]
                }
            }]
        }
        result = map_chat_completion_to_protocol(payload)
        self.assertEqual(result["type"], "tool_calls")
        self.assertEqual(len(result["calls"]), 1)
        self.assertEqual(result["calls"][0]["name"], "list_dir")

    def test_thought_json(self) -> None:
        payload = {
            "choices": [{
                "message": {"content": '{"type":"thought","content":"thinking..."}'}
            }]
        }
        result = map_chat_completion_to_protocol(payload)
        self.assertEqual(result["type"], "thought")
        self.assertEqual(result["content"], "thinking...")

    def test_final_json(self) -> None:
        payload = {
            "choices": [{
                "message": {"content": '{"type":"final","content":"answer here"}'}
            }]
        }
        result = map_chat_completion_to_protocol(payload)
        self.assertEqual(result["type"], "final")
        self.assertEqual(result["content"], "answer here")

    def test_plain_text_becomes_final(self) -> None:
        payload = {
            "choices": [{
                "message": {"content": "just a plain answer"}
            }]
        }
        result = map_chat_completion_to_protocol(payload)
        self.assertEqual(result["type"], "final")
        self.assertEqual(result["content"], "just a plain answer")

    def test_missing_choices_returns_retry(self) -> None:
        result = map_chat_completion_to_protocol({})
        self.assertEqual(result["type"], "retry")

    def test_empty_choices_returns_retry(self) -> None:
        result = map_chat_completion_to_protocol({"choices": []})
        self.assertEqual(result["type"], "retry")

    def test_missing_message_returns_retry(self) -> None:
        result = map_chat_completion_to_protocol({"choices": [{}]})
        self.assertEqual(result["type"], "retry")

    def test_code_fence_stripped(self) -> None:
        payload = {
            "choices": [{
                "message": {"content": '```json\n{"type":"final","content":"ok"}\n```'}
            }]
        }
        result = map_chat_completion_to_protocol(payload)
        self.assertEqual(result["type"], "final")
        self.assertEqual(result["content"], "ok")

    def test_final_answer_normalized(self) -> None:
        payload = {
            "choices": [{
                "message": {"content": '{"type":"final_answer","content":"done"}'}
            }]
        }
        result = map_chat_completion_to_protocol(payload)
        self.assertEqual(result["type"], "final")

    def test_list_content(self) -> None:
        payload = {
            "choices": [{
                "message": {"content": [{"type": "text", "text": "hello"}, {"type": "text", "text": "world"}]}
            }]
        }
        result = map_chat_completion_to_protocol(payload)
        self.assertEqual(result["type"], "final")
        self.assertIn("hello", result["content"])
        self.assertIn("world", result["content"])


if __name__ == "__main__":
    unittest.main()
