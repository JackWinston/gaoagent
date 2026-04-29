from __future__ import annotations

import unittest

from gaoagent.mcp.mcp_client_compat import (
    MCPServerConfig,
    _extract_tools,
    _serialize_call_result,
    build_mcp_tools_cache_payload,
    _run_coro,
)


class TestMCPServerConfigFromDict(unittest.TestCase):
    def test_stdio_config(self) -> None:
        payload = {"type": "stdio", "command": "python", "args": ["-m", "server"]}
        cfg = MCPServerConfig.from_dict(payload)
        self.assertEqual(cfg.type, "stdio")
        self.assertEqual(cfg.command, "python")
        self.assertEqual(cfg.args, ["-m", "server"])

    def test_sse_config(self) -> None:
        payload = {"type": "sse", "url": "http://localhost:8080"}
        cfg = MCPServerConfig.from_dict(payload)
        self.assertEqual(cfg.type, "sse")
        self.assertEqual(cfg.url, "http://localhost:8080")

    def test_streamable_http_config(self) -> None:
        payload = {"type": "streamable_http", "url": "http://localhost:8080"}
        cfg = MCPServerConfig.from_dict(payload)
        self.assertEqual(cfg.type, "streamable_http")

    def test_timeout_int(self) -> None:
        cfg = MCPServerConfig.from_dict({"type": "stdio", "command": "x", "timeout": 30})
        self.assertEqual(cfg.timeout, 30)

    def test_timeout_string(self) -> None:
        cfg = MCPServerConfig.from_dict({"type": "stdio", "command": "x", "timeout": "30"})
        self.assertEqual(cfg.timeout, 30)

    def test_timeout_invalid_string(self) -> None:
        cfg = MCPServerConfig.from_dict({"type": "stdio", "command": "x", "timeout": "abc"})
        self.assertIsNone(cfg.timeout)

    def test_env_filtered(self) -> None:
        payload = {"type": "stdio", "command": "x", "env": {"KEY": "val", 123: "bad"}}
        cfg = MCPServerConfig.from_dict(payload)
        self.assertEqual(cfg.env, {"KEY": "val"})

    def test_headers_filtered(self) -> None:
        payload = {"type": "sse", "url": "http://x", "headers": {"Auth": "Bearer t", 123: "bad"}}
        cfg = MCPServerConfig.from_dict(payload)
        self.assertEqual(cfg.headers, {"Auth": "Bearer t"})

    def test_disabled_true(self) -> None:
        cfg = MCPServerConfig.from_dict({"type": "stdio", "command": "x", "disabled": True})
        self.assertTrue(cfg.disabled)

    def test_disabled_false(self) -> None:
        cfg = MCPServerConfig.from_dict({"type": "stdio", "command": "x", "disabled": False})
        self.assertFalse(cfg.disabled)

    def test_disabled_non_bool(self) -> None:
        cfg = MCPServerConfig.from_dict({"type": "stdio", "command": "x", "disabled": "yes"})
        self.assertFalse(cfg.disabled)


class TestExtractTools(unittest.TestCase):
    def test_valid_tools(self) -> None:
        class FakeTool:
            def __init__(self, name, desc="", schema=None):
                self.name = name
                self.description = desc
                self.inputSchema = schema

        class FakeResult:
            tools = [FakeTool("tool1", "desc1", {"type": "object"})]

        result = _extract_tools(FakeResult())
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["name"], "tool1")
        self.assertEqual(result[0]["description"], "desc1")

    def test_no_tools_attribute(self) -> None:
        result = _extract_tools(object())
        self.assertEqual(result, [])

    def test_empty_tools(self) -> None:
        class FakeResult:
            tools = []
        result = _extract_tools(FakeResult())
        self.assertEqual(result, [])

    def test_tool_without_name(self) -> None:
        class FakeTool:
            name = ""
            description = "d"
            inputSchema = {}
        class FakeResult:
            tools = [FakeTool()]
        result = _extract_tools(FakeResult())
        self.assertEqual(result, [])


class TestSerializeCallResult(unittest.TestCase):
    def test_none(self) -> None:
        result = _serialize_call_result(None)
        self.assertEqual(result, {"content": []})

    def test_content_with_text_items(self) -> None:
        class FakeItem:
            def __init__(self, text):
                self.text = text
        class FakeResult:
            content = [FakeItem("hello"), FakeItem("world")]
        result = _serialize_call_result(FakeResult())
        self.assertEqual(len(result["content"]), 2)
        self.assertEqual(result["content"][0]["text"], "hello")

    def test_content_with_dict_items(self) -> None:
        class FakeResult:
            content = [{"type": "text", "text": "ok"}]
        result = _serialize_call_result(FakeResult())
        self.assertEqual(result["content"][0]["text"], "ok")


class TestBuildMcpToolsCachePayload(unittest.TestCase):
    def test_basic_flow(self) -> None:
        servers = {"srv1": {"type": "stdio", "command": "x"}}
        tools = [{"name": "tool1", "description": "d1", "inputSchema": {"type": "object"}}]
        payload = build_mcp_tools_cache_payload(
            servers,
            connect_and_list_tools=lambda name, body: tools,
            generated_at="test",
        )
        self.assertIn("srv1", payload["servers"])
        self.assertIn("exported_map", payload)
        self.assertTrue(len(payload["exported_map"]) > 0)

    def test_disabled_server_skipped(self) -> None:
        servers = {"srv1": {"type": "stdio", "command": "x", "disabled": True}}
        payload = build_mcp_tools_cache_payload(
            servers,
            connect_and_list_tools=lambda name, body: [],
            generated_at="test",
        )
        self.assertNotIn("srv1", payload["servers"])

    def test_error_in_connect(self) -> None:
        servers = {"srv1": {"type": "stdio", "command": "x"}}
        def fail(name, body):
            raise RuntimeError("connection failed")
        payload = build_mcp_tools_cache_payload(servers, connect_and_list_tools=fail, generated_at="test")
        self.assertIn("error", payload["servers"]["srv1"])

    def test_non_dict_servers_skipped(self) -> None:
        servers = {"srv1": "not a dict", "srv2": {"type": "stdio", "command": "x"}}
        payload = build_mcp_tools_cache_payload(
            servers,
            connect_and_list_tools=lambda name, body: [{"name": "t", "description": "", "inputSchema": {}}],
            generated_at="test",
        )
        self.assertNotIn("srv1", payload["servers"])
        self.assertIn("srv2", payload["servers"])


class TestRunCoro(unittest.TestCase):
    def test_basic_coro(self) -> None:
        async def coro():
            return 42
        result = _run_coro(coro())
        self.assertEqual(result, 42)


if __name__ == "__main__":
    unittest.main()
