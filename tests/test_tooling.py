from __future__ import annotations

import unittest

from gaoagent.core.runner.tooling import ToolCall, ToolRegistry, default_tool_registry


class TestToolCall(unittest.TestCase):
    def test_defaults(self) -> None:
        tc = ToolCall(name="test")
        self.assertEqual(tc.name, "test")
        self.assertEqual(tc.arguments, {})
        self.assertEqual(tc.description, "")
        self.assertIsNone(tc.tool_call_id)

    def test_with_args(self) -> None:
        tc = ToolCall(name="read", arguments={"path": "/tmp"}, description="desc", tool_call_id="id1")
        self.assertEqual(tc.arguments, {"path": "/tmp"})
        self.assertEqual(tc.tool_call_id, "id1")


class TestToolRegistry(unittest.TestCase):
    def test_register_and_list(self) -> None:
        reg = ToolRegistry()
        reg.register("a", lambda ctx, args: "a")
        reg.register("b", lambda ctx, args: "b")
        self.assertEqual(reg.list_names(), ["a", "b"])

    def test_register_empty_name_raises(self) -> None:
        reg = ToolRegistry()
        with self.assertRaises(ValueError):
            reg.register("", lambda ctx, args: None)

    def test_register_non_string_raises(self) -> None:
        reg = ToolRegistry()
        with self.assertRaises(ValueError):
            reg.register(123, lambda ctx, args: None)

    def test_call_known_tool(self) -> None:
        reg = ToolRegistry()
        reg.register("echo", lambda ctx, args: f"echoed: {args.get('msg')}")
        result = reg.call(None, ToolCall(name="echo", arguments={"msg": "hi"}))
        self.assertEqual(result, "echoed: hi")

    def test_call_unknown_tool_raises(self) -> None:
        reg = ToolRegistry()
        with self.assertRaises(KeyError):
            reg.call(None, ToolCall(name="nonexistent"))

    def test_call_returns_json_for_non_string(self) -> None:
        reg = ToolRegistry()
        reg.register("dict_tool", lambda ctx, args: {"key": "value"})
        result = reg.call(None, ToolCall(name="dict_tool"))
        self.assertIn("key", result)


class TestDefaultToolRegistry(unittest.TestCase):
    def test_has_expected_tools(self) -> None:
        reg = default_tool_registry()
        names = reg.list_names()
        for expected in ["list_dir", "read_file", "ask_user", "write_file", "run_command",
                         "search_workspace", "rag_search", "a2a_call"]:
            self.assertIn(expected, names, f"Missing tool: {expected}")

    def test_list_dir_with_valid_path(self) -> None:
        import tempfile, os
        reg = default_tool_registry()
        with tempfile.TemporaryDirectory() as tmpdir:
            open(os.path.join(tmpdir, "test.txt"), "w").close()
            result = reg.call(None, ToolCall(name="list_dir", arguments={"path": tmpdir}))
            self.assertIn("test.txt", result)

    def test_read_file_valid(self) -> None:
        import tempfile, os
        reg = default_tool_registry()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hello content")
            f.flush()
            result = reg.call(None, ToolCall(name="read_file", arguments={"path": f.name}))
            self.assertIn("hello content", result)

    def test_read_file_missing_path(self) -> None:
        reg = default_tool_registry()
        result = reg.call(None, ToolCall(name="read_file", arguments={}))
        self.assertIn("error", result.lower())

    def test_run_command_echo(self) -> None:
        import tempfile, os
        reg = default_tool_registry()
        with tempfile.TemporaryDirectory(dir=os.getcwd()) as tmpdir:
            result = reg.call(None, ToolCall(name="run_command", arguments={"workdir": tmpdir, "command": "echo hello"}))
            self.assertIn("hello", result)

    def test_write_file(self) -> None:
        import tempfile, os
        reg = default_tool_registry()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "output.txt")
            result = reg.call(None, ToolCall(name="write_file", arguments={"path": path, "content": "test data"}))
            self.assertIn("success", result.lower())
            with open(path) as f:
                self.assertEqual(f.read(), "test data")


if __name__ == "__main__":
    unittest.main()
