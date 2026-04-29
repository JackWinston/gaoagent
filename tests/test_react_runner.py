from __future__ import annotations

import unittest

from gaoagent.core.runner.react_runner import ReActRunner


class TestEnabledMcpServers(unittest.TestCase):
    def test_none_input(self) -> None:
        result = ReActRunner._enabled_mcp_servers(None)
        self.assertEqual(result, {})

    def test_non_dict_input(self) -> None:
        result = ReActRunner._enabled_mcp_servers("not a dict")
        self.assertEqual(result, {})

    def test_filters_disabled(self) -> None:
        servers = {
            "a": {"command": "cmd", "disabled": True},
            "b": {"command": "cmd"},
            "c": {"command": "cmd", "disabled": False},
        }
        result = ReActRunner._enabled_mcp_servers(servers)
        self.assertIn("b", result)
        self.assertIn("c", result)
        self.assertNotIn("a", result)

    def test_filters_non_dict_values(self) -> None:
        servers = {"a": "not a dict", "b": {"command": "cmd"}}
        result = ReActRunner._enabled_mcp_servers(servers)
        self.assertNotIn("a", result)
        self.assertIn("b", result)

    def test_filters_non_string_keys(self) -> None:
        servers = {123: {"command": "cmd"}, "valid": {"command": "cmd"}}
        result = ReActRunner._enabled_mcp_servers(servers)
        self.assertNotIn(123, result)
        self.assertIn("valid", result)


class TestFilterExportedMapForServers(unittest.TestCase):
    def test_none_exported_map(self) -> None:
        result = ReActRunner._filter_exported_map_for_servers(None, {"s1": {}})
        self.assertEqual(result, {})

    def test_filters_by_valid_server(self) -> None:
        exported = {
            "tool_a": {"server": "s1", "tool": "t1"},
            "tool_b": {"server": "s2", "tool": "t2"},
        }
        result = ReActRunner._filter_exported_map_for_servers(exported, {"s1": {}})
        self.assertIn("tool_a", result)
        self.assertNotIn("tool_b", result)

    def test_filters_invalid_meta(self) -> None:
        exported = {
            "tool_a": "not a dict",
            "tool_b": {"server": "s1", "tool": ""},
            "tool_c": {"server": "s1", "tool": "valid"},
        }
        result = ReActRunner._filter_exported_map_for_servers(exported, {"s1": {}})
        self.assertNotIn("tool_a", result)
        self.assertNotIn("tool_b", result)
        self.assertIn("tool_c", result)

    def test_empty_inputs(self) -> None:
        result = ReActRunner._filter_exported_map_for_servers({}, {})
        self.assertEqual(result, {})


class TestReActRunnerInit(unittest.TestCase):
    def test_default_config(self) -> None:
        runner = ReActRunner()
        self.assertEqual(runner.mode, "react")
        self.assertEqual(runner.runner_config.max_steps, 32)
        self.assertIsNotNone(runner.runner_config.tools)

    def test_custom_config(self) -> None:
        from gaoagent.core.runner.base_runner import RunnerConfig
        cfg = RunnerConfig(max_steps=10, llm_invalid_retry=1)
        runner = ReActRunner(config=cfg)
        self.assertEqual(runner.runner_config.max_steps, 10)


class TestReActRunnerRunEmptyQuestion(unittest.TestCase):
    def test_empty_question_returns_failure(self) -> None:
        runner = ReActRunner()
        result = runner.run("")
        self.assertFalse(result.success)
        self.assertEqual(result.error, "Invalid question")

    def test_none_question_returns_failure(self) -> None:
        runner = ReActRunner()
        result = runner.run(None)
        self.assertFalse(result.success)


if __name__ == "__main__":
    unittest.main()
