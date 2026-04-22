import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

from gaoagent.core.runner.Tooling import ToolCall, default_tool_registry


class TestRunCommandTool(unittest.TestCase):
    def test_run_command_in_current_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_cwd = Path.cwd()
            os.chdir(tmp)
            try:
                tools = default_tool_registry()
                result_raw = tools.call(
                    None,
                    ToolCall(
                        name="run_command",
                        arguments={
                            "workdir": ".",
                            "command": f"\"{sys.executable}\" -c \"print('ok')\"",
                        },
                    ),
                )
                result = json.loads(result_raw)
            finally:
                os.chdir(old_cwd)

        self.assertTrue(result["success"])
        self.assertEqual(result["exit_code"], 0)
        self.assertIn("ok", result["stdout"])

    def test_run_command_rejects_parent_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "project"
            child = project / "child"
            child.mkdir(parents=True, exist_ok=True)

            old_cwd = Path.cwd()
            os.chdir(project)
            try:
                tools = default_tool_registry()
                result_raw = tools.call(
                    None,
                    ToolCall(
                        name="run_command",
                        arguments={
                            "workdir": "..",
                            "command": f"\"{sys.executable}\" -c \"print('blocked')\"",
                        },
                    ),
                )
                result = json.loads(result_raw)
            finally:
                os.chdir(old_cwd)

        self.assertFalse(result["success"])
        self.assertEqual(result["error"]["type"], "PermissionError")


if __name__ == "__main__":
    unittest.main()
