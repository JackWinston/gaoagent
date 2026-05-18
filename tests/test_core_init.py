from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gaoagent.core.core_init import CoreInit
from gaoagent.core.runner.base_runner import RequestBaseInfo, RunResult


class TestCoreInitProjectOverview(unittest.TestCase):
    def test_init_writes_project_overview_markdown_from_llm(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            home_dir = temp_root / "home"
            global_dir = home_dir / ".gaoagent"
            global_dir.mkdir(parents=True, exist_ok=True)

            project_root = temp_root / "demo_project"
            project_root.mkdir(parents=True, exist_ok=True)
            (project_root / ".gitignore").write_text("*.log\n", encoding="utf-8")
            (project_root / "README.md").write_text(
                "# Demo Project\n\n"
                "一个用于验证项目概览生成的示例工程。\n\n"
                "## 功能亮点\n\n"
                "- 统一 CLI 入口\n"
                "- 提供测试目录\n",
                encoding="utf-8",
            )
            (project_root / "pyproject.toml").write_text(
                "[project]\n"
                'name = "demo-project"\n'
                'requires-python = ">=3.12"\n',
                encoding="utf-8",
            )
            (project_root / "demo_pkg").mkdir()
            (project_root / "demo_pkg" / "__init__.py").write_text("", encoding="utf-8")
            (project_root / "demo_pkg" / "cli.py").write_text(
                '"""CLI 入口文件，用于暴露命令行能力。"""\n\n'
                "def main() -> None:\n"
                "    pass\n",
                encoding="utf-8",
            )
            (project_root / "tests").mkdir()
            (project_root / "tests" / "test_cli.py").write_text(
                "def test_placeholder() -> None:\n"
                "    assert True\n",
                encoding="utf-8",
            )
            (project_root / "custom.config").write_text("alpha=1\n", encoding="utf-8")
            (project_root / "NOTICE").write_text("plain text notice\n", encoding="utf-8")
            llm_markdown = (
                "# 项目概览：demo_project\n\n"
                "## 1. 项目的主要功能\n"
                "- 这是由 LLM 为目标项目生成的概览。\n\n"
                "## 2. 项目的技术架构\n"
                "- 使用 Python 项目结构。\n\n"
                "## 3. 项目的文件树形结构\n"
                "```text\n"
                "demo_project/\n"
                "```\n\n"
                "## 4. 每个文件/文件夹的功能\n"
                "- `demo_pkg/cli.py`：CLI 入口。\n"
            )

            core_init = CoreInit()
            with (
                patch("gaoagent.core.core_init.Path.home", return_value=home_dir),
                patch("gaoagent.core.core_init.Path.cwd", return_value=project_root),
                patch.object(
                    core_init,
                    "_load_global_apis",
                    return_value={"openai": {"base_url": "https://example.com", "api_key": "x", "models": {"gpt-4.1": {}}}},
                ),
                patch.object(core_init, "_prompt_default_api_and_model", return_value=("openai", "gpt-4.1")),
                patch.object(core_init, "_load_global_mcp_configs", return_value={}),
                patch.object(core_init, "_prompt_select_mcp", return_value={}),
                patch.object(core_init, "_prompt_select_skills", return_value=[]),
                patch.object(core_init, "_prompt_select_rag", return_value=[]),
                patch.object(core_init, "_load_global_a2a_configs", return_value={}),
                patch.object(core_init, "_prompt_select_a2a", return_value={}),
                patch.object(core_init, "_request_project_overview_from_llm", return_value=llm_markdown) as mock_llm,
            ):
                core_init.init()

            overview_file = project_root / ".gaoagent" / "project.md"
            self.assertTrue(overview_file.exists())

            content = overview_file.read_text(encoding="utf-8")
            self.assertEqual(llm_markdown, content)
            self.assertIn("这是由 LLM 为目标项目生成的概览", content)
            self.assertIn("`demo_pkg/cli.py`", content)
            mock_llm.assert_called_once()
            called_project_root = mock_llm.call_args.args[0]
            called_context = mock_llm.call_args.args[1]
            self.assertEqual(project_root, called_project_root)
            self.assertIn("demo_project/", called_context)
            self.assertIn("README.md", called_context)
            self.assertIn("pyproject.toml", called_context)
            self.assertIn("custom.config", called_context)
            self.assertIn("NOTICE", called_context)

    def test_collect_context_file_snippets_accepts_all_readable_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            (project_root / "README.md").write_text("# Demo\n", encoding="utf-8")
            (project_root / "custom.config").write_text("alpha=1\nbeta=2\n", encoding="utf-8")
            (project_root / "NOTICE").write_text("plain text\n", encoding="utf-8")

            tool = CoreInit()
            _, entries, _ = tool._collect_project_entries(project_root)
            snippets = tool._collect_context_file_snippets(entries)

            self.assertIn("`custom.config`", snippets)
            self.assertIn("alpha=1", snippets)
            self.assertIn("`NOTICE`", snippets)
            self.assertIn("plain text", snippets)

    def test_request_project_overview_from_llm_uses_react_runner(self) -> None:
        tool = CoreInit()
        with (
            patch(
                "gaoagent.core.runner.utils.load_request_base_info",
                return_value=RequestBaseInfo(
                    baseurl="https://example.com",
                    api_key="x",
                    modules="gpt-4.1",
                ),
            ),
            patch(
                "gaoagent.core.runner.react_runner.ReActRunner.run",
                return_value=RunResult(success=True, final_result="## 1. 项目的主要功能\n- ok\n"),
            ) as mock_run,
        ):
            content = tool._request_project_overview_from_llm(Path("demo_project"), "context body")

        self.assertIn("## 1. 项目的主要功能", content)
        mock_run.assert_called_once()
        prompt = mock_run.call_args.args[0]
        self.assertIn("`.gaoagent/project.md`", prompt)
        self.assertIn("demo_project", prompt)
        self.assertIn("context body", prompt)

    def test_collect_project_entries_ignores_generated_and_dependency_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            (project_root / "src").mkdir()
            (project_root / "src" / "main.py").write_text("print('ok')\n", encoding="utf-8")
            (project_root / ".gaoagent").mkdir()
            (project_root / ".gaoagent" / "project.md").write_text("generated\n", encoding="utf-8")
            (project_root / ".venv").mkdir()
            (project_root / ".venv" / "ignored.py").write_text("pass\n", encoding="utf-8")

            tree_text, entries, truncated = CoreInit()._collect_project_entries(project_root)

            self.assertFalse(truncated)
            self.assertIn("src/", tree_text)
            self.assertNotIn(".gaoagent", tree_text)
            self.assertNotIn(".venv", tree_text)
            self.assertEqual(["src", "src/main.py"], [entry["relative_path"] for entry in entries])


if __name__ == "__main__":
    unittest.main()
