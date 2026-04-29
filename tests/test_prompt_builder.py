from __future__ import annotations

import unittest

from gaoagent.core.runner.PromptBuilder import (
    build_system_prompt,
    build_plan_system_text,
    build_reflection_evaluation_prompt,
    build_react_system_text,
)


class TestBuildSystemPrompt(unittest.TestCase):
    def test_react_mode(self) -> None:
        result = build_system_prompt(["list_dir", "read_file"], mode="react")
        self.assertIn("list_dir", result)
        self.assertIn("read_file", result)
        self.assertIn("thought", result)

    def test_plan_mode(self) -> None:
        result = build_system_prompt([], mode="plan")
        self.assertIn("规划", result)

    def test_retry_mode(self) -> None:
        result = build_system_prompt(["tool1"], mode="retry")
        self.assertIn("tool1", result)


class TestBuildPlanSystemText(unittest.TestCase):
    def test_contains_key_sections(self) -> None:
        text = build_plan_system_text()
        self.assertIn("规划", text)
        self.assertIn("OS", text)


class TestBuildReflectionEvaluationPrompt(unittest.TestCase):
    def test_contains_question_and_result(self) -> None:
        prompt = build_reflection_evaluation_prompt("my question", "my result")
        self.assertIn("my question", prompt)
        self.assertIn("my result", prompt)
        self.assertIn("is_finished", prompt)


class TestBuildReactSystemText(unittest.TestCase):
    def test_with_tools(self) -> None:
        text = build_react_system_text(tool_names=["a", "b"])
        self.assertIn("a", text)
        self.assertIn("b", text)

    def test_without_tools(self) -> None:
        text = build_react_system_text(tool_names=None)
        self.assertIn("智能代理", text)


if __name__ == "__main__":
    unittest.main()
