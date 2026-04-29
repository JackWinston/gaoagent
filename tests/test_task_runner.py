from __future__ import annotations

import unittest
from unittest.mock import patch, MagicMock

from gaoagent.core.task_runner import TaskRunner
from gaoagent.core.runner.base_runner import RunResult, RunnerConfig


class TestTaskRunnerModeRouting(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = TaskRunner()

    @patch("gaoagent.core.task_runner.create_run_logger")
    @patch("gaoagent.core.task_runner.ReActRunner")
    def test_react_mode(self, MockReAct, mock_logger) -> None:
        mock_logger.return_value = MagicMock()
        mock_instance = MagicMock()
        mock_instance.run.return_value = RunResult(success=True, final_result="ok")
        MockReAct.return_value = mock_instance
        self.runner.run("test question", "react")
        MockReAct.assert_called_once()
        mock_instance.run.assert_called_once()

    @patch("gaoagent.core.task_runner.create_run_logger")
    @patch("gaoagent.core.task_runner.PlanAndExecuteRunner")
    def test_plan_mode(self, MockPlan, mock_logger) -> None:
        mock_logger.return_value = MagicMock()
        mock_instance = MagicMock()
        mock_instance.run.return_value = RunResult(success=True, final_result="ok")
        MockPlan.return_value = mock_instance
        self.runner.run("test question", "plan")
        MockPlan.assert_called_once()

    @patch("gaoagent.core.task_runner.create_run_logger")
    @patch("gaoagent.core.task_runner.ReflectionRunner")
    @patch("gaoagent.core.task_runner.ReActRunner")
    def test_retry_mode(self, MockReAct, MockReflection, mock_logger) -> None:
        mock_logger.return_value = MagicMock()
        mock_react = MagicMock()
        MockReAct.return_value = mock_react
        mock_reflect = MagicMock()
        mock_reflect.run.return_value = RunResult(success=True, final_result="ok")
        MockReflection.return_value = mock_reflect
        self.runner.run("test question", "retry")
        MockReflection.assert_called_once()

    @patch("gaoagent.core.task_runner.create_run_logger")
    @patch("gaoagent.core.task_runner.ReflectionRunner")
    @patch("gaoagent.core.task_runner.PlanAndExecuteRunner")
    def test_retry_plan_combo(self, MockPlan, MockReflection, mock_logger) -> None:
        mock_logger.return_value = MagicMock()
        mock_plan = MagicMock()
        MockPlan.return_value = mock_plan
        mock_reflect = MagicMock()
        mock_reflect.run.return_value = RunResult(success=True, final_result="ok")
        MockReflection.return_value = mock_reflect
        self.runner.run("test question", "retry,plan")
        MockPlan.assert_called_once()
        MockReflection.assert_called_once()

    @patch("gaoagent.core.task_runner.create_run_logger")
    @patch("gaoagent.core.task_runner.ReflectionRunner")
    @patch("gaoagent.core.task_runner.ReActRunner")
    def test_retry_react_combo(self, MockReAct, MockReflection, mock_logger) -> None:
        mock_logger.return_value = MagicMock()
        mock_react = MagicMock()
        MockReAct.return_value = mock_react
        mock_reflect = MagicMock()
        mock_reflect.run.return_value = RunResult(success=True, final_result="ok")
        MockReflection.return_value = mock_reflect
        self.runner.run("test question", "retry,react")
        MockReAct.assert_called_once()

    @patch("gaoagent.core.task_runner.create_run_logger")
    def test_invalid_mode(self, mock_logger) -> None:
        mock_logger.return_value = MagicMock()
        self.runner.run("test question", "invalid_mode")

    @patch("gaoagent.core.task_runner.create_run_logger")
    def test_too_many_modes(self, mock_logger) -> None:
        mock_logger.return_value = MagicMock()
        self.runner.run("test question", "react,plan,retry")

    @patch("gaoagent.core.task_runner.create_run_logger")
    def test_double_retry(self, mock_logger) -> None:
        mock_logger.return_value = MagicMock()
        self.runner.run("test question", "retry,retry")


if __name__ == "__main__":
    unittest.main()
