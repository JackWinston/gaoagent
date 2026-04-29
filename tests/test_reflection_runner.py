from __future__ import annotations

import unittest

from gaoagent.core.runner.reflection_runner import ReflectionRunner
from gaoagent.core.runner.base_runner import RunResult, RunnerConfig


class TestReflectionRunnerDecide(unittest.TestCase):
    def test_decide_raises_not_implemented(self) -> None:
        from gaoagent.core.runner.react_runner import ReActRunner
        target = ReActRunner()
        runner = ReflectionRunner(target_runner=target)
        from gaoagent.core.runner.base_runner import RunnerContext
        ctx = RunnerContext(step=0, history=[])
        with self.assertRaises(NotImplementedError):
            runner.decide(ctx)


class TestReflectionRunnerRun(unittest.TestCase):
    def test_empty_question(self) -> None:
        from gaoagent.core.runner.react_runner import ReActRunner
        target = ReActRunner()
        runner = ReflectionRunner(target_runner=target)
        result = runner.run("")
        self.assertFalse(result.success)
        self.assertEqual(result.error, "Invalid question")


class TestReflectionRunnerReflect(unittest.TestCase):
    def test_immediate_finish(self) -> None:
        from unittest.mock import MagicMock, patch
        from gaoagent.core.runner.react_runner import ReActRunner
        target = MagicMock()
        runner = ReflectionRunner(target_runner=target, max_reflections=3)
        runner._client = MagicMock()
        runner.request_base_info = MagicMock()
        runner.request_base_info.modules = "test-model"
        runner._client.post_chat_completions.return_value = MagicMock(
            ok=True,
            json={"choices": [{"message": {"content": '{"is_finished": true, "feedback": "done!"}'}}]},
            text='{"is_finished": true}',
        )
        initial_result = RunResult(success=True, final_result="task completed")
        result = runner.reflect("test question", initial_result)
        self.assertTrue(result.success)

    def test_max_reflections_reached(self) -> None:
        from unittest.mock import MagicMock
        from gaoagent.core.runner.react_runner import ReActRunner
        target = MagicMock()
        target.run.return_value = RunResult(success=True, final_result="retry result")
        runner = ReflectionRunner(target_runner=target, max_reflections=2)
        runner._client = MagicMock()
        runner.request_base_info = MagicMock()
        runner.request_base_info.modules = "test-model"
        runner._client.post_chat_completions.return_value = MagicMock(
            ok=True,
            json={"choices": [{"message": {"content": '{"is_finished": false, "feedback": "needs work"}'}}]},
            text='{"is_finished": false}',
        )
        initial_result = RunResult(success=True, final_result="initial")
        result = runner.reflect("test question", initial_result)
        self.assertTrue(target.run.call_count >= 1)


if __name__ == "__main__":
    unittest.main()
