from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from gaoagent.core.runner.base_runner import (
    RunResult, RunnerContext, RunnerConfig, StepResult, RequestBaseInfo, BaseRunner
)


class TestRunResult(unittest.TestCase):
    def test_success_defaults(self) -> None:
        r = RunResult(success=True)
        self.assertTrue(r.success)
        self.assertIsNone(r.final_result)
        self.assertIsNone(r.error)

    def test_failure(self) -> None:
        r = RunResult(success=False, error="oops")
        self.assertFalse(r.success)
        self.assertEqual(r.error, "oops")


class TestRunnerContext(unittest.TestCase):
    def test_defaults(self) -> None:
        ctx = RunnerContext(step=0)
        self.assertEqual(ctx.step, 0)
        self.assertEqual(ctx.history, [])

    def test_custom_history(self) -> None:
        ctx = RunnerContext(step=1, history=[{"role": "user", "content": "hi"}])
        self.assertEqual(len(ctx.history), 1)


class TestRunnerConfig(unittest.TestCase):
    def test_defaults(self) -> None:
        cfg = RunnerConfig()
        self.assertEqual(cfg.max_steps, 32)
        self.assertIsNone(cfg.tools)
        self.assertEqual(cfg.llm_invalid_retry, 2)

    def test_custom(self) -> None:
        tools = MagicMock()
        cfg = RunnerConfig(max_steps=10, tools=tools, llm_invalid_retry=5)
        self.assertEqual(cfg.max_steps, 10)
        self.assertEqual(cfg.tools, tools)


class TestStepResult(unittest.TestCase):
    def test_defaults(self) -> None:
        sr = StepResult(decision="final")
        self.assertEqual(sr.decision, "final")
        self.assertIsNone(sr.tool_calls)
        self.assertIsNone(sr.content)
        self.assertEqual(sr.raw, {})


class TestRequestBaseInfo(unittest.TestCase):
    def test_construction(self) -> None:
        info = RequestBaseInfo(baseurl="http://x", api_key="k", modules="m")
        self.assertEqual(info.baseurl, "http://x")
        self.assertEqual(info.context_window, 4096)


class TestBaseRunner(unittest.TestCase):
    def test_abstract_decide_raises(self) -> None:
        cfg = RunnerConfig()
        runner = BaseRunner(mode="react", runner_config=cfg)
        with self.assertRaises(NotImplementedError):
            runner.decide(RunnerContext(step=0))

    def test_abstract_run_raises(self) -> None:
        cfg = RunnerConfig()
        runner = BaseRunner(mode="react", runner_config=cfg)
        with self.assertRaises(NotImplementedError):
            runner.run("test")


if __name__ == "__main__":
    unittest.main()
