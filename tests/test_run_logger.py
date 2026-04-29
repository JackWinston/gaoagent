from __future__ import annotations

import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

from gaoagent.core.runner.RunLogger import (
    RunLogger,
    _to_jsonable,
    _safe_json_dumps,
    get_current_run_logger,
    set_current_run_logger,
    reset_current_run_logger,
)


class TestToJsonable(unittest.TestCase):
    def test_primitives(self) -> None:
        self.assertEqual(_to_jsonable("str"), "str")
        self.assertEqual(_to_jsonable(42), 42)
        self.assertEqual(_to_jsonable(3.14), 3.14)
        self.assertEqual(_to_jsonable(True), True)
        self.assertIsNone(_to_jsonable(None))

    def test_dict(self) -> None:
        result = _to_jsonable({"a": 1, "b": "two"})
        self.assertEqual(result, {"a": 1, "b": "two"})

    def test_list(self) -> None:
        result = _to_jsonable([1, "two", True])
        self.assertEqual(result, [1, "two", True])

    def test_tuple(self) -> None:
        result = _to_jsonable((1, 2, 3))
        self.assertEqual(result, [1, 2, 3])

    def test_nested(self) -> None:
        data = {"a": [1, {"b": (2, 3)}]}
        result = _to_jsonable(data)
        self.assertEqual(result["a"][1]["b"], [2, 3])

    def test_non_serializable(self) -> None:
        result = _to_jsonable(object())
        self.assertIsInstance(result, str)

    def test_dataclass(self) -> None:
        from dataclasses import dataclass
        @dataclass
        class Point:
            x: int
            y: int
        result = _to_jsonable(Point(1, 2))
        self.assertEqual(result, {"x": 1, "y": 2})


class TestSafeJsonDumps(unittest.TestCase):
    def test_simple(self) -> None:
        result = _safe_json_dumps({"a": 1})
        self.assertEqual(result, '{"a": 1}')

    def test_non_serializable(self) -> None:
        result = _safe_json_dumps(object())
        self.assertIsInstance(result, str)


class TestRunLogger(unittest.TestCase):
    def test_log_event_creates_file(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"
            logger = RunLogger(log_path)
            logger.log_event("test_event", {"key": "value"}, step=1)
            self.assertTrue(log_path.exists())
            content = log_path.read_text()
            self.assertIn("test_event", content)
            self.assertIn("key", content)

    def test_log_event_without_step(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"
            logger = RunLogger(log_path)
            logger.log_event("init", {"msg": "start"})
            content = log_path.read_text()
            self.assertIn("init", content)
            self.assertNotIn("step", content)


class TestContextVarLifecycle(unittest.TestCase):
    def test_set_and_get(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = RunLogger(Path(tmpdir) / "test.log")
            token = set_current_run_logger(logger)
            self.assertIs(get_current_run_logger(), logger)
            reset_current_run_logger(token)
            self.assertIsNone(get_current_run_logger())


if __name__ == "__main__":
    unittest.main()
