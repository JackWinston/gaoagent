from __future__ import annotations

import unittest
from unittest.mock import patch, MagicMock

from gaoagent.core.runner.Console import Console, _is_debug_enabled


class TestConsoleMethods(unittest.TestCase):
    @patch("gaoagent.core.runner.Console._rich")
    def test_echo(self, mock_rich) -> None:
        Console.echo("hello")
        mock_rich.print.assert_called_once_with("hello", end="\n")

    @patch("gaoagent.core.runner.Console._rich")
    def test_info(self, mock_rich) -> None:
        Console.info("info msg")
        mock_rich.print.assert_called_once_with("info msg")

    @patch("gaoagent.core.runner.Console._rich")
    def test_error_uses_stderr(self, mock_rich) -> None:
        stderr_at_call_time = []
        mock_rich.print.side_effect = lambda *a, **kw: stderr_at_call_time.append(mock_rich.stderr)
        Console.error("err msg")
        mock_rich.print.assert_called_once_with("[error]err msg[/error]")
        self.assertEqual(stderr_at_call_time, [True])

    @patch("gaoagent.core.runner.Console._rich")
    def test_fatal_uses_stderr(self, mock_rich) -> None:
        stderr_at_call_time = []
        mock_rich.print.side_effect = lambda *a, **kw: stderr_at_call_time.append(mock_rich.stderr)
        Console.fatal("fatal msg")
        mock_rich.print.assert_called_once_with("[fatal]fatal msg[/fatal]")
        self.assertEqual(stderr_at_call_time, [True])

    @patch("gaoagent.core.runner.Console._rich")
    def test_warn(self, mock_rich) -> None:
        Console.warn("warning")
        mock_rich.print.assert_called_once_with("[warning]warning[/warning]")

    def test_interaction_text_returns_string(self) -> None:
        result = Console.interaction_text("test")
        self.assertIsInstance(result, str)


class TestIsDebugEnabled(unittest.TestCase):
    @patch("gaoagent.core.runner.Console.global_config_dir")
    def test_no_env_file(self, mock_dir) -> None:
        mock_dir.return_value = MagicMock()
        mock_dir.return_value.__truediv__ = lambda self, x: MagicMock(exists=lambda: False)
        self.assertFalse(_is_debug_enabled())


if __name__ == "__main__":
    unittest.main()
