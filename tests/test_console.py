from __future__ import annotations

import unittest
from unittest.mock import patch, MagicMock

from gaoagent.core.runner.Console import Console, _is_debug_enabled


class TestConsoleMethods(unittest.TestCase):
    @patch("gaoagent.core.runner.Console.click.echo")
    def test_echo(self, mock_echo) -> None:
        Console.echo("hello")
        mock_echo.assert_called_once_with("hello", err=False, nl=True)

    @patch("gaoagent.core.runner.Console.click.echo")
    def test_info(self, mock_echo) -> None:
        Console.info("info msg")
        mock_echo.assert_called_once()

    @patch("gaoagent.core.runner.Console.click.echo")
    def test_error_uses_stderr(self, mock_echo) -> None:
        Console.error("err msg")
        args, kwargs = mock_echo.call_args
        self.assertTrue(kwargs.get("err"))

    @patch("gaoagent.core.runner.Console.click.echo")
    def test_fatal_uses_stderr(self, mock_echo) -> None:
        Console.fatal("fatal msg")
        args, kwargs = mock_echo.call_args
        self.assertTrue(kwargs.get("err"))

    @patch("gaoagent.core.runner.Console.click.echo")
    def test_warn(self, mock_echo) -> None:
        Console.warn("warning")
        mock_echo.assert_called_once()

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
