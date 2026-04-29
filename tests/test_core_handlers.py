from __future__ import annotations

import unittest
from unittest.mock import patch

from gaoagent.core.core_handlers import CoreHandlers


class TestCoreHandlers(unittest.TestCase):
    def test_init_calls_core_init(self) -> None:
        handlers = CoreHandlers()
        with patch("gaoagent.core.core_handlers.CoreInit") as mock_core_init:
            handlers.init()
            mock_core_init.assert_called_once_with()
            mock_core_init.return_value.init.assert_called_once_with()

    def test_config_calls_core_config_default(self) -> None:
        handlers = CoreHandlers()
        with patch(
            "gaoagent.core.core_handlers.CoreConfigDefault"
        ) as mock_core_config_default:
            handlers.config()
            mock_core_config_default.assert_called_once_with()
            mock_core_config_default.return_value.config.assert_called_once_with()

    def test_chat_forwards_all_args(self) -> None:
        handlers = CoreHandlers()
        with patch("gaoagent.core.core_handlers.ChatRunner") as mock_chat_runner:
            handlers.chat(
                new=True,
                prompt="你好",
                api="openai",
                model="gpt-4.1",
                context_size=20,
            )
            mock_chat_runner.assert_called_once_with()
            mock_chat_runner.return_value.run.assert_called_once_with(
                new=True,
                prompt="你好",
                api="openai",
                model="gpt-4.1",
                context_size=20,
                images=None,
            )

    def test_chat_uses_defaults(self) -> None:
        handlers = CoreHandlers()
        with patch("gaoagent.core.core_handlers.ChatRunner") as mock_chat_runner:
            handlers.chat()
            mock_chat_runner.assert_called_once_with()
            mock_chat_runner.return_value.run.assert_called_once_with(
                new=False,
                prompt=None,
                api=None,
                model=None,
                context_size=None,
                images=None,
            )

    def test_task_forwards_required_and_optional_args(self) -> None:
        handlers = CoreHandlers()
        with patch("gaoagent.core.core_handlers.TaskRunner") as mock_task_runner:
            handlers.task("写一个总结", "react", id="session-1")
            mock_task_runner.assert_called_once_with()
            mock_task_runner.return_value.run.assert_called_once_with(
                "写一个总结", "react", id="session-1", images=None
            )

    def test_task_id_none_is_forwarded(self) -> None:
        handlers = CoreHandlers()
        with patch("gaoagent.core.core_handlers.TaskRunner") as mock_task_runner:
            handlers.task("继续", "plan")
            mock_task_runner.assert_called_once_with()
            mock_task_runner.return_value.run.assert_called_once_with(
                "继续", "plan", id=None, images=None
            )


if __name__ == "__main__":
    unittest.main()
