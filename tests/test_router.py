from __future__ import annotations

import unittest
from unittest.mock import MagicMock

try:
    from gaoagent.router import Route, dispatch, ROUTES
    HAS_ROUTER = True
except ImportError:
    HAS_ROUTER = False


@unittest.skipUnless(HAS_ROUTER, "router dependencies not available")
class TestRoute(unittest.TestCase):
    def test_dispatch_calls_factory_and_method(self) -> None:
        mock_handler = MagicMock()
        mock_factory = MagicMock(return_value=mock_handler)
        route = Route(factory=mock_factory, method_name="do_something")
        route.dispatch(x=1, y=2)
        mock_factory.assert_called_once()
        mock_handler.do_something.assert_called_once_with(x=1, y=2)

    def test_dispatch_with_no_kwargs(self) -> None:
        mock_handler = MagicMock()
        route = Route(factory=lambda: mock_handler, method_name="run")
        route.dispatch()
        mock_handler.run.assert_called_once_with()


@unittest.skipUnless(HAS_ROUTER, "router dependencies not available")
class TestRoutes(unittest.TestCase):
    def test_expected_keys_exist(self) -> None:
        expected = [
            "init", "config", "chat", "task",
            "mcp.list", "mcp.add", "mcp.remove", "mcp.enable", "mcp.disable", "mcp.test",
            "skills.list", "skills.add", "skills.remove",
            "rag.list", "rag.add", "rag.update", "rag.remove", "rag.search",
            "api.list", "api.add", "api.remove", "api.edit", "api.default",
            "agent.list", "agent.add", "agent.remove", "agent.serve",
        ]
        for key in expected:
            self.assertIn(key, ROUTES, f"Missing route key: {key}")


@unittest.skipUnless(HAS_ROUTER, "router dependencies not available")
class TestDispatch(unittest.TestCase):
    def test_unknown_action_raises_key_error(self) -> None:
        with self.assertRaises(KeyError):
            dispatch("nonexistent.action")

    def test_valid_action_calls_route(self) -> None:
        mock_handler = MagicMock()
        mock_factory = MagicMock(return_value=mock_handler)
        original_route = ROUTES["init"]
        try:
            ROUTES["init"] = Route(factory=mock_factory, method_name=original_route.method_name)
            dispatch("init")
            mock_factory.assert_called_once()
            mock_handler.init.assert_called_once_with()
        finally:
            ROUTES["init"] = original_route


if __name__ == "__main__":
    unittest.main()
