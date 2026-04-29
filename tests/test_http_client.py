from __future__ import annotations

import unittest

from gaoagent.core.runner.HttpClient import OpenAICompatibleHttpClient


class TestBuildChatCompletionsUrl(unittest.TestCase):
    def _make_client(self, base_url: str) -> OpenAICompatibleHttpClient:
        return OpenAICompatibleHttpClient(base_url=base_url, api_key="test")

    def test_plain_domain(self) -> None:
        client = self._make_client("https://api.example.com")
        self.assertEqual(client._build_chat_completions_url(), "https://api.example.com/v1/chat/completions")

    def test_with_v1_suffix(self) -> None:
        client = self._make_client("https://api.example.com/v1")
        self.assertEqual(client._build_chat_completions_url(), "https://api.example.com/v1/chat/completions")

    def test_with_chat_completions_suffix(self) -> None:
        client = self._make_client("https://api.example.com/v1/chat/completions")
        self.assertEqual(client._build_chat_completions_url(), "https://api.example.com/v1/chat/completions")

    def test_trailing_slash(self) -> None:
        client = self._make_client("https://api.example.com/v1/")
        self.assertEqual(client._build_chat_completions_url(), "https://api.example.com/v1/chat/completions")

    def test_extra_whitespace(self) -> None:
        client = self._make_client("  https://api.example.com  ")
        self.assertEqual(client._build_chat_completions_url(), "https://api.example.com/v1/chat/completions")


class TestHttpResponse(unittest.TestCase):
    def test_default_fields(self) -> None:
        from gaoagent.core.runner.HttpClient import HttpResponse
        resp = HttpResponse(ok=True)
        self.assertTrue(resp.ok)
        self.assertIsNone(resp.status)
        self.assertIsNone(resp.reason)
        self.assertIsNone(resp.json)
        self.assertIsNone(resp.text)

    def test_with_fields(self) -> None:
        from gaoagent.core.runner.HttpClient import HttpResponse
        resp = HttpResponse(ok=False, status=500, reason="error", json={"msg": "fail"}, text="fail")
        self.assertFalse(resp.ok)
        self.assertEqual(resp.status, 500)


if __name__ == "__main__":
    unittest.main()
