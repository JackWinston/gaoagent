import unittest
from unittest.mock import patch


from gaoagent.core.runner.BaseRunner import RequestBaseInfo, RunnerConfig, RunnerContext
from gaoagent.core.runner.HttpClient import HttpResponse
from gaoagent.core.runner.ReActRunner import ReActRunner


def _stream_payload_with_message(message: dict) -> dict:
    return {
        "object": "chat.completion",
        "created": 0,
        "model": "mock",
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "stop",
            }
        ],
    }


class TestInvalidLLMRetry(unittest.TestCase):
    def _runner(self, *, retries: int) -> ReActRunner:
        runner = ReActRunner(config=RunnerConfig(llm_invalid_retry=retries))
        runner.request_base_info = RequestBaseInfo(
            baseurl="http://mock",
            api_key="mock",
            modules="mock",
        )
        return runner

    def test_retries_on_empty_stream_message_then_succeeds(self):
        invalid = HttpResponse(ok=True, status=200, json=_stream_payload_with_message({"role": "assistant"}))
        valid = HttpResponse(
            ok=True,
            status=200,
            json=_stream_payload_with_message(
                {"role": "assistant", "content": '{"type":"final","content":"ok"}'}
            ),
        )

        runner = self._runner(retries=2)
        ctx = RunnerContext(step=1, history=[{"role": "user", "content": "hi"}])

        with patch(
            "gaoagent.core.runner.ReActRunner.OpenAICompatibleHttpClient.post_chat_completions",
            side_effect=[invalid, valid],
        ) as mocked:
            out = runner._callLLM(ctx)

        self.assertEqual(mocked.call_count, 2)
        self.assertEqual(out.decision, "final")
        self.assertEqual(out.content, "ok")

    def test_no_retry_when_disabled(self):
        invalid = HttpResponse(ok=True, status=200, json=_stream_payload_with_message({"role": "assistant"}))
        runner = self._runner(retries=0)
        ctx = RunnerContext(step=1, history=[{"role": "user", "content": "hi"}])

        with patch(
            "gaoagent.core.runner.ReActRunner.OpenAICompatibleHttpClient.post_chat_completions",
            return_value=invalid,
        ) as mocked:
            out = runner._callLLM(ctx)

        self.assertEqual(mocked.call_count, 1)
        self.assertEqual(out.decision, "final")
        self.assertEqual(out.content, "LLM 未返回可执行 tool_call，也未返回文本结果")

    def test_exhaust_retries_returns_retry_failure_message(self):
        invalid = HttpResponse(ok=True, status=200, json=_stream_payload_with_message({"role": "assistant"}))
        runner = self._runner(retries=1)
        ctx = RunnerContext(step=1, history=[{"role": "user", "content": "hi"}])

        with patch(
            "gaoagent.core.runner.ReActRunner.OpenAICompatibleHttpClient.post_chat_completions",
            side_effect=[invalid, invalid],
        ) as mocked:
            out = runner._callLLM(ctx)

        self.assertEqual(mocked.call_count, 2)
        self.assertEqual(out.decision, "final")
        self.assertIn("已自动重试仍失败", out.content or "")


if __name__ == "__main__":
    unittest.main()

