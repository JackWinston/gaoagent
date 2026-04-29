from __future__ import annotations

import hashlib
import unittest
from pathlib import Path

from gaoagent.core.ChatRunner import _extract_content, _trim_messages
from gaoagent.core.runner.FunctionCallProtocol import (
    http_error_to_final,
    parse_tool_arguments,
)
from gaoagent.mcp.MCPClientCompat import _sanitize_token, export_tool_name
from gaoagent.rag.RagStorePath import (
    is_internal_rag_store_dir_name,
    resolve_bm25_index_db,
    resolve_chroma_store_dir,
    resolve_index_meta_file,
)


class TestChatRunnerHelpers(unittest.TestCase):
    def test_trim_messages_keeps_system_and_latest(self) -> None:
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2"},
        ]
        trimmed = _trim_messages(messages, context_size=3)
        self.assertEqual(
            trimmed,
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "u2"},
                {"role": "assistant", "content": "a2"},
            ],
        )

    def test_trim_messages_without_system(self) -> None:
        messages = [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
        ]
        trimmed = _trim_messages(messages, context_size=2)
        self.assertEqual(
            trimmed,
            [
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "u2"},
            ],
        )

    def test_extract_content(self) -> None:
        payload = {"choices": [{"message": {"content": "hello"}}]}
        self.assertEqual(_extract_content(payload), "hello")
        self.assertIsNone(_extract_content({"choices": []}))
        self.assertIsNone(_extract_content({"choices": [{"message": {"content": " "}}]}))


class TestRagStorePathHelpers(unittest.TestCase):
    def test_resolve_paths_share_same_hash_root(self) -> None:
        kb_dir = Path.cwd() / ".gaoagent" / "rag" / "demo"
        kb_name = "  我的知识库  "
        token = hashlib.sha256("我的知识库".encode("utf-8")).hexdigest()[:24]
        expected_root = kb_dir.expanduser().resolve().parent / ".chrome_store" / f"kb_{token}"

        self.assertEqual(resolve_chroma_store_dir(kb_dir=kb_dir, kb_name=kb_name), expected_root)
        self.assertEqual(
            resolve_index_meta_file(kb_dir=kb_dir, kb_name=kb_name),
            expected_root / "index_meta.json",
        )
        self.assertEqual(
            resolve_bm25_index_db(kb_dir=kb_dir, kb_name=kb_name),
            expected_root / "bm25_index.db",
        )

    def test_internal_store_dir_name(self) -> None:
        self.assertTrue(is_internal_rag_store_dir_name(".chrome_store"))
        self.assertTrue(is_internal_rag_store_dir_name("  .chrome_store  "))
        self.assertFalse(is_internal_rag_store_dir_name(".chrome_store2"))


class TestFunctionCallProtocolHelpers(unittest.TestCase):
    def test_parse_tool_arguments(self) -> None:
        self.assertEqual(parse_tool_arguments({"a": 1}), {"a": 1})
        self.assertEqual(parse_tool_arguments('{"a": 1}'), {"a": 1})
        self.assertEqual(parse_tool_arguments("[1,2]"), {"value": [1, 2]})
        self.assertEqual(parse_tool_arguments("not-json"), {"_raw": "not-json"})
        self.assertEqual(parse_tool_arguments(None), {})

    def test_http_error_to_final(self) -> None:
        out = http_error_to_final(500, "boom", "x" * 600)
        self.assertEqual(out["type"], "final")
        self.assertIn("status=500", out["content"])
        self.assertIn("reason=boom", out["content"])


class TestMcpCompatHelpers(unittest.TestCase):
    def test_sanitize_token(self) -> None:
        self.assertEqual(_sanitize_token(" hello/world  "), "hello_world")
        self.assertEqual(_sanitize_token("$$$"), "x")

    def test_export_tool_name_has_prefix_and_limit(self) -> None:
        name = export_tool_name("server!!name", "tool name", max_len=64)
        self.assertTrue(name.startswith("mcp__"))
        self.assertLessEqual(len(name), 64)

        short_name = export_tool_name("a", "b")
        self.assertEqual(short_name, "mcp__a__b")


if __name__ == "__main__":
    unittest.main()
