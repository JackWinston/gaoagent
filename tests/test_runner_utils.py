from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from gaoagent.core.runner.utils import (
    now_ms,
    truncate_text,
    safe_json_dumps,
    summarize,
    redact,
    normalize_exception,
    parse_skill_frontmatter_with_reason,
    scan_skills_metadata,
    global_config_dir,
    load_project_registry_paths,
    try_project_root_dir,
    project_config_dir,
)


class TestNowMs(unittest.TestCase):
    def test_returns_int(self) -> None:
        result = now_ms()
        self.assertIsInstance(result, int)

    def test_returns_positive(self) -> None:
        self.assertGreater(now_ms(), 0)


class TestTruncateText(unittest.TestCase):
    def test_short_text_unchanged(self) -> None:
        self.assertEqual(truncate_text("hello", 10), "hello")

    def test_exact_limit_unchanged(self) -> None:
        self.assertEqual(truncate_text("hello", 5), "hello")

    def test_long_text_truncated(self) -> None:
        result = truncate_text("hello world", 5)
        self.assertEqual(result, "hell…")
        self.assertLessEqual(len(result), 5)

    def test_zero_limit(self) -> None:
        result = truncate_text("hello", 0)
        self.assertEqual(result, "…")

    def test_limit_one(self) -> None:
        result = truncate_text("hello", 1)
        self.assertEqual(result, "…")


class TestSafeJsonDumps(unittest.TestCase):
    def test_simple_dict(self) -> None:
        result = safe_json_dumps({"a": 1})
        self.assertEqual(result, '{"a": 1}')

    def test_sorted_keys(self) -> None:
        result = safe_json_dumps({"b": 2, "a": 1})
        self.assertEqual(result, '{"a": 1, "b": 2}')

    def test_non_serializable_falls_back_to_repr(self) -> None:
        result = safe_json_dumps(object())
        self.assertIsInstance(result, str)
        self.assertTrue(result.startswith("<object object"))

    def test_unicode(self) -> None:
        result = safe_json_dumps({"中文": "值"})
        self.assertIn("中文", result)


class TestSummarize(unittest.TestCase):
    def test_none_returns_null(self) -> None:
        self.assertEqual(summarize(None), "null")

    def test_string_within_limit(self) -> None:
        self.assertEqual(summarize("hello", 10), "hello")

    def test_string_truncated(self) -> None:
        result = summarize("hello world", 5)
        self.assertLessEqual(len(result), 5)

    def test_int_value(self) -> None:
        self.assertEqual(summarize(42), "42")

    def test_dict_value(self) -> None:
        result = summarize({"a": 1}, 20)
        self.assertIn("a", result)

    def test_default_limit(self) -> None:
        long_str = "x" * 1000
        result = summarize(long_str)
        self.assertLessEqual(len(result), 400)


class TestRedact(unittest.TestCase):
    def test_redacts_api_key(self) -> None:
        data = {"api_key": "sk-secret", "name": "test"}
        result = redact(data)
        self.assertEqual(result["api_key"], "***")
        self.assertEqual(result["name"], "test")

    def test_redacts_token(self) -> None:
        data = {"token": "abc123"}
        self.assertEqual(redact(data)["token"], "***")

    def test_redacts_password(self) -> None:
        data = {"password": "secret"}
        self.assertEqual(redact(data)["password"], "***")

    def test_case_insensitive(self) -> None:
        data = {"API_KEY": "val", "Token": "val"}
        result = redact(data)
        self.assertEqual(result["API_KEY"], "***")
        self.assertEqual(result["Token"], "***")

    def test_recursive_in_list(self) -> None:
        data = [{"api_key": "secret"}, {"name": "ok"}]
        result = redact(data)
        self.assertEqual(result[0]["api_key"], "***")
        self.assertEqual(result[1]["name"], "ok")

    def test_non_dict_passthrough(self) -> None:
        self.assertEqual(redact("hello"), "hello")
        self.assertEqual(redact(42), 42)

    def test_nested_dict(self) -> None:
        data = {"outer": {"inner": {"token": "secret"}}}
        result = redact(data)
        self.assertEqual(result["outer"]["inner"]["token"], "***")


class TestNormalizeException(unittest.TestCase):
    def test_returns_dict_with_expected_keys(self) -> None:
        try:
            raise ValueError("test error")
        except ValueError as e:
            result = normalize_exception(e)
        self.assertEqual(result["type"], "ValueError")
        self.assertEqual(result["message"], "test error")
        self.assertIn("ValueError", result["traceback"])
        self.assertIn("test error", result["traceback"])


class TestParseSkillFrontmatterWithReason(unittest.TestCase):
    def test_valid_frontmatter(self) -> None:
        content = "---\nname: myskill\ndescription: A test skill\n---\n# Body\n"
        with patch.object(Path, "read_text", return_value=content):
            meta, reason = parse_skill_frontmatter_with_reason(Path("fake"))
        self.assertIsNotNone(meta)
        self.assertEqual(meta["name"], "myskill")
        self.assertEqual(meta["description"], "A test skill")
        self.assertIsNone(reason)

    def test_missing_name(self) -> None:
        content = "---\ndescription: desc only\n---\n"
        with patch.object(Path, "read_text", return_value=content):
            meta, reason = parse_skill_frontmatter_with_reason(Path("fake"))
        self.assertIsNone(meta)
        self.assertIn("name", reason)

    def test_missing_description(self) -> None:
        content = "---\nname: test\n---\n"
        with patch.object(Path, "read_text", return_value=content):
            meta, reason = parse_skill_frontmatter_with_reason(Path("fake"))
        self.assertIsNone(meta)
        self.assertIn("description", reason)

    def test_no_frontmatter(self) -> None:
        content = "# Just a heading\nNo frontmatter here.\n"
        with patch.object(Path, "read_text", return_value=content):
            meta, reason = parse_skill_frontmatter_with_reason(Path("fake"))
        self.assertIsNone(meta)
        self.assertIn("frontmatter", reason.lower())

    def test_multiline_description(self) -> None:
        content = "---\nname: test\ndescription: >\n  First line\n  Second line\n---\n"
        with patch.object(Path, "read_text", return_value=content):
            meta, reason = parse_skill_frontmatter_with_reason(Path("fake"))
        self.assertIsNotNone(meta)
        self.assertIn("First line", meta["description"])


class TestScanSkillsMetadata(unittest.TestCase):
    def test_nonexistent_dir(self) -> None:
        skills, invalid = scan_skills_metadata(Path("/nonexistent"))
        self.assertEqual(skills, [])
        self.assertEqual(invalid, [])

    def test_empty_dir(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            skills, invalid = scan_skills_metadata(Path(tmpdir))
            self.assertEqual(skills, [])
            self.assertEqual(invalid, [])


class TestGlobalConfigDir(unittest.TestCase):
    def test_returns_path(self) -> None:
        result = global_config_dir()
        self.assertIsInstance(result, Path)
        self.assertEqual(result.name, ".gaoagent")


class TestLoadProjectRegistryPaths(unittest.TestCase):
    def test_returns_list(self) -> None:
        result = load_project_registry_paths()
        self.assertIsInstance(result, list)


class TestProjectConfigDir(unittest.TestCase):
    def test_returns_path_or_none(self) -> None:
        result = project_config_dir()
        if result is not None:
            self.assertIsInstance(result, Path)
            self.assertEqual(result.name, ".gaoagent")


if __name__ == "__main__":
    unittest.main()
