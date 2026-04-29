from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from gaoagent.skills.SkillsHandlers import SkillsHandlers


class TestSkillsResolveScopeAndPaths(unittest.TestCase):
    def test_returns_tuple(self) -> None:
        h = SkillsHandlers()
        result = h._resolve_scope_and_paths()
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        scope, path = result
        self.assertIn(scope, ["项目", "全局"])
        self.assertIsInstance(path, Path)


class TestSkillsCopyDir(unittest.TestCase):
    def test_copy_new_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "src"
            dst = Path(tmpdir) / "dst"
            src.mkdir()
            (src / "file.txt").write_text("hello")
            h = SkillsHandlers()
            h._copy_dir(src, dst)
            self.assertTrue((dst / "file.txt").exists())
            self.assertEqual((dst / "file.txt").read_text(), "hello")

    def test_copy_overwrites_existing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "src"
            dst = Path(tmpdir) / "dst"
            src.mkdir()
            dst.mkdir()
            (src / "new.txt").write_text("new")
            (dst / "old.txt").write_text("old")
            h = SkillsHandlers()
            h._copy_dir(src, dst)
            self.assertTrue((dst / "new.txt").exists())
            self.assertFalse((dst / "old.txt").exists())


if __name__ == "__main__":
    unittest.main()
