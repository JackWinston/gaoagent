from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from gaoagent.rag.rag_handlers import RagHandlers
from gaoagent.core.handler_utils import rewrite_index_meta_store_dir


class TestRagListKbNames(unittest.TestCase):
    def test_empty_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            h = RagHandlers()
            result = h._list_kb_names(Path(tmpdir))
            self.assertEqual(result, [])

    def test_with_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "kb1").mkdir()
            (Path(tmpdir) / "kb2").mkdir()
            h = RagHandlers()
            result = h._list_kb_names(Path(tmpdir))
            self.assertEqual(sorted(result), ["kb1", "kb2"])

    def test_filters_chrome_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "kb1").mkdir()
            (Path(tmpdir) / ".chrome_store").mkdir()
            h = RagHandlers()
            result = h._list_kb_names(Path(tmpdir))
            self.assertEqual(result, ["kb1"])


class TestRagRewriteIndexMetaStoreDir(unittest.TestCase):
    def test_rewrites_store_dir(self) -> None:
        from gaoagent.rag.rag_store_path import resolve_chroma_store_dir
        with tempfile.TemporaryDirectory() as tmpdir:
            kb_dir = Path(tmpdir) / "test_kb"
            kb_dir.mkdir()
            store_dir = resolve_chroma_store_dir(kb_dir=kb_dir, kb_name="test_kb")
            store_dir.mkdir(parents=True)
            meta_file = store_dir / "index_meta.json"
            meta_file.write_text('{"kb_name": "test_kb", "store_dir": "/old/path"}')
            h = RagHandlers()
            rewrite_index_meta_store_dir(kb_dir=kb_dir, kb_name="test_kb")
            import json
            data = json.loads(meta_file.read_text())
            self.assertNotEqual(data["store_dir"], "/old/path")

    def test_missing_meta_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            kb_dir = Path(tmpdir) / "test_kb"
            kb_dir.mkdir()
            h = RagHandlers()
            rewrite_index_meta_store_dir(kb_dir=kb_dir, kb_name="test_kb")


if __name__ == "__main__":
    unittest.main()
