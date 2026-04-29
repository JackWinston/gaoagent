from __future__ import annotations

import unittest

from gaoagent.rag.rag_chroma_retriever import RagChromaRetriever
from gaoagent.core.handler_utils import tokenize_for_bm25, sanitize_collection_name


class TestRrfFuse(unittest.TestCase):
    def _make_item(self, doc_id: str, doc: str = "text") -> dict:
        return {"id": doc_id, "document": doc, "metadata": {}, "distance": 0.0,
                "vector_distance": 0.0, "bm25_score": 0.0, "score": 0.0}

    def test_empty_inputs(self) -> None:
        result = RagChromaRetriever._rrf_fuse(None, vector_items=[], bm25_items=[], top_k=5)
        self.assertEqual(result, [])

    def test_only_vector_items(self) -> None:
        items = [self._make_item("a"), self._make_item("b")]
        result = RagChromaRetriever._rrf_fuse(None, vector_items=items, bm25_items=[], top_k=5)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["id"], "a")

    def test_only_bm25_items(self) -> None:
        items = [self._make_item("x"), self._make_item("y")]
        result = RagChromaRetriever._rrf_fuse(None, vector_items=[], bm25_items=items, top_k=5)
        self.assertEqual(len(result), 2)

    def test_overlap_items_get_higher_score(self) -> None:
        v_items = [self._make_item("a"), self._make_item("b")]
        b_items = [self._make_item("b"), self._make_item("c")]
        result = RagChromaRetriever._rrf_fuse(None, vector_items=v_items, bm25_items=b_items, top_k=5)
        ids = [r["id"] for r in result]
        self.assertIn("a", ids)
        self.assertIn("b", ids)
        self.assertIn("c", ids)
        b_item = next(r for r in result if r["id"] == "b")
        self.assertIn("vector", b_item["retrieval_sources"])
        self.assertIn("bm25", b_item["retrieval_sources"])

    def test_top_k_limits_output(self) -> None:
        items = [self._make_item(f"i{i}") for i in range(10)]
        result = RagChromaRetriever._rrf_fuse(None, vector_items=items, bm25_items=[], top_k=3)
        self.assertEqual(len(result), 3)

    def test_score_is_rrf_score(self) -> None:
        items = [self._make_item("a")]
        result = RagChromaRetriever._rrf_fuse(None, vector_items=items, bm25_items=[], top_k=5)
        self.assertAlmostEqual(result[0]["score"], result[0]["rrf_score"])


class TestTokenizeForBm25(unittest.TestCase):
    def test_english_words(self) -> None:
        tokens = tokenize_for_bm25("Hello World Test")
        self.assertEqual(tokens, ["hello", "world", "test"])

    def test_chinese_chars(self) -> None:
        tokens = tokenize_for_bm25("你好世界")
        self.assertEqual(tokens, ["你", "好", "世", "界"])

    def test_mixed(self) -> None:
        tokens = tokenize_for_bm25("hello你好world")
        self.assertEqual(tokens, ["hello", "你", "好", "world"])

    def test_empty(self) -> None:
        tokens = tokenize_for_bm25("")
        self.assertEqual(tokens, [])

    def test_numbers(self) -> None:
        tokens = tokenize_for_bm25("test123 abc")
        self.assertEqual(tokens, ["test123", "abc"])


class TestSanitizeCollectionName(unittest.TestCase):
    def test_normal_name(self) -> None:
        result = sanitize_collection_name("my_knowledge_base")
        self.assertEqual(result, "kb_my_knowledge_base")

    def test_special_chars(self) -> None:
        result = sanitize_collection_name("my-kb!@#name")
        self.assertTrue(result.startswith("kb_"))
        self.assertNotIn("!", result)
        self.assertNotIn("@", result)

    def test_empty_name(self) -> None:
        result = sanitize_collection_name("")
        self.assertEqual(result, "kb_default")

    def test_whitespace_only(self) -> None:
        result = sanitize_collection_name("   ")
        self.assertEqual(result, "kb_default")

    def test_long_name_truncated(self) -> None:
        long_name = "a" * 200
        result = sanitize_collection_name(long_name)
        self.assertLessEqual(len(result), 120 + 3)  # kb_ prefix


class TestIsHnswIndexError(unittest.TestCase):
    def test_hnsw_error(self) -> None:
        self.assertTrue(RagChromaRetriever._is_hnsw_index_error(None, "Error loading HNSW index"))

    def test_hnsw_segment_reader(self) -> None:
        self.assertTrue(RagChromaRetriever._is_hnsw_index_error(None, "constructing hnsw segment reader failed"))

    def test_backfill_request(self) -> None:
        self.assertTrue(RagChromaRetriever._is_hnsw_index_error(None, "backfill request to compactor"))

    def test_normal_error(self) -> None:
        self.assertFalse(RagChromaRetriever._is_hnsw_index_error(None, "file not found"))

    def test_case_insensitive(self) -> None:
        self.assertTrue(RagChromaRetriever._is_hnsw_index_error(None, "ERROR LOADING HNSW INDEX"))


if __name__ == "__main__":
    unittest.main()
