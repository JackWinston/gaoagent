from __future__ import annotations

import json
import urllib.request
import urllib.error
import re
import math
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from pathlib import Path
from typing import Any

from gaoagent.rag.RagApiConfig import RagApiConfigStore
from gaoagent.rag.RagChromaIndexer import RagChromaIndexer, RagChromaIndexerConfig
from gaoagent.rag.RagStorePath import (
    resolve_chroma_store_dir,
    resolve_index_meta_file,
    resolve_bm25_index_db,
)
from gaoagent.core.runner.Utils import _find_config_file


class RagChromaRetriever:
    """RAG 知识库检索器。

    职责:
    - 根据查询语句从指定知识库检索最相关切片。
    - 自动兼容本地 embedding 与远程 embedding 两种检索模式。
    - 对常见索引损坏场景（如 HNSW 加载失败）返回可读错误。
    """
    def __init__(self, kb_name: str) -> None:
        """初始化检索器并解析知识库相关路径。"""
        self.kb_name = kb_name
        self.rag_dir = _find_config_file("rag").resolve()
        self.kb_dir = self.rag_dir / kb_name
        self.store_dir = resolve_chroma_store_dir(kb_dir=self.kb_dir, kb_name=kb_name)
        self.meta_file = resolve_index_meta_file(kb_dir=self.kb_dir, kb_name=kb_name)
        self.bm25_db = resolve_bm25_index_db(kb_dir=self.kb_dir, kb_name=kb_name)

    def search(self, query: str, top_k: int = 3, score_threshold: float = 0.0) -> dict[str, Any]:
        """
        执行混合检索主流程（向量召回 + BM25 召回 + RRF 融合）。

        架构定位:
        - 当前方法是 `RagChromaRetriever` 的统一编排入口，负责把“多路召回 + 融合排序”
          串成单次检索调用，对上层 CLI/Tooling 暴露稳定返回结构。

        调用链:
        - 第一步：并行触发 `_vector_recall_entry()` 与 `_bm25_recall_entry()` 两路召回；
        - 第二步：向量路内部调用 `_vector_recall()`，关键词路内部调用 `_bm25_recall()`；
        - 第三步：`_rrf_fuse()` 按 RRF 进行排名融合，输出统一候选列表。

        职责边界:
        - 负责：collection 打开、两路召回调度、融合结果组装、异常收敛。
        - 不负责：分片构建、向量入库、索引元信息写入（这些由 Indexer 负责）。
        
        参数:
        - query: 用户的提问或检索词。
        - top_k: 最终返回的最大文档切片数。
        - score_threshold: 分数阈值 (针对 L2 距离，越小越相似)。
        
        返回:
        - 包含 success, error 或 items 列表的字典。
        """
        if not self.kb_dir.exists() or not self.kb_dir.is_dir():
            return {"success": False, "error": f"知识库不存在：{self.kb_name}"}
        if not self.meta_file.exists():
            return {"success": False, "error": f"知识库索引不完整：{self.kb_name}"}

        try:
            meta = json.loads(self.meta_file.read_text(encoding="utf-8"))
            collection_name = self._sanitize_collection_name(meta.get("kb_name", self.kb_name))
            embedding_mode = meta.get("embedding_mode", "local")
            embedding_model = meta.get("embedding_model", "all-MiniLM-L6-v2")
            store_dir = self.store_dir
            if not store_dir.exists() or not store_dir.is_dir():
                return {"success": False, "error": f"知识库索引不完整：{self.kb_name}"}
            bm25_ok, bm25_reason = self._ensure_bm25_index_ready()
            if not bm25_ok:
                return {"success": False, "error": bm25_reason}
            recall_top_k = max(top_k * 2, top_k)
            with ThreadPoolExecutor(max_workers=2) as executor:
                vector_future = executor.submit(
                    self._vector_recall_entry,
                    collection_name=collection_name,
                    embedding_mode=embedding_mode,
                    embedding_model=embedding_model,
                    store_dir=store_dir,
                    query=query,
                    top_k=recall_top_k,
                    score_threshold=score_threshold,
                )
                bm25_future = executor.submit(
                    self._bm25_recall_entry,
                    collection_name=collection_name,
                    embedding_mode=embedding_mode,
                    embedding_model=embedding_model,
                    store_dir=store_dir,
                    query=query,
                    top_k=recall_top_k,
                )
                vector_items = vector_future.result()
                bm25_items = bm25_future.result()

            items = self._rrf_fuse(
                vector_items=vector_items,
                bm25_items=bm25_items,
                top_k=top_k,
            )
            return {"success": True, "items": items, "kb_name": self.kb_name}

        except Exception as e:
            reason = str(e)
            if self._is_hnsw_index_error(reason):
                return {
                    "success": False,
                    "error": (
                        f"知识库索引损坏（HNSW 加载失败）：{self.kb_name}；"
                        "请重建该知识库索引（清理 store_dir 后重新入库）"
                    ),
                }
            return {"success": False, "error": reason}

    def _ensure_bm25_index_ready(self) -> tuple[bool, str]:
        """
        确保 BM25 索引文件可用；缺失或损坏时自动触发一次内部 update。
        """
        if self._is_bm25_sqlite_ready():
            return (True, "")

        ok, reason = self._run_internal_update()
        if not ok:
            return (False, f"BM25 索引缺失且内部更新失败：{reason}")

        if not self._is_bm25_sqlite_ready():
            return (False, f"BM25 索引仍不可用：{self.bm25_db}")
        return (True, "")

    def _run_internal_update(self) -> tuple[bool, str]:
        """
        内部执行一次增量更新（等价于 `rag update <kb_name>` 的核心入库流程）。
        """
        try:
            config_store = RagApiConfigStore(kb_name=self.kb_name, kb_dir=self.kb_dir)
            indexer_config: RagChromaIndexerConfig = config_store.resolve_indexer_config(
                local_embedding_model="all-MiniLM-L6-v2",
                chunk_size=1200,
                chunk_overlap=200,
                batch_size=64,
            )
            indexer = RagChromaIndexer(indexer_config)
            return indexer.update_knowledge_base(kb_name=self.kb_name, kb_dir=self.kb_dir)
        except Exception as e:
            return (False, str(e))

    def _vector_recall_entry(
        self,
        *,
        collection_name: str,
        embedding_mode: str,
        embedding_model: str,
        store_dir: Path,
        query: str,
        top_k: int,
        score_threshold: float,
    ) -> list[dict[str, Any]]:
        """向量召回并行入口：在线程内独立打开 collection 后执行向量检索。"""
        collection, remote_cfg = self._open_collection(
            collection_name=collection_name,
            embedding_mode=embedding_mode,
            embedding_model=embedding_model,
            store_dir=store_dir,
        )
        return self._vector_recall(
            collection=collection,
            query=query,
            top_k=top_k,
            score_threshold=score_threshold,
            embedding_mode=embedding_mode,
            remote_cfg=remote_cfg,
        )

    def _bm25_recall_entry(
        self,
        *,
        collection_name: str,
        embedding_mode: str,
        embedding_model: str,
        store_dir: Path,
        query: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        """BM25 召回并行入口：在线程内独立打开 collection 后执行 BM25。"""
        collection, _ = self._open_collection(
            collection_name=collection_name,
            embedding_mode=embedding_mode,
            embedding_model=embedding_model,
            store_dir=store_dir,
        )
        return self._bm25_recall(
            collection=collection,
            query=query,
            top_k=top_k,
        )

    def _open_collection(
        self,
        *,
        collection_name: str,
        embedding_mode: str,
        embedding_model: str,
        store_dir: Path,
    ) -> tuple[Any, dict[str, Any] | None]:
        """打开 Chroma collection，并返回远程模式下 query 向量化所需配置。

        说明:
        - local 模式：绑定本地 embedding function，让 Chroma 在 query 时自动向量化；
        - remote 模式：仅打开 collection，本方法额外返回远程 embedding 请求参数，
          供 `_vector_recall()` 在 query 前先拿到 query embedding。
        """
        try:
            from chromadb import PersistentClient
        except ImportError as e:
            raise RuntimeError(f"导入 chromadb 失败：{e}") from e

        client = PersistentClient(path=str(store_dir))
        if embedding_mode == "remote":
            config_store = RagApiConfigStore(kb_name=self.kb_name, kb_dir=self.kb_dir)
            cfg = config_store.resolve_indexer_config(
                local_embedding_model=embedding_model,
                chunk_size=1000,
                chunk_overlap=0,
                batch_size=1,
            )
            if not cfg.remote_base_url or not cfg.remote_api_key:
                raise RuntimeError("远程 Embedding 配置缺失")
            collection = client.get_collection(name=collection_name)
            return (
                collection,
                {
                    "remote_base_url": cfg.remote_base_url,
                    "remote_api_key": cfg.remote_api_key,
                    "remote_embedding_model": cfg.remote_embedding_model,
                    "remote_timeout_sec": cfg.remote_timeout_sec,
                },
            )

        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

        embedding_fn = SentenceTransformerEmbeddingFunction(model_name=embedding_model)
        collection = client.get_collection(name=collection_name, embedding_function=embedding_fn)
        return (collection, None)

    def _vector_recall(
        self,
        *,
        collection: Any,
        query: str,
        top_k: int,
        score_threshold: float,
        embedding_mode: str,
        remote_cfg: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """执行向量召回并规范化输出结构。

        结果语义:
        - `distance/vector_distance`：向量检索原始距离（越小越相似）；
        - `bm25_score`：该阶段固定为 0，仅在 BM25 召回阶段赋值；
        - `retrieval_source`：标记当前候选来源为 `vector`。
        """
        if embedding_mode == "remote":
            if remote_cfg is None:
                raise RuntimeError("远程检索缺少配置")
            query_embedding = self._embed_remote(
                query,
                str(remote_cfg["remote_base_url"]),
                str(remote_cfg["remote_api_key"]),
                str(remote_cfg["remote_embedding_model"]),
                int(remote_cfg["remote_timeout_sec"]),
            )
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
            )
        else:
            results = collection.query(
                query_texts=[query],
                n_results=top_k,
            )

        docs = results.get("documents", [[]])[0] if results.get("documents") else []
        metas = results.get("metadatas", [[]])[0] if results.get("metadatas") else []
        distances = results.get("distances", [[]])[0] if results.get("distances") else []
        ids = results.get("ids", [[]])[0] if results.get("ids") else []

        items: list[dict[str, Any]] = []
        for doc, meta, dist, cid in zip(docs, metas, distances, ids):
            dist_value = float(dist)
            if score_threshold > 0 and dist_value > score_threshold:
                continue
            items.append(
                {
                    "id": str(cid),
                    "document": str(doc),
                    "metadata": dict(meta) if isinstance(meta, dict) else {},
                    "distance": dist_value,
                    "vector_distance": dist_value,
                    "bm25_score": 0.0,
                    "score": 0.0,
                    "retrieval_source": "vector",
                }
            )
        return items

    def _bm25_recall(self, *, collection: Any, query: str, top_k: int) -> list[dict[str, Any]]:
        """执行 BM25 关键词召回，返回按 BM25 分数降序的候选。

        数据来源:
        - 仅使用持久化 BM25 索引数据库（`bm25_index.db`）；
        - 不再回退到 Chroma 现算，索引缺失/损坏将直接报错。

        结果语义:
        - `bm25_score`：BM25 原始相关性分数（越大越相关）；
        - `distance/vector_distance`：该阶段固定为 0；
        - `retrieval_source`：标记当前候选来源为 `bm25`。
        """
        _ = collection  # 保留参数以维持方法签名兼容；严格模式下不使用 collection 回退。
        return self._bm25_recall_from_sqlite(query=query, top_k=top_k)

    def _ensure_bm25_sqlite_required(self) -> None:
        """严格校验 BM25 SQLite；缺失或损坏时直接抛错。"""
        if not self._is_bm25_sqlite_ready():
            raise RuntimeError(
                f"BM25 索引不可用：{self.bm25_db}；请执行 `gaoagent rag update {self.kb_name}` 重建双索引"
            )

    def _is_bm25_sqlite_ready(self) -> bool:
        """检测 BM25 SQLite 是否存在且包含必要表。"""
        if not self.bm25_db.exists() or not self.bm25_db.is_file():
            return False
        try:
            with sqlite3.connect(str(self.bm25_db)) as conn:
                rows = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('bm25_docs', 'bm25_terms', 'bm25_meta')"
                ).fetchall()
                names = {str(r[0]) for r in rows}
                return {"bm25_docs", "bm25_terms", "bm25_meta"}.issubset(names)
        except Exception:
            return False

    def _bm25_recall_from_sqlite(self, *, query: str, top_k: int) -> list[dict[str, Any]]:
        """基于持久化 BM25 SQLite 执行召回。"""
        self._ensure_bm25_sqlite_required()
        with sqlite3.connect(str(self.bm25_db)) as conn:
            row = conn.execute("SELECT COUNT(*), COALESCE(AVG(doc_len), 0.0) FROM bm25_docs").fetchone()
            doc_count = int(row[0]) if row is not None else 0
            avgdl = float(row[1]) if row is not None else 0.0
            if doc_count <= 0 or avgdl <= 0.0:
                return []

            query_tokens = self._tokenize_for_bm25(query)
            if not query_tokens:
                return []

            query_tf = Counter(query_tokens)
            k1 = 1.2
            b = 0.75
            k3 = 1.0
            scores: dict[str, float] = {}
            candidate_ids: set[str] = set()
            postings_by_term: dict[str, list[tuple[str, int]]] = {}
            for token in query_tf.keys():
                rows = conn.execute(
                    "SELECT doc_id, tf FROM bm25_terms WHERE term = ?",
                    (token,),
                ).fetchall()
                parsed = [(str(r[0]), int(r[1])) for r in rows]
                postings_by_term[token] = parsed
                for doc_id, _ in parsed:
                    candidate_ids.add(doc_id)

            if not candidate_ids:
                return []

            placeholders = ",".join(["?"] * len(candidate_ids))
            doc_rows = conn.execute(
                f"SELECT doc_id, doc_len, document, metadata_json FROM bm25_docs WHERE doc_id IN ({placeholders})",
                list(candidate_ids),
            ).fetchall()
            doc_map: dict[str, tuple[int, str, str]] = {
                str(r[0]): (int(r[1]), str(r[2]), str(r[3])) for r in doc_rows
            }

            for token, qf in query_tf.items():
                token_postings = postings_by_term.get(token, [])
                token_df = len(token_postings)
                if token_df <= 0:
                    continue
                idf = math.log(((doc_count - token_df + 0.5) / (token_df + 0.5)) + 1.0)
                qtf_weight = ((k3 + 1.0) * qf) / (k3 + qf)
                for doc_id, token_tf in token_postings:
                    doc_entry = doc_map.get(doc_id)
                    if doc_entry is None:
                        continue
                    dl = float(doc_entry[0])
                    if dl <= 0:
                        continue
                    norm_k = k1 * (1.0 - b + b * (dl / avgdl))
                    tf_weight = ((k1 + 1.0) * token_tf) / (norm_k + token_tf)
                    scores[doc_id] = float(scores.get(doc_id, 0.0)) + (idf * tf_weight * qtf_weight)

            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            items: list[dict[str, Any]] = []
            for doc_id, score in ranked[:top_k]:
                doc_entry = doc_map.get(doc_id)
                if doc_entry is None:
                    continue
                metadata_json = doc_entry[2]
                try:
                    metadata = json.loads(metadata_json)
                except Exception:
                    metadata = {}
                items.append(
                    {
                        "id": doc_id,
                        "document": doc_entry[1],
                        "metadata": metadata if isinstance(metadata, dict) else {},
                        "distance": 0.0,
                        "vector_distance": 0.0,
                        "bm25_score": float(score),
                        "score": float(score),
                        "retrieval_source": "bm25",
                    }
                )
            return items

    def _rrf_fuse(
        self,
        *,
        vector_items: list[dict[str, Any]],
        bm25_items: list[dict[str, Any]],
        top_k: int,
        rrf_k: int = 60,
    ) -> list[dict[str, Any]]:
        """执行 RRF 融合，输出最终排序候选。

        融合原则:
        - 仅使用“名次”而非原始分值做融合，降低不同检索器分数尺度不一致的问题；
        - 同一 chunk 同时出现在两路召回时会被累积加权，通常得到更稳定的前排结果。

        输出补充:
        - `score` 对齐为 `rrf_score`，便于上层统一读取融合排序分数；
        - `retrieval_source/retrieval_sources` 标识命中来源（vector/bm25/两者）。
        """
        merged: dict[str, dict[str, Any]] = {}

        for rank, item in enumerate(vector_items, start=1):
            cid = str(item.get("id") or "")
            if not cid:
                continue
            current = merged.setdefault(
                cid,
                {
                    "id": cid,
                    "document": item.get("document", ""),
                    "metadata": item.get("metadata", {}),
                    "distance": float(item.get("distance", 0.0) or 0.0),
                    "vector_distance": float(item.get("vector_distance", item.get("distance", 0.0)) or 0.0),
                    "bm25_score": 0.0,
                    "score": 0.0,
                    "rrf_score": 0.0,
                    "retrieval_sources": set(),
                },
            )
            current["rrf_score"] = float(current["rrf_score"]) + (1.0 / (rrf_k + rank))
            current["retrieval_sources"].add("vector")

        for rank, item in enumerate(bm25_items, start=1):
            cid = str(item.get("id") or "")
            if not cid:
                continue
            current = merged.setdefault(
                cid,
                {
                    "id": cid,
                    "document": item.get("document", ""),
                    "metadata": item.get("metadata", {}),
                    "distance": 0.0,
                    "vector_distance": 0.0,
                    "bm25_score": 0.0,
                    "score": 0.0,
                    "rrf_score": 0.0,
                    "retrieval_sources": set(),
                },
            )
            if not current.get("document"):
                current["document"] = item.get("document", "")
            if not current.get("metadata"):
                current["metadata"] = item.get("metadata", {})
            current["bm25_score"] = max(float(current.get("bm25_score", 0.0)), float(item.get("bm25_score", 0.0)))
            current["rrf_score"] = float(current["rrf_score"]) + (1.0 / (rrf_k + rank))
            current["retrieval_sources"].add("bm25")

        fused = list(merged.values())
        fused.sort(key=lambda x: float(x.get("rrf_score", 0.0)), reverse=True)
        final_items: list[dict[str, Any]] = []
        for item in fused[:top_k]:
            sources = item.get("retrieval_sources", set())
            item["retrieval_source"] = "+".join(sorted(sources)) if isinstance(sources, set) else str(sources)
            item["score"] = float(item.get("rrf_score", 0.0))
            item["retrieval_sources"] = sorted(sources) if isinstance(sources, set) else [str(sources)]
            final_items.append(item)
        return final_items

    def _tokenize_for_bm25(self, text: str) -> list[str]:
        """BM25 轻量分词。

        规则:
        - 英文/数字/下划线：按词切分；
        - 中文连续片段：按单字展开。

        设计意图:
        - 在不引入额外中文分词依赖的前提下，提供可用的稀疏召回能力；
        - 若后续需要更高中文检索精度，可替换为领域分词器实现。
        """
        normalized = str(text or "").lower()
        tokens: list[str] = []
        for seg in re.findall(r"[\u4e00-\u9fff]+|[a-z0-9_]+", normalized):
            if re.fullmatch(r"[\u4e00-\u9fff]+", seg):
                tokens.extend(list(seg))
            else:
                tokens.append(seg)
        return tokens

    def _sanitize_collection_name(self, kb_name: str) -> str:
        """保持与 Indexer 一致的 sanitize 逻辑"""
        raw = kb_name.strip()
        base = re.sub(r"[^a-zA-Z0-9._-]+", "_", raw)
        base = re.sub(r"_+", "_", base)
        base = base.strip("._-")
        if not base:
            base = "default"
        base = base[:120].strip("._-")
        if not base:
            base = "default"
        return f"kb_{base}"

    def _embed_remote(self, text: str, base_url: str, api_key: str, model: str, timeout: int) -> list[float]:
        """保持与 Indexer 一致的远程请求逻辑，但只针对单条 query"""
        base = base_url.strip().rstrip("/")
        url = f"{base}/embeddings" if base.endswith("/v1") else f"{base}/v1/embeddings"
        payload = json.dumps({"model": model, "input": [text]}, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            url=url,
            method="POST",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key.strip()}",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8", errors="replace")
            data = json.loads(raw)
            items = data.get("data")
            if isinstance(items, list) and len(items) > 0:
                emb = items[0].get("embedding")
                if isinstance(emb, list):
                    return [float(x) for x in emb]
            if isinstance(data.get("embedding"), list):
                return [float(x) for x in data.get("embedding")]
            raise RuntimeError("无法解析远程 embedding 响应")
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"远程 embedding 请求失败：http={e.code}, body={body}") from e
        except Exception as e:
            raise RuntimeError(f"远程 embedding 请求失败：{e}") from e

    def _is_hnsw_index_error(self, message: str) -> bool:
        """判断异常信息是否属于 HNSW 索引加载类故障。"""
        text = str(message).lower()
        keywords = [
            "error loading hnsw index",
            "hnsw segment reader",
            "constructing hnsw segment reader",
            "creating hnsw segment reader",
            "backfill request to compactor",
        ]
        return any(k in text for k in keywords)
