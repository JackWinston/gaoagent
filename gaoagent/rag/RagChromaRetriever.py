from __future__ import annotations

import json
import urllib.request
import urllib.error
import re
from pathlib import Path
from typing import Any

from gaoagent.rag.RagApiConfig import RagApiConfigStore
from gaoagent.rag.RagStorePath import resolve_chroma_store_dir, resolve_index_meta_file
from gaoagent.core.runner.Utils import _find_config_file


class RagChromaRetriever:
    """
    RAG 知识库检索器。
    用于根据用户查询 (query) 检索指定知识库中最相关的文档切片。
    """
    def __init__(self, kb_name: str) -> None:
        self.kb_name = kb_name
        self.rag_dir = _find_config_file("rag").resolve()
        self.kb_dir = self.rag_dir / kb_name
        self.store_dir = resolve_chroma_store_dir(kb_dir=self.kb_dir, kb_name=kb_name)
        self.meta_file = resolve_index_meta_file(kb_dir=self.kb_dir, kb_name=kb_name)

    def search(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> dict[str, Any]:
        """
        执行相似度检索。
        
        参数:
        - query: 用户的提问或检索词。
        - top_k: 返回的最大文档切片数。
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

            try:
                from chromadb import PersistentClient
            except ImportError as e:
                return {"success": False, "error": f"导入 chromadb 失败：{e}"}

            client = PersistentClient(path=str(store_dir))

            if embedding_mode == "remote":
                collection = client.get_collection(name=collection_name)
                config_store = RagApiConfigStore()
                cfg = config_store.resolve_indexer_config(
                    local_embedding_model=embedding_model,
                    chunk_size=1000, chunk_overlap=0, batch_size=1
                )
                if not cfg.remote_base_url or not cfg.remote_api_key:
                    return {"success": False, "error": "远程 Embedding 配置缺失"}

                query_embedding = self._embed_remote(
                    query,
                    cfg.remote_base_url,
                    cfg.remote_api_key,
                    cfg.remote_embedding_model,
                    cfg.remote_timeout_sec
                )
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k
                )
            else:
                from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
                embedding_fn = SentenceTransformerEmbeddingFunction(model_name=embedding_model)
                collection = client.get_collection(name=collection_name, embedding_function=embedding_fn)
                results = collection.query(
                    query_texts=[query],
                    n_results=top_k
                )

            docs = results.get("documents", [[]])[0] if results.get("documents") else []
            metas = results.get("metadatas", [[]])[0] if results.get("metadatas") else []
            distances = results.get("distances", [[]])[0] if results.get("distances") else []
            ids = results.get("ids", [[]])[0] if results.get("ids") else []

            items = []
            for doc, m, dist, cid in zip(docs, metas, distances, ids):
                if score_threshold > 0 and dist > score_threshold:
                    continue
                items.append({
                    "id": cid,
                    "document": doc,
                    "metadata": m,
                    "distance": dist
                })

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
        text = str(message).lower()
        keywords = [
            "error loading hnsw index",
            "hnsw segment reader",
            "constructing hnsw segment reader",
            "creating hnsw segment reader",
            "backfill request to compactor",
        ]
        return any(k in text for k in keywords)
