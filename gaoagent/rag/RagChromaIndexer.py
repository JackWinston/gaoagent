from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
from typing import Any
from collections import Counter
import json
import re
import time
import gc
import urllib.request
import urllib.error

from gaoagent.rag.RagStorePath import (
    resolve_chroma_store_dir,
    resolve_index_meta_file,
    resolve_bm25_index_file,
)


@dataclass
class RagChromaIndexerConfig:
    """
    RAG 入库参数。

    - embedding_model: 生成向量使用的 embedding 模型名称。
    - chunk_size: 单个切片最大字符长度。
    - chunk_overlap: 相邻切片重叠长度，避免语义断裂。
    - batch_size: 批量写入向量库时每批大小。
    - remote_base_url: 远程 embedding 服务地址（OpenAI 兼容）。
    - remote_api_key: 远程 embedding 服务鉴权 key。
    - remote_embedding_model: 远程 embedding 模型名。
    - remote_timeout_sec: 远程请求超时秒数。
    - chunker_py_file: 自定义切片器 python 文件路径；为空时使用内置切片策略。
    """
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 1200
    chunk_overlap: int = 200
    batch_size: int = 64
    remote_base_url: str = ""
    remote_api_key: str = ""
    remote_embedding_model: str = ""
    remote_timeout_sec: int = 120
    chunker_py_file: str = ""


class RagChromaIndexer:
    """
    基于 ChromaDB 的知识库入库器。

    当前流程：
    1. 扫描知识库目录，仅收集 `.md/.txt` 文件；
    2. 读取文本并按配置切片；
    3. 使用 Chroma embedding_function 生成向量并持久化写入；
    4. 写入 `.chrome_store/.../index_meta.json` 记录本次索引基本信息。
    """
    _SUPPORTED_SUFFIX = {".md", ".txt"}

    def __init__(self, config: RagChromaIndexerConfig | None = None) -> None:
        """创建入库器实例。

        参数:
        - `config`: 可选入库配置；为空时使用 `RagChromaIndexerConfig` 默认值。

        说明:
        - 自定义切片器函数会在首次使用时懒加载并缓存到 `_external_chunker`。
        """
        self._config = config or RagChromaIndexerConfig()
        self._external_chunker: Any | None = None

    def ingest_knowledge_base(self, *, kb_name: str, kb_dir: Path) -> tuple[bool, str]:
        """
        对一个知识库目录执行完整入库。

        参数：
        - kb_name: 知识库名称，用于 collection 名称、chunk id 与元数据。
        - kb_dir: 知识库目录（包含待入库源文件）。

        返回：
        - (True, ""): 入库成功。
        - (False, reason): 入库失败，并返回可直接展示给用户的失败原因。
        """
        if not kb_dir.exists() or not kb_dir.is_dir():
            return (False, f"知识库目录不存在：{kb_dir}")

        # 切片步长 = chunk_size - chunk_overlap，overlap 不可大于等于 chunk_size。
        if self._config.chunk_overlap >= self._config.chunk_size:
            return (False, "chunk_overlap 必须小于 chunk_size")

        # 仅允许特定文本文件类型，避免误读二进制文件。
        source_files = self._list_source_files(kb_dir)
        if not source_files:
            return (False, "未发现可入库文件（仅支持 .md/.txt）")

        chunks: list[dict[str, Any]] = []
        for file_path in source_files:
            text = self._read_text(file_path)
            if not text:
                # 无法读取或内容为空时跳过该文件，不中断整体流程。
                print(f"无法读取或内容为空文件：{file_path}")
                continue
            try:
                chunks.extend(
                    self._chunk_document(
                        kb_name=kb_name,
                        kb_dir=kb_dir,
                        file_path=file_path,
                        text=text,
                    )
                )
            except Exception as e:
                return (False, f"文档切片失败（{file_path.name}）：{e}")

        if not chunks:
            return (False, "可入库文件内容为空")

        try:
            # 延迟导入，避免在未安装依赖时影响其他命令启动。
            from chromadb import PersistentClient
        except Exception as e:
            return (False, f"导入 chromadb 失败：{e}")

        # 将 Chroma 数据存放到 ASCII 安全路径，规避中文目录下 HNSW 文件异常。
        store_dir = resolve_chroma_store_dir(kb_dir=kb_dir, kb_name=kb_name)
        store_dir.mkdir(parents=True, exist_ok=True)
        collection_name = self._sanitize_collection_name(kb_name)

        client: Any | None = None
        try:
            client = PersistentClient(path=str(store_dir))

            use_remote = self._use_remote_embedding()
            if use_remote:
                # 远程 embedding 模式：embedding 在本地通过 HTTP 请求拿到，再显式写入 Chroma。
                collection = client.get_or_create_collection(
                    name=collection_name,
                    metadata={
                        "kb_name": kb_name,
                        "embedding_mode": "remote",
                        "embedding_model": self._config.remote_embedding_model,
                    },
                )
            else:
                # 本地 embedding 模式：由 Chroma 的 embedding_function 自动生成向量。
                try:
                    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
                except Exception as e:
                    return (False, f"导入本地 embedding 依赖失败：{e}")
                embedding_fn = SentenceTransformerEmbeddingFunction(model_name=self._config.embedding_model)
                collection = client.get_or_create_collection(
                    name=collection_name,
                    embedding_function=embedding_fn,
                    metadata={
                        "kb_name": kb_name,
                        "embedding_mode": "local",
                        "embedding_model": self._config.embedding_model,
                    },
                )

            total_chunks = len(chunks)
            total_batches = (total_chunks + self._config.batch_size - 1) // self._config.batch_size
            print(f"[RAG] 开始写入向量：total={total_chunks}, batch_size={self._config.batch_size}, batches={total_batches}")
            for i in range(0, len(chunks), self._config.batch_size):
                # 分批 add，减少单次请求体积，避免大文档时内存/请求过大。
                batch = chunks[i: i + self._config.batch_size]
                ids = [str(item["id"]) for item in batch]
                docs = [str(item["document"]) for item in batch]
                metas = [dict(item["metadata"]) for item in batch]
                if use_remote:
                    embeddings = self._embed_remote(docs)
                    collection.upsert(
                        ids=ids,
                        documents=docs,
                        metadatas=metas,
                        embeddings=embeddings,
                    )
                else:
                    collection.upsert(
                        ids=ids,
                        documents=docs,
                        metadatas=metas,
                    )
                done = min(i + len(batch), total_chunks)
                progress = (done / total_chunks * 100.0) if total_chunks > 0 else 100.0
                print(f"[RAG] 写入进度：done={done}/{total_chunks} ({progress:.2f}%)")
            final_chunk_count = int(collection.count())
        except Exception as e:
            # 尽快释放底层 sqlite 句柄，降低 Windows 清理索引目录时的占用概率。
            client = None
            gc.collect()
            return (False, f"写入向量库失败：{e}")

        self._write_index_meta(
            kb_dir=kb_dir,
            kb_name=kb_name,
            source_file_count=len(source_files),
            chunk_count=final_chunk_count,
        )
        bm25_sync_ok, bm25_reason = self._sync_bm25_index(
            kb_dir=kb_dir,
            kb_name=kb_name,
            collection=collection,
            incremental_chunks=None,
            force_rebuild=True,
        )
        if not bm25_sync_ok:
            return (False, f"写入 BM25 稀疏索引失败：{bm25_reason}")
        return (True, "")

    def update_knowledge_base(self, *, kb_name: str, kb_dir: Path) -> tuple[bool, str]:
        """
        对已存在知识库执行增量更新。

        行为:
        1. 扫描并切片当前语料；
        2. 检查向量索引与 BM25 稀疏索引是否存在；
        3. 缺失则创建，存在则仅增量插入“新 chunk”；
        4. 统一刷新 `index_meta.json`。
        """
        if not kb_dir.exists() or not kb_dir.is_dir():
            return (False, f"知识库目录不存在：{kb_dir}")
        if self._config.chunk_overlap >= self._config.chunk_size:
            return (False, "chunk_overlap 必须小于 chunk_size")

        source_files = self._list_source_files(kb_dir)
        if not source_files:
            return (False, "未发现可入库文件（仅支持 .md/.txt）")

        chunks: list[dict[str, Any]] = []
        for file_path in source_files:
            text = self._read_text(file_path)
            if not text:
                print(f"无法读取或内容为空文件：{file_path}")
                continue
            try:
                chunks.extend(
                    self._chunk_document(
                        kb_name=kb_name,
                        kb_dir=kb_dir,
                        file_path=file_path,
                        text=text,
                    )
                )
            except Exception as e:
                return (False, f"文档切片失败（{file_path.name}）：{e}")

        if not chunks:
            return (False, "可入库文件内容为空")

        try:
            from chromadb import PersistentClient
        except Exception as e:
            return (False, f"导入 chromadb 失败：{e}")

        store_dir = resolve_chroma_store_dir(kb_dir=kb_dir, kb_name=kb_name)
        store_dir.mkdir(parents=True, exist_ok=True)
        collection_name = self._sanitize_collection_name(kb_name)
        bm25_index_file = resolve_bm25_index_file(kb_dir=kb_dir, kb_name=kb_name)

        client: Any | None = None
        try:
            client = PersistentClient(path=str(store_dir))
            use_remote = self._use_remote_embedding()
            if use_remote:
                collection = client.get_or_create_collection(
                    name=collection_name,
                    metadata={
                        "kb_name": kb_name,
                        "embedding_mode": "remote",
                        "embedding_model": self._config.remote_embedding_model,
                    },
                )
            else:
                try:
                    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
                except Exception as e:
                    return (False, f"导入本地 embedding 依赖失败：{e}")
                embedding_fn = SentenceTransformerEmbeddingFunction(model_name=self._config.embedding_model)
                collection = client.get_or_create_collection(
                    name=collection_name,
                    embedding_function=embedding_fn,
                    metadata={
                        "kb_name": kb_name,
                        "embedding_mode": "local",
                        "embedding_model": self._config.embedding_model,
                    },
                )

            # 仅插入当前 collection 中不存在的新 chunk，实现增量更新。
            existing_ids: set[str] = set()
            batch_size = self._config.batch_size
            for i in range(0, len(chunks), batch_size):
                probe_ids = [str(item["id"]) for item in chunks[i: i + batch_size]]
                existing_ids.update(self._get_existing_ids_in_collection(collection=collection, ids=probe_ids))

            new_chunks = [item for item in chunks if str(item["id"]) not in existing_ids]
            if new_chunks:
                total_new = len(new_chunks)
                total_batches = (total_new + batch_size - 1) // batch_size
                print(f"[RAG] 增量写入向量：new={total_new}, batch_size={batch_size}, batches={total_batches}")
                for i in range(0, total_new, batch_size):
                    batch = new_chunks[i: i + batch_size]
                    ids = [str(item["id"]) for item in batch]
                    docs = [str(item["document"]) for item in batch]
                    metas = [dict(item["metadata"]) for item in batch]
                    if use_remote:
                        embeddings = self._embed_remote(docs)
                        collection.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)
                    else:
                        collection.upsert(ids=ids, documents=docs, metadatas=metas)
            else:
                print("[RAG] 未检测到新增 chunk，跳过向量增量写入。")

            final_chunk_count = int(collection.count())
            bm25_sync_ok, bm25_reason = self._sync_bm25_index(
                kb_dir=kb_dir,
                kb_name=kb_name,
                collection=collection,
                incremental_chunks=new_chunks,
                force_rebuild=(not bm25_index_file.exists()),
            )
            if not bm25_sync_ok:
                return (False, f"写入 BM25 稀疏索引失败：{bm25_reason}")
        except Exception as e:
            client = None
            gc.collect()
            return (False, f"增量写入向量库失败：{e}")

        self._write_index_meta(
            kb_dir=kb_dir,
            kb_name=kb_name,
            source_file_count=len(source_files),
            chunk_count=final_chunk_count,
        )
        return (True, "")

    def _list_source_files(self, kb_dir: Path) -> list[Path]:
        """递归列出知识库内可入库文件（仅 `.md/.txt`）。"""
        files: list[Path] = []
        for p in kb_dir.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in self._SUPPORTED_SUFFIX:
                continue
            files.append(p)
        # 固定排序，保证同样输入下分片与写入顺序稳定。
        files.sort(key=lambda x: str(x))
        return files

    def _read_text(self, file_path: Path) -> str:
        """
        读取文本文件内容。

        优先 utf-8，其次 utf-8-sig，再尝试 gbk，覆盖常见中文文本编码场景。
        """
        for encoding in ("utf-8", "utf-8-sig", "gbk"):
            try:
                return file_path.read_text(encoding=encoding).strip()
            except Exception:
                continue
        return ""

    def _chunk_document(self, *, kb_name: str, kb_dir: Path, file_path: Path, text: str) -> list[dict[str, Any]]:
        """
        将单个文档切片为可入库记录。

        每个切片结构包含：
        - id: 稳定主键（知识库 + 相对路径 + 序号）；
        - document: 切片文本；
        - metadata: 检索与追踪所需元数据。
        """
        if self._config.chunker_py_file.strip():
            external_chunks = self._chunk_document_external(
                kb_name=kb_name,
                kb_dir=kb_dir,
                file_path=file_path,
                text=text,
            )
            return self._normalize_external_chunks(
                kb_name=kb_name,
                kb_dir=kb_dir,
                file_path=file_path,
                chunks=external_chunks,
            )
        return self._chunk_document_default(
            kb_name=kb_name,
            kb_dir=kb_dir,
            file_path=file_path,
            text=text,
        )

    def _chunk_document_default(self, *, kb_name: str, kb_dir: Path, file_path: Path, text: str) -> list[dict[str, Any]]:
        """内置固定窗口切片实现。

        切片规则:
        - 步长为 `chunk_size - chunk_overlap`。
        - 基于知识库内相对路径构造稳定 chunk id。
        - 为每个切片补齐 `kb_name/source_file/chunk_index/start/end` 元数据。
        """
        step = self._config.chunk_size - self._config.chunk_overlap
        # 使用知识库内相对路径，避免机器绝对路径污染索引元数据。
        rel_path = str(file_path.resolve().relative_to(kb_dir.resolve()))
        chunks: list[dict[str, Any]] = []
        index = 0
        for start in range(0, len(text), step):
            content = text[start: start + self._config.chunk_size].strip()
            if not content:
                continue
            # chunk_id 采用稳定格式，便于后续幂等重建与问题排查。
            chunk_id = f"{kb_name}:{rel_path}:{index}"
            chunks.append(
                {
                    "id": chunk_id,
                    "document": content,
                    "metadata": {
                        "kb_name": kb_name,
                        "source_file": rel_path,
                        "chunk_index": index,
                        "start": start,
                        "end": start + len(content),
                    },
                }
            )
            index += 1
        return chunks

    def _chunk_document_external(self, *, kb_name: str, kb_dir: Path, file_path: Path, text: str) -> list[dict[str, Any]]:
        """调用外部自定义切片器生成切片列表。

        要求:
        - 外部函数签名需兼容 `chunk_document(...)`。
        - 返回值必须为 `list`，后续由 `_normalize_external_chunks()` 归一化。
        """
        chunker = self._load_external_chunker()
        result = chunker(
            kb_name=kb_name,
            kb_dir=kb_dir,
            file_path=file_path,
            text=text,
            chunk_size=self._config.chunk_size,
            chunk_overlap=self._config.chunk_overlap,
        )
        if not isinstance(result, list):
            raise RuntimeError("自定义切片器返回值必须是 list")
        return result

    def _load_external_chunker(self) -> Any:
        """加载并缓存外部切片器函数。

        加载规则:
        - 从 `chunker_py_file` 指向的 Python 文件动态导入模块。
        - 模块中必须存在可调用的 `chunk_document` 函数。
        """
        if self._external_chunker is not None:
            return self._external_chunker

        file_path = Path(self._config.chunker_py_file).expanduser().resolve()
        if not file_path.exists() or not file_path.is_file():
            raise RuntimeError(f"自定义切片器文件不存在：{file_path}")

        spec = importlib.util.spec_from_file_location("gaoagent_rag_custom_chunker", str(file_path))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"无法加载自定义切片器模块：{file_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        chunker = getattr(module, "chunk_document", None)
        if not callable(chunker):
            raise RuntimeError("自定义切片器必须定义可调用函数：chunk_document(...)")
        self._external_chunker = chunker
        return chunker

    def _normalize_external_chunks(
        self,
        *,
        kb_name: str,
        kb_dir: Path,
        file_path: Path,
        chunks: list[Any],
    ) -> list[dict[str, Any]]:
        """将外部切片器输出统一为标准入库结构。

        支持输入项:
        - `str`: 仅表示 document 文本。
        - `dict`: 支持显式提供 `id/document/metadata`。

        归一化策略:
        - 空文本切片会被丢弃。
        - 未提供 `id` 时自动生成稳定 id。
        - 自动补齐最小元数据字段（`kb_name/source_file/chunk_index`）。
        """
        rel_path = str(file_path.resolve().relative_to(kb_dir.resolve()))
        normalized: list[dict[str, Any]] = []
        for i, item in enumerate(chunks):
            if isinstance(item, str):
                content = item.strip()
                meta: dict[str, Any] = {}
                cid = ""
            elif isinstance(item, dict):
                content = str(item.get("document") or "").strip()
                cid = str(item.get("id") or "")
                raw_meta = item.get("metadata")
                meta = dict(raw_meta) if isinstance(raw_meta, dict) else {}
            else:
                raise RuntimeError("自定义切片器返回项仅支持 str 或 dict")

            if not content:
                continue

            chunk_id = cid or f"{kb_name}:{rel_path}:{i}"
            meta.setdefault("kb_name", kb_name)
            meta.setdefault("source_file", rel_path)
            meta.setdefault("chunk_index", i)
            normalized.append(
                {
                    "id": chunk_id,
                    "document": content,
                    "metadata": meta,
                }
            )
        return normalized

    def _sync_bm25_index(
        self,
        *,
        kb_dir: Path,
        kb_name: str,
        collection: Any,
        incremental_chunks: list[dict[str, Any]] | None,
        force_rebuild: bool,
    ) -> tuple[bool, str]:
        """
        同步 BM25 稀疏索引文件。

        策略:
        - `force_rebuild=True` 或索引文件缺失/损坏：全量重建；
        - 其余情况：对新增 chunk 执行增量插入。
        """
        bm25_file = resolve_bm25_index_file(kb_dir=kb_dir, kb_name=kb_name)
        try:
            if force_rebuild or (not bm25_file.exists()):
                records = self._load_collection_records(collection=collection)
                payload = self._build_bm25_index_payload(kb_name=kb_name, records=records)
                self._write_bm25_index(bm25_file=bm25_file, payload=payload)
                return (True, "")

            payload = self._read_bm25_index(bm25_file=bm25_file)
            if payload is None or not self._is_valid_bm25_index_payload(payload):
                records = self._load_collection_records(collection=collection)
                rebuilt = self._build_bm25_index_payload(kb_name=kb_name, records=records)
                self._write_bm25_index(bm25_file=bm25_file, payload=rebuilt)
                return (True, "")

            chunks = incremental_chunks or []
            if chunks:
                self._apply_incremental_chunks_to_bm25(payload=payload, chunks=chunks)
                self._write_bm25_index(bm25_file=bm25_file, payload=payload)
            return (True, "")
        except Exception as e:
            return (False, str(e))

    def _load_collection_records(self, *, collection: Any) -> list[dict[str, Any]]:
        """读取 collection 全量文档记录，用于构建 BM25 索引。"""
        records: list[dict[str, Any]] = []
        page_size = 512
        offset = 0
        while True:
            payload = collection.get(include=["documents", "metadatas"], limit=page_size, offset=offset)
            page_ids = payload.get("ids") or []
            page_docs = payload.get("documents") or []
            page_metas = payload.get("metadatas") or []
            if not page_ids:
                break
            for cid, doc, meta in zip(page_ids, page_docs, page_metas):
                records.append(
                    {
                        "id": str(cid),
                        "document": str(doc),
                        "metadata": dict(meta) if isinstance(meta, dict) else {},
                    }
                )
            if len(page_ids) < page_size:
                break
            offset += len(page_ids)
        return records

    def _build_bm25_index_payload(self, *, kb_name: str, records: list[dict[str, Any]]) -> dict[str, Any]:
        """基于全量记录构建 BM25 倒排索引载荷。"""
        postings: dict[str, dict[str, Any]] = {}
        doc_len: dict[str, int] = {}
        documents: dict[str, str] = {}
        metadatas: dict[str, dict[str, Any]] = {}
        total_len = 0

        for item in records:
            cid = str(item.get("id") or "")
            if not cid:
                continue
            text = str(item.get("document") or "")
            meta = item.get("metadata")
            metadata = dict(meta) if isinstance(meta, dict) else {}
            tokens = self._tokenize_for_bm25(text)
            tf = Counter(tokens)
            dl = len(tokens)
            doc_len[cid] = dl
            documents[cid] = text
            metadatas[cid] = metadata
            total_len += dl
            for token, term_tf in tf.items():
                bucket = postings.setdefault(token, {"df": 0, "docs": {}})
                bucket["docs"][cid] = int(term_tf)

        for bucket in postings.values():
            docs_map = bucket.get("docs", {})
            bucket["df"] = len(docs_map) if isinstance(docs_map, dict) else 0

        doc_count = len(doc_len)
        avgdl = (total_len / doc_count) if doc_count > 0 else 0.0
        return {
            "version": 1,
            "kb_name": kb_name,
            "doc_count": doc_count,
            "avgdl": avgdl,
            "doc_len": doc_len,
            "documents": documents,
            "metadatas": metadatas,
            "postings": postings,
        }

    def _apply_incremental_chunks_to_bm25(self, *, payload: dict[str, Any], chunks: list[dict[str, Any]]) -> None:
        """将新增 chunk 增量写入已存在的 BM25 索引载荷。"""
        doc_len = payload.setdefault("doc_len", {})
        documents = payload.setdefault("documents", {})
        metadatas = payload.setdefault("metadatas", {})
        postings = payload.setdefault("postings", {})

        for item in chunks:
            cid = str(item.get("id") or "")
            if not cid or cid in doc_len:
                continue
            text = str(item.get("document") or "")
            meta = item.get("metadata")
            metadata = dict(meta) if isinstance(meta, dict) else {}
            tokens = self._tokenize_for_bm25(text)
            tf = Counter(tokens)
            doc_len[cid] = len(tokens)
            documents[cid] = text
            metadatas[cid] = metadata
            for token, term_tf in tf.items():
                bucket = postings.setdefault(token, {"df": 0, "docs": {}})
                docs_map = bucket.setdefault("docs", {})
                if cid not in docs_map:
                    docs_map[cid] = int(term_tf)
                bucket["df"] = len(docs_map)

        doc_count = len(doc_len)
        total_len = sum(int(v) for v in doc_len.values())
        payload["doc_count"] = doc_count
        payload["avgdl"] = (total_len / doc_count) if doc_count > 0 else 0.0

    def _write_bm25_index(self, *, bm25_file: Path, payload: dict[str, Any]) -> None:
        """原子写入 BM25 索引文件。"""
        bm25_file.parent.mkdir(parents=True, exist_ok=True)
        tmp_file = bm25_file.with_name(f"{bm25_file.name}.tmp")
        tmp_file.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        tmp_file.replace(bm25_file)

    def _read_bm25_index(self, *, bm25_file: Path) -> dict[str, Any] | None:
        """读取 BM25 索引文件；读取失败返回 None。"""
        if not bm25_file.exists() or not bm25_file.is_file():
            return None
        try:
            data = json.loads(bm25_file.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else None
        except Exception:
            return None

    def _is_valid_bm25_index_payload(self, payload: dict[str, Any]) -> bool:
        """校验 BM25 索引载荷结构是否满足最小要求。"""
        required = ("doc_count", "avgdl", "doc_len", "documents", "metadatas", "postings")
        for key in required:
            if key not in payload:
                return False
        return isinstance(payload.get("postings"), dict) and isinstance(payload.get("doc_len"), dict)

    def _tokenize_for_bm25(self, text: str) -> list[str]:
        """轻量 BM25 分词：英文按词，中文按单字。"""
        normalized = str(text or "").lower()
        tokens: list[str] = []
        for seg in re.findall(r"[\u4e00-\u9fff]+|[a-z0-9_]+", normalized):
            if re.fullmatch(r"[\u4e00-\u9fff]+", seg):
                tokens.extend(list(seg))
            else:
                tokens.append(seg)
        return tokens

    def _sanitize_collection_name(self, kb_name: str) -> str:
        """将知识库名转换为 Chroma 可接受的 collection 名称。"""
        # Chroma 约束：
        # - 仅允许 [a-zA-Z0-9._-]
        # - 首尾必须是 [a-zA-Z0-9]
        raw = kb_name.strip()
        base = re.sub(r"[^a-zA-Z0-9._-]+", "_", raw)
        base = re.sub(r"_+", "_", base)
        base = base.strip("._-")
        if not base:
            base = "default"

        # 限制长度并再次确保首尾合法，避免截断后尾部变成符号。
        base = base[:120].strip("._-")
        if not base:
            base = "default"
        return f"kb_{base}"

    def _use_remote_embedding(self) -> bool:
        """判断当前配置是否启用远程 embedding 模式。"""
        return bool(
            self._config.remote_base_url.strip()
            and self._config.remote_api_key.strip()
            and self._config.remote_embedding_model.strip()
        )

    def _embed_remote(self, texts: list[str]) -> list[list[float]]:
        """
        调用 OpenAI 兼容 embedding 接口。

        请求：
        POST {remote_base_url}/embeddings
        Authorization: Bearer {remote_api_key}
        {"model": "...", "input": [...]}
        """
        url = self._build_remote_embeddings_url(self._config.remote_base_url)
        payload = json.dumps(
            {
                "model": self._config.remote_embedding_model,
                "input": texts,
            },
            ensure_ascii=False,
        ).encode("utf-8")
        request = urllib.request.Request(
            url=url,
            method="POST",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._config.remote_api_key.strip()}",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self._config.remote_timeout_sec) as response:
                raw = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"远程 embedding 请求失败：http={e.code}, body={body}") from e
        except Exception as e:
            raise RuntimeError(f"远程 embedding 请求失败：{e}") from e

        try:
            data = json.loads(raw)
        except Exception as e:
            raise RuntimeError(f"远程 embedding 响应不是合法 JSON：{e}") from e

        embeddings = self._parse_remote_embeddings(data=data, expect_count=len(texts), raw=raw)

        if len(embeddings) != len(texts):
            raise RuntimeError(
                f"远程 embedding 数量不匹配：expect={len(texts)}, actual={len(embeddings)}"
            )
        return embeddings

    def _build_remote_embeddings_url(self, base_url: str) -> str:
        """
        兼容两类 base_url：
        - https://host
        - https://host/v1
        """
        base = base_url.strip().rstrip("/")
        if base.endswith("/v1"):
            return f"{base}/embeddings"
        return f"{base}/v1/embeddings"

    def _parse_remote_embeddings(self, *, data: Any, expect_count: int, raw: str) -> list[list[float]]:
        """
        兼容多种远程 embedding 响应格式：
        1) OpenAI: {"data":[{"embedding":[...]}]}
        2) 单条向量: {"embedding":[...]}（仅当输入条数为 1）
        3) 多条向量: {"embeddings":[[...],[...]]}
        4) 直接数组: [[...],[...]]
        """
        if isinstance(data, dict):
            err = data.get("error")
            if err is not None:
                raise RuntimeError(f"远程 embedding 返回错误：{err}")

            items = data.get("data")
            if isinstance(items, list):
                return self._extract_embeddings_from_data_list(items)

            if isinstance(data.get("embeddings"), list):
                return self._to_embeddings_list(data.get("embeddings"))

            if isinstance(data.get("embedding"), list):
                if expect_count != 1:
                    raise RuntimeError(
                        f"远程 embedding 响应仅包含单条 embedding，但输入为 {expect_count} 条"
                    )
                return [self._to_single_embedding(data.get("embedding"))]

        if isinstance(data, list):
            return self._to_embeddings_list(data)

        raw_preview = raw[:500]
        raise RuntimeError(f"远程 embedding 响应格式不支持，响应片段：{raw_preview}")

    def _extract_embeddings_from_data_list(self, items: list[Any]) -> list[list[float]]:
        """解析 OpenAI 风格 `data[*].embedding` 响应结构。"""
        embeddings: list[list[float]] = []
        for item in items:
            emb = item.get("embedding") if isinstance(item, dict) else None
            if not isinstance(emb, list):
                raise RuntimeError("远程 embedding 响应格式错误：data[*].embedding 缺失")
            embeddings.append(self._to_single_embedding(emb))
        return embeddings

    def _to_embeddings_list(self, value: Any) -> list[list[float]]:
        """将远程响应中的 embedding 字段规范为二维浮点数组。"""
        if not isinstance(value, list):
            raise RuntimeError("远程 embedding 响应格式错误：embeddings 不是数组")
        if not value:
            return []
        first = value[0]
        if isinstance(first, (int, float)):
            return [self._to_single_embedding(value)]
        return [self._to_single_embedding(x) for x in value]

    def _to_single_embedding(self, value: Any) -> list[float]:
        """将单条向量转换为 `list[float]` 并校验结构。"""
        if not isinstance(value, list):
            raise RuntimeError("远程 embedding 向量格式错误：单条向量必须是数组")
        return [float(x) for x in value]

    def _get_existing_ids_in_collection(self, *, collection: Any, ids: list[str]) -> set[str]:
        """
        查询一批 id 在 collection 中是否已存在，用于估算本批“新增量”。
        """
        if not ids:
            return set()
        payload = collection.get(ids=ids, include=["metadatas"])
        raw_ids = payload.get("ids")
        if not isinstance(raw_ids, list):
            return set()
        return {str(x) for x in raw_ids if x is not None}

    def _check_vector_store_health(
        self,
        *,
        store_dir: Path,
        collection_name: str,
        expect_min_count: int,
        probe_ids: list[str] | None = None,
    ) -> tuple[bool, str]:
        """
        入库过程中的最小健康检查。

        目标：
        1) 能正常打开 collection；
        2) count 至少达到当前应完成数量；
        3) 当前批次至少有一条记录可读（优先探测本批首条 id）；
        4) 执行一次最小向量检索，提前暴露 HNSW 加载类问题。
        """
        try:
            from chromadb import PersistentClient
        except Exception as e:
            return (False, f"导入 chromadb 失败：{e}")

        try:
            client = PersistentClient(path=str(store_dir))
            collection = client.get_collection(name=collection_name)
            actual_count = int(collection.count())
            if actual_count < expect_min_count:
                return (
                    False,
                    f"chunk 数量不足：expect_at_least={expect_min_count}, actual={actual_count}",
                )

            if actual_count > 0:
                # 优先探测当前批次 id，确保“刚写入”数据可读；没有传入则退化为首条探测。
                if probe_ids:
                    payload = collection.get(
                        ids=[str(probe_ids[0])],
                        include=["documents", "metadatas"],
                    )
                else:
                    payload = collection.get(
                        include=["documents", "metadatas"],
                        limit=1,
                        offset=0,
                    )
                ids = payload.get("ids")
                if not isinstance(ids, list) or len(ids) < 1:
                    return (False, "读取探测失败：collection.get 未返回有效 ids")

                docs = payload.get("documents")
                if not isinstance(docs, list) or len(docs) < 1:
                    return (False, "读取探测失败：collection.get 未返回有效 documents")
                probe_doc = str(docs[0]).strip()
                if not probe_doc:
                    return (False, "读取探测失败：probe document 为空")

                # 走一次 query 路径，触发 HNSW reader 初始化。
                if self._use_remote_embedding():
                    probe_embedding = self._embed_remote([probe_doc])[0]
                    collection.query(
                        query_embeddings=[probe_embedding],
                        n_results=1,
                    )
                else:
                    collection.query(
                        query_texts=[probe_doc],
                        n_results=1,
                    )
            return (True, "")
        except Exception as e:
            return (False, str(e))

    def _write_index_meta(self, *, kb_dir: Path, kb_name: str, source_file_count: int, chunk_count: int) -> None:
        """写入索引元信息，便于展示、迁移和运维排障。

        关键字段:
        - `embedding_mode/embedding_model`: 记录建库时采用的向量化模式与模型。
        - `source_file_count/chunk_count`: 记录语料规模，便于核对重建结果。
        - `store_dir`: 指向当前知识库真实 Chroma 存储目录。
        """
        used_model = (
            self._config.remote_embedding_model
            if self._use_remote_embedding()
            else self._config.embedding_model
        )
        mode = "remote" if self._use_remote_embedding() else "local"
        meta = {
            "kb_name": kb_name,
            "embedding_mode": mode,
            "embedding_model": used_model,
            "source_file_count": source_file_count,
            "chunk_count": chunk_count,
            "store": "chromadb",
            "store_dir": str(resolve_chroma_store_dir(kb_dir=kb_dir, kb_name=kb_name).resolve()),
        }
        meta_file = resolve_index_meta_file(kb_dir=kb_dir, kb_name=kb_name)
        meta_file.parent.mkdir(parents=True, exist_ok=True)
        meta_file.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
