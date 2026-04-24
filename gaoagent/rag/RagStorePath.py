from __future__ import annotations

import hashlib
from pathlib import Path


def resolve_chroma_store_dir(*, kb_dir: Path, kb_name: str) -> Path:
    """
    计算 Chroma 持久化目录（ASCII 安全路径）。

    目录规则：
    - 统一放在 `.gaoagent/rag/.chrome_store/` 下，避免知识库名包含非 ASCII 字符导致
      HNSW 文件在某些 Windows 环境下无法稳定生成/读取。
    - 使用知识库名 hash 作为子目录名。
    """
    resolved_kb_dir = kb_dir.expanduser().resolve()
    rag_root = resolved_kb_dir.parent
    chroma_store_root = rag_root / ".chrome_store"
    key = kb_name.strip()
    token = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
    return chroma_store_root / f"kb_{token}"


def resolve_index_meta_file(*, kb_dir: Path, kb_name: str) -> Path:
    """
    index_meta.json 与向量库放在同一个 hash 子目录，避免项目/全局互拷后路径错配。
    """
    return resolve_chroma_store_dir(kb_dir=kb_dir, kb_name=kb_name) / "index_meta.json"


def resolve_bm25_index_db(*, kb_dir: Path, kb_name: str) -> Path:
    """
    BM25 稀疏索引 SQLite 文件路径。

    说明:
    - 与 Chroma 与 index_meta 保持同目录，便于项目/全局同步时整体迁移。
    - 该数据库用于保存倒排统计与文档快照，供增量更新与关键词检索复用。
    """
    return resolve_chroma_store_dir(kb_dir=kb_dir, kb_name=kb_name) / "bm25_index.db"


def is_internal_rag_store_dir_name(name: str) -> bool:
    """判断目录名是否为 RAG 内部管理目录。

    用途:
    - 在 `rag list`、知识库扫描等用户可见场景中过滤内部目录，避免混入业务列表。
    """
    return name.strip() == ".chrome_store"
