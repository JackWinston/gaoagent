from __future__ import annotations

import hashlib
from pathlib import Path


def resolve_chroma_store_dir(*, kb_dir: Path, kb_name: str) -> Path:
    """
    计算 Chroma 持久化目录（ASCII 安全路径）。

    目录规则：
    - 统一放在 `.gaoagent/rag/chroma_store/` 下，避免知识库名包含非 ASCII 字符导致
      HNSW 文件在某些 Windows 环境下无法稳定生成/读取。
    - 使用知识库名 hash 作为子目录名。
    """
    resolved_kb_dir = kb_dir.expanduser().resolve()
    rag_root = resolved_kb_dir.parent
    chroma_store_root = rag_root / "chroma_store"
    key = kb_name.strip()
    token = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
    return chroma_store_root / f"kb_{token}"
