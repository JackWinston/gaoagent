from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gaoagent.rag.RagChromaIndexer import RagChromaIndexerConfig
from gaoagent.rag.RagStorePath import resolve_chroma_store_dir


class RagApiConfigStore:
    """RAG API 配置存储器（知识库级）。

    定位:
    - 管理单个知识库对应的 `gao_client_rag_api_config.json`。
    - 该配置用于“入库时 embedding 策略决策”：本地模型或远程 OpenAI 兼容接口。

    职责:
    - 提供配置文件路径解析（与 `.chrome_store/kb_<hash>` 路径体系保持一致）。
    - 负责安全加载/保存配置（异常容错 + 原子写入）。
    - 将配置解析为 `RagChromaIndexerConfig`，供 `RagChromaIndexer` 直接使用。
    """
    _FILENAME = "gao_client_rag_api_config.json"

    def __init__(self, *, kb_name: str, kb_dir: Path) -> None:
        """初始化知识库级配置存储对象。

        参数:
        - `kb_name`: 知识库名称。
        - `kb_dir`: 知识库目录（会在构造时 `resolve()` 规范化）。
        """
        self._kb_name = kb_name
        self._kb_dir = kb_dir.resolve()

    def config_file(self) -> Path:
        """返回当前知识库的 RAG API 配置文件路径。"""
        store_dir = resolve_chroma_store_dir(kb_dir=self._kb_dir, kb_name=self._kb_name)
        return store_dir / self._FILENAME

    def _default_payload(self) -> dict[str, Any]:
        """返回标准默认配置（无远程 API，无自定义切片器）。"""
        return {"remote_api": {}, "chunker_py_file": ""}

    def load(self) -> dict[str, Any]:
        """加载并规范化配置。

        行为:
        - 文件不存在、JSON 非法或结构不合法时回退默认值。
        - 仅保留 `remote_api(dict)` 与 `chunker_py_file(str)` 两个字段。
        """
        try:
            path = self.config_file()
        except Exception:
            return self._default_payload()
        if not path.exists() or not path.is_file():
            return self._default_payload()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return self._default_payload()
        if not isinstance(payload, dict):
            return self._default_payload()
        remote_api = payload.get("remote_api")
        if not isinstance(remote_api, dict):
            remote_api = {}
        chunker_py_file = payload.get("chunker_py_file")
        if not isinstance(chunker_py_file, str):
            chunker_py_file = ""
        return {"remote_api": remote_api, "chunker_py_file": chunker_py_file}

    def save(self, payload: dict[str, Any]) -> None:
        """保存配置到磁盘（原子替换写入）。"""
        path = self.config_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.tmp")
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        tmp.replace(path)

    def resolve_indexer_config(
        self,
        *,
        local_embedding_model: str,
        chunk_size: int,
        chunk_overlap: int,
        batch_size: int,
    ) -> RagChromaIndexerConfig:
        """将存储配置解析为 `RagChromaIndexerConfig`。

        解析策略:
        - 始终保留调用方传入的本地参数（模型、切片、批量大小）。
        - 若 `remote_api` 不存在或关键字段缺失，自动回退本地 embedding 模式。
        - 若远程字段完整，则返回包含远程 embedding 配置的 indexer config。
        """
        payload = self.load()
        remote_api = payload.get("remote_api")
        chunker_py_file = str(payload.get("chunker_py_file") or "").strip()
        remote_body = remote_api if isinstance(remote_api, dict) else None

        if remote_body is None:
            return RagChromaIndexerConfig(
                embedding_model=local_embedding_model,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                batch_size=batch_size,
                chunker_py_file=chunker_py_file,
            )

        base_url = str(remote_body.get("base_url") or "").strip()
        api_key = str(remote_body.get("api_key") or "").strip()
        model = str(remote_body.get("embedding_model") or "").strip()
        timeout_raw = remote_body.get("timeout_sec")
        timeout = int(timeout_raw) if isinstance(timeout_raw, int) and timeout_raw > 0 else 120

        if not (base_url and api_key and model):
            # 远程配置缺字段时自动回退本地，避免影响主流程。
            return RagChromaIndexerConfig(
                embedding_model=local_embedding_model,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                batch_size=batch_size,
                chunker_py_file=chunker_py_file,
            )

        return RagChromaIndexerConfig(
            embedding_model=local_embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            batch_size=batch_size,
            remote_base_url=base_url,
            remote_api_key=api_key,
            remote_embedding_model=model,
            remote_timeout_sec=timeout,
            chunker_py_file=chunker_py_file,
        )
