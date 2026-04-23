from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gaoagent.rag.RagChromaIndexer import RagChromaIndexerConfig
from gaoagent.rag.RagStorePath import resolve_chroma_store_dir


class RagApiConfigStore:
    _FILENAME = "gao_client_rag_api_config.json"

    def __init__(self, *, kb_name: str, kb_dir: Path) -> None:
        self._kb_name = kb_name
        self._kb_dir = kb_dir.resolve()

    def config_file(self) -> Path:
        store_dir = resolve_chroma_store_dir(kb_dir=self._kb_dir, kb_name=self._kb_name)
        return store_dir / self._FILENAME

    def _default_payload(self) -> dict[str, Any]:
        return {"remote_api": {}, "chunker_py_file": ""}

    def load(self) -> dict[str, Any]:
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
