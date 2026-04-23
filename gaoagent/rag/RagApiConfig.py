from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gaoagent.rag.RagChromaIndexer import RagChromaIndexerConfig
from gaoagent.core.runner.Utils import _find_config_file


class RagApiConfigStore:
    _FILENAME = "gao_client_rag_api_config.json"

    def config_file(self) -> Path:
        rag_dir = _find_config_file("rag").resolve()
        return rag_dir / ".chrome_store" / self._FILENAME

    def _default_payload(self) -> dict[str, Any]:
        return {"kb_remote_apis": {}, "chunker_py_file": ""}

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
        kb_remote_apis = payload.get("kb_remote_apis")
        if not isinstance(kb_remote_apis, dict):
            kb_remote_apis = {}
        chunker_py_file = payload.get("chunker_py_file")
        if not isinstance(chunker_py_file, str):
            chunker_py_file = ""
        return {"kb_remote_apis": kb_remote_apis, "chunker_py_file": chunker_py_file}

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
        kb_name: str,
        local_embedding_model: str,
        chunk_size: int,
        chunk_overlap: int,
        batch_size: int,
    ) -> RagChromaIndexerConfig:
        payload = self.load()
        kb_remote_apis = payload.get("kb_remote_apis")
        chunker_py_file = str(payload.get("chunker_py_file") or "").strip()
        remote_body: dict[str, Any] | None = None

        if isinstance(kb_remote_apis, dict):
            body = kb_remote_apis.get(kb_name)
            if isinstance(body, dict):
                remote_body = body

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
