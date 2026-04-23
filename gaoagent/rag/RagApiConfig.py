from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gaoagent.rag.RagChromaIndexer import RagChromaIndexerConfig


class RagApiConfigStore:
    _FILENAME = "gao_client_rag_api_config.json"

    def config_file(self) -> Path:
        return Path.home() / ".gaoagent" / self._FILENAME

    def load(self) -> dict[str, Any]:
        path = self.config_file()
        if not path.exists() or not path.is_file():
            return {"default_remote": "", "remote_apis": {}, "chunker_py_file": ""}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {"default_remote": "", "remote_apis": {}, "chunker_py_file": ""}
        if not isinstance(payload, dict):
            return {"default_remote": "", "remote_apis": {}, "chunker_py_file": ""}
        remotes = payload.get("remote_apis")
        if not isinstance(remotes, dict):
            remotes = {}
        default_remote = payload.get("default_remote")
        if not isinstance(default_remote, str):
            default_remote = ""
        chunker_py_file = payload.get("chunker_py_file")
        if not isinstance(chunker_py_file, str):
            chunker_py_file = ""
        return {"default_remote": default_remote, "remote_apis": remotes, "chunker_py_file": chunker_py_file}

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
        remotes = payload.get("remote_apis")
        default_remote = payload.get("default_remote")
        chunker_py_file = str(payload.get("chunker_py_file") or "").strip()
        remote_name: str | None = None
        remote_body: dict[str, Any] | None = None

        if isinstance(default_remote, str) and isinstance(remotes, dict) and default_remote in remotes:
            body = remotes.get(default_remote)
            if isinstance(body, dict):
                remote_name = default_remote
                remote_body = body

        if remote_body is None and isinstance(remotes, dict):
            for name in sorted(remotes.keys()):
                body = remotes.get(name)
                if isinstance(name, str) and isinstance(body, dict):
                    remote_name = name
                    remote_body = body
                    break

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

        _ = remote_name
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
