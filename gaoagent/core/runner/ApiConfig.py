from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ApiSelection:
    base_url: str
    api_key: str
    model: str
    api_name: str


def default_api_config_path() -> Path:
    return Path.home() / ".gaoagent" / "gao_client_api_config.json"


def load_api_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return json.loads(path.read_text(encoding="utf-8"))


def select_api_and_model(config_payload: dict[str, Any], memory: dict[str, Any]) -> ApiSelection:
    apis = config_payload.get("apis") if isinstance(config_payload, dict) else None
    if not isinstance(apis, dict) or not apis:
        raise ValueError("API 配置无效：缺少 apis 或 apis 为空")

    api_name = str(memory.get("api_name") or "").strip()
    selected_api: dict[str, Any] | None = None
    selected_api_name = ""
    if api_name:
        candidate = apis.get(api_name)
        if isinstance(candidate, dict):
            selected_api = candidate
            selected_api_name = api_name
        else:
            raise KeyError(f"未找到指定 API 配置：{api_name}")
    else:
        selected_api_name, selected_api = next(iter(apis.items()))
        if not isinstance(selected_api, dict):
            raise ValueError(f"API 配置格式无效：{selected_api_name}")

    base_url = str(selected_api.get("base_url") or "").strip()
    api_key = str(selected_api.get("api_key") or "").strip()
    models = selected_api.get("models")
    if not base_url:
        raise ValueError(f"API 配置缺少 base_url：{selected_api_name}")
    if not api_key:
        raise ValueError(f"API 配置缺少 api_key：{selected_api_name}")
    if not isinstance(models, dict) or not models:
        raise ValueError(f"API 配置缺少 models：{selected_api_name}")

    model = str(memory.get("model") or "").strip()
    if model:
        if model not in models:
            raise KeyError(f"未找到指定模型：{model}")
    else:
        model = next(iter(models.keys()))

    return ApiSelection(base_url=base_url, api_key=api_key, model=model, api_name=selected_api_name)
