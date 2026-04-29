from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click
from gaoagent.core.runner.console import Console
from gaoagent.core.runner.utils import try_project_root_dir
from gaoagent.core.handler_utils import read_json_file, write_json, prompt_non_empty_str, prompt_positive_int


class ApiHandlers:
    """API 配置命令处理器。

    作用域规则:
    - 当前目录属于已初始化项目时，优先操作项目配置
      `项目/.gaoagent/gao_client_api_config.json`。
    - 否则操作全局配置 `~/.gaoagent/gao_client_api_config.json`。

    支持能力:
    - `list`: 展示当前作用域 API 列表（名称、URL、模型列表、是否默认）。
    - `add`: 新增 API；项目作用域下会同步写入全局并设为默认。
    - `remove`: 删除指定 API（按当前作用域）。
    - `edit`: 编辑指定 API（仅当前作用域）。
    - `default`: 设置指定 API 为当前作用域默认 API。
    """

    _API_CONFIG_FILENAME = "gao_client_api_config.json"

    def list(self) -> None:
        """列出当前作用域 API 配置信息。"""
        (scope, config_file) = self._resolve_scope_and_config_path()
        payload = self._load_api_payload(config_file)
        apis = payload["apis"]
        if not apis:
            Console.info(f"未检测到{scope} API 配置：{config_file}")
            return

        default_api = payload.get("default_api")
        if not isinstance(default_api, str) or default_api not in apis:
            default_api = next(iter(apis.keys()))
        Console.info(f"{scope} API 列表：")
        for idx, name in enumerate(sorted(apis.keys()), start=1):
            body = apis.get(name)
            if not isinstance(body, dict):
                continue
            base_url = str(body.get("base_url") or "")
            models = body.get("models")
            model_names = sorted(list(models.keys())) if isinstance(models, dict) else []
            is_default = "是" if name == default_api else "否"
            Console.info(f"{idx}. name={name}")
            Console.info(f"   url={base_url}")
            Console.info(f"   models={model_names}")
            Console.info(f"   default={is_default}")

    def add(self) -> None:
        """新增 API 配置。"""
        (scope, config_file) = self._resolve_scope_and_config_path()
        payload = self._load_api_payload(config_file)
        apis = payload["apis"]

        name = self._prompt_non_empty_str("请输入 API 配置名称")
        if name in apis:
            Console.info(f"API 已存在：{name}")
            return

        api_body = self._prompt_api_body(existing=None)
        apis[name] = api_body
        payload["apis"] = apis

        if scope == "项目":
            payload["default_api"] = name
            payload["default_model"] = self._pick_first_model(api_body)

        self._write_api_payload(config_file, payload)
        Console.info(f"已添加{scope} API：{name}")

        if scope == "项目":
            global_file = self._global_config_dir() / self._API_CONFIG_FILENAME
            global_payload = self._load_api_payload(global_file)
            global_payload["apis"][name] = api_body
            global_payload["default_api"] = name
            global_payload["default_model"] = self._pick_first_model(api_body)
            self._write_api_payload(global_file, global_payload)
            Console.info(f"已同步到全局配置并设为默认：{global_file}")

    def remove(self, name: str) -> None:
        """删除指定 API 配置（按当前作用域）。"""
        target = (name or "").strip()
        if not target:
            Console.info("请提供 name 参数")
            return

        (scope, config_file) = self._resolve_scope_and_config_path()
        payload = self._load_api_payload(config_file)
        apis = payload["apis"]
        if target not in apis:
            Console.info(f"未找到 API：{target}")
            return

        del apis[target]
        payload["apis"] = apis
        if not apis:
            payload["default_api"] = ""
            payload["default_model"] = ""
        elif payload.get("default_api") == target:
            fallback = next(iter(apis.keys()))
            payload["default_api"] = fallback
            payload["default_model"] = self._pick_first_model(apis.get(fallback))

        self._write_api_payload(config_file, payload)
        Console.info(f"已删除{scope} API：{target}")

    def edit(self, name: str) -> None:
        """编辑指定 API 配置（仅当前作用域）。"""
        target = (name or "").strip()
        if not target:
            Console.info("请提供 name 参数")
            return

        (scope, config_file) = self._resolve_scope_and_config_path()
        payload = self._load_api_payload(config_file)
        apis = payload["apis"]
        old = apis.get(target)
        if not isinstance(old, dict):
            Console.info(f"未找到 API：{target}")
            return

        apis[target] = self._prompt_api_body(existing=old)
        payload["apis"] = apis
        if payload.get("default_api") == target:
            payload["default_model"] = self._pick_first_model(apis[target])
        self._write_api_payload(config_file, payload)
        Console.info(f"已更新{scope} API：{target}")

    def default(self, name: str) -> None:
        """设置指定 API 为当前作用域默认 API。"""
        target = (name or "").strip()
        if not target:
            Console.info("请提供 name 参数")
            return

        (scope, config_file) = self._resolve_scope_and_config_path()
        payload = self._load_api_payload(config_file)
        apis = payload["apis"]
        body = apis.get(target)
        if not isinstance(body, dict):
            Console.info(f"未找到 API：{target}")
            return

        payload["default_api"] = target
        payload["default_model"] = self._pick_first_model(body)
        self._write_api_payload(config_file, payload)
        Console.info(f"已设置{scope}默认 API：{target}")

    def _resolve_scope_and_config_path(self) -> tuple[str, Path]:
        """解析当前作用域和 API 配置文件路径。"""
        project_root = try_project_root_dir()
        if project_root is not None:
            return ("项目", project_root / ".gaoagent" / self._API_CONFIG_FILENAME)
        return ("全局", self._global_config_dir() / self._API_CONFIG_FILENAME)

    def _global_config_dir(self) -> Path:
        """返回全局配置目录。"""
        return Path.home() / ".gaoagent"

    def _read_json_file(self, file_path: Path) -> Any | None:
        """读取 JSON 文件。"""
        return read_json_file(file_path)

    def _load_api_payload(self, file_path: Path) -> dict[str, Any]:
        """读取 API 配置并做结构标准化。"""
        payload = self._read_json_file(file_path)
        if not isinstance(payload, dict):
            payload = {}
        apis_raw = payload.get("apis")
        apis = apis_raw if isinstance(apis_raw, dict) else {}
        default_api = payload.get("default_api")
        default_model = payload.get("default_model")
        return {
            "apis": apis,
            "default_api": (default_api if isinstance(default_api, str) else ""),
            "default_model": (default_model if isinstance(default_model, str) else ""),
        }

    def _write_api_payload(self, file_path: Path, payload: dict[str, Any]) -> None:
        """原子写入 API 配置文件。"""
        write_json(file_path, payload)

    def _prompt_api_body(self, *, existing: dict[str, Any] | None) -> dict[str, Any]:
        """交互采集 API 配置体（base_url/api_key/models）。"""
        base_url_default = str(existing.get("base_url") or "") if isinstance(existing, dict) else ""
        api_key_default = str(existing.get("api_key") or "") if isinstance(existing, dict) else ""
        models_default = existing.get("models") if isinstance(existing, dict) else {}
        models_default = models_default if isinstance(models_default, dict) else {}

        base_url = Console.prompt("请输入 API Base URL", type=str, default=base_url_default, show_default=bool(base_url_default)).strip()
        while not base_url:
            Console.info("Base URL 不能为空")
            base_url = Console.prompt("请输入 API Base URL", type=str).strip()

        api_key = Console.prompt(
            "请输入 API Key（留空表示保持原值）",
            type=str,
            default="",
            show_default=False,
            hide_input=True,
        ).strip()
        if not api_key:
            api_key = api_key_default
        while not api_key:
            Console.info("API Key 不能为空")
            api_key = Console.prompt("请输入 API Key", type=str, hide_input=True).strip()

        if models_default and Console.confirm("是否复用现有模型列表？", default=True):
            models = models_default
        else:
            models = self._prompt_models()

        return {
            "base_url": base_url,
            "api_key": api_key,
            "models": models,
        }

    def _prompt_models(self) -> dict[str, Any]:
        """交互采集模型列表，至少包含一个模型。"""
        models: dict[str, Any] = {}
        while True:
            model_id = self._prompt_non_empty_str("请输入模型名")
            if model_id in models:
                Console.info("模型名重复，请重新输入")
                continue
            context_window = self._prompt_positive_int("请输入该模型 context window", default=8192)
            vision = Console.confirm("该模型是否支持图片（vision）？", default=False)
            tools = Console.confirm("该模型是否支持工具调用（tools）？", default=True)
            reasoning = Console.confirm("该模型是否支持推理（reasoning）？", default=False)
            aliases_raw = Console.prompt("请输入别名（逗号分隔，可留空）", type=str, default="", show_default=False).strip()
            aliases = [a.strip() for a in aliases_raw.split(",") if a.strip()] if aliases_raw else []
            models[model_id] = {
                "id": model_id,
                "context_window": context_window,
                "capabilities": {"vision": vision, "tools": tools, "reasoning": reasoning},
                "aliases": aliases,
            }
            if not Console.confirm("继续添加模型？", default=False):
                break
        return models

    def _pick_first_model(self, api_body: Any) -> str:
        """从 API 配置中选取一个默认模型名。"""
        if not isinstance(api_body, dict):
            return ""
        models = api_body.get("models")
        if not isinstance(models, dict) or not models:
            return ""
        return next(iter(models.keys()))

    def _prompt_non_empty_str(self, text: str) -> str:
        """获取非空字符串输入。"""
        return prompt_non_empty_str(text)

    def _prompt_positive_int(self, text: str, *, default: int) -> int:
        """获取正整数输入。"""
        return prompt_positive_int(text, default=default)
