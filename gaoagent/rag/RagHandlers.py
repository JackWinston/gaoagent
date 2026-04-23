import click
import shutil
import json
from pathlib import Path
from typing import Any

from gaoagent.core.CoreConfigDefault import CoreConfigDefault
from gaoagent.rag.RagApiConfig import RagApiConfigStore
from gaoagent.rag.RagStorePath import (
    resolve_chroma_store_dir,
    resolve_index_meta_file,
    is_internal_rag_store_dir_name,
)


class RagHandlers:
    def list(self) -> None:
        (scope, rag_dir) = self._resolve_scope_and_rag_dir()
        if not rag_dir.exists() or not rag_dir.is_dir():
            click.echo(f"未检测到{scope} RAG 目录：{rag_dir}")
            return

        names = self._list_kb_names(rag_dir)
        if not names:
            click.echo(f"当前{scope}无知识库")
            return

        click.echo(f"当前{scope}知识库列表：")
        for idx, name in enumerate(names, start=1):
            click.echo(f"{idx}. {name}")

    def add(self, name: str | None = None, chunker_py_file: str | None = None) -> None:
        kb_name = (name or "").strip()
        if not kb_name:
            kb_name = click.prompt("请输入要新增的知识库名称", type=str).strip()
        chunker_file = (chunker_py_file or "").strip() or None

        project_root = self._detect_project_root()
        config_default = CoreConfigDefault()
        if project_root is None:
            config_default.create_rag_knowledge_base(kb_name=kb_name, chunker_py_file=chunker_file)
            return

        project_rag_dir = project_root / ".gaoagent" / "rag"
        created = config_default.create_rag_knowledge_base(
            rag_root_dir=project_rag_dir,
            kb_name=kb_name,
            chunker_py_file=chunker_file,
        )
        if not created:
            return

        global_rag_dir = self._global_rag_dir()
        global_rag_dir.mkdir(parents=True, exist_ok=True)
        src_dir = project_rag_dir / kb_name
        dst_dir = global_rag_dir / kb_name
        if dst_dir.exists():
            should_overwrite = click.confirm(
                f"全局知识库已存在：{dst_dir}，是否覆盖？",
                default=False,
            )
            if not should_overwrite:
                click.echo(f"已跳过同步到全局目录：{dst_dir}")
                return
        self._copy_dir(src_dir, dst_dir)
        src_store_dir = resolve_chroma_store_dir(kb_dir=src_dir, kb_name=kb_name)
        dst_store_dir = resolve_chroma_store_dir(kb_dir=dst_dir, kb_name=kb_name)
        if not src_store_dir.exists() or not src_store_dir.is_dir():
            click.echo(f"同步失败：未找到 Chroma 存储目录：{src_store_dir}")
            return
        self._copy_dir(src_store_dir, dst_store_dir)
        self._rewrite_index_meta_store_dir(kb_dir=dst_dir, kb_name=kb_name)
        click.echo(f"已同步知识库到全局目录：{dst_dir}")

    def remove(self, name: str | None = None) -> None:
        (scope, rag_dir) = self._resolve_scope_and_rag_dir()
        if not rag_dir.exists() or not rag_dir.is_dir():
            click.echo(f"未检测到{scope} RAG 目录：{rag_dir}")
            return

        names = self._list_kb_names(rag_dir)
        if not names:
            click.echo(f"当前{scope}无可移除的知识库")
            return

        target = name if (name is not None and name in names) else click.prompt(
            "请输入要移除的知识库名称",
            type=click.Choice(names, case_sensitive=False),
            default=names[0],
            show_default=True,
        )
        kb_dir = rag_dir / target
        if not kb_dir.exists() or not kb_dir.is_dir():
            click.echo(f"知识库不存在：{target}")
            return

        store_dir = resolve_chroma_store_dir(kb_dir=kb_dir, kb_name=target)
        if store_dir.exists() and store_dir.is_dir():
            shutil.rmtree(store_dir)
        shutil.rmtree(kb_dir)
        click.echo(f"已移除{scope}知识库：{target}")

    def api_list(self) -> None:
        store = RagApiConfigStore()
        payload = store.load()
        remotes = payload.get("remote_apis")
        if not isinstance(remotes, dict) or not remotes:
            click.echo("未配置远程 embedding，将使用本地 embedding_model")
            click.echo(f"配置文件：{store.config_file()}")
            return

        default_remote = str(payload.get("default_remote") or "")
        click.echo("远程 embedding 配置列表：")
        for idx, name in enumerate(sorted(remotes.keys()), start=1):
            body = remotes.get(name)
            if not isinstance(body, dict):
                continue
            base_url = str(body.get("base_url") or "")
            model = str(body.get("embedding_model") or "")
            suffix = " (default)" if name == default_remote else ""
            click.echo(f"{idx}. {name}{suffix}")
            click.echo(f"   - base_url: {base_url}")
            click.echo(f"   - embedding_model: {model}")

        click.echo(f"配置文件：{store.config_file()}")

    def api_add(self) -> None:
        store = RagApiConfigStore()
        payload = store.load()
        remotes = payload.get("remote_apis")
        if not isinstance(remotes, dict):
            remotes = {}

        name = self._prompt_non_empty_str("请输入远程配置名称")
        if name in remotes:
            click.echo(f"配置已存在：{name}")
            return

        body = self._prompt_remote_config_body(existing=None)
        remotes[name] = body
        payload["remote_apis"] = remotes

        if not payload.get("default_remote"):
            payload["default_remote"] = name
        elif click.confirm("是否设为默认远程配置？", default=True):
            payload["default_remote"] = name

        store.save(payload)
        click.echo(f"已新增远程 embedding 配置：{name}")

    def api_edit(self) -> None:
        store = RagApiConfigStore()
        payload = store.load()
        remotes = payload.get("remote_apis")
        if not isinstance(remotes, dict) or not remotes:
            click.echo("未检测到任何远程 embedding 配置")
            return

        name = self._prompt_remote_name(remotes, action="编辑")
        existing = remotes.get(name)
        if not isinstance(existing, dict):
            click.echo(f"配置无效：{name}")
            return

        remotes[name] = self._prompt_remote_config_body(existing=existing)
        payload["remote_apis"] = remotes
        if click.confirm("是否设为默认远程配置？", default=(payload.get("default_remote") == name)):
            payload["default_remote"] = name
        store.save(payload)
        click.echo(f"已更新远程 embedding 配置：{name}")

    def api_remove(self, name: str | None = None) -> None:
        store = RagApiConfigStore()
        payload = store.load()
        remotes = payload.get("remote_apis")
        if not isinstance(remotes, dict) or not remotes:
            click.echo("未检测到任何远程 embedding 配置")
            return

        target = name if (name is not None and name in remotes) else self._prompt_remote_name(remotes, action="移除")
        if target not in remotes:
            click.echo(f"未找到远程配置：{target}")
            return
        del remotes[target]
        payload["remote_apis"] = remotes

        default_remote = str(payload.get("default_remote") or "")
        if default_remote == target:
            payload["default_remote"] = sorted(remotes.keys())[0] if remotes else ""

        store.save(payload)
        click.echo(f"已移除远程 embedding 配置：{target}")

    def search(self, kb_name: str, query: str, top_k: int = 5) -> None:
        from gaoagent.rag.RagChromaRetriever import RagChromaRetriever
        
        click.echo(f"正在知识库 '{kb_name}' 中检索: {query} (top_k={top_k})...")
        retriever = RagChromaRetriever(kb_name=kb_name)
        res = retriever.search(query=query, top_k=top_k)
        
        if not res.get("success"):
            click.echo(f"检索失败: {res.get('error')}")
            return
            
        items = res.get("items", [])
        if not items:
            click.echo("未找到相关内容。")
            return
            
        for idx, item in enumerate(items, 1):
            doc = str(item.get("document", "")).replace("\n", " ")[:150]
            dist = item.get("distance", 0)
            meta = item.get("metadata", {})
            src = meta.get("source_file", "unknown")
            click.echo(f"\n[{idx}] (相似度距离: {dist:.4f}) [来源: {src}]")
            click.echo(f"    {doc}...")

    def _resolve_scope_and_rag_dir(self) -> tuple[str, Path]:
        project_root = self._detect_project_root()
        if project_root is not None:
            return ("项目", project_root / ".gaoagent" / "rag")
        return ("全局", self._global_rag_dir())

    def _global_rag_dir(self) -> Path:
        return Path.home() / ".gaoagent" / "rag"

    def _list_kb_names(self, rag_dir: Path) -> list[str]:
        return sorted(
            [
                p.name
                for p in rag_dir.iterdir()
                if p.is_dir() and not is_internal_rag_store_dir_name(p.name)
            ]
        )

    def _project_registry_file(self) -> Path:
        return Path.home() / ".gaoagent" / "inited_projects.txt"

    def _load_project_registry_paths(self) -> list[Path]:
        registry_file = self._project_registry_file()
        if not registry_file.exists() or not registry_file.is_file():
            return []
        try:
            lines = registry_file.read_text(encoding="utf-8").splitlines()
        except Exception:
            return []

        roots: list[Path] = []
        seen: set[str] = set()
        for line in lines:
            raw = line.strip()
            if not raw:
                continue
            try:
                root = Path(raw).expanduser().resolve()
            except Exception:
                continue
            key = str(root)
            if key in seen:
                continue
            seen.add(key)
            roots.append(root)
        return roots

    def _detect_project_root(self) -> Path | None:
        cwd = Path.cwd().resolve()
        config_dir = cwd / ".gaoagent"
        if config_dir.exists() and config_dir.is_dir():
            return cwd

        candidates: list[Path] = []
        for root in self._load_project_registry_paths():
            config = root / ".gaoagent"
            if not (root.exists() and root.is_dir() and config.exists() and config.is_dir()):
                continue
            if root == cwd or root in cwd.parents:
                candidates.append(root)

        if not candidates:
            return None
        candidates.sort(key=lambda p: len(p.parts), reverse=True)
        return candidates[0]

    def _copy_dir(self, src: Path, dst: Path) -> None:
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

    def _rewrite_index_meta_store_dir(self, *, kb_dir: Path, kb_name: str) -> None:
        meta_file = resolve_index_meta_file(kb_dir=kb_dir, kb_name=kb_name)
        if not meta_file.exists() or not meta_file.is_file():
            return
        try:
            data = json.loads(meta_file.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(data, dict):
            return
        data["store_dir"] = str(resolve_chroma_store_dir(kb_dir=kb_dir, kb_name=kb_name).resolve())
        meta_file.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _prompt_non_empty_str(self, prompt: str, *, hide_input: bool = False, default: str | None = None) -> str:
        while True:
            if default is None:
                value = click.prompt(prompt, type=str, hide_input=hide_input).strip()
            else:
                value = click.prompt(prompt, type=str, hide_input=hide_input, default=default, show_default=True).strip()
            if value:
                return value
            click.echo("输入不能为空，请重新输入")

    def _prompt_positive_int(self, prompt: str, *, default: int) -> int:
        while True:
            value = click.prompt(prompt, type=int, default=default, show_default=True)
            if value > 0:
                return value
            click.echo("请输入正整数")

    def _prompt_remote_name(self, remotes: dict[str, Any], *, action: str) -> str:
        names = sorted([str(x) for x in remotes.keys()])
        return click.prompt(
            f"请选择要{action}的远程配置",
            type=click.Choice(names, case_sensitive=False),
            default=names[0],
            show_default=True,
        )

    def _prompt_remote_config_body(self, *, existing: dict[str, Any] | None) -> dict[str, Any]:
        base_default = str(existing.get("base_url") or "") if isinstance(existing, dict) else None
        key_default = str(existing.get("api_key") or "") if isinstance(existing, dict) else ""
        model_default = str(existing.get("embedding_model") or "") if isinstance(existing, dict) else None
        timeout_default = (
            int(existing.get("timeout_sec"))
            if isinstance(existing, dict) and isinstance(existing.get("timeout_sec"), int)
            else 120
        )

        base_url = self._prompt_non_empty_str("请输入远程 Base URL（OpenAI 兼容）", default=base_default)
        if isinstance(existing, dict):
            api_key_input = click.prompt(
                "请输入远程 API Key（留空保持不变）",
                type=str,
                hide_input=True,
                default="",
                show_default=False,
            ).strip()
            api_key = api_key_input if api_key_input else key_default
        else:
            api_key = self._prompt_non_empty_str("请输入远程 API Key", hide_input=True)
        embedding_model = self._prompt_non_empty_str("请输入远程 Embedding Model", default=model_default)
        timeout_sec = self._prompt_positive_int("请输入请求超时秒数", default=timeout_default)
        return {
            "base_url": base_url.rstrip("/"),
            "api_key": api_key,
            "embedding_model": embedding_model,
            "timeout_sec": timeout_sec,
        }
