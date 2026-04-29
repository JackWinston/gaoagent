from __future__ import annotations

import click
from gaoagent.core.runner.Console import Console
import shutil
import json
from pathlib import Path

from gaoagent.core.CoreConfigDefault import CoreConfigDefault
from gaoagent.core.runner.Utils import try_project_root_dir
from gaoagent.rag.RagStorePath import (
    resolve_chroma_store_dir,
    resolve_index_meta_file,
    is_internal_rag_store_dir_name,
)


class RagHandlers:
    """RAG 命令处理器（`gaoagent rag` 子命令入口）。

    定位:
    - 负责知识库的增删查与作用域分发（项目优先、全局兜底）。
    - 封装 CLI 层的交互逻辑与路径选择，底层索引/检索由专用组件实现。

    核心能力:
    - `list`: 列出当前作用域知识库。
    - `add`: 创建知识库并在项目/全局间做同步。
    - `update`: 对已有知识库执行增量入库并维护索引。
    - `remove`: 删除知识库及其 Chroma 存储目录。
    - `search`: 在指定知识库执行向量检索并格式化输出。
    """
    def list(self) -> None:
        """列出当前作用域下可见的知识库名称。"""
        (scope, rag_dir) = self._resolve_scope_and_rag_dir()
        if not rag_dir.exists() or not rag_dir.is_dir():
            Console.info(f"未检测到{scope} RAG 目录：{rag_dir}")
            return

        names = self._list_kb_names(rag_dir)
        if not names:
            Console.info(f"当前{scope}无知识库")
            return

        Console.info(f"当前{scope}知识库列表：")
        for idx, name in enumerate(names, start=1):
            Console.info(f"{idx}. {name}")

    def add(self, name: str | None = None, chunker_py_file: str | None = None) -> None:
        """新增知识库，并在项目与全局目录之间按规则同步。

        行为:
        - 若当前不在项目作用域，仅在全局创建知识库。
        - 若在项目作用域，先在项目创建，再按用户选择同步到全局目录。
        - 同步时会复制 `.chrome_store` 对应目录并修正 `index_meta.store_dir`。
        """
        kb_name = (name or "").strip()
        if not kb_name:
            kb_name = Console.prompt("请输入要新增的知识库名称", type=str).strip()
        chunker_file = (chunker_py_file or "").strip() or None

        project_root = try_project_root_dir()
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
            should_overwrite = Console.confirm(
                f"全局知识库已存在：{dst_dir}，是否覆盖？",
                default=False,
            )
            if not should_overwrite:
                Console.info(f"已跳过同步到全局目录：{dst_dir}")
                return
        self._copy_dir(src_dir, dst_dir)
        src_store_dir = resolve_chroma_store_dir(kb_dir=src_dir, kb_name=kb_name)
        dst_store_dir = resolve_chroma_store_dir(kb_dir=dst_dir, kb_name=kb_name)
        if not src_store_dir.exists() or not src_store_dir.is_dir():
            Console.info(f"同步失败：未找到 Chroma 存储目录：{src_store_dir}")
            return
        self._copy_dir(src_store_dir, dst_store_dir)
        self._rewrite_index_meta_store_dir(kb_dir=dst_dir, kb_name=kb_name)
        Console.info(f"已同步知识库到全局目录：{dst_dir}")

    def remove(self, name: str | None = None) -> None:
        """删除知识库目录及对应 Chroma 存储目录。"""
        (scope, rag_dir) = self._resolve_scope_and_rag_dir()
        if not rag_dir.exists() or not rag_dir.is_dir():
            Console.info(f"未检测到{scope} RAG 目录：{rag_dir}")
            return

        names = self._list_kb_names(rag_dir)
        if not names:
            Console.info(f"当前{scope}无可移除的知识库")
            return

        target = name if (name is not None and name in names) else Console.prompt(
            "请输入要移除的知识库名称",
            type=click.Choice(names, case_sensitive=False),
            default=names[0],
            show_default=True,
        )
        kb_dir = rag_dir / target
        if not kb_dir.exists() or not kb_dir.is_dir():
            Console.info(f"知识库不存在：{target}")
            return

        store_dir = resolve_chroma_store_dir(kb_dir=kb_dir, kb_name=target)
        if store_dir.exists() and store_dir.is_dir():
            shutil.rmtree(store_dir)
        shutil.rmtree(kb_dir)
        Console.info(f"已移除{scope}知识库：{target}")

    def update(self, name: str | None = None, chunker_py_file: str | None = None) -> None:
        """
        更新知识库（增量入库）。

        行为:
        - 优先在项目作用域更新；不在项目时更新全局作用域；
        - 更新过程会检查并维护 Chroma + BM25 两套索引；
        - 项目作用域更新成功后，同步覆盖到全局目录。
        """
        target_name = (name or "").strip()
        chunker_file = (chunker_py_file or "").strip() or None
        config_default = CoreConfigDefault()
        project_root = try_project_root_dir()
        if project_root is None:
            global_rag_dir = self._global_rag_dir()
            names = self._list_kb_names(global_rag_dir) if global_rag_dir.exists() else []
            if not names:
                Console.info("当前全局无可更新知识库")
                return
            if target_name not in names:
                target_name = Console.prompt(
                    "请输入要更新的知识库名称",
                    type=click.Choice(names, case_sensitive=False),
                    default=names[0],
                    show_default=True,
                )
            config_default.update_rag_knowledge_base(
                kb_name=target_name,
                rag_root_dir=global_rag_dir,
                chunker_py_file=chunker_file,
            )
            return

        project_rag_dir = project_root / ".gaoagent" / "rag"
        names = self._list_kb_names(project_rag_dir) if project_rag_dir.exists() else []
        if not names:
            Console.info("当前项目无可更新知识库")
            return
        if target_name not in names:
            target_name = Console.prompt(
                "请输入要更新的知识库名称",
                type=click.Choice(names, case_sensitive=False),
                default=names[0],
                show_default=True,
            )

        updated = config_default.update_rag_knowledge_base(
            kb_name=target_name,
            rag_root_dir=project_rag_dir,
            chunker_py_file=chunker_file,
        )
        if not updated:
            return

        # 与 add 保持一致：项目更新后同步到全局目录，避免两处知识库状态漂移。
        global_rag_dir = self._global_rag_dir()
        global_rag_dir.mkdir(parents=True, exist_ok=True)
        src_dir = project_rag_dir / target_name
        dst_dir = global_rag_dir / target_name
        self._copy_dir(src_dir, dst_dir)
        src_store_dir = resolve_chroma_store_dir(kb_dir=src_dir, kb_name=target_name)
        dst_store_dir = resolve_chroma_store_dir(kb_dir=dst_dir, kb_name=target_name)
        if src_store_dir.exists() and src_store_dir.is_dir():
            self._copy_dir(src_store_dir, dst_store_dir)
        self._rewrite_index_meta_store_dir(kb_dir=dst_dir, kb_name=target_name)
        Console.info(f"已同步更新后的知识库到全局目录：{dst_dir}")

    def search(self, kb_name: str, query: str, top_k: int = 3) -> None:
        """在指定知识库执行检索并输出结果摘要。"""
        from gaoagent.rag.RagChromaRetriever import RagChromaRetriever
        
        Console.info(f"正在知识库 '{kb_name}' 中检索: {query} (top_k={top_k})...")
        retriever = RagChromaRetriever(kb_name=kb_name)
        res = retriever.search(query=query, top_k=top_k)
        
        if not res.get("success"):
            Console.info(f"检索失败: {res.get('error')}")
            return
            
        items = res.get("items", [])
        if not items:
            Console.info("未找到相关内容。")
            return
            
        for idx, item in enumerate(items, 1):
            doc = str(item.get("document", "")).replace("\n", " ")[:150]
            dist = item.get("distance", 0)
            meta = item.get("metadata", {})
            src = meta.get("source_file", "unknown")
            Console.info(f"\n[{idx}] (相似度距离: {dist:.4f}) [来源: {src}]")
            Console.info(f"    {doc}...")

    def _resolve_scope_and_rag_dir(self) -> tuple[str, Path]:
        """解析当前作用域并返回对应 RAG 根目录。"""
        project_root = try_project_root_dir()
        if project_root is not None:
            return ("项目", project_root / ".gaoagent" / "rag")
        return ("全局", self._global_rag_dir())

    def _global_rag_dir(self) -> Path:
        """返回全局 RAG 根目录（`~/.gaoagent/rag`）。"""
        return Path.home() / ".gaoagent" / "rag"

    def _list_kb_names(self, rag_dir: Path) -> list[str]:
        """列出知识库目录名，并过滤内部管理目录（如 `.chrome_store`）。"""
        return sorted(
            [
                p.name
                for p in rag_dir.iterdir()
                if p.is_dir() and not is_internal_rag_store_dir_name(p.name)
            ]
        )

    def _copy_dir(self, src: Path, dst: Path) -> None:
        """目录覆盖复制：目标存在时先删后拷。"""
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

    def _rewrite_index_meta_store_dir(self, *, kb_dir: Path, kb_name: str) -> None:
        """修正 `index_meta.json` 的 `store_dir` 为当前目录实际路径。"""
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
