from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

import click
from gaoagent.core.runner.console import Console


# ── 文件操作 ──────────────────────────────────────────────


def write_json(file_path: Path, payload: Any) -> None:
    """以临时文件替换方式原子写入 JSON 文件。"""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = file_path.with_name(f"{file_path.name}.tmp")
    tmp_file.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    tmp_file.replace(file_path)


def read_json_file(file_path: Path) -> Any | None:
    """读取 JSON 文件；不存在或解析失败时返回 None。"""
    if not file_path.exists() or not file_path.is_file():
        return None
    try:
        return json.loads(file_path.read_text(encoding="utf-8"))
    except Exception as e:
        Console.error(f"读取 JSON 文件失败：{file_path}，{e}")
        return None


def copy_dir(src: Path, dst: Path) -> None:
    """目录覆盖复制：目标存在时先删除后复制。"""
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def rewrite_index_meta_store_dir(*, kb_dir: Path, kb_name: str) -> None:
    """修正 index_meta.json 中的 store_dir 到当前实际路径。"""
    from gaoagent.rag.rag_store_path import resolve_chroma_store_dir, resolve_index_meta_file

    meta_file = resolve_index_meta_file(kb_dir=kb_dir, kb_name=kb_name)
    if not meta_file.exists() or not meta_file.is_file():
        return
    try:
        data = json.loads(meta_file.read_text(encoding="utf-8"))
    except Exception as e:
        Console.error(f"读取索引元数据失败：{meta_file}，{e}")
        return
    if not isinstance(data, dict):
        return
    data["store_dir"] = str(resolve_chroma_store_dir(kb_dir=kb_dir, kb_name=kb_name).resolve())
    meta_file.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


# ── 作用域解析 ─────────────────────────────────────────────


def resolve_scope_and_config_path(config_filename: str) -> tuple[str, Path]:
    """解析当前作用域并返回 (scope_label, config_file_path)。

    项目优先，其次全局。
    """
    from gaoagent.core.runner.utils import try_project_root_dir, global_config_dir

    project_config_root = try_project_root_dir()
    if project_config_root is not None:
        return ("项目", project_config_root / ".gaoagent" / config_filename)
    return ("全局", global_config_dir() / config_filename)


# ── 交互输入 ─────────────────────────────────────────────


def prompt_non_empty_str(text: str, *, hide_input: bool = False) -> str:
    """获取非空字符串输入；为空则提示并重新输入。"""
    while True:
        value = Console.prompt(text, type=str, hide_input=hide_input).strip()
        if value:
            return value
        Console.info("输入不能为空，请重新输入")


def prompt_positive_int(text: str, *, default: int, show_default: bool = True) -> int:
    """获取正整数输入；非正整数则提示并重新输入。"""
    while True:
        value = Console.prompt(text, type=int, default=default, show_default=show_default)
        if value > 0:
            return value
        Console.info("请输入正整数")


def prompt_multi_select(prompt: str, options: list[str]) -> list[str]:
    """通用多选输入解析器（支持序号、名称与全选）。

    输入规则:
    - 回车: 返回空列表（表示跳过）。
    - all/*: 返回全部选项。
    - 1,3 或 nameA,nameB: 支持逗号分隔混合输入。
    - 非法输入会提示并重试。

    返回:
    - 去重且保持输入顺序的选项列表。
    """
    if not options:
        return []

    while True:
        raw = Console.prompt(prompt, type=str, default="", show_default=False).strip()
        if raw == "":
            return []

        lowered = raw.lower()
        if lowered in ("all", "*"):
            return options

        parts = [p.strip() for p in raw.split(",") if p.strip()]
        selected: list[str] = []
        ok = True
        for part in parts:
            if part.isdigit():
                idx = int(part)
                if idx < 1 or idx > len(options):
                    ok = False
                    break
                selected.append(options[idx - 1])
                continue

            match = None
            for opt in options:
                if opt.lower() == part.lower():
                    match = opt
                    break
            if match is None:
                ok = False
                break
            selected.append(match)

        if ok:
            deduped: list[str] = []
            seen: set[str] = set()
            for item in selected:
                if item in seen:
                    continue
                seen.add(item)
                deduped.append(item)
            return deduped

        Console.info("输入不合法，请重新输入")


# ── RAG / BM25 工具 ──────────────────────────────────────


def tokenize_for_bm25(text: str) -> list[str]:
    """轻量 BM25 分词：英文按词，中文按单字。"""
    normalized = str(text or "").lower()
    tokens: list[str] = []
    for seg in re.findall(r"[\u4e00-\u9fff]+|[a-z0-9_]+", normalized):
        if re.fullmatch(r"[\u4e00-\u9fff]+", seg):
            tokens.extend(list(seg))
        else:
            tokens.append(seg)
    return tokens


def sanitize_collection_name(kb_name: str) -> str:
    """将知识库名转换为 Chroma 可接受的 collection 名称。

    约束:
    - 仅允许 [a-zA-Z0-9._-]
    - 首尾必须是 [a-zA-Z0-9]
    """
    raw = kb_name.strip()
    base = re.sub(r"[^a-zA-Z0-9._-]+", "_", raw)
    base = re.sub(r"_+", "_", base)
    base = base.strip("._-")
    if not base:
        base = "default"
    base = base[:120].strip("._-")
    if not base:
        base = "default"
    return f"kb_{base}"
