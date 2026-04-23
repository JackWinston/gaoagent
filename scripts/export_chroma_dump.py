from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

from chromadb import PersistentClient

from gaoagent.rag.RagStorePath import resolve_chroma_store_dir, resolve_index_meta_file


def _default_rag_root() -> Path:
    project_rag = Path.cwd() / ".gaoagent" / "rag"
    if project_rag.exists() and project_rag.is_dir():
        return project_rag
    return Path.home() / ".gaoagent" / "rag"


def _timestamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _list_kb_dirs(rag_root: Path) -> list[Path]:
    if not rag_root.exists() or not rag_root.is_dir():
        return []
    return sorted([p for p in rag_root.iterdir() if p.is_dir()])


def _safe_read_json(file_path: Path) -> dict[str, Any] | None:
    if not file_path.exists() or not file_path.is_file():
        return None
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _write_json(file_path: Path, payload: Any) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    # 兼容 numpy.ndarray / numpy scalar 等对象。
    if hasattr(value, "tolist"):
        try:
            converted = value.tolist()
            return converted if isinstance(converted, list) else [converted]
        except Exception:
            pass
    try:
        return list(value)
    except Exception:
        return [value]


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(x) for x in value]
    if hasattr(value, "tolist"):
        try:
            return _to_jsonable(value.tolist())
        except Exception:
            return str(value)
    return str(value)


def _iter_collection_rows(
    collection: Any,
    *,
    batch_size: int,
    include_embeddings: bool,
) -> tuple[int, list[dict[str, Any]]]:
    include_fields = ["documents", "metadatas"]
    if include_embeddings:
        include_fields.append("embeddings")

    total = int(collection.count())
    rows: list[dict[str, Any]] = []
    offset = 0
    while offset < total:
        current_limit = min(batch_size, total - offset)
        payload = collection.get(
            include=include_fields,
            limit=current_limit,
            offset=offset,
        )
        ids = _as_list(payload.get("ids"))
        docs = _as_list(payload.get("documents"))
        metas = _as_list(payload.get("metadatas"))
        embeddings = _as_list(payload.get("embeddings"))

        for i, item_id in enumerate(ids):
            row: dict[str, Any] = {
                "id": item_id,
                "document": _to_jsonable(docs[i]) if i < len(docs) else "",
                "metadata": _to_jsonable(metas[i]) if i < len(metas) else {},
            }
            if include_embeddings:
                row["embedding"] = _to_jsonable(embeddings[i]) if i < len(embeddings) else []
            rows.append(row)
        offset += current_limit
    return (total, rows)


def export_kb(
    kb_dir: Path,
    *,
    output_root: Path,
    batch_size: int,
    include_embeddings: bool,
) -> dict[str, Any]:
    kb_name = kb_dir.name
    kb_out_dir = output_root / kb_name
    kb_out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "kb_name": kb_name,
        "kb_dir": str(kb_dir),
        "status": "ok",
        "message": "",
        "collections": [],
    }

    index_meta = _safe_read_json(resolve_index_meta_file(kb_dir=kb_dir, kb_name=kb_name))
    if index_meta is not None:
        _write_json(kb_out_dir / "index_meta.json", index_meta)
        summary["index_meta"] = index_meta

    store_dir = resolve_chroma_store_dir(kb_dir=kb_dir, kb_name=kb_name)
    if not store_dir.exists() or not store_dir.is_dir():
        summary["status"] = "skip"
        summary["message"] = f"未找到 Chroma 存储目录: {store_dir}"
        return summary

    try:
        client = PersistentClient(path=str(store_dir))
        collections = client.list_collections()
    except Exception as e:
        summary["status"] = "error"
        summary["message"] = f"打开 ChromaDB 失败: {e}"
        return summary

    for collection in collections:
        try:
            col = client.get_collection(name=collection.name)
            total, rows = _iter_collection_rows(
                col,
                batch_size=batch_size,
                include_embeddings=include_embeddings,
            )
            out_file = kb_out_dir / f"{collection.name}.jsonl"
            with out_file.open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            summary["collections"].append(
                {
                    "name": collection.name,
                    "count": total,
                    "output_file": str(out_file),
                }
            )
        except Exception as e:
            summary["collections"].append(
                {
                    "name": collection.name,
                    "status": "error",
                    "message": str(e),
                }
            )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="批量导出本地 ChromaDB 内容（按知识库和集合导出为 JSONL）",
    )
    parser.add_argument(
        "--rag-root",
        type=Path,
        default=_default_rag_root(),
        help="RAG 根目录（默认优先当前项目 .gaoagent/rag，否则 ~/.gaoagent/rag）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd() / "_debug" / f"chroma_export_{_timestamp()}",
        help="导出目录",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=200,
        help="分页读取批大小",
    )
    parser.add_argument(
        "--include-embeddings",
        action="store_true",
        help="是否导出 embedding 向量（体积较大）",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    rag_root = args.rag_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    batch_size = max(1, int(args.batch_size))
    include_embeddings = bool(args.include_embeddings)

    kb_dirs = _list_kb_dirs(rag_root)
    if not kb_dirs:
        print(f"未找到知识库目录: {rag_root}")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "generated_at": dt.datetime.now().isoformat(),
        "rag_root": str(rag_root),
        "output_dir": str(output_dir),
        "include_embeddings": include_embeddings,
        "knowledge_bases": [],
    }

    print(f"RAG_ROOT: {rag_root}")
    print(f"OUTPUT_DIR: {output_dir}")
    print(f"KB_COUNT: {len(kb_dirs)}")

    for kb_dir in kb_dirs:
        print(f"\n[导出] {kb_dir.name}")
        summary = export_kb(
            kb_dir,
            output_root=output_dir,
            batch_size=batch_size,
            include_embeddings=include_embeddings,
        )
        report["knowledge_bases"].append(summary)
        status = summary.get("status")
        message = summary.get("message") or ""
        print(f"  status={status} {message}".rstrip())
        for col in summary.get("collections", []):
            if col.get("status") == "error":
                print(f"  - {col.get('name')}: error={col.get('message')}")
            else:
                print(f"  - {col.get('name')}: count={col.get('count')}")

    report_file = output_dir / "report.json"
    _write_json(report_file, report)
    print(f"\n导出完成，汇总文件: {report_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
