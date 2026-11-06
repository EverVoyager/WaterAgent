"""离线迁移：先 export 完整旧表备份，再 import 到 Qdrant v2 和 SQL 统计表。

在迁移期间停止应用写入和 Curator。不会删除或修改旧表/旧向量集合。
"""
import argparse
import hashlib
import json
import sys
from datetime import date, datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for _path in (_ROOT, _ROOT / "backend"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(_ROOT / "backend" / ".env")

from agent.memory import vector_index  # noqa: E402
from agent.memory.records import validate_body  # noqa: E402
from agent.memory.usage_store import get_usage_store  # noqa: E402
from agent.utils import now_iso  # noqa: E402

TABLES = {"semantic": "agent_semantic", "episode": "agent_episodes",
          "procedure": "agent_procedures"}


def _json_default(value):
    if isinstance(value, datetime | date):
        return value.isoformat()
    raise TypeError(f"备份不支持的 SQL 类型：{type(value).__name__}")


def _canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      default=_json_default)


def _checksum(tables: dict) -> str:
    return hashlib.sha256(_canonical(tables).encode("utf-8")).hexdigest()


def export_legacy(output: Path, store=None, page_size: int = 500) -> dict:
    """完整备份原字段；主键分页避免 SELECT 的隐式条数上限。"""
    if output.exists():
        raise FileExistsError("备份已存在，请指定新文件，禁止覆盖")
    store = store or get_usage_store()
    tables = {}
    with store._get_conn() as conn, conn.cursor() as cur:
        # 同一个只读快照覆盖三个旧表；任何导出错误都不会启动导入。
        cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
        cur.execute("START TRANSACTION WITH CONSISTENT SNAPSHOT, READ ONLY")
        for kind, table in TABLES.items():
            rows, last_id = [], 0
            while True:
                cur.execute(f"SELECT * FROM {table} WHERE id > %s ORDER BY id LIMIT %s",
                            (last_id, page_size))
                page = cur.fetchall()
                if not page:
                    break
                rows.extend(page)
                last_id = page[-1]["id"]
            tables[kind] = rows
    # shortcut: 小型记忆库在内存中编码备份，备份达到数百 MB 时改为流式格式。
    document = {"format_version": 1, "exported_at": now_iso(),
                "tables": tables, "sha256": _checksum(tables)}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as file:
        file.write(_canonical(document) + "\n")
    # 重新读取落盘文件并核对摘要，后续导入只信任已验证的备份。
    return load_backup(output)


def load_backup(path: Path) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    tables = document.get("tables")
    if (document.get("format_version") != 1 or not isinstance(tables, dict)
            or set(tables) != set(TABLES) or any(not isinstance(rows, list) for rows in tables.values())
            or _checksum(tables) != document.get("sha256")):
        raise ValueError("备份格式或完整性校验失败，未执行导入")
    return document


def _parsed(value, default):
    if value is None or value == "":
        return default
    return json.loads(value) if isinstance(value, str) else value


def convert_legacy(kind: str, row: dict) -> tuple[dict, dict]:
    """不伪造缺失证据；旧字段原文仍完整保留在备份和旧表中。"""
    mid = vector_index.point_id(row["id"])
    ref = {"legacy_table": TABLES[kind], "legacy_id": mid}
    stats = {"hit_count": row.get("hit_count", 0), "success_count": 0, "refined_count": 0}
    if kind == "semantic":
        provenance = _parsed(row.get("provenance_json"), {})
        body = {"title": row["title"], "content": row["content"],
                "reuse_when": provenance.get("reuse_when", ""),
                "evidence": provenance.get("evidence", [])}
    elif kind == "episode":
        summary = row["event_summary"]
        query = row.get("query_summary")
        if query and query not in summary:
            summary += f"\n原始请求：{query}"
        body = {"event_summary": summary, "resolution": row.get("resolution") or "",
                "outcome": row.get("outcome") or "partial",
                "happened_at": str(row.get("happened_at") or ""),
                "source_refs": [ref], "event_id": f"legacy:episode:{mid}"}
    else:
        body = {"name": row["name"], "applicability": row["applicability"],
                "steps": _parsed(row.get("steps_json"), []),
                "source_refs": [ref], "status": row.get("status") or "active"}
        stats.update(hit_count=row.get("use_count", 0), success_count=row.get("success_count", 0),
                     refined_count=row.get("refined_count", 0))
    counts = {key: int(value or 0) for key, value in stats.items()}
    if any(count < 0 for count in counts.values()):
        raise ValueError("旧统计值不能为负")
    return validate_body(kind, body), counts


def import_backup(document: dict, store=None) -> dict:
    """可重复导入；已存在的不同正文一律报告冲突，不覆盖新内容。"""
    if (document.get("format_version") != 1 or set(document["tables"]) != set(TABLES)
            or any(not isinstance(rows, list) for rows in document["tables"].values())
            or _checksum(document["tables"]) != document.get("sha256")):
        raise ValueError("备份完整性校验失败")
    store = store or get_usage_store()
    store._ensure_tables()  # 在任何 Qdrant 写入前确认 SQL 统计表可写。
    report = {"imported": 0, "unchanged": 0, "conflicts": [], "skipped": [], "failed": []}
    for kind, rows in document["tables"].items():
        for row in rows:
            ref = {"kind": kind, "id": row.get("id") if isinstance(row, dict) else None}
            try:
                mid = vector_index.point_id(row["id"])
                body, counts = convert_legacy(kind, row)
                created = str(row.get("created_at") or row.get("happened_at") or document["exported_at"])
                updated = str(row.get("updated_at") or created)
            except (ValueError, TypeError, KeyError, AttributeError) as e:
                report["skipped"].append({**ref, "error": type(e).__name__})
                continue
            try:
                collection = vector_index.COLLECTIONS[kind]
                if not vector_index._collection_ready(collection):
                    raise RuntimeError("Qdrant 不可用")
                occupied = vector_index.get_qdrant_client().retrieve(
                    collection_name=collection, ids=[mid], with_payload=True,
                )
                expected = {**body, "id": mid, "created_at": created, "updated_at": updated}
                if occupied and vector_index._decode(kind, occupied[0]) != expected:
                    report["conflicts"].append(ref)
                    continue
                if not occupied:
                    vector_index.put_record(kind, body, memory_id=mid,
                                            created_at=created, updated_at=updated)
                store.import_stats(kind, mid, counts)
                if vector_index.get_records(kind, [mid]) != [expected]:
                    raise RuntimeError("导入后正文核对失败")
                actual_stats = store.get_stats(kind, [mid])
                if actual_stats is None or any(
                    actual_stats.get(str(mid), {}).get(key, -1) < count
                    for key, count in counts.items()
                ):
                    raise RuntimeError("导入后统计核对失败")
                report["unchanged" if occupied else "imported"] += 1
            except Exception as e:
                report["failed"].append({**ref, "error": type(e).__name__})
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("export", "import"))
    parser.add_argument("--backup", type=Path, required=True, help="完整旧表 JSON 备份路径")
    args = parser.parse_args()
    try:
        if args.action == "export":
            document = export_legacy(args.backup)
            print(json.dumps({"exported": {k: len(v) for k, v in document["tables"].items()}},
                             ensure_ascii=False))
            return 0
        report = import_backup(load_backup(args.backup))
        print(json.dumps(report, ensure_ascii=False))
        return 2 if any(report[key] for key in ("conflicts", "skipped", "failed")) else 0
    except Exception as e:
        print(f"迁移未完成：{type(e).__name__}；旧数据未删除，请检查备份和服务状态。", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
