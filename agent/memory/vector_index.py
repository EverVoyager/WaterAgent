"""三类记忆的 Qdrant 主存：payload 是正文，不从 SQL 重建或回表读取。"""
import logging
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID, uuid4

from qdrant_client.http import models as qmodels

from agent.memory.records import validate_body
from agent.rag.embedding import embed_query
from agent.utils import now_iso
from app.core.llm import get_qdrant_client, get_qdrant_config

logger = logging.getLogger(__name__)

SEMANTIC_COLLECTION = "agent_semantic_v2"
EPISODE_COLLECTION = "agent_episodes_v2"
PROCEDURE_COLLECTION = "agent_procedures_v2"
COLLECTIONS = {"semantic": SEMANTIC_COLLECTION, "episode": EPISODE_COLLECTION,
               "procedure": PROCEDURE_COLLECTION}
MIN_SCORE = 0.35


def is_older(value: str, days: int) -> bool:
    try:
        stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return False
    # 旧数据时区未知时保留原文，不根据猜测的时间执行删除。
    return stamp.tzinfo is not None and stamp < datetime.now(timezone.utc) - timedelta(days=days)


def point_id(value: int | str | UUID) -> int | str:
    """Qdrant 原生 ID：兼容迁移前的整数，新记录使用 UUID。"""
    if type(value) is int and 0 <= value < 2**64:
        return value
    if isinstance(value, str) and value.isdecimal():
        number = int(value)
        if 0 <= number < 2**64:
            return number
    try:
        return str(UUID(str(value)))
    except (ValueError, TypeError, AttributeError) as e:
        raise ValueError("无效的记忆 ID") from e


def _collection_ready(collection: str) -> bool:
    try:
        client = get_qdrant_client()
        if not client.collection_exists(collection):
            try:
                client.create_collection(
                    collection_name=collection,
                    vectors_config=qmodels.VectorParams(
                        size=get_qdrant_config()["vector_size"], distance=qmodels.Distance.COSINE,
                    ),
                )
            except Exception:
                if not client.collection_exists(collection):
                    raise
        config = client.get_collection(collection).config.params.vectors
        if (not isinstance(config, qmodels.VectorParams)
                or config.size != get_qdrant_config()["vector_size"]
                or config.distance != qmodels.Distance.COSINE):
            raise RuntimeError("记忆集合与当前 embedding 维度或距离类型不一致")
        return True
    except Exception as e:
        logger.warning("[memory-qdrant] 集合不可用 %s：%s", collection, type(e).__name__)
        return False


def _filter(kind: str, status: str | None = None) -> qmodels.Filter:
    conditions = [qmodels.FieldCondition(key="memory_type", match=qmodels.MatchValue(value=kind)),
                  qmodels.FieldCondition(key="schema_version", match=qmodels.MatchValue(value=2))]
    if status:
        conditions.append(qmodels.FieldCondition(key="status", match=qmodels.MatchValue(value=status)))
    return qmodels.Filter(must=conditions)


def _decode(kind: str, record) -> dict | None:
    payload = dict(record.payload or {})
    if payload.pop("memory_type", None) != kind or payload.pop("schema_version", None) != 2:
        return None
    created_at, updated_at = payload.pop("created_at", ""), payload.pop("updated_at", "")
    try:
        body = validate_body(kind, payload)
    except (ValueError, TypeError):
        logger.warning("[memory-qdrant] 跳过不符合正文结构或安全校验的记录 %s id=%s", kind, record.id)
        return None
    return {**body, "id": record.id, "created_at": created_at, "updated_at": updated_at}


def get_records(kind: str, ids: list[int | str]) -> list[dict]:
    if not ids:
        return []
    collection = COLLECTIONS[kind]
    if not _collection_ready(collection):
        raise RuntimeError("Qdrant 记忆正文不可用")
    points = get_qdrant_client().retrieve(
        collection_name=collection, ids=[point_id(mid) for mid in ids], with_payload=True,
    )
    return [row for record in points if (row := _decode(kind, record)) is not None]


def iter_records(kind: str):
    collection = COLLECTIONS[kind]
    if not _collection_ready(collection):
        raise RuntimeError("Qdrant 记忆正文不可用")
    offset = None
    while True:
        points, offset = get_qdrant_client().scroll(
            collection_name=collection, scroll_filter=_filter(kind), limit=100,
            offset=offset, with_payload=True,
        )
        for record in points:
            if (row := _decode(kind, record)) is not None:
                yield row
        if offset is None:
            break


def list_records(kind: str, limit: int = 100, days_back: int | None = None,
                 status: str | None = None, outcome: str | None = None) -> list[dict]:
    # shortcut: 当前小库全量排序，记录规模明显增大时改用 payload 索引分页。
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days_back)).date().isoformat() if days_back else ""
    rows = [row for row in iter_records(kind)
            if (not status or row.get("status") == status)
            and (not outcome or row.get("outcome") == outcome)
            and (not cutoff or str(row.get("created_at", ""))[:10] >= cutoff)]
    rows.sort(key=lambda row: str(row.get("created_at", "")), reverse=True)
    return rows[:limit]


def put_record(kind: str, body: dict[str, Any], memory_id: int | str | None = None,
               created_at: str | None = None, updated_at: str | None = None) -> int | str:
    body = validate_body(kind, body)
    memory_id = point_id(memory_id) if memory_id is not None else str(uuid4())
    collection = COLLECTIONS[kind]
    if not _collection_ready(collection):
        raise RuntimeError("Qdrant 记忆正文不可用")
    if kind == "semantic":
        text = f"{body['title']}。{body['content']}。适用：{body['reuse_when']}"
    elif kind == "episode":
        text = f"{body['event_summary']}。{body['resolution']}"
    else:
        text = f"{body['name']}。{body['applicability']}"
    vector = embed_query(text)
    if vector is None:
        raise RuntimeError("记忆 embedding 失败，原正文未修改")
    vector = vector.tolist() if hasattr(vector, "tolist") else list(vector)
    stamp = now_iso()
    payload = {**body, "memory_type": kind, "schema_version": 2,
               "created_at": created_at or stamp, "updated_at": updated_at or stamp}
    try:
        get_qdrant_client().upsert(
            collection_name=collection,
            points=[qmodels.PointStruct(id=memory_id, vector=vector, payload=payload)], wait=True,
        )
    except Exception as e:
        raise RuntimeError(f"记忆写入未确认，请按 ID 核查后重试：{memory_id}") from e
    return memory_id


def update_record(kind: str, memory_id: int | str, changes: dict) -> bool:
    rows = get_records(kind, [memory_id])
    if not rows:
        return False
    old = rows[0]
    body = {key: value for key, value in old.items() if key not in ("id", "created_at", "updated_at")}
    body.update(changes)
    body = validate_body(kind, body)
    if set(changes) <= {"status"}:
        get_qdrant_client().set_payload(
            collection_name=COLLECTIONS[kind],
            payload={**{key: body[key] for key in changes}, "updated_at": now_iso()},
            points=[old["id"]], wait=True,
        )
        return True
    # shortcut: 并发编辑采用后完成写入生效，多管理员并发时升级条件更新。
    put_record(kind, body, memory_id=old["id"], created_at=old["created_at"])
    return True


def delete_records(kind: str, ids: list[int | str]) -> int:
    if not ids:
        return 0
    collection = COLLECTIONS[kind]
    if not _collection_ready(collection):
        raise RuntimeError("Qdrant 记忆正文不可用")
    points = get_qdrant_client().retrieve(
        collection_name=collection, ids=[point_id(mid) for mid in ids], with_payload=True,
    )
    existing = [p.id for p in points if (p.payload or {}).get("memory_type") == kind
                and (p.payload or {}).get("schema_version") == 2]
    if existing:
        get_qdrant_client().delete(
            collection_name=collection,
            points_selector=qmodels.PointIdsList(points=existing), wait=True,
        )
    return len(existing)


def _search(kind: str, query: str, top_k: int, min_score: float = MIN_SCORE) -> list[dict] | None:
    if not query:
        return []
    try:
        if not _collection_ready(COLLECTIONS[kind]):
            return None
        if get_qdrant_client().count(
            collection_name=COLLECTIONS[kind], count_filter=_filter(kind), exact=True,
        ).count == 0:
            return []
        vector = embed_query(query)
        if vector is None:
            return None
        response = get_qdrant_client().query_points(
            collection_name=COLLECTIONS[kind],
            query=vector.tolist() if hasattr(vector, "tolist") else list(vector),
            query_filter=_filter(kind, "active" if kind == "procedure" else None),
            limit=top_k, with_payload=True, score_threshold=min_score,
        )
        return [{**row, "score": float(record.score)} for record in response.points
                if float(record.score) >= min_score and (row := _decode(kind, record)) is not None]
    except Exception as e:
        logger.warning("[memory-qdrant] 召回不可用 %s：%s", kind, type(e).__name__)
        return None


def search_semantic(query: str, top_k: int = 3) -> list[dict] | None:
    return _search("semantic", query, top_k)


def search_episodes(query: str, top_k: int = 2) -> list[dict] | None:
    return _search("episode", query, top_k)


def search_procedures(query: str, top_k: int = 2) -> list[dict] | None:
    return _search("procedure", query, top_k)
