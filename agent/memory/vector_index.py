"""记忆语义索引（Qdrant 向量检索）。

借鉴 Hermes Agent（SQLite FTS5 记忆检索）与 Letta（archival memory 向量检索）：
为 MySQL 中的长期记忆/技能记忆建立向量索引，注入时按语义相关性检索，
替代时间倒序盲注（"问太原天气却注入吴堡站水位"的问题根源）。

Collections（与法规检索的 water_regulations 隔离）：
- agent_memories_vec — 长期记忆（point id = MySQL agent_memories.id）
- agent_skills_vec   — 技能记忆（point id = MySQL agent_skills.id）

降级策略：Qdrant 不可达或 embedding 失败时返回 None，调用方回退到
时间倒序 / LIKE 检索；返回 [] 表示"索引可用但无语义相关结果"，
此时不回退（避免又注入无关记忆）。
"""
import logging
from typing import Any

from qdrant_client.http import models as qmodels

from agent.rag.embedding import embed_query, embed_texts
from app.core.llm import get_qdrant_client, get_qdrant_config

logger = logging.getLogger(__name__)

MEMORY_COLLECTION = "agent_memories_vec"
SKILL_COLLECTION = "agent_skills_vec"

# 语义相关性阈值：低于此分数视为无关（与 skill matcher 的 0.55 相比略低，
# 因为记忆内容与 query 的表述差异通常更大）
MIN_SCORE = 0.35


def _collection_ready(collection: str) -> bool:
    """检查 Qdrant 可达且 collection 存在（不存在则创建）。"""
    try:
        client = get_qdrant_client()
        cols = [c.name for c in client.get_collections().collections]
        if collection not in cols:
            client.create_collection(
                collection_name=collection,
                vectors_config=qmodels.VectorParams(
                    size=get_qdrant_config()["vector_size"],
                    distance=qmodels.Distance.COSINE,
                ),
            )
            # 为过滤字段建索引
            try:
                field = "memory_type" if collection == MEMORY_COLLECTION else "kind"
                client.create_payload_index(
                    collection_name=collection,
                    field_name=field,
                    field_schema=qmodels.PayloadSchemaType.KEYWORD,
                )
            except Exception:
                pass
            logger.info("[memory-vec] 已创建 collection: %s", collection)
        return True
    except Exception as e:
        logger.debug("[memory-vec] Qdrant 不可用（%s）：%s", collection, e)
        return False


# ====== 长期记忆索引 ======

def index_memory(mem_id: int, memory_type: str, content: str, tags: list[str] | None = None) -> bool:
    """写入/更新单条记忆的向量索引。失败返回 False（不影响主流程）。"""
    if not isinstance(mem_id, int) or not content:
        return False
    try:
        if not _collection_ready(MEMORY_COLLECTION):
            return False
        vec = embed_query(content)
        if vec is None:
            logger.warning("[memory-vec] 记忆 embedding 失败 id=%s", mem_id)
            return False
        get_qdrant_client().upsert(
            collection_name=MEMORY_COLLECTION,
            points=[qmodels.PointStruct(
                id=mem_id,
                vector=vec.tolist(),
                payload={
                    "memory_type": memory_type,
                    "content": content,
                    "tags": ",".join(tags) if tags else "",
                },
            )],
        )
        return True
    except Exception as e:
        logger.debug("[memory-vec] index_memory 失败 id=%s：%s", mem_id, e)
        return False


def remove_memory(mem_id: int) -> bool:
    """从索引中删除单条记忆（MySQL 行删除后调用，保持两侧一致）。"""
    if not isinstance(mem_id, int):
        return False
    try:
        if not _collection_ready(MEMORY_COLLECTION):
            return False
        get_qdrant_client().delete(
            collection_name=MEMORY_COLLECTION,
            points_selector=qmodels.PointIdsList(points=[mem_id]),
        )
        return True
    except Exception as e:
        logger.debug("[memory-vec] remove_memory 失败 id=%s：%s", mem_id, e)
        return False


def search_memories(
    query: str,
    memory_types: list[str] | None = None,
    top_k: int = 10,
    min_score: float = MIN_SCORE,
) -> list[dict[str, Any]] | None:
    """按语义相关性检索记忆。

    Args:
        query: 当前用户查询
        memory_types: 限定记忆类型（None 表示不过滤）
        top_k: 返回前 K 条
        min_score: 最低余弦相似度

    Returns:
        [{"id", "memory_type", "content", "score"}, ...]；
        None 表示索引不可用（调用方应降级），[] 表示无相关结果。
    """
    if not query:
        return None
    try:
        if not _collection_ready(MEMORY_COLLECTION):
            return None
        vec = embed_query(query)
        if vec is None:
            return None
        query_filter = None
        if memory_types:
            query_filter = qmodels.Filter(
                should=[
                    qmodels.FieldCondition(
                        key="memory_type", match=qmodels.MatchValue(value=t)
                    ) for t in memory_types
                ]
            )
        response = get_qdrant_client().query_points(
            collection_name=MEMORY_COLLECTION,
            query=vec.tolist(),
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )
        results = []
        for scored in response.points:
            if float(scored.score) < min_score:
                continue
            payload = scored.payload or {}
            results.append({
                "id": scored.id,
                "memory_type": payload.get("memory_type", ""),
                "content": payload.get("content", ""),
                "score": round(float(scored.score), 4),
            })
        return results
    except Exception as e:
        logger.debug("[memory-vec] search_memories 失败：%s", e)
        return None


def sync_memory_type(memory_type: str, rows: list[dict[str, Any]]) -> int:
    """全量同步某类型记忆到索引：先 embed，成功后删旧点再 upsert 当前行。

    用于压缩/治理后对账（MySQL 是 source of truth，索引可随时重建）。
    先 embed 后删点，避免 embed 失败时把索引清空。

    Args:
        memory_type: 记忆类型
        rows: 该类型全部当前行（需含 id/content 字段）

    Returns:
        成功同步的点数（0 表示失败或无行）
    """
    try:
        if not _collection_ready(MEMORY_COLLECTION):
            return 0
        contents = [r.get("content", "") for r in rows]
        vectors = embed_texts(contents) if contents else None
        if contents and vectors is None:
            return 0  # embed 失败，不动现有索引
        client = get_qdrant_client()
        # 删除该类型所有旧点（含已从 MySQL 删除的僵尸点）
        client.delete(
            collection_name=MEMORY_COLLECTION,
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(must=[qmodels.FieldCondition(
                    key="memory_type", match=qmodels.MatchValue(value=memory_type)
                )])
            ),
        )
        if not rows:
            return 0
        points = [
            qmodels.PointStruct(
                id=int(r["id"]),
                vector=v.tolist(),
                payload={
                    "memory_type": memory_type,
                    "content": r.get("content", ""),
                    "tags": r.get("tags") or "",
                },
            )
            for r, v in zip(rows, vectors, strict=False)
        ]
        client.upsert(collection_name=MEMORY_COLLECTION, points=points)
        logger.info("[memory-vec] 同步 %s 索引：%d points", memory_type, len(points))
        return len(points)
    except Exception as e:
        logger.debug("[memory-vec] sync_memory_type 失败 type=%s：%s", memory_type, e)
        return 0


# ====== 技能记忆索引 ======

def index_skill(skill_id: int, query_pattern: str) -> bool:
    """写入/更新单条技能的向量索引。"""
    if not isinstance(skill_id, int) or not query_pattern:
        return False
    try:
        if not _collection_ready(SKILL_COLLECTION):
            return False
        vec = embed_query(query_pattern)
        if vec is None:
            return False
        get_qdrant_client().upsert(
            collection_name=SKILL_COLLECTION,
            points=[qmodels.PointStruct(
                id=skill_id,
                vector=vec.tolist(),
                payload={"kind": "skill", "query_pattern": query_pattern},
            )],
        )
        return True
    except Exception as e:
        logger.debug("[memory-vec] index_skill 失败 id=%s：%s", skill_id, e)
        return False


def search_skills(
    query: str,
    top_k: int = 3,
    min_score: float = MIN_SCORE,
) -> list[dict[str, Any]] | None:
    """按语义相关性检索技能。

    Returns:
        [{"id", "query_pattern", "score"}, ...]；None=索引不可用；[]=无相关技能。
    """
    if not query:
        return None
    try:
        if not _collection_ready(SKILL_COLLECTION):
            return None
        vec = embed_query(query)
        if vec is None:
            return None
        response = get_qdrant_client().query_points(
            collection_name=SKILL_COLLECTION,
            query=vec.tolist(),
            limit=top_k,
            with_payload=True,
        )
        results = []
        for scored in response.points:
            if float(scored.score) < min_score:
                continue
            payload = scored.payload or {}
            results.append({
                "id": scored.id,
                "query_pattern": payload.get("query_pattern", ""),
                "score": round(float(scored.score), 4),
            })
        return results
    except Exception as e:
        logger.debug("[memory-vec] search_skills 失败：%s", e)
        return None


def sync_skills(rows: list[dict[str, Any]]) -> int:
    """全量同步技能索引（rows 需含 id/query_pattern）。"""
    try:
        if not _collection_ready(SKILL_COLLECTION):
            return 0
        patterns = [r.get("query_pattern", "") for r in rows]
        vectors = embed_texts(patterns) if patterns else None
        if patterns and vectors is None:
            return 0
        client = get_qdrant_client()
        client.delete(
            collection_name=SKILL_COLLECTION,
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(must=[qmodels.FieldCondition(
                    key="kind", match=qmodels.MatchValue(value="skill")
                )])
            ),
        )
        if not rows:
            return 0
        points = [
            qmodels.PointStruct(
                id=int(r["id"]),
                vector=v.tolist(),
                payload={"kind": "skill", "query_pattern": r.get("query_pattern", "")},
            )
            for r, v in zip(rows, vectors, strict=False)
        ]
        client.upsert(collection_name=SKILL_COLLECTION, points=points)
        logger.info("[memory-vec] 同步技能索引：%d points", len(points))
        return len(points)
    except Exception as e:
        logger.debug("[memory-vec] sync_skills 失败：%s", e)
        return 0
