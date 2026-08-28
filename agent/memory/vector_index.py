"""记忆语义索引（Qdrant 向量检索）——五类记忆架构版。

为 MySQL 中的三类可检索记忆（语义/情景/程序）建立向量索引，注入时按语义
相关性检索，替代时间倒序盲注（"问太原天气却注入吴堡站水位"的问题根源）。

Collections（与法规检索的 water_regulations 隔离，point id = MySQL 行 id）：
- agent_semantic_vec   — 语义记忆（embed title+content）
- agent_episodes_vec   — 情景记忆（embed event_summary+resolution）
- agent_procedures_vec — 程序记忆（embed applicability）

三态约定（调用方据此降级）：
- None = 索引不可用（Qdrant/embedding 失败）→ 调用方回退时间倒序/LIKE
- []   = 索引可用但无语义相关结果 → 不回退不注入（防无关记忆污染）
- 非空 = 按相关性排序的命中
"""
import logging
from typing import Any

from qdrant_client.http import models as qmodels

from agent.rag.embedding import embed_query, embed_texts
from app.core.llm import get_qdrant_client, get_qdrant_config

logger = logging.getLogger(__name__)

SEMANTIC_COLLECTION = "agent_semantic_vec"
EPISODE_COLLECTION = "agent_episodes_vec"
PROCEDURE_COLLECTION = "agent_procedures_vec"

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
            logger.info("[memory-vec] 已创建 collection: %s", collection)
        return True
    except Exception as e:
        logger.debug("[memory-vec] Qdrant 不可用（%s）：%s", collection, e)
        return False


# ====== 泛化操作（三 collection 参数化） ======

def _index_one(collection: str, row_id: int, embed_text: str,
               payload: dict[str, Any]) -> bool:
    """写入/更新单条向量。失败返回 False（不影响主流程）。"""
    if not isinstance(row_id, int) or not embed_text:
        return False
    try:
        if not _collection_ready(collection):
            return False
        vec = embed_query(embed_text)
        if vec is None:
            logger.warning("[memory-vec] embedding 失败 %s id=%s", collection, row_id)
            return False
        get_qdrant_client().upsert(
            collection_name=collection,
            points=[qmodels.PointStruct(id=row_id, vector=vec.tolist(), payload=payload)],
        )
        return True
    except Exception as e:
        logger.debug("[memory-vec] index 失败 %s id=%s：%s", collection, row_id, e)
        return False


def _remove_one(collection: str, row_id: int) -> bool:
    """删除单条向量（MySQL 行删除后调用，保持两侧一致）。"""
    if not isinstance(row_id, int):
        return False
    try:
        if not _collection_ready(collection):
            return False
        get_qdrant_client().delete(
            collection_name=collection,
            points_selector=qmodels.PointIdsList(points=[row_id]),
        )
        return True
    except Exception as e:
        logger.debug("[memory-vec] remove 失败 %s id=%s：%s", collection, row_id, e)
        return False


def _search(
    collection: str,
    query: str,
    top_k: int = 5,
    min_score: float = MIN_SCORE,
) -> list[dict[str, Any]] | None:
    """按语义相关性检索（三态返回：None 不可用 / [] 无相关 / 非空命中）。"""
    if not query:
        return None
    try:
        if not _collection_ready(collection):
            return None
        vec = embed_query(query)
        if vec is None:
            return None
        response = get_qdrant_client().query_points(
            collection_name=collection,
            query=vec.tolist(),
            limit=top_k,
            with_payload=True,
        )
        results = []
        for scored in response.points:
            if float(scored.score) < min_score:
                continue
            results.append({
                "id": scored.id,
                "score": round(float(scored.score), 4),
                **(scored.payload or {}),
            })
        return results
    except Exception as e:
        logger.debug("[memory-vec] search 失败 %s：%s", collection, e)
        return None


def _sync_all(collection: str, rows: list[dict[str, Any]]) -> int:
    """全量同步索引：先 embed 成功再删旧点再 upsert（MySQL 是 source of truth）。

    先 embed 后删点，避免 embed 失败时把索引清空。
    rows 各项需含 id / embed_text。
    """
    try:
        if not _collection_ready(collection):
            return 0
        texts = [r.get("embed_text", "") for r in rows]
        vectors = embed_texts(texts) if texts else None
        if texts and vectors is None:
            return 0  # embed 失败，不动现有索引
        client = get_qdrant_client()
        client.delete(
            collection_name=collection,
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(must=[qmodels.FieldCondition(
                    key="kind", match=qmodels.MatchValue(value="memory")
                )])
            ),
        )
        if not rows:
            return 0
        points = [
            qmodels.PointStruct(
                id=int(r["id"]),
                vector=v.tolist(),
                payload={"kind": "memory", **r.get("payload", {})},
            )
            for r, v in zip(rows, vectors, strict=False)
        ]
        client.upsert(collection_name=collection, points=points)
        logger.info("[memory-vec] 同步 %s：%d points", collection, len(points))
        return len(points)
    except Exception as e:
        logger.debug("[memory-vec] sync 失败 %s：%s", collection, e)
        return 0


# ====== 语义记忆 ======

def index_semantic(mem_id: int, title: str, content: str) -> bool:
    return _index_one(
        SEMANTIC_COLLECTION, mem_id, f"{title}。{content}",
        {"title": title, "content": content},
    )


def remove_semantic(mem_id: int) -> bool:
    return _remove_one(SEMANTIC_COLLECTION, mem_id)


def search_semantic(query: str, top_k: int = 3) -> list[dict[str, Any]] | None:
    """语义记忆检索（synthesizer 注入用）。"""
    return _search(SEMANTIC_COLLECTION, query, top_k)


def sync_semantic(rows: list[dict[str, Any]]) -> int:
    """全量同步（rows 需含 id/title/content）。"""
    return _sync_all(
        SEMANTIC_COLLECTION,
        [{"id": r["id"], "embed_text": f"{r.get('title', '')}。{r.get('content', '')}",
          "payload": {"title": r.get("title", ""), "content": r.get("content", "")}}
         for r in rows],
    )


# ====== 情景记忆 ======

def index_episode(episode_id: int, event_summary: str, resolution: str = "") -> bool:
    return _index_one(
        EPISODE_COLLECTION, episode_id, f"{event_summary}。{resolution}",
        {"event_summary": event_summary, "resolution": resolution},
    )


def remove_episode(episode_id: int) -> bool:
    return _remove_one(EPISODE_COLLECTION, episode_id)


def search_episodes(query: str, top_k: int = 2) -> list[dict[str, Any]] | None:
    """情景记忆检索（planner 注入"历史类似情形"）。"""
    return _search(EPISODE_COLLECTION, query, top_k)


def sync_episodes(rows: list[dict[str, Any]]) -> int:
    """全量同步（rows 需含 id/event_summary/resolution）。"""
    return _sync_all(
        EPISODE_COLLECTION,
        [{"id": r["id"],
          "embed_text": f"{r.get('event_summary', '')}。{r.get('resolution', '')}",
          "payload": {"event_summary": r.get("event_summary", ""),
                      "resolution": r.get("resolution", "")}}
         for r in rows],
    )


# ====== 程序记忆 ======

def index_procedure(proc_id: int, applicability: str) -> bool:
    return _index_one(
        PROCEDURE_COLLECTION, proc_id, applicability,
        {"applicability": applicability},
    )


def remove_procedure(proc_id: int) -> bool:
    return _remove_one(PROCEDURE_COLLECTION, proc_id)


def search_procedures(query: str, top_k: int = 2) -> list[dict[str, Any]] | None:
    """程序记忆检索（planner 注入"推荐方法"）。"""
    return _search(PROCEDURE_COLLECTION, query, top_k)


def sync_procedures(rows: list[dict[str, Any]]) -> int:
    """全量同步（rows 需含 id/applicability）。"""
    return _sync_all(
        PROCEDURE_COLLECTION,
        [{"id": r["id"], "embed_text": r.get("applicability", ""),
          "payload": {"applicability": r.get("applicability", "")}}
         for r in rows],
    )
