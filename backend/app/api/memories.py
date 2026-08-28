"""记忆管理接口（治理面）。

借鉴 Letta 的系统级 memory API：让记忆可被外部显式查询/删除，
而非"写入即永久不可变"。当 Agent 出现"固执拒绝做某事"等异常行为时，
可通过此接口排查并清理问题记忆。

接口：
- GET /api/memories        列表（支持 type 过滤、分页）
- DELETE /api/memories/{id}  删除单条
- POST /api/memories/compact  手动触发压缩（按类型）
- GET /api/memories/reflections  反思日志列表（审计）
"""
import logging

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from agent.memory.memory_store import (
    MemoryType,
    get_memory_store,
    is_memory_enabled,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/memories", tags=["memories"])


class MemoryItem(BaseModel):
    """单条记忆的响应结构。"""
    id: int
    memory_type: str
    content: str
    context: str | None = None
    tags: str | None = None
    hit_count: int = 0
    created_at: str | None = None


class MemoryListResponse(BaseModel):
    """记忆列表响应。"""
    total: int
    items: list[MemoryItem]
    memory_enabled: bool


class DeleteResponse(BaseModel):
    """删除响应。"""
    deleted: bool
    id: int


class CompactRequest(BaseModel):
    """压缩请求。"""
    memory_type: str  # tool_failure / user_preference / ...


class CompactResponse(BaseModel):
    """压缩响应。"""
    memory_type: str
    deleted: int  # 负数表示失败


class ReflectionItem(BaseModel):
    """反思日志条目。"""
    id: int
    user_query: str
    trigger_reason: str
    tool_calls_summary: str | None = None
    final_answer: str | None = None
    reflection_text: str
    memories_created: int = 0
    created_at: str | None = None


class ReflectionListResponse(BaseModel):
    """反思日志列表响应。"""
    total: int
    items: list[ReflectionItem]
    memory_enabled: bool


def _to_memory_item(row: dict) -> MemoryItem:
    """把 DB 行转为响应模型。"""
    return MemoryItem(
        id=row.get("id", 0),
        memory_type=row.get("memory_type", ""),
        content=row.get("content", ""),
        context=str(row.get("context") or "") if row.get("context") else None,
        tags=row.get("tags"),
        hit_count=row.get("hit_count", 0) or 0,
        created_at=str(row.get("created_at")) if row.get("created_at") else None,
    )


def _to_reflection_item(row: dict) -> ReflectionItem:
    """把反思日志 DB 行转为响应模型。"""
    return ReflectionItem(
        id=row.get("id", 0),
        user_query=row.get("user_query", ""),
        trigger_reason=row.get("trigger_reason", ""),
        tool_calls_summary=row.get("tool_calls_summary"),
        final_answer=row.get("final_answer"),
        reflection_text=row.get("reflection_text", ""),
        memories_created=row.get("memories_created", 0) or 0,
        created_at=str(row.get("created_at")) if row.get("created_at") else None,
    )


@router.get("", response_model=MemoryListResponse)
def list_memories(
    memory_type: str | None = Query(None, description="按类型过滤：tool_failure/user_preference/..."),
    limit: int = Query(50, ge=1, le=500, description="返回上限"),
    days_back: int | None = Query(None, ge=1, le=365, description="只返回最近 N 天的记录"),
):
    """列出长期记忆（支持按类型、时间过滤）。"""
    enabled = is_memory_enabled()
    if not enabled:
        return MemoryListResponse(total=0, items=[], memory_enabled=False)

    store = get_memory_store()
    mem_type = None
    if memory_type:
        try:
            mem_type = MemoryType(memory_type)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"未知记忆类型: {memory_type}。可选: {[t.value for t in MemoryType]}",
            ) from None

    rows = store.get_memories(
        memory_type=mem_type,
        limit=limit,
        days_back=days_back,
    )
    items = [_to_memory_item(r) for r in rows]
    return MemoryListResponse(total=len(items), items=items, memory_enabled=True)


@router.delete("/{memory_id}", response_model=DeleteResponse)
def delete_memory(memory_id: int):
    """删除单条记忆。

    用于治理：当某条记忆导致 Agent 行为异常时，可通过此接口删除。
    同步清理向量索引，防止僵尸点继续被语义检索命中。
    """
    if not is_memory_enabled():
        raise HTTPException(status_code=503, detail="记忆模块未启用")
    store = get_memory_store()
    deleted = store.delete_memory(memory_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"记忆 id={memory_id} 不存在或删除失败")
    try:
        from agent.memory import vector_index
        vector_index.remove_memory(memory_id)
    except Exception as e:
        logger.debug("[api] 清理记忆向量索引失败 id=%s：%s", memory_id, e)
    logger.info("[api] 删除记忆 id=%s", memory_id)
    return DeleteResponse(deleted=True, id=memory_id)


@router.post("/compact", response_model=CompactResponse)
def compact_memories(req: CompactRequest):
    """手动触发某类型记忆的压缩（LLM 语义合并）。

    用于治理：积累过多相似记忆时手动整理。
    压缩后同步向量索引，保持 MySQL 与索引一致。
    """
    if not is_memory_enabled():
        raise HTTPException(status_code=503, detail="记忆模块未启用")
    try:
        mem_type = MemoryType(req.memory_type)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"未知记忆类型: {req.memory_type}",
        ) from None

    store = get_memory_store()
    # 复用 reflection.py 的 LLM 压缩函数
    from agent.memory.reflection import _llm_compact_memories, _sync_type_index
    deleted = store.compact_memories(mem_type, _llm_compact_memories)
    if deleted > 0:
        # 压缩改变了行集合，同步向量索引
        try:
            _sync_type_index(store, mem_type)
        except Exception as e:
            logger.debug("[api] 同步向量索引失败：%s", e)
    return CompactResponse(memory_type=req.memory_type, deleted=deleted)


@router.get("/reflections", response_model=ReflectionListResponse)
def list_reflections(
    limit: int = Query(20, ge=1, le=100, description="返回上限"),
):
    """列出反思日志（审计用）。

    反思日志记录每次反思的完整过程，便于排查 Agent 行为异常的根因。
    """
    enabled = is_memory_enabled()
    if not enabled:
        return ReflectionListResponse(total=0, items=[], memory_enabled=False)

    store = get_memory_store()
    if not store.enabled:
        return ReflectionListResponse(total=0, items=[], memory_enabled=False)

    try:
        import pymysql.cursors
        with store._get_conn() as conn, conn.cursor(pymysql.cursors.DictCursor) as cur:
            cur.execute(
                "SELECT id, user_query, trigger_reason, tool_calls_summary, "
                "final_answer, reflection_text, memories_created, created_at "
                "FROM agent_reflections ORDER BY created_at DESC LIMIT %s",
                (limit,),
            )
            rows = cur.fetchall()
        items = [_to_reflection_item(r) for r in rows]
        return ReflectionListResponse(total=len(items), items=items, memory_enabled=True)
    except Exception as e:
        logger.warning("[api] 查询反思日志失败：%s", e)
        raise HTTPException(status_code=500, detail=f"查询失败: {e}") from e
