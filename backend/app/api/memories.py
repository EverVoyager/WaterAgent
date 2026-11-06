"""记忆管理接口（五类记忆治理面）。

借鉴 Letta 的 memory 治理 API + Claude Code 的 /memory 命令：让记忆可被
外部显式查询/编辑/删除，而非"写入即永久不可变"。

接口：
- GET/PUT  /api/memories/longterm       MEMORY.md 用户手册读写（Agent 只读的那层）
- GET      /api/memories/auto           Agent 自动记忆概览（memory/ 目录）
- PUT/DELETE /api/memories/auto/{topic} 主题文件编辑/删除
- GET/POST /api/memories/semantic       语义记忆列表/手动添加
- DELETE   /api/memories/semantic/{id}  删除语义记忆
- GET      /api/memories/episodes       情景记忆列表
- GET      /api/memories/procedures     程序记忆列表
- POST     /api/memories/procedures/{id}/promote  手动晋升为 Skill
- POST     /api/memories/compact        手动触发一轮 Curator 治理（后台异步执行）
- POST     /api/memories/review         指定历史反思复盘，只返回并审计候选建议
- GET      /api/memories/reflections    反思日志（审计）

鉴权：MEMORY_ADMIN_API_KEY 非空时全路由要求 X-API-Key 头。
"""
import logging
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel

from agent.memory import longterm, longterm_files
from agent.memory.episode_store import get_episode_store
from agent.memory.memory_store import get_memory_store, is_memory_enabled
from agent.memory.procedure_store import get_procedure_store
from agent.memory.records import EpisodeBody, ProcedureBody, SemanticBody
from agent.memory.review import (
    MemoryReviewRequest,
    MemoryReviewResponse,
    ReflectionNotFoundError,
    review_reflections,
)
from agent.memory.semantic_store import get_semantic_store
from agent.memory.usage_store import with_usage

logger = logging.getLogger(__name__)


async def _require_admin_key(x_api_key: str | None = Header(default=None)) -> None:
    """可选治理钥匙：MEMORY_ADMIN_API_KEY 非空时强制校验 X-API-Key 头。

    本 API 可改写用户手册/删除记忆/触发治理，公网部署应配置钥匙；
    留空保持本地开发的匿名可用（前端当前不依赖此 API）。
    """
    from app.core.config import get_settings
    expected = getattr(get_settings(), "MEMORY_ADMIN_API_KEY", "")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="缺少或无效的 X-API-Key")


router = APIRouter(prefix="/api/memories", tags=["memories"],
                   dependencies=[Depends(_require_admin_key)])


# ====== 响应模型 ======

class SemanticItem(SemanticBody):
    id: int | str
    hit_count: int | None = None
    created_at: str | None = None
    updated_at: str | None = None


class SemanticListResponse(BaseModel):
    total: int
    items: list[SemanticItem]
    memory_enabled: bool


class SemanticCreateRequest(SemanticBody):
    pass


class EpisodeItem(EpisodeBody):
    id: int | str
    hit_count: int | None = None
    created_at: str | None = None
    updated_at: str | None = None


class EpisodeListResponse(BaseModel):
    total: int
    items: list[EpisodeItem]
    memory_enabled: bool


class ProcedureItem(ProcedureBody):
    id: int | str
    hit_count: int | None = None
    success_count: int | None = None
    refined_count: int | None = None
    created_at: str | None = None
    updated_at: str | None = None


class ProcedureListResponse(BaseModel):
    total: int
    items: list[ProcedureItem]
    memory_enabled: bool


class PromoteResponse(BaseModel):
    ok: bool
    skill_name: str = ""
    reason: str = ""


class DeleteResponse(BaseModel):
    deleted: bool
    id: int | str


class LongtermResponse(BaseModel):
    content: str


class LongtermUpdateRequest(BaseModel):
    content: str


class TopicRequest(BaseModel):
    content: str


class AutoMemoryResponse(BaseModel):
    index: str
    topics: list[dict[str, Any]]


class ReflectionItem(BaseModel):
    id: int
    user_query: str
    trigger_reason: str
    reflection_text: str
    memories_created: int = 0
    created_at: str | None = None
    context: dict[str, Any] | None = None


class ReflectionListResponse(BaseModel):
    total: int
    items: list[ReflectionItem]
    memory_enabled: bool


# ====== 长期记忆（用户手册） ======

@router.get("/longterm", response_model=LongtermResponse)
async def get_longterm():
    """读 MEMORY.md 用户手册原文。"""
    path = longterm._manual_file()
    content = ""
    if path.exists():
        content = path.read_text(encoding="utf-8")
    return LongtermResponse(content=content)


@router.put("/longterm", response_model=LongtermResponse)
async def update_longterm(req: LongtermUpdateRequest):
    """人工编辑 MEMORY.md（Agent 权威手册，Agent 自身只读）。"""
    path = longterm._manual_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    longterm_files._atomic_write(path, req.content.rstrip() + "\n")
    return LongtermResponse(content=longterm_files._read(path))


# ====== Agent 自动记忆（memory/ 目录） ======

@router.get("/auto", response_model=AutoMemoryResponse)
async def get_auto_memory():
    """Agent 自动记忆概览：索引 + 主题文件清单与内容。"""
    overview = longterm.get_auto_memory_overview()
    return AutoMemoryResponse(**overview)


@router.put("/auto/{topic}")
async def update_topic(topic: str, req: TopicRequest):
    """人工编辑指定主题文件。"""
    if not longterm.write_topic(topic, req.content):
        raise HTTPException(status_code=400, detail=f"非法主题名：{topic}")
    return {"ok": True, "topic": topic}


@router.delete("/auto/{topic}")
async def delete_topic(topic: str):
    """删除主题文件并更新索引。"""
    if not longterm.delete_topic(topic):
        raise HTTPException(status_code=404, detail=f"主题不存在：{topic}")
    return {"ok": True, "topic": topic}


# ====== 语义记忆 ======

def _items(model, kind: str, rows: list[dict]) -> list:
    return [model(**{k: v for k, v in row.items() if k in model.model_fields})
            for row in with_usage(kind, rows)]


def _unavailable(error: Exception) -> HTTPException:
    logger.warning("[memories-api] 正文操作失败：%s", type(error).__name__)
    return HTTPException(status_code=503, detail="Qdrant 记忆正文服务不可用")


def _memory_id(value: UUID | int) -> int | str:
    from agent.memory.vector_index import point_id

    try:
        return point_id(value)
    except ValueError:
        raise HTTPException(status_code=422, detail="无效的记忆 ID")


@router.get("/semantic", response_model=SemanticListResponse)
async def list_semantic(limit: int = Query(100, ge=1, le=500),
                        days_back: int | None = Query(None, ge=1)):
    try:
        rows = get_semantic_store().list_semantic(limit=limit, days_back=days_back)
        items = _items(SemanticItem, "semantic", rows)
    except Exception as e:
        raise _unavailable(e)
    return SemanticListResponse(total=len(items), items=items, memory_enabled=True)


@router.post("/semantic", response_model=SemanticItem, status_code=201)
async def create_semantic(req: SemanticCreateRequest):
    try:
        store = get_semantic_store()
        mem_id = store.add_semantic(**req.model_dump())
    except ValueError:
        raise HTTPException(status_code=422, detail="正文、适用条件或证据未通过校验")
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise _unavailable(e)
    try:
        return _items(SemanticItem, "semantic", store.get_by_ids([mem_id]))[0]
    except Exception:
        raise HTTPException(status_code=503, detail=f"记忆 {mem_id} 已保存，读取回执失败，请按 ID 核查")


@router.delete("/semantic/{semantic_id}", response_model=DeleteResponse)
async def delete_semantic(semantic_id: UUID | int):
    semantic_id = _memory_id(semantic_id)
    try:
        deleted = get_semantic_store().delete_semantic(semantic_id)
    except Exception as e:
        raise _unavailable(e)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"语义记忆 {semantic_id} 不存在")
    return DeleteResponse(deleted=True, id=semantic_id)


# ====== 情景记忆 ======

@router.get("/episodes", response_model=EpisodeListResponse)
async def list_episodes(limit: int = Query(100, ge=1, le=500),
                        days_back: int | None = Query(None, ge=1),
                        outcome: str | None = Query(None)):
    try:
        rows = get_episode_store().list_episodes(limit=limit, days_back=days_back, outcome=outcome)
        items = _items(EpisodeItem, "episode", rows)
    except Exception as e:
        raise _unavailable(e)
    return EpisodeListResponse(total=len(items), items=items, memory_enabled=True)


# ====== 程序记忆 ======

@router.get("/procedures", response_model=ProcedureListResponse)
async def list_procedures(limit: int = Query(100, ge=1, le=500),
                          status: str | None = Query(None)):
    try:
        rows = get_procedure_store().list_procedures(limit=limit, status=status)
        items = _items(ProcedureItem, "procedure", rows)
    except Exception as e:
        raise _unavailable(e)
    return ProcedureListResponse(total=len(items), items=items, memory_enabled=True)


@router.post("/procedures/{procedure_id}/promote", response_model=PromoteResponse)
async def promote_procedure(procedure_id: UUID | int):
    """手动把程序记忆晋升为候选 Skill（enabled=false，去技能管理页确认启用）。"""
    if not is_memory_enabled():
        raise HTTPException(status_code=503, detail="MySQL 未配置")
    try:
        result = get_procedure_store().promote_to_skill(_memory_id(procedure_id), auto_enable=False)
    except HTTPException:
        raise
    except Exception as e:
        raise _unavailable(e)
    if not result.get("ok"):
        raise HTTPException(status_code=409, detail=result.get("reason", "晋升失败"))
    return PromoteResponse(**result)


# ====== 压缩 / 审计 ======

class CompactResponse(BaseModel):
    started: bool


@router.post("/review", response_model=MemoryReviewResponse)
def review_memories(req: MemoryReviewRequest):
    """复盘指定的历史任务；同步返回候选，记忆和 Skill 均不改写。"""
    if not is_memory_enabled():
        raise HTTPException(status_code=503, detail="MySQL 未配置，历史复盘不可用")
    from agent.memory.reflection import _reflection_rate_exceeded
    if _reflection_rate_exceeded():
        raise HTTPException(status_code=429, detail="反思及复盘请求过多，请稍后重试")
    try:
        return review_reflections(req)
    except ReflectionNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.warning("[memories-api] 历史复盘失败：%s", type(e).__name__)
        raise HTTPException(status_code=503, detail="复盘或审计服务不可用，未应用任何候选")


@router.post("/compact", response_model=CompactResponse, status_code=202)
async def compact_memories():
    """手动触发一轮 Curator 治理（后台线程异步执行）。

    完整治理含多次 LLM 调用（分钟级），在请求内同步执行
    会阻塞事件循环，因此提交后台线程后立即返回。
    """
    import threading

    from agent.memory.curator import run_curation_once

    def _run():
        try:
            run_curation_once()
        except Exception as e:
            logger.warning("[memories-api] 手动治理失败：%s", e)

    threading.Thread(target=_run, daemon=True, name="manual-curator").start()
    return CompactResponse(started=True)


@router.get("/reflections", response_model=ReflectionListResponse)
async def list_reflections(limit: int = Query(50, ge=1, le=200),
                           days_back: int | None = Query(None, ge=1)):
    if not is_memory_enabled():
        return ReflectionListResponse(total=0, items=[], memory_enabled=False)
    rows = get_memory_store().list_reflections(limit=limit, days_back=days_back)
    items = [ReflectionItem(
        id=r["id"], user_query=r["user_query"],
        trigger_reason=r["trigger_reason"], reflection_text=r["reflection_text"],
        memories_created=r.get("memories_created", 0),
        created_at=str(r.get("created_at", "")),
        context=r.get("context"),
    ) for r in rows]
    return ReflectionListResponse(total=len(items), items=items, memory_enabled=True)
