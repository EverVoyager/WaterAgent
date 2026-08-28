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
- POST     /api/memories/compact        手动触发一轮 Curator 治理
- GET      /api/memories/reflections    反思日志（审计）
"""
import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from agent.memory import longterm
from agent.memory.episode_store import get_episode_store
from agent.memory.memory_store import get_memory_store, is_memory_enabled
from agent.memory.procedure_store import get_procedure_store
from agent.memory.semantic_store import get_semantic_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/memories", tags=["memories"])


# ====== 响应模型 ======

class SemanticItem(BaseModel):
    id: int
    title: str
    content: str
    source: str = "reflection"
    tags: str | None = None
    hit_count: int = 0
    created_at: str | None = None


class SemanticListResponse(BaseModel):
    total: int
    items: list[SemanticItem]
    memory_enabled: bool


class SemanticCreateRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=256)
    content: str = Field(..., min_length=1)
    tags: str = ""


class EpisodeItem(BaseModel):
    id: int
    happened_at: str | None = None
    event_summary: str
    resolution: str | None = None
    outcome: str = "success"
    query_summary: str | None = None
    tags: str | None = None


class EpisodeListResponse(BaseModel):
    total: int
    items: list[EpisodeItem]
    memory_enabled: bool


class ProcedureStep(BaseModel):
    step: int = 1
    action: str = ""
    tool: str | None = None


class ProcedureItem(BaseModel):
    id: int
    name: str
    applicability: str
    steps: list[ProcedureStep] = []
    tool_sequence: list[str] = []
    source: str = "reflection"
    use_count: int = 0
    success_count: int = 0
    refined_count: int = 0
    status: str = "active"
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
    id: int


class LongtermResponse(BaseModel):
    content: str


class LongtermUpdateRequest(BaseModel):
    content: str


class TopicRequest(BaseModel):
    content: str


class AutoMemoryResponse(BaseModel):
    index: str
    topics: list[dict[str, Any]]


class CompactResponse(BaseModel):
    compacted: int


class ReflectionItem(BaseModel):
    id: int
    user_query: str
    trigger_reason: str
    reflection_text: str
    memories_created: int = 0
    created_at: str | None = None


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
    longterm._atomic_write(path, req.content.rstrip() + "\n")
    return LongtermResponse(content=longterm._read(path))


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

@router.get("/semantic", response_model=SemanticListResponse)
async def list_semantic(limit: int = Query(100, ge=1, le=500),
                        days_back: int | None = Query(None, ge=1)):
    if not is_memory_enabled():
        return SemanticListResponse(total=0, items=[], memory_enabled=False)
    rows = get_semantic_store().list_semantic(limit=limit, days_back=days_back)
    items = [SemanticItem(
        id=r["id"], title=r["title"], content=r["content"],
        source=r.get("source", ""), tags=r.get("tags"),
        hit_count=r.get("hit_count", 0),
        created_at=str(r.get("created_at", "")),
    ) for r in rows]
    return SemanticListResponse(total=len(items), items=items, memory_enabled=True)


@router.post("/semantic", response_model=SemanticItem, status_code=201)
async def create_semantic(req: SemanticCreateRequest):
    if not is_memory_enabled():
        raise HTTPException(status_code=503, detail="MySQL 未配置，语义记忆不可用")
    from agent.memory import vector_index
    mem_id = get_semantic_store().add_semantic(
        title=req.title, content=req.content, source="manual", tags=req.tags)
    if not isinstance(mem_id, int):
        raise HTTPException(status_code=500, detail="写入失败")
    vector_index.index_semantic(mem_id, req.title, req.content)
    return SemanticItem(id=mem_id, title=req.title, content=req.content,
                        source="manual", tags=req.tags or None,
                        created_at=str(datetime.now()))


@router.delete("/semantic/{semantic_id}", response_model=DeleteResponse)
async def delete_semantic(semantic_id: int):
    if not is_memory_enabled():
        raise HTTPException(status_code=503, detail="MySQL 未配置")
    from agent.memory import vector_index
    if not get_semantic_store().delete_semantic(semantic_id):
        raise HTTPException(status_code=404, detail=f"语义记忆 {semantic_id} 不存在")
    vector_index.remove_semantic(semantic_id)
    return DeleteResponse(deleted=True, id=semantic_id)


# ====== 情景记忆 ======

@router.get("/episodes", response_model=EpisodeListResponse)
async def list_episodes(limit: int = Query(100, ge=1, le=500),
                        days_back: int | None = Query(None, ge=1),
                        outcome: str | None = Query(None)):
    if not is_memory_enabled():
        return EpisodeListResponse(total=0, items=[], memory_enabled=False)
    rows = get_episode_store().list_episodes(limit=limit, days_back=days_back,
                                             outcome=outcome)
    items = [EpisodeItem(
        id=r["id"], happened_at=str(r.get("happened_at", "")),
        event_summary=r["event_summary"], resolution=r.get("resolution"),
        outcome=r.get("outcome", ""), query_summary=r.get("query_summary"),
        tags=r.get("tags"),
    ) for r in rows]
    return EpisodeListResponse(total=len(items), items=items, memory_enabled=True)


# ====== 程序记忆 ======

@router.get("/procedures", response_model=ProcedureListResponse)
async def list_procedures(limit: int = Query(100, ge=1, le=500),
                          status: str | None = Query(None)):
    if not is_memory_enabled():
        return ProcedureListResponse(total=0, items=[], memory_enabled=False)
    rows = get_procedure_store().list_procedures(limit=limit, status=status)
    import json as _json
    items = []
    for r in rows:
        try:
            steps = _json.loads(r.get("steps_json") or "[]")
        except (ValueError, TypeError):
            steps = []
        try:
            tool_seq = _json.loads(r.get("tool_sequence_json") or "[]")
        except (ValueError, TypeError):
            tool_seq = []
        items.append(ProcedureItem(
            id=r["id"], name=r["name"], applicability=r["applicability"],
            steps=[ProcedureStep(**s) if isinstance(s, dict) else ProcedureStep(action=str(s))
                   for s in steps],
            tool_sequence=tool_seq, source=r.get("source", ""),
            use_count=r.get("use_count", 0), success_count=r.get("success_count", 0),
            refined_count=r.get("refined_count", 0), status=r.get("status", ""),
            updated_at=str(r.get("updated_at", "")),
        ))
    return ProcedureListResponse(total=len(items), items=items, memory_enabled=True)


@router.post("/procedures/{procedure_id}/promote", response_model=PromoteResponse)
async def promote_procedure(procedure_id: int):
    """手动把程序记忆晋升为候选 Skill（enabled=false，去技能管理页确认启用）。"""
    if not is_memory_enabled():
        raise HTTPException(status_code=503, detail="MySQL 未配置")
    result = get_procedure_store().promote_to_skill(procedure_id, auto_enable=False)
    if not result.get("ok"):
        raise HTTPException(status_code=409, detail=result.get("reason", "晋升失败"))
    return PromoteResponse(**result)


# ====== 压缩 / 审计 ======

@router.post("/compact", response_model=CompactResponse)
async def compact_memories():
    """手动触发一轮 Curator 治理（剪枝+压缩+提炼+晋升+对账）。"""
    from agent.memory.curator import run_curation_once
    stats = run_curation_once()
    return CompactResponse(compacted=stats.get("compacted", 0))


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
    ) for r in rows]
    return ReflectionListResponse(total=len(items), items=items, memory_enabled=True)
