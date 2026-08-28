"""会话管理接口（P1-b）。

将会话持久化从前端 localStorage 迁移到后端 MySQL。
硬失败策略：MySQL 不可用时返回 500 错误（不降级到 localStorage）。

接口：
- GET    /api/sessions           列出所有会话（含消息，前端启动时全量加载）
- POST   /api/sessions           创建会话
- GET    /api/sessions/{id}      获取单个会话（含消息）
- PUT    /api/sessions/{id}      全量同步会话（标题 + 消息）
- PATCH  /api/sessions/{id}      更新会话标题
- DELETE /api/sessions/{id}      删除会话（级联删除消息）
"""
import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from agent.memory.session_store import get_session_store, is_session_enabled

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


# ====== 请求/响应模型 ======

class CreateSessionRequest(BaseModel):
    """创建会话请求。"""
    id: str = Field(..., max_length=32, description="前端生成的会话 ID")
    title: str = Field("新会话", max_length=256)


class SyncSessionRequest(BaseModel):
    """全量同步会话请求（标题 + 消息）。"""
    title: str = Field("新会话", max_length=256)
    messages: list[dict[str, Any]] = Field(default_factory=list, description="完整消息列表")


class UpdateTitleRequest(BaseModel):
    """更新标题请求。"""
    title: str = Field(..., max_length=256)


class SessionResponse(BaseModel):
    """会话响应（含消息）。"""
    id: str
    title: str
    createdAt: int
    updatedAt: int
    messages: list[dict[str, Any]] = []


class SessionListResponse(BaseModel):
    """会话列表响应。"""
    sessions: list[SessionResponse]
    total: int


# ====== 错误检查 ======

def _ensure_enabled():
    """检查 MySQL 是否可用，不可用时抛 500。"""
    if not is_session_enabled():
        raise HTTPException(
            status_code=500,
            detail="会话持久化未启用：MYSQL_PASSWORD 未配置。请配置 MySQL 后重启服务。",
        )


def _handle_store_error(e: Exception):
    """统一处理 SessionStore 异常。"""
    if "未启用" in str(e) or "MYSQL_PASSWORD" in str(e):
        raise HTTPException(status_code=500, detail=str(e))
    logger.exception("[sessions] SessionStore 操作失败")
    raise HTTPException(status_code=500, detail=f"会话存储操作失败：{e}")


# ====== 路由 ======

@router.get("", response_model=SessionListResponse)
async def list_sessions():
    """列出所有会话（含消息，前端启动时全量加载）。"""
    _ensure_enabled()
    try:
        store = get_session_store()
        sessions = store.list_sessions_with_messages()
        return SessionListResponse(
            sessions=[SessionResponse(**s) for s in sessions],
            total=len(sessions),
        )
    except HTTPException:
        raise
    except Exception as e:
        _handle_store_error(e)


@router.post("", response_model=SessionResponse, status_code=201)
async def create_session(req: CreateSessionRequest):
    """创建会话。"""
    _ensure_enabled()
    try:
        store = get_session_store()
        store.create_session(req.id, req.title)
        session = store.get_session(req.id)
        if not session:
            raise HTTPException(status_code=500, detail="创建会话后查询失败")
        return SessionResponse(**session)
    except HTTPException:
        raise
    except Exception as e:
        _handle_store_error(e)


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str):
    """获取单个会话（含消息）。"""
    _ensure_enabled()
    try:
        store = get_session_store()
        session = store.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail=f"会话 {session_id} 不存在")
        return SessionResponse(**session)
    except HTTPException:
        raise
    except Exception as e:
        _handle_store_error(e)


@router.put("/{session_id}", response_model=SessionResponse)
async def sync_session(session_id: str, req: SyncSessionRequest):
    """全量同步会话：更新标题 + 替换所有消息。

    用于前端 persistActiveSession：流式完成后一次性同步整个会话状态。
    """
    _ensure_enabled()
    try:
        store = get_session_store()
        store.sync_session(session_id, req.title, req.messages)
        session = store.get_session(session_id)
        if not session:
            raise HTTPException(status_code=500, detail="同步会话后查询失败")
        return SessionResponse(**session)
    except HTTPException:
        raise
    except Exception as e:
        _handle_store_error(e)


@router.patch("/{session_id}")
async def update_title(session_id: str, req: UpdateTitleRequest):
    """更新会话标题。"""
    _ensure_enabled()
    try:
        store = get_session_store()
        if not store.update_session_title(session_id, req.title):
            raise HTTPException(status_code=404, detail=f"会话 {session_id} 不存在")
        return {"ok": True}
    except HTTPException:
        raise
    except Exception as e:
        _handle_store_error(e)


@router.delete("/{session_id}")
async def delete_session(session_id: str):
    """删除会话（级联删除消息）。"""
    _ensure_enabled()
    try:
        store = get_session_store()
        if not store.delete_session(session_id):
            raise HTTPException(status_code=404, detail=f"会话 {session_id} 不存在")
        return {"ok": True}
    except HTTPException:
        raise
    except Exception as e:
        _handle_store_error(e)
