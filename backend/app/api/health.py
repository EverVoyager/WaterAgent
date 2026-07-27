"""健康检查接口。

M11：分层健康检查
- /api/health         liveness 存活检查（进程能响应即可）
- /api/health/ready   readiness 就绪检查（依赖服务可用）
"""
import logging
from datetime import datetime, timezone

import httpx
from fastapi import APIRouter
from pydantic import BaseModel

from app.core.config import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["health"])


class HealthResponse(BaseModel):
    """健康检查响应结构。"""

    status: str
    service: str
    version: str
    env: str
    timestamp: str


class DependencyStatus(BaseModel):
    """单个依赖的状态。"""
    name: str
    ready: bool
    latency_ms: float = 0.0
    detail: str = ""


class ReadinessResponse(BaseModel):
    """就绪检查响应。"""
    status: str  # ready / not_ready
    dependencies: list[DependencyStatus]
    timestamp: str


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """liveness：进程存活即可，不检查依赖。"""
    settings = get_settings()
    return HealthResponse(
        status="ok",
        service=settings.APP_NAME,
        version=settings.APP_VERSION,
        env=settings.APP_ENV,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@router.get("/health/ready", response_model=ReadinessResponse)
def readiness_check() -> ReadinessResponse:
    """M11：readiness 检查依赖服务是否就绪。

    轻量探测（短超时 + 只检查连通性），失败不阻塞服务启动。
    """
    settings = get_settings()
    deps: list[DependencyStatus] = []

    # 1. Qdrant 向量库
    deps.append(_check_qdrant(settings))

    # 2. LLM API（仅检查 base_url 可达，不消耗 token）
    deps.append(_check_llm_api(settings))

    # 3. 水文数据源（可选，不阻塞）
    deps.append(_check_hydro_source(settings))

    all_ready = all(d.ready for d in deps if d.name != "hydro_source")  # hydro_source 不阻塞
    return ReadinessResponse(
        status="ready" if all_ready else "not_ready",
        dependencies=deps,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def _check_qdrant(settings) -> DependencyStatus:
    """检查 Qdrant 连通性。"""
    try:
        from app.core.llm import get_qdrant_client
        client = get_qdrant_client()
        start = datetime.now(timezone.utc)
        # 轻量调用：获取集合列表
        collections = client.get_collections()
        latency = (datetime.now(timezone.utc) - start).total_seconds() * 1000
        return DependencyStatus(
            name="qdrant",
            ready=True,
            latency_ms=round(latency, 2),
            detail=f"collections={len(collections.collections)}",
        )
    except Exception as e:
        return DependencyStatus(name="qdrant", ready=False, detail=f"error: {e}")


def _check_llm_api(settings) -> DependencyStatus:
    """检查 LLM API base_url 可达性（HEAD 请求，不消耗 token）。"""
    try:
        start = datetime.now(timezone.utc)
        # 用 httpx 直接 HEAD 探测，超时 3 秒
        with httpx.Client(timeout=3.0) as client:
            resp = client.head(settings.LLM_BASE_URL)
        latency = (datetime.now(timezone.utc) - start).total_seconds() * 1000
        # LLM 服务一般 HEAD 返回 200/405/404 都说明可达
        ready = resp.status_code < 500
        return DependencyStatus(
            name="llm_api",
            ready=ready,
            latency_ms=round(latency, 2),
            detail=f"status={resp.status_code}, model={settings.LLM_MODEL}",
        )
    except Exception as e:
        return DependencyStatus(name="llm_api", ready=False, detail=f"error: {e}")


def _check_hydro_source(settings) -> DependencyStatus:
    """检查水文数据源可达性（可选依赖，失败不阻塞就绪）。"""
    try:
        start = datetime.now(timezone.utc)
        with httpx.Client(timeout=5.0) as client:
            resp = client.head(settings.HYDRO_SOURCE_URL)
        latency = (datetime.now(timezone.utc) - start).total_seconds() * 1000
        ready = resp.status_code < 500
        return DependencyStatus(
            name="hydro_source",
            ready=ready,
            latency_ms=round(latency, 2),
            detail=f"status={resp.status_code}",
        )
    except Exception as e:
        return DependencyStatus(name="hydro_source", ready=False, detail=f"error: {e}")
