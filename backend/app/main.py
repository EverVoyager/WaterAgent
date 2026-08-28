"""FastAPI 应用入口。"""
import sys
from pathlib import Path

# 将项目根目录（backend 的父目录）加入 sys.path，使 agent 等模块可被导入
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# 确认 backend 目录在 path 中（uvicorn 从 backend 启动时通常已有，这里兜底）
_BACKEND_ROOT = str(Path(__file__).resolve().parents[1])
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.api.agent import router as agent_router
from app.api.health import router as health_router
from app.api.memories import router as memories_router
from app.api.sessions import router as sessions_router
from app.api.skills import router as skills_router
from app.core.config import get_settings
from app.core.logging import setup_logging
from app.core.rate_limit import limiter


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动 Curator 后台治理线程（记忆剪枝/压缩/索引对账）。"""
    try:
        from agent.memory.curator import start_curator_thread
        start_curator_thread()
    except Exception as e:
        # 治理线程启动失败不影响服务本身
        import logging
        logging.getLogger(__name__).warning("[startup] Curator 启动失败（不影响服务）：%s", e)
    yield


def create_app() -> FastAPI:
    """应用工厂。"""
    # P4.1：尽早初始化 structlog，后续日志都走结构化
    setup_logging()

    settings = get_settings()

    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="黄河吕梁段防汛预警智能体后端服务",
        lifespan=lifespan,
    )

    # ====== Rate Limiting（P1.2 slowapi）======
    # limiter 实例在 app.core.rate_limit 中定义（避免循环导入）
    # default_limits 覆盖所有端点；敏感端点可在路由上用 @limiter.limit 收紧
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    # 必须注册中间件，default_limits 才会真正生效
    app.add_middleware(SlowAPIMiddleware)

    # ====== CORS（P1.3 production 收紧）======
    # production 模式不允许通配符，必须显式列出域名
    if settings.is_production and "*" in settings.cors_origins_list:
        raise RuntimeError(
            "Production 模式 CORS_ORIGINS 不允许使用通配符 '*'，"
            "请配置具体域名，如 https://water.example.com"
        )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        # production 收紧方法：只允许业务需要的
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"] if settings.is_production else ["*"],
        allow_headers=["Content-Type", "Authorization"] if settings.is_production else ["*"],
    )

    # ====== 路由注册 ======
    app.include_router(health_router)
    app.include_router(agent_router)
    app.include_router(skills_router)
    app.include_router(memories_router)
    app.include_router(sessions_router)

    # ====== 全局异常处理（P2 异常分类）======
    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        """捕获未处理异常，避免堆栈直接返回给客户端。"""
        import structlog
        log = structlog.get_logger()
        log.error("unhandled_exception",
                  path=request.url.path,
                  error=str(exc),
                  error_type=type(exc).__name__)
        return JSONResponse(
            status_code=500,
            content={"detail": "服务内部错误，请联系管理员"},
        )

    return app


app = create_app()
