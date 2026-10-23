"""请求级上下文：request_id 生成、日志绑定与响应头回传。

- RequestIDMiddleware：每个 HTTP 请求生成 12 位 request_id，绑定进
  structlog contextvars（该请求内所有日志自动携带，stdlib 日志同样），
  并写入响应头 X-Request-ID 供前端展示 / 运维按 ID 检索日志；
  请求结束时输出一条含状态码与耗时的访问日志（健康探针除外）
- request_id_var：跨线程传递的 ContextVar。contextvars 不随手动创建的
  线程自动继承，两类线程需显式处理：
    1. SSE agent worker（app.api.agent._stream_generator）：读该值后 bind
    2. executor 工具线程池（agent.graph.nodes._run_stage）：copy_context 传播
"""
import contextvars
import time
import uuid

import structlog
from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default=""
)

# 健康探针不记访问日志（高频轮询会刷屏），request_id 照常绑定
_QUIET_PATH_PREFIXES = ("/api/health",)


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]


def get_request_id() -> str:
    return request_id_var.get()


class RequestIDMiddleware:
    """纯 ASGI 中间件：request_id 生成 / 绑定 / 响应头 / 访问日志。"""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        rid = new_request_id()
        rid_token = request_id_var.set(rid)
        structlog.contextvars.bind_contextvars(request_id=rid)
        quiet = scope.get("path", "").startswith(_QUIET_PATH_PREFIXES)
        start = time.perf_counter()
        status_code = 500

        async def send_with_headers(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                MutableHeaders(scope=message).append("X-Request-ID", rid)
            await send(message)

        try:
            await self.app(scope, receive, send_with_headers)
        finally:
            duration_ms = round((time.perf_counter() - start) * 1000, 1)
            if not quiet:
                structlog.get_logger("request").info(
                    "http_request",
                    method=scope.get("method", ""),
                    path=scope.get("path", ""),
                    status=status_code,
                    duration_ms=duration_ms,
                )
            structlog.contextvars.clear_contextvars()
            request_id_var.reset(rid_token)
