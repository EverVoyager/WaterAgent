"""request_id 贯穿与统一日志管道的回归测试（离线，不起 LLM）。

覆盖：
- RequestIDMiddleware：每请求独立 request_id、X-Request-ID 响应头回传、
  健康探针静默（无访问日志）、非探针路径输出访问日志（状态码 + 耗时）
- 统一管道：stdlib printf 风格日志携带 bind_context 绑定的 request_id；
  clear_context 后不再携带；线程池经 copy_context 传播同样携带

捕获方式注意：stdlib 日志的 request_id 由 ProcessorFormatter 在格式化时
经 foreign_pre_chain 合并（不在 record.msg 里），捕获 handler 必须复用
root 上的 ProcessorFormatter 并记录 format(record) 而非 getMessage()。
"""
import contextvars
import logging
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from app.core.logging import bind_context, clear_context, setup_logging
from app.main import create_app


class _Capture(logging.Handler):
    """按 root 格式化器渲染后捕获（含 contextvars 合并的 request_id）。"""

    def __init__(self):
        super().__init__()
        self.records: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(self.format(record))


def _attach_capture() -> _Capture:
    """在 root 上挂捕获 handler，复用首个 root handler 的 ProcessorFormatter。"""
    root = logging.getLogger()
    assert root.handlers, "setup_logging 未初始化（应先 create_app / setup_logging）"
    cap = _Capture()
    cap.setFormatter(root.handlers[0].formatter)
    root.addHandler(cap)
    return cap


def _detach(cap: _Capture) -> None:
    logging.getLogger().removeHandler(cap)


@pytest.fixture()
def client():
    return TestClient(create_app())


class TestRequestIDMiddleware:
    def test_unique_request_id_per_request(self, client):
        r1 = client.get("/api/health")
        r2 = client.get("/api/health")
        rid1 = r1.headers.get("X-Request-ID", "")
        rid2 = r2.headers.get("X-Request-ID", "")
        assert r1.status_code == 200
        assert len(rid1) == 12 and len(rid2) == 12
        assert rid1 != rid2, "每个请求应有独立 request_id"

    def test_health_probe_is_quiet(self, client):
        cap = _attach_capture()
        try:
            # /api/health/ready 同样命中 /api/health 前缀，均应静默
            client.get("/api/health")
            client.get("/api/health/ready")
            assert not [m for m in cap.records if "http_request" in m]
        finally:
            _detach(cap)

    def test_access_log_emitted_for_normal_path(self, client):
        cap = _attach_capture()
        try:
            client.get("/openapi.json")
            hits = [m for m in cap.records if "http_request" in m]
            assert any("openapi.json" in m and "status" in m for m in hits)
        finally:
            _detach(cap)


class TestUnifiedPipeline:
    """setup_logging 统一管道：stdlib 日志同样携带 contextvars 绑定。"""

    def test_stdlib_log_carries_request_id(self):
        setup_logging(json_mode=True, level="DEBUG")
        cap = _attach_capture()
        try:
            log = logging.getLogger("test.unified")
            bind_context(request_id="rid-unified-1")
            log.info("[stdlib] printf %s", "arg")
            clear_context()
            log.info("[stdlib] after clear")
        finally:
            _detach(cap)
        with_rid = [m for m in cap.records if "rid-unified-1" in m]
        assert len(with_rid) == 1 and "printf arg" in with_rid[0]
        after = [m for m in cap.records if "after clear" in m]
        assert after and "rid-unified-1" not in after[0]

    def test_pool_thread_inherits_context_via_copy_context(self):
        """executor 线程池同款传播方式：copy_context().run 内日志保留绑定。"""
        setup_logging(json_mode=True, level="DEBUG")
        cap = _attach_capture()
        try:
            bind_context(request_id="rid-pool-1")
            pool = ThreadPoolExecutor(2)
            futures = [
                pool.submit(contextvars.copy_context().run,
                            lambda i=i: logging.getLogger("test.pool").info("[pool] task %d", i))
                for i in range(2)
            ]
            [f.result() for f in futures]
            pool.shutdown()
            clear_context()
        finally:
            _detach(cap)
        pool_logs = [m for m in cap.records if "[pool] task" in m]
        assert len(pool_logs) == 2
        assert all("rid-pool-1" in m for m in pool_logs)
