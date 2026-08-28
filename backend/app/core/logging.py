"""结构化日志配置（structlog）。

提供：
- 统一的 JSON/Console 双模式输出
- 上下文绑定（request_id、user、tool 等）
- 与标准 logging 模块兼容

使用方式：
    import structlog
    log = structlog.get_logger()
    log.info("event_name", key=value, ...)

输出示例（console）：
    2026-07-21 10:00:00 [info] event_name key=value tool=get_hydrology
"""
import logging
import sys
from typing import Any

import structlog

# 普通日志级别到 structlog 方法的映射
_LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


def _shared_processors() -> list:
    """所有模式共享的预处理器（按顺序执行）。"""
    return [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=False),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]


def setup_logging(json_mode: bool = False, level: str = "INFO") -> None:
    """初始化 structlog + 标准 logging 的统一配置。

    Args:
        json_mode: True 输出 JSON（适合生产/ELK 采集），False 输出彩色 Console
        level: 日志级别 DEBUG/INFO/WARNING/ERROR
    """
    log_level = _LOG_LEVELS.get(level.upper(), logging.INFO)

    timestamper = structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S")

    if json_mode:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[
            *_shared_processors(),
            timestamper,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # 同时配置标准 logging，使三方库（uvicorn / qdrant_client 等）的日志走同一管道
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,
    )

    # 收紧常用库的噪音
    for noisy in ("httpx", "httpcore", "openai._base_client", "urllib3.connectionpool"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # structlog 自身用 print 输出，不再走标准 logging
    structlog.get_logger("logging").info(
        "logging_initialized", json_mode=json_mode, level=level
    )


def bind_context(**kwargs: Any) -> None:
    """绑定全局日志上下文（如 request_id、user_id）。

    在中间件/请求入口调用，后续该请求内所有日志自动携带。
    """
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_context() -> None:
    """清除当前协程的日志上下文（请求结束时调用）。"""
    structlog.contextvars.clear_contextvars()


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """获取一个结构化 logger。"""
    return structlog.get_logger(name)
