"""结构化日志配置（structlog 前端 + stdlib 后端统一管道）。

所有日志——无论经 structlog 还是标准 logging.getLogger 发出——汇入同一组
root handler，共享同一套 processor（含 contextvars 合并），保证：

- 统一格式：JSON 模式（生产 / ELK / Loki 采集）或彩色 Console（开发）
- 统一上下文：bind_context(request_id=...) 后，该上下文内的所有日志
  （stdlib 的 printf 风格同样）自动携带 request_id——关键在 merge_contextvars
  同时挂在 structlog processor 链与 ProcessorFormatter.foreign_pre_chain 上
- 统一出口：stdout + 可选 RotatingFileHandler 落盘轮转（LOG_FILE 非空时）

使用方式：
    from app.core.logging import get_logger
    log = get_logger(__name__)
    log.info("event_name", key=value)          # structlog 风格 kv

    # 项目内既有代码沿用标准 logging，格式化后同样携带 request_id：
    import logging
    logger = logging.getLogger(__name__)
    logger.info("[module] printf %s", arg)
"""
import logging
import sys
from logging.handlers import RotatingFileHandler
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
    """structlog 原生与 stdlib 外来记录共享的预处理器（按序执行）。

    merge_contextvars 放最前：把 bind_context 绑定的 request_id 等合并进
    event_dict；经 ProcessorFormatter.foreign_pre_chain 对 stdlib 记录同样
    生效，这是「标准 logging 日志也带 request_id」的关键。
    """
    return [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
        structlog.processors.StackInfoRenderer(),
    ]


def setup_logging(
    json_mode: bool = False,
    level: str = "INFO",
    log_file: str = "",
    log_file_max_mb: int = 50,
    log_file_backup_count: int = 5,
) -> None:
    """初始化 structlog + 标准 logging 的统一管道。

    Args:
        json_mode: True 输出 JSON（适合生产/ELK 采集），False 输出彩色 Console
        level: 日志级别 DEBUG/INFO/WARNING/ERROR
        log_file: 非空时追加 RotatingFileHandler 落盘（目录需已存在或可创建）
        log_file_max_mb: 单文件大小上限（MB），超过即轮转
        log_file_backup_count: 轮转保留的历史文件数
    """
    log_level = _LOG_LEVELS.get(level.upper(), logging.INFO)

    if json_mode:
        renderer = structlog.processors.JSONRenderer(ensure_ascii=False)
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    # ProcessorFormatter 是两条管道的汇合点：
    # - stdlib 记录（foreign）先过 foreign_pre_chain 再渲染
    # - structlog 记录经 wrap_for_formatter 打包后走同一组渲染 processor
    # format_exc_info 仅 JSON 模式需要（把 exc_info 转成字符串字段）；
    # Console 模式交给 ConsoleRenderer 自带的异常排版（挂了反而触发
    # structlog 的 "Remove format_exc_info ... pretty exceptions" 警告）
    render_processors = [
        structlog.stdlib.ProcessorFormatter.remove_processors_meta,
    ]
    if json_mode:
        render_processors.append(structlog.processors.format_exc_info)
    render_processors.append(renderer)

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=render_processors,
        foreign_pre_chain=_shared_processors(),
    )

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        try:
            handlers.append(
                RotatingFileHandler(
                    log_file,
                    maxBytes=log_file_max_mb * 1024 * 1024,
                    backupCount=log_file_backup_count,
                    encoding="utf-8",
                )
            )
        except OSError as e:
            # 路径非法 / 无写权限：退回 stdout-only，不阻断启动
            print(
                f"[logging] LOG_FILE={log_file} 初始化失败，仅输出 stdout：{e}",
                file=sys.stderr,
            )

    for h in handlers:
        h.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(log_level)
    for h in list(root.handlers):
        root.removeHandler(h)
    for h in handlers:
        root.addHandler(h)

    # structlog 作为前端：日志经 stdlib LoggerFactory 落到上面配置的 root
    # handler，与标准 logging 走完全相同的出口与格式
    structlog.configure(
        processors=[
            *_shared_processors(),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # 收紧常用库的噪音
    for noisy in ("httpx", "httpcore", "openai._base_client", "urllib3.connectionpool"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    structlog.get_logger("logging").info(
        "logging_initialized",
        json_mode=json_mode, level=level, log_file=log_file or "-",
    )


def bind_context(**kwargs: Any) -> None:
    """绑定请求级日志上下文（request_id 等）。

    在中间件/线程入口调用，该上下文内后续所有日志（structlog + stdlib）
    自动携带。注意 contextvars 不随新线程继承：手动创建的线程
    （如 SSE agent worker）需读取 request_id_var 后自行 bind。
    """
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_context() -> None:
    """清除当前上下文的日志绑定（请求/线程结束时调用）。"""
    structlog.contextvars.clear_contextvars()


def get_logger(name: str | None = None):
    """获取一个结构化 logger（经统一管道输出）。"""
    return structlog.get_logger(name)
