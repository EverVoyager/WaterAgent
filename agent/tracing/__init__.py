"""LLM 调用追踪模块。

提供 LangFuse 集成（可选启用）。未配置 LangFuse 时所有追踪函数为 no-op。
"""
from agent.tracing.langfuse_client import (
    get_langfuse,
    is_langfuse_enabled,
    trace_llm_call,
    flush,
)

__all__ = [
    "get_langfuse",
    "is_langfuse_enabled",
    "trace_llm_call",
    "flush",
]
