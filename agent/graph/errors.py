"""LLM 异常分类（P2）。

从 workflow.py 拆分而来，无内部依赖。
"""
import httpx
from openai import (
    APIConnectionError,
    APITimeoutError,
    APIError,
    RateLimitError,
)


class LLMError(RuntimeError):
    """LLM 调用异常基类，携带分类信息便于前端展示。"""

    def __init__(self, kind: str, message: str, status_code: int = 500):
        self.kind = kind            # timeout / rate_limit / connection / api_error / format_error
        self.status_code = status_code
        super().__init__(f"[{kind}] {message}")


def _classify_llm_error(e: Exception) -> LLMError:
    """把 OpenAI SDK 异常分类为 LLMError，便于上层决定 HTTP 状态码和重试策略。"""
    if isinstance(e, APITimeoutError):
        return LLMError("timeout", f"LLM 调用超时：{e}", status_code=504)
    if isinstance(e, RateLimitError):
        return LLMError("rate_limit", f"LLM 触发限流：{e}", status_code=429)
    if isinstance(e, APIConnectionError):
        return LLMError("connection", f"LLM 连接失败：{e}", status_code=502)
    if isinstance(e, APIError):
        # 通用 API 错误，保留原始 status_code
        sc = getattr(e, "status_code", None) or 500
        return LLMError("api_error", f"LLM API 错误：{e}", status_code=sc)
    if isinstance(e, httpx.TimeoutException):
        return LLMError("timeout", f"HTTP 超时：{e}", status_code=504)
    return LLMError("unknown", f"LLM 未知错误：{e}", status_code=500)
