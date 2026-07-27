"""LangFuse LLM 追踪客户端。

通过环境变量 LANGFUSE_HOST / LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY 启用。
三者任一为空时，追踪功能为 no-op（不抛错，不记录）。

使用方式（在 LLM 调用处）：
    from agent.tracing import trace_llm_call, is_langfuse_enabled

    with trace_llm_call(
        name="planner",
        model="qwen-plus",
        input={"system": sys_prompt, "user": user_prompt},
    ) as gen:
        resp = client.chat.completions.create(...)
        gen.set_output({"content": resp.choices[0].message.content})
        gen.set_usage(resp.usage)
"""
import logging
import uuid
from contextlib import contextmanager
from functools import lru_cache
from typing import Any, Dict, Iterator, Optional

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _create_client():
    """创建 LangFuse 客户端单例。未配置时返回 None。"""
    try:
        from langfuse import Langfuse
    except ImportError:
        logger.info("[langfuse] 包未安装，追踪功能禁用")
        return None

    from app.core.config import get_settings
    settings = get_settings()

    if not (settings.LANGFUSE_HOST and settings.LANGFUSE_PUBLIC_KEY and settings.LANGFUSE_SECRET_KEY):
        logger.info("[langfuse] 未配置 host/public_key/secret_key，追踪功能禁用")
        return None

    try:
        client = Langfuse(
            host=settings.LANGFUSE_HOST,
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
        )
        logger.info("[langfuse] 客户端已初始化，host=%s", settings.LANGFUSE_HOST)
        return client
    except Exception as e:
        logger.warning("[langfuse] 客户端初始化失败：%s — 追踪功能禁用", e)
        return None


def get_langfuse():
    """获取 LangFuse 客户端（单例）。未启用时返回 None。"""
    return _create_client()


def is_langfuse_enabled() -> bool:
    """检查 LangFuse 是否已启用。"""
    return get_langfuse() is not None


def flush() -> None:
    """手动刷新待发送的追踪数据（应用关闭时调用）。"""
    client = get_langfuse()
    if client:
        try:
            client.flush()
        except Exception as e:
            logger.warning("[langfuse] flush 失败：%s", e)


class _NoOpGeneration:
    """未启用 LangFuse 时的占位 generation，所有方法为 no-op。"""

    def set_output(self, output: Dict[str, Any]) -> None:
        pass

    def set_usage(self, usage: Any) -> None:
        pass

    def end(self, **kwargs) -> None:
        pass


class _GenerationWrapper:
    """LangFuse generation 的包装器，提供 set_output / set_usage / end 方法。"""

    def __init__(self, trace, generation):
        self._trace = trace
        self._generation = generation
        self._ended = False

    def set_output(self, output: Dict[str, Any]) -> None:
        """设置 LLM 输出。"""
        if self._generation:
            try:
                self._generation.update(output=output)
            except Exception as e:
                logger.debug("[langfuse] set_output 失败：%s", e)

    def set_usage(self, usage: Any) -> None:
        """设置 token 使用量。

        Args:
            usage: OpenAI resp.usage 对象（有 prompt_tokens/completion_tokens/total_tokens 属性）
        """
        if not self._generation or usage is None:
            return
        try:
            usage_dict = {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
                "total_tokens": getattr(usage, "total_tokens", 0) or 0,
            }
            self._generation.update(usage=usage_dict)
        except Exception as e:
            logger.debug("[langfuse] set_usage 失败：%s", e)

    def end(self, **kwargs) -> None:
        """结束 generation 记录。"""
        if self._generation and not self._ended:
            try:
                self._generation.end(**kwargs)
            except Exception as e:
                logger.debug("[langfuse] end 失败：%s", e)
            self._ended = True


@contextmanager
def trace_llm_call(
    name: str,
    model: str = "",
    input: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    trace_id: Optional[str] = None,
) -> Iterator[_GenerationWrapper]:
    """LLM 调用追踪上下文管理器。

    自动记录：
    - 调用开始时间、结束时间、耗时
    - input（system + user prompt）
    - output（通过 gen.set_output 设置）
    - token 使用量（通过 gen.set_usage 设置）

    用法：
        with trace_llm_call(name="planner", model="qwen-plus",
                            input={"system": sys, "user": usr}) as gen:
            resp = client.chat.completions.create(...)
            gen.set_output({"content": resp.choices[0].message.content})
            gen.set_usage(resp.usage)

    未启用 LangFuse 时为 no-op，gen.set_output / gen.set_usage 不做任何事。
    """
    client = get_langfuse()
    if client is None:
        yield _NoOpGeneration()
        return

    # 创建独立 trace（每个 LLM 调用一个 trace）
    _trace_id = trace_id or f"llm-{name}-{uuid.uuid4().hex[:8]}"
    try:
        trace = client.trace(
            id=_trace_id,
            name=name,
            metadata=metadata or {},
        )
        generation = trace.generation(
            name=name,
            model=model,
            input=input or {},
        )
    except Exception as e:
        logger.debug("[langfuse] 创建 trace/generation 失败：%s — 降级为 no-op", e)
        yield _NoOpGeneration()
        return

    wrapper = _GenerationWrapper(trace, generation)
    try:
        yield wrapper
    except Exception as e:
        # LLM 调用抛异常也要记录到 trace
        try:
            wrapper.set_output({"error": str(e), "error_type": type(e).__name__})
            wrapper.end(level="ERROR", status_message=str(e))
        except Exception:
            pass
        raise
    finally:
        wrapper.end()
