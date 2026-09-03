"""LLM usage 统计：前缀缓存（KV Cache）命中率观测。

三后端 usage 字段命名兼容（防御式提取，字段缺失或后端不支持时归零不报错）：
- OpenAI 风格 / 阿里云 MaaS / vLLM 新版：usage.prompt_tokens_details.cached_tokens
- DeepSeek 原生：usage.prompt_cache_hit_tokens（命中 token 数）
- vLLM 旧版：usage.cached_tokens

按节点（planner / synthesizer / chat / ...）进程内聚合，周期性输出 INFO 日志，
用于验证前缀稳定化改造的缓存命中效果。无外部依赖，单进程内有效。
"""
import logging
import threading

logger = logging.getLogger(__name__)

# node -> {"calls", "prompt_tokens", "cached_tokens"}
_STATS: dict[str, dict[str, int]] = {}
_STATS_LOCK = threading.Lock()

# 每 N 次调用输出一次汇总日志（避免高频刷屏）
_LOG_EVERY = 10


def _extract_cached_tokens(usage) -> tuple[int, int]:
    """从 usage 对象提取 (prompt_tokens, cached_tokens)。

    兼容 OpenAI 对象属性、dict、以及三后端的字段命名差异；
    usage 为 None、字段缺失或类型异常时返回 (0, 0)，绝不抛错——
    观测模块的任何异常都不能影响 LLM 调用主路径。
    """
    if usage is None:
        return 0, 0
    try:
        if isinstance(usage, dict):
            def get(obj, key):
                return obj.get(key)
        else:
            def get(obj, key):
                return getattr(obj, key, None)

        prompt = int(get(usage, "prompt_tokens") or 0)
        cached = 0

        # 1) OpenAI 风格：prompt_tokens_details.cached_tokens
        details = (
            usage.get("prompt_tokens_details")
            if isinstance(usage, dict)
            else getattr(usage, "prompt_tokens_details", None)
        )
        if details is not None:
            cached = int(get(details, "cached_tokens") or 0)

        # 2) DeepSeek 原生：prompt_cache_hit_tokens
        if not cached:
            cached = int(get(usage, "prompt_cache_hit_tokens") or 0)

        # 3) vLLM 旧版：cached_tokens
        if not cached:
            cached = int(get(usage, "cached_tokens") or 0)

        return prompt, cached
    except Exception:
        return 0, 0


def record_llm_usage(node: str, usage) -> None:
    """记录一次 LLM 调用的 usage（非流式传 resp.usage，流式传末 chunk 的 usage）。

    Args:
        node: 调用节点标识（planner / synthesizer_phase1 / synthesizer_phase2 / chat ...）
        usage: OpenAI Usage 对象、dict 或 None
    """
    prompt, cached = _extract_cached_tokens(usage)
    with _STATS_LOCK:
        s = _STATS.setdefault(
            node, {"calls": 0, "prompt_tokens": 0, "cached_tokens": 0}
        )
        s["calls"] += 1
        s["prompt_tokens"] += prompt
        s["cached_tokens"] += cached
        calls = s["calls"]

    if cached:
        logger.debug(
            "[llm-cache] node=%s prompt_tokens=%d cached_tokens=%d", node, prompt, cached
        )
    if calls % _LOG_EVERY == 0:
        log_cache_summary()


def log_cache_summary() -> None:
    """输出各节点的缓存命中率汇总（INFO）。"""
    with _STATS_LOCK:
        snapshot = {k: dict(v) for k, v in _STATS.items()}
    for node, s in sorted(snapshot.items()):
        rate = (
            s["cached_tokens"] / s["prompt_tokens"] * 100
            if s["prompt_tokens"]
            else 0.0
        )
        logger.info(
            "[llm-cache] node=%s calls=%d prompt_tokens=%d cached_tokens=%d hit_rate=%.1f%%",
            node, s["calls"], s["prompt_tokens"], s["cached_tokens"], rate,
        )


def get_cache_stats() -> dict[str, dict[str, int]]:
    """返回统计快照（测试用）。"""
    with _STATS_LOCK:
        return {k: dict(v) for k, v in _STATS.items()}


def reset_cache_stats() -> None:
    """清空统计（测试用）。"""
    with _STATS_LOCK:
        _STATS.clear()
