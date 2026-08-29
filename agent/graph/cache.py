"""工具结果缓存（P6）。

从 workflow.py 拆分而来。依赖 agent.tools.mock_executor。
"""
import json
import logging
import time
from typing import Any

from agent.tools.mock_executor import execute_tool

logger = logging.getLogger(__name__)

# 进程级 LRU + TTL 缓存：key = (tool_name, sorted_args_json)
# 仅缓存幂等工具（不缓存 generate_plan 这类需要 LLM 生成的）
_CACHEABLE_TOOLS = {"get_weather", "get_hydrology", "predict_runoff", "query_gis_terrain", "search_regulation"}
_TOOL_RESULT_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_TOOL_CACHE_TTL = 300.0  # 5 分钟
_CACHE_MAX_ENTRIES = 256  # 容量上限：防止参数多样化时缓存无限增长


def _evict_if_needed(now: float) -> None:
    """容量超限时先清过期条目，仍超则淘汰最旧条目。"""
    if len(_TOOL_RESULT_CACHE) < _CACHE_MAX_ENTRIES:
        return
    expired = [k for k, (ts, _) in _TOOL_RESULT_CACHE.items()
               if now - ts >= _TOOL_CACHE_TTL]
    for k in expired:
        del _TOOL_RESULT_CACHE[k]
    if len(_TOOL_RESULT_CACHE) >= _CACHE_MAX_ENTRIES:
        oldest_key = min(_TOOL_RESULT_CACHE, key=lambda k: _TOOL_RESULT_CACHE[k][0])
        del _TOOL_RESULT_CACHE[oldest_key]


def _cache_key(tool_name: str, arguments: dict[str, Any]) -> str:
    try:
        return f"{tool_name}:{json.dumps(arguments, sort_keys=True, ensure_ascii=False)}"
    except (TypeError, ValueError):
        return f"{tool_name}:{arguments}"


def clear_tool_result_cache() -> None:
    """清空工具结果缓存。

    评估回放上下文切换 case 时调用：前一个 case 的 overrides 结果
    若残留在 TTL 缓存中，会被同参数的下一个 case 命中，破坏确定性回放。
    """
    _TOOL_RESULT_CACHE.clear()


def _cached_execute_tool(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """带缓存的工具执行。仅缓存幂等工具，TTL 5 分钟。

    注意：hydrology/weather 已有各自的内部缓存（30min/10min），
    本层缓存是针对同一会话内 LLM 重复决策同一调用的快速命中。
    评估回放期间绕过缓存：回放要求每次调用都按当前 case 的 overrides 重新生成。
    """
    from agent.tools.mock_executor import is_replay_active

    if is_replay_active() or tool_name not in _CACHEABLE_TOOLS:
        return execute_tool(tool_name, arguments)

    key = _cache_key(tool_name, arguments)
    now = time.time()
    cached = _TOOL_RESULT_CACHE.get(key)
    if cached and now - cached[0] < _TOOL_CACHE_TTL:
        logger.debug("[cache] hit %s", tool_name)
        # 返回副本避免上层修改污染缓存
        result = dict(cached[1])
        result["_from_cache"] = True
        return result

    result = execute_tool(tool_name, arguments)
    if isinstance(result, dict) and not result.get("error"):
        _evict_if_needed(now)
        _TOOL_RESULT_CACHE[key] = (now, result)
    return result
