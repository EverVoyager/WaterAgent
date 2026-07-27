"""工具结果缓存（P6）。

从 workflow.py 拆分而来。依赖 agent.tools.mock_executor。
"""
import json
import logging
import time
from typing import Any, Dict

from agent.tools.mock_executor import execute_tool

logger = logging.getLogger(__name__)

# 进程级 LRU + TTL 缓存：key = (tool_name, sorted_args_json)
# 仅缓存幂等工具（不缓存 generate_plan 这类需要 LLM 生成的）
_CACHEABLE_TOOLS = {"get_weather", "get_hydrology", "predict_runoff", "query_gis_terrain", "search_regulation"}
_TOOL_RESULT_CACHE: Dict[str, tuple[float, Dict[str, Any]]] = {}
_TOOL_CACHE_TTL = 300.0  # 5 分钟


def _cache_key(tool_name: str, arguments: Dict[str, Any]) -> str:
    try:
        return f"{tool_name}:{json.dumps(arguments, sort_keys=True, ensure_ascii=False)}"
    except (TypeError, ValueError):
        return f"{tool_name}:{arguments}"


def _cached_execute_tool(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """带缓存的工具执行。仅缓存幂等工具，TTL 5 分钟。

    注意：hydrology/weather 已有各自的内部缓存（30min/10min），
    本层缓存是针对同一会话内 LLM 重复决策同一调用的快速命中。
    """
    if tool_name not in _CACHEABLE_TOOLS:
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
        _TOOL_RESULT_CACHE[key] = (now, result)
    return result
