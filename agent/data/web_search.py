"""Tavily 联网搜索客户端。

文档：https://docs.tavily.com/docs/rest-api/api-reference

返回结构化搜索结果（标题、摘要、URL），用于 Agent 的 web_search 工具。
搜索结果中的网页内容可作为 Citation Grounding 的引用来源。
"""
import logging
import time
from typing import Any

import requests

from agent.utils import now_iso as _now_iso
from app.core.config import get_settings

logger = logging.getLogger(__name__)

# 简单缓存：避免高频重复搜索
_cache: dict[str, tuple[float, dict[str, Any]]] = {}
_CACHE_TTL = 600  # 10 分钟


def search_web(query: str, max_results: int = 5) -> dict[str, Any]:
    """调用 Tavily API 执行联网搜索。

    Args:
        query: 搜索关键词
        max_results: 最大返回结果数（默认 5）

    Returns:
        {
            "query": str,
            "results": [
                {
                    "title": str,       # 网页标题
                    "snippet": str,     # 内容摘要
                    "url": str,         # 网页链接
                    "score": float,     # 相关度分数
                },
                ...
            ],
            "result_count": int,
            "searched_at": iso,
            "source": "tavily",
        }

    Raises:
        RuntimeError: API 调用失败或未配置 key
    """
    settings = get_settings()
    if not settings.TAVILY_API_KEY:
        raise RuntimeError("TAVILY_API_KEY 未配置，请在 backend/.env 中填入 Tavily API Key")

    cache_key = f"{query}:{max_results}"
    now = time.time()

    # 检查缓存
    if cache_key in _cache:
        cached_at, cached_data = _cache[cache_key]
        if now - cached_at < _CACHE_TTL:
            logger.debug("[web_search] 命中缓存: %s", query[:60])
            return cached_data

    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": settings.TAVILY_API_KEY,
                "query": query,
                "max_results": max_results,
                "search_depth": "basic",
                "include_answer": False,
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        logger.exception("[web_search] Tavily API 调用失败")
        raise RuntimeError(f"Tavily 搜索 API 调用失败：{e}") from e

    # 解析结果
    raw_results = data.get("results", [])
    results: list[dict[str, Any]] = []
    for r in raw_results:
        results.append({
            "title": r.get("title", ""),
            "snippet": (r.get("content", "") or "").strip(),
            "url": r.get("url", ""),
            "score": r.get("score", 0.0),
        })

    result = {
        "query": query,
        "results": results,
        "result_count": len(results),
        "searched_at": _now_iso(),
        "source": "tavily",
    }

    # 写入缓存
    _cache[cache_key] = (now, result)
    return result
