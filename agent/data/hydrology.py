"""水文站实时数据爬虫。

数据源：qqjjsj.com 每日发布黄河水文站水位流量（来源于水利部全国水雨情信息网公开数据）

工作流程：
1. 抓 http://www.qqjjsj.com/list226a1/ 列表页
2. 找最新一篇"黄河水文站实时水位"文章
3. 解析 HTML 表格，提取吴堡/龙门等站数据
4. 缓存结果，TTL 由 HYDRO_CACHE_TTL 控制（默认 30 分钟）

黄河吕梁段主要水文站真实特征值（参考水利部公开资料）：
- 吴堡站：基准水位 636.0m，警戒水位 640.0m，保证水位 642.0m，多年平均流量 ~900 m³/s
- 龙门站：基准水位 377.0m，警戒水位 382.0m，保证水位 385.0m，多年平均流量 ~1000 m³/s

注意：数据源覆盖黄河干流主要站点（玛曲/兰州/石嘴山/巴彦高勒/头道拐/吴堡/龙门/华县/潼关/
三门峡/小浪底/黑石关/武陟/花园口/高村/泺口/利津），不包含府谷站。
"""
import logging
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests

from app.core.config import get_settings

logger = logging.getLogger(__name__)

# 黄河吕梁段主要水文站基准参数（真实值，用于阈值判断和降级 mock）
# 仅包含数据源实际发布的站点
STATION_PARAMS = {
    "吴堡": {
        "base_level_m": 636.0,
        "warning_level_m": 640.0,
        "guaranteed_level_m": 642.0,
        "base_flow_m3_s": 900,
        "warning_flow_m3_s": 5000,
        "river": "黄河",
    },
    "龙门": {
        "base_level_m": 377.0,
        "warning_level_m": 382.0,
        "guaranteed_level_m": 385.0,
        "base_flow_m3_s": 1000,
        "warning_flow_m3_s": 7000,
        "river": "黄河",
    },
}

# 缓存：{cache_key: (fetched_at, data)}
_cache: Dict[str, tuple[float, Dict[str, Any]]] = {}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fetch_url(url: str, timeout: int = 10) -> str:
    """抓 URL 内容。"""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def _find_latest_yellow_river_article(list_url: str) -> tuple[str, str]:
    """从列表页找到最新一篇黄河水文站文章。"""
    html = _fetch_url(list_url)
    # 匹配 detail 链接 + 标题
    pattern = re.compile(r'href="([^"]*id=\d+[^"]*)"[^>]*>([^<]*黄河[^<]*)</a>')
    links = pattern.findall(html)
    # 优先"实时水位"/"水位情况"
    candidates = [
        (url, title.strip())
        for url, title in links
        if "实时水位" in title or "水位情况" in title
    ]
    if not candidates:
        candidates = links
    if not candidates:
        raise RuntimeError("qqjjsj.com 列表页未找到黄河水文站文章")
    return candidates[0]


def _parse_hydro_table(article_url: str) -> List[Dict[str, str]]:
    """解析文章中的水文站表格。"""
    html = _fetch_url(article_url)
    table_match = re.search(r"<table[^>]*>(.*?)</table>", html, re.S)
    if not table_match:
        raise RuntimeError(f"文章页未找到 table: {article_url}")
    table_html = table_match.group(1)
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", table_html, re.S)
    results = []
    for row in rows:
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.S)
        cells = [re.sub(r"<[^>]+>", "", c).strip() for c in cells]
        if len(cells) >= 5:
            results.append({
                "river": cells[0],
                "station": cells[1],
                "time": cells[2],
                "water_level": cells[3],
                "flow": cells[4],
            })
    return results


def _parse_float(s: str) -> Optional[float]:
    """安全解析浮点。"""
    s = re.sub(r"[^\d.]", "", s)
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def fetch_hydrology(station: str, metric: str = "both") -> Dict[str, Any]:
    """查询水文站实时水情。

    Args:
        station: 水文站名（"吴堡"/"龙门"/"府谷"）
        metric: water_level / flow / both

    Returns:
        与 mock_get_hydrology 兼容的 dict：
        {
            "station": str,
            "water_level_m": float,
            "warning_level_m": float,
            "guaranteed_level_m": float,
            "flow_m3_s": float,
            "warning_flow_m3_s": float,
            "fetched_at": iso,
            "source": "qqjjsj_realtime",
            "observed_time": str  # 数据源观测时间，如 "2026-7-21 8:00"
        }

    Raises:
        RuntimeError: 数据源访问失败或站点不存在
    """
    if station not in STATION_PARAMS:
        raise RuntimeError(f"不支持的水文站：{station}（支持：{list(STATION_PARAMS.keys())}）")

    settings = get_settings()
    cache_key = f"hydro:{station}"
    now = time.time()

    # 检查缓存
    if cache_key in _cache:
        cached_at, cached_data = _cache[cache_key]
        if now - cached_at < settings.HYDRO_CACHE_TTL:
            logger.debug("[hydrology] 命中缓存 %s", cache_key)
            return _filter_metric(cached_data, metric)

    # 抓列表 + 详情
    try:
        article_url, article_title = _find_latest_yellow_river_article(settings.HYDRO_SOURCE_URL)
        logger.info("[hydrology] 抓取文章: %s", article_title)
        records = _parse_hydro_table(article_url)
    except requests.RequestException as e:
        logger.exception("[hydrology] 数据源访问失败")
        raise RuntimeError(f"水文数据源访问失败：{e}") from e
    except (RuntimeError, ValueError) as e:
        logger.exception("[hydrology] 数据解析失败")
        raise RuntimeError(f"水文数据解析失败：{e}") from e

    # 过滤目标站点
    target = next((r for r in records if r["station"] == station), None)
    if not target:
        raise RuntimeError(
            f"数据源未找到站点 {station}（数据源共 {len(records)} 条记录，"
            f"可用站点：{[r['station'] for r in records][:10]}）"
        )

    water_level = _parse_float(target.get("water_level", ""))
    flow = _parse_float(target.get("flow", ""))
    params = STATION_PARAMS[station]

    if water_level is None and flow is None:
        raise RuntimeError(f"站点 {station} 数据解析失败: {target}")

    result: Dict[str, Any] = {
        "station": station,
        "river": params["river"],
        "observed_time": target.get("time", ""),
        "fetched_at": _now_iso(),
        "source": "qqjjsj_realtime",
        "article_title": article_title,
    }

    if water_level is not None:
        result["water_level_m"] = round(water_level, 2)
        result["warning_level_m"] = params["warning_level_m"]
        result["guaranteed_level_m"] = params["guaranteed_level_m"]
        # 超警幅度（正=超警）
        result["above_warning_m"] = round(water_level - params["warning_level_m"], 2)

    if flow is not None:
        result["flow_m3_s"] = round(flow, 0)
        result["warning_flow_m3_s"] = params["warning_flow_m3_s"]
        result["above_warning_flow_m3_s"] = round(flow - params["warning_flow_m3_s"], 0)

    # 写入缓存
    _cache[cache_key] = (now, result)
    return _filter_metric(result, metric)


def _filter_metric(data: Dict[str, Any], metric: str) -> Dict[str, Any]:
    """按 metric 过滤返回字段。"""
    if metric == "water_level":
        return {k: v for k, v in data.items() if "flow" not in k.lower() or k == "warning_flow_m3_s"}
    elif metric == "flow":
        return {k: v for k, v in data.items() if "level" not in k.lower() or k == "warning_level_m"}
    return data
