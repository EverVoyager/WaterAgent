"""工具结果摘要、引用来源编号与阈值说明。"""
import json
import logging
from typing import Any

from agent.utils import (
    WARNING_THRESHOLDS,
)

logger = logging.getLogger(__name__)


def _format_tool_results_for_llm(
    tool_results: dict[str, Any],
) -> tuple[str, dict[int, dict[str, Any]]]:
    """把工具结果格式化为 LLM 可读文本，并构建来源注册表。

    编号策略（借鉴 Perplexica：模型看到的编号 = 用户看到的引用，天然对齐）：
    - web_search 结果与法规检索条款带 [编号]，进入 source_registry（可引用、可校验）；
    - 其他工具数据（水文/天气/径流/GIS/阈值）用【语义标签】替代编号——
      模型无法给它们标 [N]，从源头杜绝"编号空间错位"导致的编造引用。

    source_registry 结构：
        编号 -> {source_type, title, snippet, url, text}
    其中 text 是用于 Citation Grounding 校验的原文（与展示给 LLM 的文本一致），
    LLM 的 quote 必须是其子串。
    """
    parts: list[str] = []
    registry: dict[int, dict[str, Any]] = {}
    counter = [0]

    def _next_id() -> int:
        counter[0] += 1
        return counter[0]

    if not tool_results:
        parts.append("(暂无工具结果)")
    else:
        for key, val in tool_results.items():
            if not isinstance(val, dict):
                continue
            # 联网搜索结果：每条一个编号，进入 source_registry（可引用）
            if "web_search" in key:
                results = val.get("results", [])
                parts.append(f"{key}: 搜索到 {len(results)} 条结果：")
                for r in results:
                    title = r.get("title", "")
                    snippet = (r.get("snippet", "") or "").strip()
                    url = r.get("url", "")
                    ref_id = _next_id()
                    parts.append(
                        f"  [{ref_id}] {title}\n"
                        f"      {snippet}\n"
                        f"      链接：{url}"
                    )
                    registry[ref_id] = {
                        "source_type": "web_search",
                        "title": title,
                        "snippet": snippet,
                        "url": url,
                        "text": snippet,  # Citation Grounding 校验用
                    }
            # 法规检索结果：每条一个编号，进入 source_registry（可引用、可校验）
            elif "search_regulation" in key:
                hits = val.get("hits", [])
                parts.append(f"{key}: 检索到 {len(hits)} 条法规条款：")
                for h in hits:
                    content = (h.get("content", "") or "")[:200]
                    ref_id = _next_id()
                    parts.append(
                        f"  [{ref_id}] {h.get('title', '')} {h.get('article', '')}\n"
                        f"      {content}"
                    )
                    registry[ref_id] = {
                        "source_type": "regulation",
                        "title": h.get("title", ""),
                        "snippet": content,
                        "url": "",
                        "text": content,  # 与展示给 LLM 的文本一致，保证子串校验可过
                    }
            # GIS 分析结果：语义标签（不可引用）
            elif "query_gis_terrain" in key:
                gis_lines = ["【GIS 地形分析】"]
                if val.get("slope"):
                    s = val["slope"]
                    gis_lines.append(
                        f"  坡度：均值 {s.get('mean_degree')}°，最大 {s.get('max_degree')}°，"
                        f"高风险区 {s.get('high_risk_area_km2')}km²"
                    )
                if val.get("channel_cross_section"):
                    c = val["channel_cross_section"]
                    gis_lines.append(
                        f"  河床断面：河宽 {c.get('width_m')}m，最大水深 {c.get('max_depth_m')}m"
                    )
                if val.get("inundation"):
                    f = val["inundation"]
                    gis_lines.append(
                        f"  淹没范围：面积 {f.get('inundated_area_km2')}km²，"
                        f"受影响村庄 {f.get('affected_villages')} 个"
                    )
                parts.append("\n".join(gis_lines))
            # 水文结果：语义标签（不可引用）
            elif "get_hydrology" in key:
                snippet = _extract_hydrology_summary(val)
                parts.append(f"【实时水情】{json.dumps(snippet, ensure_ascii=False)}")
            # 天气结果：语义标签（不可引用）
            elif "get_weather" in key:
                snippet = _extract_weather_summary(val)
                parts.append(f"【天气预报】{json.dumps(snippet, ensure_ascii=False)}")
            # 径流预测：语义标签（不可引用）
            elif "predict_runoff" in key:
                snippet = _extract_runoff_summary(val)
                parts.append(f"【径流预测】{json.dumps(snippet, ensure_ascii=False)}")
            # 其他工具结果：语义标签（不可引用）
            else:
                text = json.dumps(val, ensure_ascii=False)
                if len(text) > 500:
                    text = text[:500] + "...(truncated)"
                parts.append(f"【{key}】{text}")

    # 追加阈值标准（语义标签，不作为引用展示）
    parts.append(f"【预警等级阈值标准】{_build_threshold_source()}")

    return "\n".join(parts), registry


def _build_threshold_source() -> str:
    """构造预警等级阈值标准的原文描述（供 LLM 引用 + 校验）。"""
    f1 = WARNING_THRESHOLDS["flow_level1"]
    f2 = WARNING_THRESHOLDS["flow_level2"]
    f3 = WARNING_THRESHOLDS["flow_level3"]
    r1 = WARNING_THRESHOLDS["rain_level1"]
    r2 = WARNING_THRESHOLDS["rain_level2"]
    return (
        f"防汛预警等级阈值标准："
        f"Ⅰ级（红色）：流量≥{f1}m³/s，或水位超保证水位，或24h降雨>{r1}mm；"
        f"Ⅱ级（橙色）：流量{f2}-{f1}m³/s，或水位超警戒水位，或24h降雨{r2}-{r1}mm；"
        f"Ⅲ级（黄色）：流量{f3}-{f2}m³/s，或水位接近警戒水位；"
        f"Ⅳ级（蓝色）：流量<{f3}m³/s，水位正常。"
    )


def _extract_hydrology_summary(val: dict[str, Any]) -> dict[str, Any]:
    """M9：提取水文结果关键字段，避免长文本污染上下文。"""
    keys = [
        "station", "water_level_m", "flow_m3_s", "warning_level_m",
        "guaranteed_level_m", "above_warning_m", "observation_time", "source",
    ]
    return {k: val[k] for k in keys if k in val}


def _extract_weather_summary(val: dict[str, Any]) -> dict[str, Any]:
    """M9：提取天气结果统计摘要，丢弃逐小时 series。

    注意：fetch_weather 返回的温度/天气描述嵌套在 current 字典里，
    需要从 current 中提取并拍平为 current_temp_c / current_weather。
    """
    summary = {k: val[k] for k in [
        "location", "total_rainfall_mm", "max_hourly_rainfall_mm",
        "hours", "source",
    ] if k in val}
    # 从 current 嵌套字典提取实况温度和天气描述
    current = val.get("current") or {}
    if current.get("temperature"):
        summary["current_temp_c"] = current["temperature"]
    if current.get("weather"):
        summary["current_weather"] = current["weather"]
    if current.get("humidity"):
        summary["humidity"] = current["humidity"]
    if current.get("winddirection"):
        summary["wind_direction"] = current["winddirection"]
        summary["wind_power"] = current.get("windpower", "")
    # series 只保留降雨时段数和最大值，不展开
    series = val.get("series", [])
    if series:
        rainy_hours = sum(1 for s in series if (s or {}).get("rainfall_mm", 0) > 0)
        summary["rainy_hours"] = rainy_hours
        summary["series_points"] = len(series)
    return summary


def _extract_runoff_summary(val: dict[str, Any]) -> dict[str, Any]:
    """M9：提取径流预测关键指标，丢弃过程线序列。"""
    summary = {k: val[k] for k in [
        "station", "rainfall_mm", "runoff_depth_mm", "cn",
        "area_km2", "tc_hours", "base_flow_m3_s",
        "peak_flow_m3_s", "peak_time", "series_points", "source",
    ] if k in val}
    # 过程线只保留前 3 + 洪峰 + 后 3 的采样
    series = val.get("flow_series") or val.get("series") or []
    if series and len(series) > 6:
        # 找到洪峰位置
        try:
            peak_idx = max(range(len(series)), key=lambda i: (
                series[i].get("flow_m3_s", 0) if isinstance(series[i], dict) else 0
            ))
            sample_idx = sorted(set(
                list(range(3)) + [peak_idx-1, peak_idx, peak_idx+1] +
                list(range(len(series)-3, len(series)))
            ))
            sample_idx = [i for i in sample_idx if 0 <= i < len(series)]
            summary["flow_series_sample"] = [series[i] for i in sample_idx]
        except (ValueError, TypeError):
            pass
    elif series:
        summary["flow_series_sample"] = series[:6]
    return summary
