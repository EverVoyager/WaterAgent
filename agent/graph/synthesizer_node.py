"""综合研判节点（synthesizer）。

从 workflow.py 拆分而来。依赖 state, errors, llm_helpers, agent.prompts, app.core.llm。

注意：`_summarize_results` 原计划放在 runner.py，但它使用本模块的
`_extract_*` 函数，且被 nodes.planner_node 调用。若放 runner.py 会造成
nodes → runner → nodes 的循环依赖，故就近放在本模块（synthesizer_node）。
"""
import json
import logging
from typing import Any, Dict

from openai import (
    APIConnectionError,
    APITimeoutError,
    APIError,
    RateLimitError,
)

from agent.graph.errors import LLMError, _classify_llm_error
from agent.graph.state import AgentState
from agent.prompts import SYNTHESIZER_PROMPT
from app.core.llm import LLM_TIMEOUTS, get_llm_client, get_llm_config

logger = logging.getLogger(__name__)


def synthesizer_node(state: AgentState) -> Dict[str, Any]:
    """综合研判节点：LLM 综合所有工具结果生成最终回答。

    LLM 调用失败时直接抛错，由上层 API 返回 500 给前端。
    """
    tool_results = state.get("tool_results", {})
    query = state["user_query"]

    synth = _synth_via_llm(query, tool_results)

    logger.info("[synthesizer] LLM synth level=%s", synth.get("warning_level", ""))
    return {
        "warning_level": synth.get("warning_level", ""),
        "reasoning": synth.get("reasoning", ""),
        "actions": synth.get("actions", []),
        "final_answer": synth.get("answer", ""),
    }


def _synth_via_llm(query: str, tool_results: Dict[str, Any]) -> Dict[str, Any]:
    """LLM 综合所有工具结果生成最终回答。

    LLM 调用失败时抛 LLMError。
    """
    tool_results_text = _format_tool_results_for_llm(tool_results)

    settings = get_llm_config()
    client = get_llm_client().with_options(timeout=LLM_TIMEOUTS["synthesizer"])

    # 自进化：注入用户偏好 + 领域知识
    preferences = ""
    try:
        from agent.memory import get_user_preferences
        preferences = get_user_preferences()
        if preferences:
            logger.info("[synthesizer] 注入用户偏好：\n%s", preferences[:200])
    except Exception as e:
        logger.debug("[synthesizer] 注入偏好失败（不影响主流程）：%s", e)

    # 在 system prompt 后追加用户偏好
    system_content = SYNTHESIZER_PROMPT
    if preferences:
        system_content = (
            SYNTHESIZER_PROMPT
            + "\n\n请在生成回答时遵循以下用户偏好和历史知识：\n"
            + preferences
        )

    try:
        resp = client.chat.completions.create(
            model=settings["model"],
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": (
                    f"用户问题：{query}\n\n"
                    f"工具返回结果：\n{tool_results_text}"
                )},
            ],
            temperature=0.3,
            max_tokens=1500,
        )
    except (APITimeoutError, RateLimitError, APIConnectionError, APIError) as e:
        logger.exception("[synthesizer] LLM 综合研判调用失败 (%s)", type(e).__name__)
        raise _classify_llm_error(e) from e
    except Exception as e:
        logger.exception("[synthesizer] LLM 未知异常")
        raise _classify_llm_error(e) from e

    content = (resp.choices[0].message.content or "").strip()
    # 兼容 ```json ``` 包裹
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()
        if content.endswith("```"):
            content = content[:-3].strip()

    try:
        result = json.loads(content)
    except json.JSONDecodeError as e:
        logger.error("[synthesizer] LLM 返回非 JSON: %s", content[:200])
        raise LLMError("format_error", f"LLM 综合研判返回格式异常（非 JSON）：{e}", status_code=502) from e

    # 规范化等级字段
    level = result.get("warning_level", "")
    if level and level not in ("I", "II", "III", "IV"):
        level_map = {"Ⅰ": "I", "Ⅱ": "II", "Ⅲ": "III", "Ⅳ": "IV",
                     "1": "I", "2": "II", "3": "III", "4": "IV"}
        level = level_map.get(level, "")
    if level:
        result["warning_level"] = level
    return result


def _synth_via_llm_stream(query: str, tool_results: Dict[str, Any]):
    """流式版本的综合研判生成器。

    两阶段设计：
      阶段 1：LLM 非流式调用，拿到结构化 JSON（warning_level/reasoning/actions/answer）
      阶段 2：把 answer 按细粒度切分（2-4 字一组），逐组 yield，模拟 token 流式

    借鉴 agent-service-toolkit 的 stream_mode=["updates", "messages"] 思路：
      - updates：节点级状态（阶段 1 完成后推送结构化结果）
      - messages：token 级流式（阶段 2 逐字推送 answer）

    yield 事件：
      - {"type": "synth_meta", "data": {warning_level, reasoning, actions}}  # 结构化结果
      - {"type": "answer_delta", "content": "..."}                            # answer 分片
    """
    # 阶段 1：非流式拿结构化 JSON
    synth = _synth_via_llm(query, tool_results)
    answer_text = synth.get("answer", "")

    # 推送结构化元数据（不含 answer，前端可提前渲染等级横幅等）
    yield {
        "type": "synth_meta",
        "data": {
            "warning_level": synth.get("warning_level", ""),
            "reasoning": synth.get("reasoning", ""),
            "actions": synth.get("actions", []),
        },
    }

    # 阶段 2：细粒度切分 answer，逐组推送
    # 按标点+长度切分，保留标点，每组 2-4 字，模拟 token 流式
    import re
    # 先按标点切分，保留标点
    segments = re.split(r"(?<=[。！？\n；：，、])", answer_text)
    for seg in segments:
        if not seg.strip():
            continue
        # 长段落再按 3-4 字切分，模拟 token 粒度
        if len(seg) > 6:
            for i in range(0, len(seg), 3):
                piece = seg[i:i + 3]
                if piece:
                    yield {"type": "answer_delta", "content": piece}
        else:
            yield {"type": "answer_delta", "content": seg}

    # 推送完整 answer 供 done 事件使用
    yield {"type": "synth_answer_full", "content": answer_text}


def _format_tool_results_for_llm(tool_results: Dict[str, Any]) -> str:
    """M9：把工具结果格式化为 LLM 可读的文本，控制长度避免上下文溢出。

    策略：
    - 法规检索：每条条款内容限制 200 字
    - GIS 分析：只保留统计摘要
    - 水文/天气：只保留关键字段，丢弃 series 等长序列
    - 径流预测：只保留洪峰/径流深/过程线摘要
    """
    if not tool_results:
        return "(暂无工具结果)"
    parts = []
    for key, val in tool_results.items():
        if not isinstance(val, dict):
            continue
        # 法规检索结果：限制每条条款长度
        if "search_regulation" in key:
            hits = val.get("hits", [])
            parts.append(f"[{key}] 检索到 {len(hits)} 条法规条款：")
            for i, h in enumerate(hits, 1):
                content = (h.get("content", "") or "")[:200]
                parts.append(
                    f"  ({i}) {h.get('title', '')} {h.get('article', '')}\n"
                    f"      {content}"
                )
        # GIS 分析结果：只保留统计摘要
        elif "query_gis_terrain" in key:
            parts.append(f"[{key}] GIS 地形分析结果：")
            if val.get("slope"):
                s = val["slope"]
                parts.append(
                    f"  坡度：均值 {s.get('mean_degree')}°，最大 {s.get('max_degree')}°，"
                    f"高风险区 {s.get('high_risk_area_km2')}km²"
                )
            if val.get("channel_cross_section"):
                c = val["channel_cross_section"]
                parts.append(
                    f"  河床断面：河宽 {c.get('width_m')}m，最大水深 {c.get('max_depth_m')}m"
                )
            if val.get("inundation"):
                f = val["inundation"]
                parts.append(
                    f"  淹没范围：面积 {f.get('inundated_area_km2')}km²，"
                    f"受影响村庄 {f.get('affected_villages')} 个"
                )
        # 水文结果：只保留关键字段，丢弃缓存标记等
        elif "get_hydrology" in key:
            snippet = _extract_hydrology_summary(val)
            parts.append(f"[{key}] {json.dumps(snippet, ensure_ascii=False)}")
        # 天气结果：丢弃 series，只保留统计
        elif "get_weather" in key:
            snippet = _extract_weather_summary(val)
            parts.append(f"[{key}] {json.dumps(snippet, ensure_ascii=False)}")
        # 径流预测：丢弃过程线序列，只保留洪峰/径流深
        elif "predict_runoff" in key:
            snippet = _extract_runoff_summary(val)
            parts.append(f"[{key}] {json.dumps(snippet, ensure_ascii=False)}")
        # 其他工具结果：截断到 500 字
        else:
            text = json.dumps(val, ensure_ascii=False)
            if len(text) > 500:
                text = text[:500] + "...(truncated)"
            parts.append(f"[{key}] {text}")
    return "\n".join(parts) if parts else "(暂无关键字段)"


def _extract_hydrology_summary(val: Dict[str, Any]) -> Dict[str, Any]:
    """M9：提取水文结果关键字段，避免长文本污染上下文。"""
    keys = [
        "station", "water_level_m", "flow_m3_s", "warning_level_m",
        "guaranteed_level_m", "above_warning_m", "observation_time", "source",
    ]
    return {k: val[k] for k in keys if k in val}


def _extract_weather_summary(val: Dict[str, Any]) -> Dict[str, Any]:
    """M9：提取天气结果统计摘要，丢弃逐小时 series。"""
    summary = {k: val[k] for k in [
        "location", "total_rainfall_mm", "max_hourly_rainfall_mm",
        "current_weather", "current_temp_c", "hours", "source",
    ] if k in val}
    # series 只保留降雨时段数和最大值，不展开
    series = val.get("series", [])
    if series:
        rainy_hours = sum(1 for s in series if (s or {}).get("rainfall_mm", 0) > 0)
        summary["rainy_hours"] = rainy_hours
        summary["series_points"] = len(series)
    return summary


def _extract_runoff_summary(val: Dict[str, Any]) -> Dict[str, Any]:
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


def _summarize_results(tool_results: Dict[str, Any]) -> str:
    """M9：把工具结果压缩为字符串摘要供 LLM 阅读（planner 用）。

    控制总长度在 800 字内，避免 planner 上下文膨胀。

    注意：此函数被 nodes.planner_node 调用，放在本模块（而非 runner.py）
    是为避免 nodes → runner → nodes 的循环依赖；同时复用本模块的 _extract_* 函数。
    """
    if not tool_results:
        return "(暂无)"
    parts = []
    for key, val in tool_results.items():
        if not isinstance(val, dict):
            continue
        # 根据工具类型提取关键字段
        if "get_hydrology" in key:
            snippet = _extract_hydrology_summary(val)
        elif "get_weather" in key:
            snippet = _extract_weather_summary(val)
        elif "predict_runoff" in key:
            snippet = _extract_runoff_summary(val)
        elif "search_regulation" in key:
            hits = val.get("hits", [])
            snippet = {"hit_count": len(hits),
                       "titles": [h.get("title", "") for h in hits[:3]]}
        elif "query_gis_terrain" in key:
            snippet = {k: val[k] for k in ["slope", "channel_cross_section", "inundation"]
                       if k in val}
        elif "generate_plan" in key:
            snippet = {k: val[k] for k in ["warning_level", "actions", "affected_area"]
                       if k in val}
        else:
            snippet = {k: val[k] for k in ["station", "source"] if k in val}
        if snippet:
            parts.append(f"{key}: {json.dumps(snippet, ensure_ascii=False)}")
    result = "\n".join(parts)
    # 总长度兜底
    if len(result) > 800:
        result = result[:800] + "...(truncated)"
    return result if result else "(暂无关键字段)"
