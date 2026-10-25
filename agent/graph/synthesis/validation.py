"""研判结果校验：JSON 解析、规则定级与引用溯源。"""
import logging
import re
from typing import Any

from agent.graph.synthesizer import compute_warning_level, get_actions_for_level
from agent.utils import (
    parse_json_from_llm,
)

logger = logging.getLogger(__name__)

# 引用校验失败时的最大重生成次数
_MAX_VERIFY_RETRIES = 2


def _parse_synthesizer_json(content: str) -> dict[str, Any] | None:
    """解析 synthesizer LLM 返回的 JSON，多策略容错。

    返回 None 表示无法解析。
    """
    return parse_json_from_llm(content)


def _normalize_level(result: dict[str, Any]) -> None:
    """规范化 warning_level 字段为 I/II/III/IV。"""
    level = result.get("warning_level", "")
    if level and level not in ("I", "II", "III", "IV"):
        level_map = {"Ⅰ": "I", "Ⅱ": "II", "Ⅲ": "III", "Ⅳ": "IV",
                     "1": "I", "2": "II", "3": "III", "4": "IV"}
        level = level_map.get(level, "")
    if level:
        result["warning_level"] = level


def _check_level_consistency(
    result: dict[str, Any],
    tool_results: dict[str, Any],
) -> tuple[bool, str, str]:
    """预警等级在线一致性门：规则引擎重算 vs LLM 输出（CoVe 式校验，零额外 LLM 成本）。

    规则引擎 compute_warning_level 是全项目单一权威来源（线上研判 + 训练等级真值
    + 奖励函数共用），LLM 等级与其不一致即视为幻觉。

    Returns:
        (是否通过, 反馈信息, 规则引擎等级)
        工具结果中没有可判级数据（流量/降雨/水位全缺）时放行，不用默认 IV 强压 LLM。
    """
    try:
        rule_level, rule_reason = compute_warning_level(tool_results)
    except Exception as e:
        logger.debug("[synthesizer] 规则引擎等级计算失败（跳过一致性校验）：%s", e)
        return True, "", ""
    if not rule_reason or "暂无足够数据" in rule_reason:
        return True, "", rule_level
    llm_level = result.get("warning_level", "")
    if llm_level == rule_level:
        return True, "", rule_level
    return (
        False,
        f"预警等级与规则引擎不一致：基于工具数据的规则判定为 {rule_level} 级"
        f"（{rule_reason}），但回答中给出 {llm_level or '(空)'} 级。"
        f"请对照【预警等级阈值标准】和工具数据，以规则判定为准修正 warning_level。",
        rule_level,
    )


def _apply_rule_level_override(
    result: dict[str, Any],
    rule_level: str,
    rule_reason: str = "",
) -> None:
    """重试用尽仍不一致：以规则引擎等级覆盖 LLM 输出，并在 reasoning 中留痕。"""
    if not rule_level:
        return
    old = result.get("warning_level", "")
    result["warning_level"] = rule_level
    note = (
        f"[等级校正] LLM 原判 {old or '(空)'} 级，"
        f"经规则引擎按工具数据校正为 {rule_level} 级"
        + (f"（{rule_reason}）" if rule_reason else "")
    )
    result["reasoning"] = (note + "。" + (result.get("reasoning") or "")).lstrip("。；")
    logger.warning("[synthesizer] %s", note)


def _rule_engine_fallback_metadata(tool_results: dict[str, Any]) -> dict[str, Any]:
    """metadata 输出异常时的规则引擎降级（永不让请求因格式问题硬失败）。

    LLM 输出非法 JSON（如被 token 截断）时，用规则引擎重算等级与标准措施，
    citations 置空，Phase 2 仍会基于此生成自然语言回答。
    """
    try:
        rule_level, rule_reason = compute_warning_level(tool_results)
    except Exception:
        rule_level, rule_reason = "", ""
    has_data = bool(rule_reason) and "暂无足够数据" not in rule_reason
    level = rule_level if has_data else ""
    return {
        "warning_level": level,
        "reasoning": (
            "（综合研判结构化输出异常，已降级为规则引擎研判）"
            + (rule_reason if has_data else "")
        ),
        "actions": get_actions_for_level(level) if has_data else [],
        "citations": [],
    }


def _has_web_sources(source_registry: dict[int, dict[str, Any]]) -> bool:
    """来源注册表中是否存在联网搜索来源（引用空缺补救的触发条件）。"""
    return any(
        isinstance(s, dict) and s.get("source_type") == "web_search"
        for s in source_registry.values()
    )


def _verify_citations(
    citations: list[dict[str, Any]],
    source_registry: dict[int, dict[str, Any]],
) -> tuple[bool, str]:
    """Citation Grounding 校验：检查每条引用的 quote 是否真实存在于来源原文中。

    Returns:
        (是否通过, 失败原因)
    """
    if not citations:
        # 无引用时：若也没有可用来源则通过；有来源却没引用也放行（不强制）
        return True, ""
    for cite in citations:
        ref_id = cite.get("ref_id")
        quote = (cite.get("quote", "") or "").strip()
        if not quote:
            return False, f"引用 [{ref_id}] 的 quote 为空"
        if ref_id not in source_registry:
            return False, f"引用编号 [{ref_id}] 在来源中不存在"
        original = source_registry[ref_id].get("text", "")
        # 精确子串匹配（先尝试原文，再尝试去除空白后匹配）
        if quote in original:
            continue
        # 宽松匹配：去除所有空白后比较（应对换行/多空格差异）
        if re.sub(r"\s+", "", quote) in re.sub(r"\s+", "", original):
            continue
        return False, f'引用 [{ref_id}] 的 quote 在来源原文中找不到：「{quote[:40]}...」'
    return True, ""


def _build_citations_with_metadata(
    raw_citations: list[dict[str, Any]],
    source_registry: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    """把 LLM 输出的引用与 source_registry 中的元数据拼合，输出给前端。

    只有 web_search 结果在 source_registry 中，所以 citations 只包含联网搜索的链接。
    过滤掉校验不通过的引用（quote 找不到原文或编号不存在）。
    """
    result = []
    seen_ref_ids = set()
    for cite in raw_citations:
        ref_id = cite.get("ref_id")
        quote = (cite.get("quote", "") or "").strip()
        if ref_id not in source_registry or not quote:
            continue
        # 二次校验（重试用尽后可能仍有无效引用）
        original = source_registry[ref_id].get("text", "")
        if quote not in original and re.sub(r"\s+", "", quote) not in re.sub(r"\s+", "", original):
            continue
        if ref_id in seen_ref_ids:
            continue
        seen_ref_ids.add(ref_id)
        src = source_registry[ref_id]
        result.append({
            "ref_id": ref_id,
            "quote": quote,
            "source_type": src.get("source_type", "web_search"),
            "title": src.get("title", ""),
            "url": src.get("url", ""),
        })
    return result
