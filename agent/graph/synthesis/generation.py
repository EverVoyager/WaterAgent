"""结构化研判生成与校验重试，供同步和流式入口共用。"""
import logging
from typing import Any

from agent.graph.synthesis.context import _build_synth_messages
from agent.graph.synthesis.transport import (
    _append_structured_retry_feedback,
    _call_synth_with_fallback,
    _extract_structured_content,
)
from agent.graph.synthesis.validation import (
    _MAX_VERIFY_RETRIES,
    _apply_rule_level_override,
    _build_citations_with_metadata,
    _check_level_consistency,
    _has_web_sources,
    _normalize_level,
    _parse_synthesizer_json,
    _rule_engine_fallback_metadata,
    _verify_citations,
)
from agent.prompts.synthesizer import SYNTH_META_SCHEMA as _SYNTH_META_SCHEMA
from agent.utils import (
    strip_citation_markers,
)
from app.core import llm
from app.core.llm import LLM_TIMEOUTS, strip_think

logger = logging.getLogger(__name__)


def _synth_via_llm(
    query: str,
    tool_results: dict[str, Any],
    history: list[dict[str, Any]] | None = None,
    skill_instructions: str = "",
    recalled_context: str = "",
    experiences: str = "",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """LLM 综合所有工具结果生成最终回答（含 Citation Grounding 校验循环）。

    非流式路径：一次 LLM 调用返回完整 JSON（含 answer）。
    流式路径请用 _synth_metadata_via_llm + _stream_answer_via_llm（两阶段真流式）。

    Args:
        history: 压缩后的历史对话（可选），用于注入上下文摘要
        skill_instructions: 匹配到的 Skill 行为指令（借鉴 Claude Skills 按需加载）
        experiences: planner round-1 检索的历史经验（方案 A 作答端注入）

    Returns:
        (synth_result, citations)
        - synth_result: 含 warning_level/reasoning/actions/answer
        - citations: 已校验的引用列表，每条含 ref_id/quote/source_type + 来源元数据

    LLM 调用失败时抛 LLMError。
    """
    messages, source_registry = _build_synth_messages(
        query, tool_results, history, skill_instructions,
        recalled_context=recalled_context,
        experiences=experiences,
    )

    settings = llm.get_llm_config()
    client = llm.get_llm_client().with_options(timeout=LLM_TIMEOUTS["synthesizer"])

    # Generate-Verify-Correct 循环
    result: dict[str, Any] | None = None
    format_retry = False  # 结构化输出是否触发过重试（反思 format_error 信号源）
    for attempt in range(_MAX_VERIFY_RETRIES + 1):
        resp = _call_synth_with_fallback(client, settings["model"], messages)
        msg = resp.choices[0].message
        # 仅取 message.content，不回退 reasoning_content（推理过程不应作为答案）
        # 工具通道优先取 tool_calls arguments，正文兜底
        raw_content = _extract_structured_content(msg)
        # 剥离 <think> 块后解析 JSON
        content = strip_think(raw_content)
        result = _parse_synthesizer_json(content)
        if result is None and not content and raw_content:
            # strip_think 剥光了（JSON 包在 <think> 里），从原始内容提取
            logger.warning("[synthesizer] strip_think 后内容为空，尝试从原始内容提取 JSON")
            result = _parse_synthesizer_json(raw_content)
        if result is None:
            logger.error("[synthesizer] LLM 返回非 JSON（raw 前 300 字）: %s", raw_content[:300])
            if attempt < _MAX_VERIFY_RETRIES:
                # 带反馈重试（常见于输出被截断或字段类型错误）
                format_retry = True
                _append_structured_retry_feedback(
                    messages, msg,
                    "你上次的输出不是合法 JSON（可能被截断）。请严格输出符合 schema 的"
                    " JSON 对象：actions 只放简短的应急措施文本（不要放链接）；"
                    "字段值中的双引号须转义；citations 只引用上下文中带 [编号] 的来源。",
                )
                continue
            # 重试用尽：规则引擎降级，不让请求因格式问题硬失败
            logger.error("[synthesizer] JSON 解析重试用尽，降级为规则引擎 metadata")
            result = _rule_engine_fallback_metadata(tool_results)
            if not result.get("answer"):
                result["answer"] = (
                    (result.get("reasoning") or "")
                    + ("\n建议措施：" + "；".join(result["actions"]) if result["actions"] else "")
                )

        _normalize_level(result)
        raw_citations = result.get("citations", []) or []

        # 引用：逐条校验，无效的统一过滤（不触发重生成，避免一次坏 quote
        # 导致 60-90s 的整轮重新调用）
        cite_ok, cite_feedback = _verify_citations(raw_citations, source_registry)
        if cite_ok:
            logger.info("[synthesizer] 引用校验通过（attempt=%d，%d 条引用）",
                        attempt, len(raw_citations))
        else:
            logger.warning("[synthesizer] 引用存在无效项，将过滤（不重生成）：%s",
                           cite_feedback)

        # 引用空缺检查：存在联网来源但模型未输出任何引用 → 视为可重试失败
        # （评估失效模式 4：web_search 用例引用可溯源率仅 25%）
        citations_missing = not raw_citations and _has_web_sources(source_registry)

        # 等级一致性门：不一致才触发重生成（安全攸关）
        level_ok, level_feedback, rule_level = _check_level_consistency(result, tool_results)
        if level_ok and not citations_missing:
            break

        if attempt < _MAX_VERIFY_RETRIES:
            # 追加校验反馈，要求 LLM 修正后重生成
            format_retry = True
            if not level_ok:
                logger.warning("[synthesizer] 等级校验失败（attempt=%d）：%s",
                               attempt, level_feedback)
                feedback = (f"校验失败：{level_feedback}\n"
                            "请重新生成，warning_level 必须与基于工具数据的规则判定一致；")
            else:
                logger.warning(
                    "[synthesizer] citations 为空但存在联网来源（attempt=%d），触发引用补救重试",
                    attempt,
                )
                feedback = ("你上次的输出缺少 citations，但上下文中存在联网搜索来源；"
                            "凡回答使用了联网搜索内容，必须在 citations 数组中逐条给出 "
                            "{ref_id, quote, source_type}，")
            _append_structured_retry_feedback(
                messages, msg,
                feedback
                + "citations 中的 quote 必须是对应编号来源中逐字摘录的原文片段，"
                "无法找到原文就移除该引用。",
            )
        elif not level_ok:
            # 重试用尽：以规则引擎等级覆盖
            logger.warning("[synthesizer] 等级校验重试用尽，以规则引擎为准覆盖")
            _apply_rule_level_override(result, rule_level)
        # 引用补救重试用尽：接受空引用（宁缺毋假），不做进一步处理

    # 构造带元数据的引用列表
    citations = _build_citations_with_metadata(
        result.get("citations", []) or [], source_registry
    )
    _normalize_level(result)
    # 展示层兜底：answer 只保留对应已验证引用（联网搜索）的 [N] 标记
    valid_ids = {c["ref_id"] for c in citations}
    if result.get("answer"):
        result["answer"] = strip_citation_markers(result["answer"], valid_ids)
    result["_format_retry"] = format_retry
    return result, citations


def _synth_metadata_via_llm(
    query: str,
    tool_results: dict[str, Any],
    history: list[dict[str, Any]] | None = None,
    skill_instructions: str = "",
    recalled_context: str = "",
    experiences: str = "",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """两阶段流式 Phase 1（同步版）：非流式 LLM 调用获取结构化 metadata。

    流式进度事件版本见 _synth_metadata_via_llm_iter（本函数是其同步包装）。
    """
    result, citations = None, None
    for ev in _synth_metadata_via_llm_iter(
        query, tool_results, history, skill_instructions,
        recalled_context=recalled_context,
        experiences=experiences,
    ):
        if ev["type"] == "_synth_meta_result":
            result, citations = ev["result"], ev["citations"]
    return result, citations


def _synth_metadata_via_llm_iter(
    query: str,
    tool_results: dict[str, Any],
    history: list[dict[str, Any]] | None = None,
    skill_instructions: str = "",
    recalled_context: str = "",
    experiences: str = "",
):
    """两阶段流式 Phase 1（生成器版）：带进度事件，消除长时间静默。

    生成式 LLM 调用可能耗时 60-90s，期间推送 reasoning_step 事件让前端
    可见"正在生成/校验/修正"的进度，避免用户误以为系统无响应。

    校验策略（性能权衡）：
    - 引用校验失败 → 只过滤无效引用，不触发重生成（一次坏 quote 的代价
      不应是一整轮 60-90s 的重新调用；过滤逻辑在 _build_citations_with_metadata）
    - 预警等级与规则引擎不一致 → 触发重生成（安全攸关），重试用尽以规则引擎为准

    Yields:
        {"type": "reasoning_step", "step": "synthesizer", "phase": "thinking|decision", ...}
        {"type": "_synth_meta_result", "result": {...}, "citations": [...]}  # 最后一个事件
    """
    messages, source_registry = _build_synth_messages(
        query, tool_results, history, skill_instructions,
        recalled_context=recalled_context,
        experiences=experiences,
    )

    settings = llm.get_llm_config()
    client = llm.get_llm_client().with_options(timeout=LLM_TIMEOUTS["synthesizer"])

    result: dict[str, Any] | None = None
    format_retry = False  # 结构化输出是否触发过重试（反思 format_error 信号源）
    for attempt in range(_MAX_VERIFY_RETRIES + 1):
        yield {"type": "reasoning_step", "step": "synthesizer", "phase": "thinking",
               "message": "正在综合工具数据生成研判结论（等级/依据/措施）...",
               "details": {"attempt": attempt}}
        resp = _call_synth_with_fallback(client, settings["model"], messages, schema=_SYNTH_META_SCHEMA)
        msg = resp.choices[0].message
        # 工具通道优先取 tool_calls arguments，正文兜底
        raw_content = _extract_structured_content(msg)
        content = strip_think(raw_content)
        result = _parse_synthesizer_json(content)
        if result is None and not content and raw_content:
            logger.warning("[synthesizer] Phase 1 strip_think 后内容为空，尝试从原始内容提取 JSON")
            result = _parse_synthesizer_json(raw_content)
        if result is None:
            logger.error("[synthesizer] Phase 1 LLM 返回非 JSON（raw 前 300 字）: %s", raw_content[:300])
            if attempt < _MAX_VERIFY_RETRIES:
                format_retry = True
                yield {"type": "reasoning_step", "step": "synthesizer", "phase": "decision",
                       "message": "结构化输出格式异常（可能被截断），正在重新生成...",
                       "details": {"attempt": attempt}}
                _append_structured_retry_feedback(
                    messages, msg,
                    "你上次的输出不是合法 JSON（可能被截断）。请严格输出符合 schema 的"
                    " JSON 对象：actions 只放简短的应急措施文本（不要放链接）；"
                    "字段值中的双引号须转义；citations 只引用上下文中带 [编号] 的来源。",
                )
                continue
            # 重试用尽：规则引擎降级，不让请求因格式问题硬失败
            logger.error("[synthesizer] Phase 1 JSON 解析重试用尽，降级为规则引擎 metadata")
            result = _rule_engine_fallback_metadata(tool_results)

        _normalize_level(result)
        raw_citations = result.get("citations", []) or []

        yield {"type": "reasoning_step", "step": "synthesizer", "phase": "thinking",
               "message": "正在校验引用原文真实性与预警等级一致性...",
               "details": {"attempt": attempt, "citations": len(raw_citations)}}

        # 引用：逐条校验，无效的后续统一过滤（不触发重生成）
        cite_ok, cite_feedback = _verify_citations(raw_citations, source_registry)
        if cite_ok:
            logger.info("[synthesizer] Phase 1 引用校验通过（attempt=%d，%d 条引用）",
                        attempt, len(raw_citations))
        else:
            logger.warning("[synthesizer] Phase 1 引用存在无效项，将过滤（不重生成）：%s",
                           cite_feedback)

        # 引用空缺检查：存在联网来源但模型未输出任何引用 → 视为可重试失败
        # （评估失效模式 4，与非流式路径同策略）
        citations_missing = not raw_citations and _has_web_sources(source_registry)

        # 等级一致性门：不一致才触发重生成
        level_ok, level_feedback, rule_level = _check_level_consistency(result, tool_results)
        if level_ok and not citations_missing:
            break

        if attempt < _MAX_VERIFY_RETRIES:
            format_retry = True
            if not level_ok:
                logger.warning("[synthesizer] Phase 1 等级校验失败（attempt=%d）：%s",
                               attempt, level_feedback)
                yield {"type": "reasoning_step", "step": "synthesizer", "phase": "decision",
                       "message": "预警等级与规则引擎不一致，正在修正重新生成...",
                       "details": {"attempt": attempt, "rule_level": rule_level}}
                feedback = (f"校验失败：{level_feedback}\n"
                            "请重新生成，warning_level 必须与基于工具数据的规则判定一致；")
            else:
                logger.warning(
                    "[synthesizer] Phase 1 citations 为空但存在联网来源（attempt=%d），触发引用补救重试",
                    attempt,
                )
                yield {"type": "reasoning_step", "step": "synthesizer", "phase": "decision",
                       "message": "检测到联网来源未标注引用，正在补充引用...",
                       "details": {"attempt": attempt}}
                feedback = ("你上次的输出缺少 citations，但上下文中存在联网搜索来源；"
                            "凡回答使用了联网搜索内容，必须在 citations 数组中逐条给出 "
                            "{ref_id, quote, source_type}，")
            _append_structured_retry_feedback(
                messages, msg,
                feedback
                + "citations 中的 quote 必须是对应编号来源中逐字摘录的原文片段，"
                "无法找到原文就移除该引用。",
            )
        elif not level_ok:
            logger.warning("[synthesizer] Phase 1 等级校验重试用尽，以规则引擎为准覆盖")
            _apply_rule_level_override(result, rule_level)
        # 引用补救重试用尽：接受空引用（宁缺毋假）

    citations = _build_citations_with_metadata(
        result.get("citations", []) or [], source_registry
    )
    _normalize_level(result)
    result["_format_retry"] = format_retry
    yield {"type": "_synth_meta_result", "result": result, "citations": citations}
