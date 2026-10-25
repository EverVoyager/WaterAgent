"""两阶段研判流：元数据事件与逐 token 答案。"""
import logging
from typing import Any

from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    RateLimitError,
)

from agent.graph.errors import _classify_llm_error
from agent.graph.stream_output import stream_answer_events
from agent.graph.synthesis.context import _build_synth_messages
from agent.graph.synthesis.generation import _synth_metadata_via_llm_iter
from app.core import llm
from app.core.llm import LLM_TIMEOUTS

logger = logging.getLogger(__name__)


def _stream_answer_via_llm(
    query: str,
    tool_results: dict[str, Any],
    metadata: dict[str, Any],
    history: list[dict[str, Any]] | None = None,
    skill_instructions: str = "",
    valid_ref_ids: set[int] | None = None,
    recalled_context: str = "",
    experiences: str = "",
):
    """两阶段流式 Phase 2：LLM stream=True 逐 token 生成 answer。

    使用 Phase 1 的 metadata 作为上下文，确保 answer 与预警等级/措施一致。
    system prompt 为 Phase 1 system + SYNTH_ANSWER_ADDENDUM（answer_only=True），
    前缀对齐以命中 Phase 1 已算的 KV Cache；追加块要求模型只输出
    纯文本自然语言回答，避免再次输出 JSON 外壳被当作 answer 流式展示。
    <think> 块剥离状态机与 direct_chat_stream 一致（推理过程不流式推给前端）。

    Args:
        valid_ref_ids: 已验证引用（联网搜索）的编号集合；不在集合内的 [N] 标记
            会在流式输出中即时剥离（None 表示全部剥离）。

    Yields:
        {"type": "answer_delta", "content": "..."}     # 真 token 流式
        {"type": "synth_answer_full", "content": "..."} # 完整 answer（think 已过滤）
    """
    # 将 Phase 1 metadata 注入 user 消息作为上下文
    meta_context = (
        "已确定的分析结论（请基于此生成详细回答，确保与以下结论一致）：\n"
        f"预警等级：{metadata.get('warning_level', '')}\n"
        f"分析推理：{metadata.get('reasoning', '')}\n"
        f"建议措施：{'；'.join(metadata.get('actions', []))}\n"
        "请生成详细、完整的最终回答，涵盖以上结论。"
    )
    messages, _ = _build_synth_messages(
        query,
        tool_results,
        history,
        skill_instructions,
        extra_context=meta_context,
        answer_only=True,
        recalled_context=recalled_context,
        experiences=experiences,  # 与 Phase 1 同一份：保住两阶段前缀对齐
    )

    settings = llm.get_llm_config()
    client = llm.get_llm_client().with_options(timeout=LLM_TIMEOUTS["synthesizer"])

    try:
        stream = client.chat.completions.create(
            model=settings["model"],
            messages=messages,
            temperature=0.3,
            max_tokens=4096,
            stream=True,
            # 末 chunk 返回 usage，用于前缀缓存命中率观测（后端不支持时字段为空，无副作用）
            stream_options={"include_usage": True},
        )
    except (APITimeoutError, RateLimitError, APIConnectionError, APIError) as e:
        logger.exception("[synthesizer] Phase 2 LLM 流式调用失败 (%s)", type(e).__name__)
        raise _classify_llm_error(e) from e
    except Exception as e:
        logger.exception("[synthesizer] Phase 2 LLM 未知异常")
        raise _classify_llm_error(e) from e

    yield from stream_answer_events(stream, "synthesizer_phase2", valid_ref_ids)


def _synth_via_llm_stream(
    query: str,
    tool_results: dict[str, Any],
    history: list[dict[str, Any]] | None = None,
    skill_instructions: str = "",
    recalled_context: str = "",
    experiences: str = "",
):
    """两阶段真流式综合研判生成器。

    Phase 1（非流式）：LLM 返回结构化 metadata（warning_level/reasoning/actions/citations）
      - 使用 _SYNTH_META_SCHEMA（不含 answer 字段），减少 token 消耗
      - 含 Citation Grounding 校验循环
      - 完成后推送 synth_meta 事件，前端可提前渲染预警等级横幅 + 引用卡片

    Phase 2（stream=True）：LLM 逐 token 生成 answer
      - 使用 Phase 1 metadata 作为上下文，确保 answer 与预警等级/措施一致
      - <think> 块自动剥离（推理过程不推给前端）
      - 逐 token 推送 answer_delta 事件，实现真 token 级流式

    yield 事件：
      - {"type": "synth_meta", "data": {warning_level, reasoning, actions, citations}}
      - {"type": "answer_delta", "content": "..."}                    # 真 token 流式
      - {"type": "synth_answer_full", "content": "..."}               # 完整 answer
    """
    # Phase 1：非流式获取 metadata（不含 answer），透传进度事件（生成/校验/修正）
    synth = citations = None
    for ev in _synth_metadata_via_llm_iter(
        query, tool_results, history, skill_instructions,
        recalled_context=recalled_context,
        experiences=experiences,
    ):
        if ev["type"] == "_synth_meta_result":
            synth, citations = ev["result"], ev["citations"]
        else:
            yield ev  # reasoning_step 进度事件直通前端

    # 推送结构化元数据（前端可提前渲染等级横幅 + 引用卡片）
    yield {
        "type": "synth_meta",
        "data": {
            "warning_level": synth.get("warning_level", ""),
            "reasoning": synth.get("reasoning", ""),
            "actions": synth.get("actions", []),
            "citations": citations,
            # 结构化输出是否触发过重试（反思 format_error 触发信号，前端可忽略）
            "format_retry": bool(synth.get("_format_retry", False)),
        },
    }

    # Phase 2：流式生成 answer（真 token 级）；引用标记只保留已验证的联网来源
    valid_ids = {c["ref_id"] for c in citations}
    yield from _stream_answer_via_llm(
        query, tool_results, synth, history, skill_instructions, valid_ids,
        recalled_context=recalled_context,
        experiences=experiences,
    )
