"""上下文 token 压缩（借鉴 Codex compact.rs + token_budget.rs）。

策略：
1. 估算全部 history 的 token 数
2. 未超预算 → 直接返回原 history（不调 LLM，零开销）
3. 超预算 → 保留最近 N 轮原文，早轮用 LLM 总结成一条 system 摘要

LLM 摘要带缓存（history 指纹），相同历史不重复调用。
LLM 摘要失败时降级为简单截断（每条保留首 200 字），不阻塞主流程。
"""
import hashlib
import logging
import re
from typing import Any

from agent.prompts.compact import COMPACT_HISTORY_SYSTEM_PROMPT
from app.core.llm import LLM_TIMEOUTS, extract_content, get_llm_client, get_llm_config

logger = logging.getLogger(__name__)

# 摘要缓存：history 指纹 → 摘要文本。避免相同 history 重复调 LLM
_SUMMARY_CACHE: dict[str, str] = {}
_MAX_CACHE_SIZE = 50


def estimate_tokens(text: str) -> int:
    """粗估 token 数。

    中文按 1.5 字/token，英文/数字/符号按 4 字符/token。
    零依赖，误差约 ±20%，用于上下文预算控制足够。
    """
    if not text:
        return 0
    cjk_count = len(re.findall(r"[\u4e00-\u9fff\u3400-\u4dbf]", text))
    other_count = len(text) - cjk_count
    return int(cjk_count / 1.5 + other_count / 4) + 1


def _history_fingerprint(history: list[dict[str, Any]]) -> str:
    """计算 history 指纹（基于每条消息的 role + content 全文）。

    注：早期版本 content 截断到前 300 字，长消息会碰撞出错误摘要缓存，
    改为全文参与指纹计算。
    """
    parts = [f"{m.get('role', '')}:{m.get('content', '')}" for m in history]
    return hashlib.md5("|".join(parts).encode("utf-8")).hexdigest()


def _split_recent(
    history: list[dict[str, Any]],
    keep_rounds: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """把 history 分为 (待摘要部分, 保留原文部分)。

    keep_rounds: 保留最近几轮（1 轮 = 1 问 1 答 = 2 条消息）
    """
    keep_msgs = keep_rounds * 2
    if len(history) <= keep_msgs:
        return [], history
    return history[:-keep_msgs], history[-keep_msgs:]


def _summarize_via_llm(history: list[dict[str, Any]]) -> str:
    """调用 LLM 把历史对话总结成摘要。

    带缓存：相同 history 指纹返回缓存结果。
    LLM 失败时返回空字符串（由调用方降级为截断）。
    """
    fp = _history_fingerprint(history)
    if fp in _SUMMARY_CACHE:
        logger.debug("[compact] 命中摘要缓存 fp=%s", fp[:8])
        return _SUMMARY_CACHE[fp]

    conversation_text = "\n".join(
        f"{m.get('role', 'user')}：{m.get('content', '')[:500]}"
        for m in history
    )

    system_prompt = COMPACT_HISTORY_SYSTEM_PROMPT
    user_prompt = f"待压缩的对话历史：\n{conversation_text}\n\n请输出摘要："

    try:
        settings = get_llm_config()
        client = get_llm_client().with_options(timeout=LLM_TIMEOUTS["chat"])
        resp = client.chat.completions.create(
            model=settings["model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=512,
        )
        summary = extract_content(resp.choices[0].message).strip()
        if not summary:
            logger.warning("[compact] LLM 摘要返回空内容")
            return ""

        # 写入缓存（满了先清空）
        if len(_SUMMARY_CACHE) >= _MAX_CACHE_SIZE:
            _SUMMARY_CACHE.clear()
        _SUMMARY_CACHE[fp] = summary
        logger.info("[compact] LLM 摘要成功 fp=%s 摘要=%s", fp[:8], summary[:80])
        return summary
    except Exception as e:
        logger.warning("[compact] LLM 摘要失败，降级为截断：%s", e)
        return ""


def _truncate_history(
    history: list[dict[str, Any]], max_chars_per_msg: int = 200
) -> str:
    """降级策略：LLM 摘要不可用时，每条消息保留首 N 字拼接成文本。"""
    parts = []
    for m in history:
        role = m.get("role", "user")
        content = m.get("content", "")
        truncated = content[:max_chars_per_msg]
        if len(content) > max_chars_per_msg:
            truncated += "..."
        parts.append(f"{role}：{truncated}")
    return "\n".join(parts)


def compact_history(
    history: list[dict[str, Any]],
    max_tokens: int = 4000,
    keep_recent_rounds: int = 2,
) -> list[dict[str, Any]]:
    """压缩历史对话，控制在 token 预算内。

    策略（借鉴 Codex compact.rs）：
    1. 估算全部 history 的 token
    2. 未超预算 → 直接返回原 history（不调 LLM，零开销）
    3. 超预算 → 保留最近 keep_recent_rounds 轮原文，早轮用 LLM 总结成一条 system 摘要

    Returns:
        压缩后的 history，格式为：
        [{"role":"system","content":"[历史对话摘要]\\n..."}, ...最近几轮原文]
        未超预算时返回原 history（不变）。
    """
    if not history:
        return history

    total_tokens = sum(estimate_tokens(m.get("content", "")) for m in history)
    if total_tokens <= max_tokens:
        logger.debug(
            "[compact] history tokens=%d <= budget=%d, no compaction needed",
            total_tokens, max_tokens,
        )
        return history

    logger.info(
        "[compact] history tokens=%d > budget=%d, compacting (keep %d rounds)",
        total_tokens, max_tokens, keep_recent_rounds,
    )

    to_summarize, recent = _split_recent(history, keep_recent_rounds)
    if not to_summarize:
        # 保留窗口已覆盖全部 history（keep_recent_rounds 太大）
        return history

    summary = _summarize_via_llm(to_summarize)
    if not summary:
        # LLM 摘要失败，降级为简单截断
        summary = _truncate_history(to_summarize)

    return [
        {"role": "system", "content": f"[历史对话摘要]\n{summary}"},
        *recent,
    ]


def extract_history_context(history: list[dict[str, Any]]) -> str:
    """从压缩后的 history 提取可读文本（供 planner/synthesizer 注入 prompt）。

    格式：
        [历史对话摘要]
        ...摘要内容...

        用户：...
        助手：...
    """
    if not history:
        return ""

    parts = []
    for m in history:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            parts.append(content)
        else:
            label = "用户" if role == "user" else "助手"
            parts.append(f"{label}：{content}")

    return "\n\n".join(parts)
