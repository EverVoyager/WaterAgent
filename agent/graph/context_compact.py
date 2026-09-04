"""上下文 token 压缩（借鉴 Codex compact.rs + token_budget.rs）。

策略（任务段折叠版，替换早期的一次性 LLM 合并摘要，设计依据
docs/context-compression-research.md）：
1. 估算全部 history 的 token 数
2. 未超预算 → 直接返回原 history（不调 LLM，零开销）
3. 超预算 → 交给 session_archive.compact_with_segments：
   早段折叠为冻结的结构化摘要消息（每段一条，含"续摘要"追加），
   近 N 轮原文保留；全文（含工具数据）落 MD 归档，按需匹配还原。

冻结摘要跨请求逐字一致（KV Cache 前缀稳定），摘要 LLM 失败时降级
规则提取（不冻结），embedding 不可用时整体降级单段。
"""
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


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


def is_compacted_history(history: list[dict[str, Any]]) -> bool:
    """判断 history 是否为压缩产物（开头含早段摘要 system 消息）。

    兼容两种摘要标记：旧合并摘要 "[历史对话摘要]" 与任务段摘要
    "[历史任务·N]"（session_archive 产出，可能多条）。
    """
    for m in (history or [])[:8]:
        if m.get("role") == "system":
            c = m.get("content", "") or ""
            if "[历史对话摘要]" in c or "[历史任务" in c:
                return True
    return False


def compact_history(
    history: list[dict[str, Any]],
    max_tokens: int = 4000,
    keep_recent_rounds: int = 2,
) -> list[dict[str, Any]]:
    """压缩历史对话，控制在 token 预算内。

    未超预算返回原 history（零开销）；超预算交给任务段折叠
    （早段冻结摘要 + 近 N 轮原文，见 agent/memory/session_archive.py）。
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
        "[compact] history tokens=%d > budget=%d, compacting by segments (keep %d rounds)",
        total_tokens, max_tokens, keep_recent_rounds,
    )

    from agent.memory.session_archive import compact_with_segments

    return compact_with_segments(history, keep_recent_rounds)


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
