"""会话压缩与归档：冻结摘要、磁盘指针和收尾工具轨迹。"""
import json
import logging
import threading
from datetime import datetime
from typing import Any

from agent.memory.archive_storage import ARCHIVE_READ_MAX_CHARS as ARCHIVE_READ_MAX_CHARS
from agent.memory.archive_storage import (
    _archive_filename,
    _format_round_section,
    _load_meta,
    _read_body,
    _save_segment,
    archive_rounds,
    upsert_round,
)
from agent.memory.archive_storage import cleanup_archive as cleanup_archive
from agent.memory.archive_storage import read_archive_file as read_archive_file
from agent.memory.session_recall import recall_relevant_segments as recall_relevant_segments
from agent.memory.session_segments import Round, Segment, extract_rounds, segment_rounds
from app.core.config import get_settings

logger = logging.getLogger(__name__)


# ============ 冻结摘要 ============

def ensure_summaries(seg: Segment, needed_covered: int, seg_index: int) -> list[str]:
    """确保段有覆盖 needed_covered 轮的冻结摘要，返回摘要文本列表。

    不变量 1（冻结）：已存在的摘要条目永不改写；覆盖不足时生成"续摘要"
    追加新条目。不变量 2（文件固化）：摘要生成一次写盘，后续请求只读。
    LLM 失败 → 规则提取临时文本（不写盘、下次重试）。
    """
    meta = _load_meta(seg.first_fp)
    summaries: list[dict] = (meta or {}).get("summaries", [])
    covered = summaries[-1]["covered"] if summaries else 0

    if covered < needed_covered:
        slice_ = seg.rounds[covered:needed_covered]
        text = _generate_summary_llm(slice_, seg_index, covered)
        if text is None:
            # 降级：规则提取，不冻结（不写盘，跨请求可能变化——仅异常场景）
            return [s["text"] for s in summaries] + [_fallback_summary(slice_)]
        summaries = summaries + [{"covered": needed_covered, "text": text}]
        # 冻结写盘（body 原样保留）
        if meta is not None:
            meta["summaries"] = summaries
            _save_segment(seg.first_fp, meta, _read_body(seg.first_fp))
        else:
            # 段文件尚不存在（归档未跑/失败）——落一份仅含摘要的文件保冻结
            _save_segment(seg.first_fp, {
                "first_fp": seg.first_fp,
                "round_fps": [r.fp for r in seg.rounds[:needed_covered]],
                "queries": [r.query for r in seg.rounds[:needed_covered]],
                "summaries": summaries,
                "created_at": datetime.now().isoformat(timespec="seconds"),
            }, "".join(_format_round_section(r, "") for r in seg.rounds[:needed_covered]))
    return [s["text"] for s in summaries]


def _generate_summary_llm(rounds: list[Round], seg_index: int, covered: int) -> str | None:
    """LLM 生成结构化段摘要。失败返回 None（由调用方降级）。"""
    try:
        from agent.prompts.compact import SEGMENT_SUMMARY_SYSTEM_PROMPT
        from app.core.llm import LLM_TIMEOUTS, extract_content, get_llm_client, get_llm_config

        conversation = "\n".join(
            f"用户：{r.query}\n回答：{r.answer[:400]}" for r in rounds
        )
        settings = get_llm_config()
        client = get_llm_client().with_options(timeout=LLM_TIMEOUTS["chat"])
        resp = client.chat.completions.create(
            model=settings["model"],
            messages=[
                {"role": "system", "content": SEGMENT_SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": f"待总结的任务段（第 {seg_index + 1} 个历史任务）：\n{conversation}"},
            ],
            temperature=0.1,
            max_tokens=300,
        )
        text = extract_content(resp.choices[0].message).strip()
        return text or None
    except Exception as e:
        logger.warning("[session-archive] 段摘要 LLM 生成失败（降级规则提取）：%s", e)
        return None


def _fallback_summary(rounds: list[Round]) -> str:
    """规则提取的临时摘要（不冻结）：query 列表 + 末轮答案首行。"""
    queries = "；".join(r.query[:40] for r in rounds)
    last_answer = rounds[-1].answer[:80] if rounds else ""
    return f"[历史任务] 意图：{queries}｜结论：{last_answer}"


# ============ 压缩（compact 的分段实现） ============

def compact_with_segments(
    history: list[dict[str, Any]],
    keep_recent_rounds: int,
) -> list[dict[str, Any]]:
    """分段压缩：早段 → 冻结摘要消息（每段可能多条：初始+续），近 N 轮原文。

    返回消息序列 [段摘要 system 消息...] + history[-keep_msgs:]。
    替换原 compact_history 的 LLM 合并摘要路径（有损且跨请求不稳定）。

    摘要消息末尾标注存档文件名（Claude Code 式"摘要 + 磁盘指针"）：
    LLM 在上下文中看到早段摘要相关但细节不足时，调用 read_session_archive
    工具读取该段全文（含工具数据），替代逐请求的 query-段向量匹配还原。
    指针文本由 first_fp 推导（确定性），冻结摘要本体不改写——KV Cache
    前缀稳定不变量保持成立。

    早段在产出摘要前同步归档：指针引用的文件必须已存在（读取工具依赖）。
    """
    rounds = extract_rounds(history)
    segments = segment_rounds(rounds)

    keep_msgs = keep_recent_rounds * 2
    window = history[len(history) - keep_msgs:] if keep_msgs > 0 else []
    if not window:
        return history  # 窗口覆盖全部 history，无需压缩
    window_rounds = {r.fp for r in extract_rounds(window)}

    settings = get_settings()
    archive_on = settings.SESSION_ARCHIVE_ENABLED

    # 逐段产出摘要消息；跨窗口段只摘要其窗口外轮次
    early_segments: list[Segment] = []
    msgs: list[dict[str, Any]] = []
    for idx, seg in enumerate(segments):
        outside = [r for r in seg.rounds if r.fp not in window_rounds]
        if not outside:
            continue  # 段整体在窗口内（原文保留）
        early_segments.append(seg)
        summaries = ensure_summaries(seg, needed_covered=len(outside), seg_index=idx)
        pointer = (
            f"\n[存档] 全文与工具数据存于 {_archive_filename(seg.first_fp)}"
            "（细节不足时调用 read_session_archive 读取）"
            if archive_on else ""
        )
        for s in summaries:
            msgs.append({"role": "system", "content": s + pointer})

    # 同步归档早段（幂等）：摘要指针引用的文件必须已存在；已带工具数据
    # 的既有轮次不会被覆盖（archive_rounds 按 meta.round_fps 跳过）。
    # 窗口内段维持异步归档（收尾归档还会补工具数据）
    if early_segments:
        archive_rounds(early_segments)
    rest = [s for s in segments if s not in early_segments]
    if rest:
        _archive_async(rest)
    msgs.extend(window)
    return msgs


def _archive_async(segments: list[Segment]) -> None:
    """后台线程执行入口归档（压缩主流程不等待磁盘 IO）。"""
    t = threading.Thread(target=archive_rounds, args=(segments,), daemon=True)
    t.start()


# ============ 收尾归档（补工具数据） ============

def archive_completed_round_async(
    history: list[dict[str, Any]],
    query: str,
    final_answer: str,
    tool_calls: list[dict[str, Any]] | None,
    warning_level: str = "",
) -> None:
    """请求收尾异步归档：本轮（含工具轨迹）追加进所属段。

    分段对 history+本轮 重算（确定性，无进程状态依赖；embedding 缓存使
    增量成本 ≈ 1 次 embed）。工具数据是前端 history 不含的关键增量。
    """
    settings = get_settings()
    if not settings.SESSION_ARCHIVE_ENABLED:
        return

    try:
        t = threading.Thread(
            target=_archive_completed_round_sync,
            args=(history, query, final_answer, tool_calls, warning_level),
            daemon=True,
        )
        t.start()
    except Exception as e:
        logger.debug("[session-archive] 收尾归档调度失败：%s", e)


def _archive_completed_round_sync(
    history: list[dict[str, Any]],
    query: str,
    final_answer: str,
    tool_calls: list[dict[str, Any]] | None,
    warning_level: str = "",
) -> None:
    """收尾归档同步实现（供异步包装与测试调用）。"""
    try:
        all_rounds = extract_rounds(history or []) + [Round(query=query, answer=final_answer)]
        segments = segment_rounds(all_rounds)
        if not segments:
            return
        seg = segments[-1]  # 本轮所属段
        tool_text = _format_tool_calls(tool_calls or [], warning_level)
        last_fp = seg.rounds[-1].fp
        # 段内全部轮入档（入口归档可能没跑过），当前轮带工具数据
        for r in seg.rounds:
            upsert_round(seg.first_fp, seg.rounds, r,
                         tool_text if r.fp == last_fp else "")
        logger.info("[session-archive] 本轮已归档段 fp=%s（含工具数据）", seg.first_fp)
    except Exception as e:
        logger.warning("[session-archive] 收尾归档失败（不影响响应）：%s", e)


def _format_tool_calls(tool_calls: list[dict[str, Any]], warning_level: str) -> str:
    lines = []
    if warning_level:
        lines.append(f"预警等级：{warning_level}")
    for tc in tool_calls:
        name = tc.get("tool_name", "")
        args = tc.get("arguments", {})
        result = tc.get("result", {})
        err = tc.get("error", "")
        line = f"- {name}({json.dumps(args, ensure_ascii=False)[:200]})"
        if err:
            line += f" → 失败：{err}"
        else:
            line += f" → {json.dumps(result, ensure_ascii=False)[:500]}"
        lines.append(line)
    return "\n".join(lines)
