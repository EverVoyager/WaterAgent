"""反思调度：触发判断、频率限制和异步任务编排。"""
import json
import logging
import threading
import time
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from uuid import uuid4

from agent.memory.memory_store import is_memory_enabled
from agent.memory.reflection_llm import _generate_reflection
from agent.memory.reflection_writer import (
    _demote_ineffective,
    _dispatch_episode,
    _dispatch_longterm,
    _dispatch_procedure,
    _dispatch_semantic,
    _write_audit,
)
from agent.memory.safety import check_memory_safety as check_memory_safety

logger = logging.getLogger(__name__)

# 反思专用线程池（与工具执行池隔离，避免互相阻塞）
_REFLECT_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="reflect")

# 触发关键词（用户纠正/反馈）
_CORRECTION_KEYWORDS = {"不对", "错了", "应该是", "不是", "不对吧", "搞错了", "修正"}

_FEEDBACK_KEYWORDS = {"以后", "下次", "请记住", "建议", "希望", "偏好", "不要", "不要用"}

# 反思频率上限（滚动窗口）：触发词含"不是/建议/不要"等高频子串，multi_round
# 覆盖几乎所有业务请求——超限后丢弃新任务，防反思成本随流量线性膨胀
_REFLECT_RATE_MAX = 30

_REFLECT_RATE_WINDOW_S = 3600.0

_reflect_times: deque[float] = deque()

_reflect_rate_lock = threading.Lock()


def _reflection_rate_exceeded() -> bool:
    """滑动窗口限流：窗口内提交数超限返回 True（并记录本次时刻）。"""
    now = time.monotonic()
    with _reflect_rate_lock:
        while _reflect_times and now - _reflect_times[0] > _REFLECT_RATE_WINDOW_S:
            _reflect_times.popleft()
        if len(_reflect_times) >= _REFLECT_RATE_MAX:
            return True
        _reflect_times.append(now)
        return False


def should_reflect(
    user_query: str,
    final_answer: str,
    tool_calls: list[dict[str, Any]],
    tool_errors: list[str],
    rounds: int,
    format_retry: bool = False,
) -> str | None:
    """判断是否应触发反思。返回触发原因（None 表示不反思）。"""
    # 1. 用户明确纠正
    if any(kw in user_query for kw in _CORRECTION_KEYWORDS):
        return "user_correction"
    # 2. 用户反馈偏好
    if any(kw in user_query for kw in _FEEDBACK_KEYWORDS):
        return "explicit_feedback"
    # 3. 工具调用失败
    if tool_errors:
        return "tool_failure"
    # 4. 格式错误重试
    if format_retry:
        return "format_error"
    # 5. 多轮才解决（说明问题有难度，值得反思）
    if rounds >= 2 and tool_calls:
        return "multi_round"
    return None


def _reflection_available() -> bool:
    """反思是否可运行由 SELF_EVOLUTION_ENABLED 主开关决定。

    SELF_EVOLUTION_ENABLED=False 时反思循环整体停用（对齐 config 注释语义）；
    该开关此前只被 Curator 检查，反思照跑导致"关闭自进化"不生效。
    """
    try:
        from app.core.config import get_settings
        settings = get_settings()
        return bool(getattr(settings, "SELF_EVOLUTION_ENABLED", True))
    except Exception:
        pass
    return is_memory_enabled()


def run_reflection_async(
    user_query: str,
    final_answer: str,
    tool_calls: list[dict[str, Any]],
    tool_errors: list[str],
    rounds: int,
    trigger_reason: str,
    format_retry: bool = False,
    injected_memories: list[dict[str, Any]] | None = None,
) -> None:
    """异步执行反思循环（fire-and-forget，不阻塞调用方）。

    Args:
        injected_memories: 本次请求注入到 prompt 的记忆 [{"id", "content", "kind"}]，
            供反思评估注入有效性（被注入后仍被纠正 → demote，效果闭环）
    """
    if not _reflection_available():
        logger.debug("[reflection] 记忆模块未启用，跳过反思")
        return
    if _reflection_rate_exceeded():
        logger.info(
            "[reflection] 近 %ds 内反思提交已达 %d 次，跳过（reason=%s）",
            int(_REFLECT_RATE_WINDOW_S), _REFLECT_RATE_MAX, trigger_reason,
        )
        return

    _REFLECT_EXECUTOR.submit(
        _run_reflection_sync,
        user_query=user_query,
        final_answer=final_answer,
        tool_calls=tool_calls,
        tool_errors=tool_errors,
        rounds=rounds,
        trigger_reason=trigger_reason,
        format_retry=format_retry,
        injected_memories=injected_memories,
    )
    logger.info("[reflection] 已提交异步反思任务 reason=%s", trigger_reason)


def _run_reflection_sync(
    user_query: str,
    final_answer: str,
    tool_calls: list[dict[str, Any]],
    tool_errors: list[str],
    rounds: int,
    trigger_reason: str,
    format_retry: bool,
    injected_memories: list[dict[str, Any]] | None = None,
) -> None:
    """反思循环同步实现（在线程池中执行）。"""
    try:
        # 构造反思输入
        tool_summary = _summarize_tool_calls(tool_calls)
        semantic_sources = _build_semantic_sources(user_query, tool_calls, trigger_reason)
        reflection_input = {
            "user_query": user_query,
            "tool_calls": tool_summary,
            "tool_errors": tool_errors,
            "final_answer": final_answer,
            "rounds": rounds,
            "trigger_reason": trigger_reason,
            "format_retry": format_retry,
            "injected_memories": injected_memories or [],
            "semantic_sources": semantic_sources,
        }
        source_ref = f"reflection:{uuid4().hex}"
        context = {
            "kind": "reflection", "version": 1, "source_ref": source_ref,
            "semantic_sources": {
                key: text for key, text in semantic_sources.items()
                if not check_memory_safety(text, text.replace('"', ""))
            },
        }
        event = json.dumps({
            "user_query": user_query[:500], "tool_calls": tool_summary[:1500],
            "tool_errors": [str(e)[:200] for e in tool_errors[:10]],
            "rounds": rounds,
        }, ensure_ascii=False)
        if not check_memory_safety(event, event.replace('"', "")):
            context["event"] = event

        # LLM 生成反思（reflector 超时配置：90s，防线程挂死）
        reflection = _generate_reflection(reflection_input)
        if not reflection:
            logger.warning("[reflection] LLM 未生成有效反思")
            _write_audit(user_query, trigger_reason, tool_summary, final_answer,
                         "未生成有效反思，保留来源供按需复盘", 0, context=context)
            return

        memories_created = 0

        writers = (
            ("longterm", _dispatch_longterm, (reflection, user_query, trigger_reason)),
            ("semantic", _dispatch_semantic, (reflection, semantic_sources, source_ref)),
            ("episode", _dispatch_episode, (reflection, source_ref)),
            ("procedure", _dispatch_procedure, (reflection, tool_calls, source_ref)),
        )
        for kind, writer, args in writers:
            try:
                memories_created += writer(*args)
            except Exception as e:
                context.setdefault("write_errors", []).append({"kind": kind, "error": type(e).__name__})
                logger.warning("[reflection] %s 写入未完成：%s", kind, type(e).__name__)

        # 5. 效果闭环：语义冲突待复核、程序停用（仅限本次真实注入过的 id）
        demoted = _demote_ineffective(
            reflection.get("demote") or {}, injected_memories, review_context=context,
        )

        # 6. 审计日志（agent_reflections，MySQL 可用时）
        _write_audit(
            user_query, trigger_reason, tool_summary, final_answer,
            reflection.get("reflection", ""), memories_created,
            context=context,
        )

        logger.info(
            "[reflection] 反思完成 reason=%s created=%d demoted=%d",
            trigger_reason, memories_created, demoted,
        )
    except Exception as e:
        logger.warning("[reflection] 反思失败：%s\n%s", e, traceback.format_exc())


def _build_semantic_sources(
    user_query: str, tool_calls: list[dict[str, Any]], trigger_reason: str,
) -> dict[str, str]:
    """反思与写入校验共用同一份有界原文，避免把回答中的幻觉当成证据。"""
    sources = {}
    if trigger_reason in ("user_correction", "explicit_feedback"):
        sources["user"] = user_query[:500]
    remaining = 6000
    for i, tc in enumerate(tool_calls):
        name = tc.get("tool_name", "")
        result = tc.get("result")
        if (tc.get("error") or not isinstance(result, dict) or not result
                or result.get("error") or result.get("success") is False
                or result.get("source") == "mock"
                or name in ("read_memory_topic", "read_session_archive", "generate_plan")):
            continue
        text = json.dumps(result, ensure_ascii=False, indent=2, default=str)
        excerpt = text[:min(2000, remaining)]
        sources[f"tool:{i}:{name}"] = excerpt
        remaining -= len(excerpt)
        if remaining <= 0:
            break
    return sources


def _summarize_tool_calls(tool_calls: list[dict[str, Any]]) -> str:
    """摘要工具调用记录为可读字符串。"""
    if not tool_calls:
        return "(无工具调用)"
    parts = []
    for i, tc in enumerate(tool_calls, 1):
        name = tc.get("tool_name", "")
        args = tc.get("arguments", {})
        error = tc.get("error", "")
        if error:
            parts.append(f"  {i}. {name}({args}) ❌ {error[:80]}")
        else:
            result_keys = list((tc.get("result") or {}).keys())[:5]
            parts.append(f"  {i}. {name}({args}) ✓ keys={result_keys}")
    return "\n".join(parts)
