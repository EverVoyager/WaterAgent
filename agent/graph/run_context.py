"""请求生命周期：历史压缩、按需还原、收尾归档与反思。"""
import logging
from typing import Any

from agent.graph.context_compact import compact_history
from agent.graph.state import AgentState
from app.core.config import get_settings

logger = logging.getLogger(__name__)


def _compact_history_entry(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """入口处压缩 history：根据 config 配置调用 compact_history。

    未超 token 预算时零开销返回原 history；超预算时早段折叠为冻结的
    结构化段摘要 + 近 N 轮原文（任务段落盘机制，见 session_archive.py）。
    """
    if not history:
        return history
    settings = get_settings()
    return compact_history(
        history,
        max_tokens=settings.HISTORY_MAX_TOKENS,
        keep_recent_rounds=settings.HISTORY_KEEP_RECENT_ROUNDS,
    )


def _recall_context_entry(user_query: str, raw_history: list[dict[str, Any]]) -> str:
    """入口按需还原（旧机制，SESSION_RECALL_ON_QUERY 控制，默认关闭）。

    主还原路径已改为 Claude Code 式"摘要 + 磁盘指针"：压缩产出的段摘要
    末尾标注存档文件名，planner 需要细节时调用 read_session_archive 工具
    读取——零逐请求 embedding 比较成本，按需付 token。
    置 True 恢复 query-段向量匹配自动注入（对照实验用）。
    返回空串表示无命中或机制关闭。异常全部吞掉，绝不影响主流程。
    """
    try:
        settings = get_settings()
        if not settings.SESSION_RECALL_ON_QUERY:
            return ""
        from agent.memory.session_recall import recall_relevant_segments

        return recall_relevant_segments(
            user_query, raw_history, settings.HISTORY_KEEP_RECENT_ROUNDS,
        )
    except Exception as e:
        logger.debug("[runner] 相关历史段还原失败（不影响主流程）：%s", e)
        return ""


def _maybe_archive_round(
    raw_history: list[dict[str, Any]],
    user_query: str,
    final_answer: str,
    tool_calls: list[dict[str, Any]] | None,
    warning_level: str = "",
) -> None:
    """收尾归档：本轮（含工具轨迹）异步追加进所属任务段。

    工具数据是前端 history 不含的关键增量（跨轮还原依赖它）。
    异步执行，不阻塞响应。
    """
    try:
        from agent.memory.session_archive import archive_completed_round_async

        archive_completed_round_async(
            raw_history, user_query, final_answer, tool_calls, warning_level,
        )
    except Exception as e:
        logger.debug("[runner] 收尾归档失败（不影响主流程）：%s", e)


def _maybe_trigger_reflection(state: AgentState, user_query: str, final_answer: str,
                              format_retry: bool | None = None) -> None:
    """自进化：在响应完成后异步触发反思循环（不阻塞响应发送）。

    任何异常均捕获并降级为 debug 日志，确保不影响主流程。
    同时把本次注入的记忆传给反思，评估注入有效性（效果闭环）。
    format_retry：synthesizer 结构化输出是否触发过重试（format_error 触发
    信号）；None 时从 state 读取（非流式路径经 synthesizer_node 写入 state）。
    """
    try:
        from agent.memory import run_reflection_async, should_reflect
        from agent.memory.experience import get_injected_memories
        tool_calls_state = state.get("tool_calls", [])
        tool_errors = [tc.get("error", "") for tc in tool_calls_state if tc.get("error")]
        if format_retry is None:
            format_retry = bool(state.get("format_retry", False))
        trigger_reason = should_reflect(
            user_query=user_query,
            final_answer=final_answer,
            tool_calls=tool_calls_state,
            tool_errors=tool_errors,
            rounds=state.get("rounds", 0),
            format_retry=format_retry,
        )
        if trigger_reason:
            run_reflection_async(
                user_query=user_query,
                final_answer=final_answer,
                tool_calls=tool_calls_state,
                tool_errors=tool_errors,
                rounds=state.get("rounds", 0),
                trigger_reason=trigger_reason,
                format_retry=format_retry,
                # 效果闭环：本次注入的记忆（planner 经验 + synthesizer 偏好）
                # 在同一 worker 线程内记录，反思时评估是否需要降权
                injected_memories=get_injected_memories(),
            )
    except Exception as e:
        logger.debug("[stream_v2] 反思触发失败（不影响响应）：%s", e)
