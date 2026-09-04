"""LangGraph 图构建与运行入口。

从 workflow.py 拆分而来。依赖 state, nodes, synthesizer_node。

提供：
- build_agent_graph()：构建并编译 LangGraph
- run_graph_agent()：同步入口
- run_graph_agent_stream_v2()：流式入口（手动驱动状态机，支持推理过程可视化）
- _route_after_planner / _route_after_executor：条件边
"""
import logging
import threading
from collections.abc import Iterator
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent.graph.context_compact import compact_history
from agent.graph.direct_chat_stream import _direct_chat_stream
from agent.graph.errors import LLMError
from agent.graph.nodes import (
    direct_chat_node,
    executor_node,
    get_max_rounds,
    planner_node,
)
from agent.graph.state import AgentState
from agent.graph.synthesizer_node import (
    _synth_via_llm_stream,
    synthesizer_node,
)
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
    """入口按需还原：query 与压缩窗口外的早段匹配，命中段全文注入。

    返回空串表示无命中（embedding 不可用/无相关段）。异常全部吞掉，
    绝不影响主流程。
    """
    try:
        from agent.memory.session_archive import recall_relevant_segments

        settings = get_settings()
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


# ====== 条件边 ======

def _route_after_planner(state: AgentState) -> str:
    """planner 后路由：第 1 轮无工具调用 → direct_chat（闲聊），否则 → executor。

    借鉴 OpenAI / Cohere 主流方案：移除独立路由层，由 LLM 原生 Function
    Calling 统一决策。模型不调工具即视为闲聊，调工具即视为业务。
    第 2+ 轮无工具调用则由 _route_after_executor 路由到 synthesizer。
    """
    rounds = state.get("rounds", 0)
    planned = state.get("planned_calls", [])
    if rounds == 1 and not planned:
        return "direct_chat"
    return "executor"


def _route_after_executor(state: AgentState) -> str:
    """基于 planner 设置的 should_continue 决策下一步。"""
    return "synthesizer" if not state.get("should_continue", False) else "planner"


# ====== 构建图 ======

def build_agent_graph() -> CompiledStateGraph:
    """构建并编译 LangGraph。

    拓扑（移除独立路由层，由 planner 统一决策）：
        START → planner ──(round=1 且无工具)──→ direct_chat → END
                  │
                  └──(有工具)──→ executor
                                    ↓
                             (should_continue?) ─是→ planner (循环)
                                    ↓ 否
                              synthesizer → END

    借鉴 OpenAI Agents SDK / Cohere 主流方案：LLM 原生 Function Calling
    统一决策，模型不调工具即视为闲聊，调工具即视为业务。省掉独立路由层
    （原 semantic_router embedding 二分类与 planner 重复决策）。
    """
    graph = StateGraph(AgentState)

    graph.add_node("direct_chat", direct_chat_node)
    graph.add_node("planner", planner_node)
    graph.add_node("executor", executor_node)
    graph.add_node("synthesizer", synthesizer_node)

    graph.add_edge(START, "planner")
    # planner 后：第 1 轮无工具 → direct_chat（闲聊），否则 → executor
    graph.add_conditional_edges(
        "planner",
        _route_after_planner,
        {"direct_chat": "direct_chat", "executor": "executor"},
    )
    graph.add_edge("direct_chat", END)
    # executor 后基于 should_continue 路由
    graph.add_conditional_edges(
        "executor",
        _route_after_executor,
        {"synthesizer": "synthesizer", "planner": "planner"},
    )
    graph.add_edge("synthesizer", END)

    return graph.compile()


# ====== 运行入口 ======

def run_graph_agent(user_query: str, history: list[dict[str, Any]] = None) -> dict[str, Any]:
    """运行 LangGraph Agent（非流式，保留兼容）。

    Returns:
        Dict 包含：
          - final_answer: str
          - warning_level: str（闲聊为空）
          - reasoning: str（闲聊为空）
          - actions: list（闲聊为空）
          - tool_calls: list
          - rounds: int
          - intent: str  chitchat / agent_task（由 planner 决策，非独立路由）
    """
    app = build_agent_graph()
    compacted_history = _compact_history_entry(history or [])
    initial_state: AgentState = {
        "user_query": user_query,
        "history": compacted_history,
        "recalled_context": _recall_context_entry(user_query, history or []),
        "rounds": 0,
        "tool_results": {},
        "tool_calls": [],
    }
    final_state = app.invoke(initial_state, config={"recursion_limit": 20})
    # intent 由 planner 间接决定：无工具调用走 direct_chat → chitchat；否则 agent_task
    is_chitchat = not final_state.get("tool_calls") and final_state.get("rounds", 0) <= 1
    intent = "chitchat" if is_chitchat else "agent_task"
    # 收尾归档：本轮（含工具轨迹）异步追加进所属任务段
    _maybe_archive_round(
        history or [], user_query,
        final_state.get("final_answer", ""),
        final_state.get("tool_calls", []),
        "" if is_chitchat else final_state.get("warning_level", ""),
    )
    return {
        "final_answer": final_state.get("final_answer", ""),
        "warning_level": "" if is_chitchat else final_state.get("warning_level", ""),
        "reasoning": "" if is_chitchat else final_state.get("reasoning", ""),
        "actions": [] if is_chitchat else final_state.get("actions", []),
        "tool_calls": final_state.get("tool_calls", []),
        "citations": [] if is_chitchat else final_state.get("citations", []),
        "rounds": final_state.get("rounds", 0),
        "intent": intent,
    }


def _stream_chitchat_branch(
    user_query: str,
    history: list[dict[str, Any]],
    skill_instructions: str = "",
    cancel_event: threading.Event | None = None,
    raw_history: list[dict[str, Any]] | None = None,
    recalled_context: str = "",
):
    """闲聊分支：流式 LLM 对话。

    yields reasoning_step + answer_delta 事件，最终 yield done 事件。
    raw_history：未压缩的原始 history（收尾归档用，缺省退回 history）。
    """
    yield {"type": "reasoning_step", "step": "direct_chat", "phase": "start",
           "message": "正在生成回复...", "details": {}}
    final_answer = ""
    for ev in _direct_chat_stream(
        user_query, history or [], skill_instructions,
        recalled_context=recalled_context,
    ):
        # 客户端已断开：停止消费 LLM 流，提前结束
        if cancel_event is not None and cancel_event.is_set():
            return
        if ev["type"] == "answer_delta":
            yield ev
        elif ev["type"] == "synth_answer_full":
            final_answer = ev["content"]
    yield {"type": "reasoning_step", "step": "direct_chat", "phase": "done",
           "message": "回复生成完成", "details": {}}
    # 收尾归档：本轮异步追加进所属任务段（闲聊无工具轨迹）
    _maybe_archive_round(raw_history or history or [], user_query, final_answer, [])
    yield {
        "type": "done",
        "data": {
            "answer": final_answer,
            "warning_level": "",
            "reasoning": "",
            "actions": [],
            "tool_calls": [],
            "rounds": 0,
            "intent": "chitchat",
        },
    }


def _stream_planner_executor_loop(
    state: AgentState,
    user_query: str,
    history: list[dict[str, Any]],
    cancel_event: threading.Event | None = None,
    raw_history: list[dict[str, Any]] | None = None,
):
    """planner → executor 循环，直到 should_continue=False。

    第 1 轮 planner 返回空工具调用时，转入闲聊分支（借鉴 OpenAI / Cohere
    主流方案：模型不调工具即视为闲聊）。

    yields reasoning_step / tool_call / tool_result 事件。
    返回 True 表示已走闲聊分支（调用方应终止），False 表示正常完成进 synthesizer。
    """
    while True:
        # 客户端已断开：在轮次边界提前终止，不再发起后续 LLM/工具调用
        if cancel_event is not None and cancel_event.is_set():
            return True
        round_num = state.get("rounds", 0) + 1
        # ===== planner =====
        yield {"type": "reasoning_step", "step": "planner", "phase": "start",
               "message": f"第 {round_num} 轮规划：正在决策调用哪些工具...",
               "details": {"round": round_num}}
        planner_update = planner_node(state)
        state.update(planner_update)
        planned = state.get("planned_calls", [])

        if not planned:
            yield {"type": "reasoning_step", "step": "planner", "phase": "decision",
                   "message": "无需调用工具，信息已充分",
                   "details": {"round": round_num, "tools": []}}
            # 第 1 轮无工具 → 闲聊分支；第 2+ 轮无工具 → 进 synthesizer
            if round_num == 1:
                yield from _stream_chitchat_branch(
                    user_query, history, state.get("skill_instructions", ""),
                    cancel_event, raw_history=raw_history,
                    recalled_context=state.get("recalled_context", ""),
                )
                return True
            # 后续轮次无工具，信息已充分，结束循环进 synthesizer
            break
        else:
            tool_names = [c.get("name", "") for c in planned]
            yield {"type": "reasoning_step", "step": "planner", "phase": "decision",
                   "message": f"决定调用工具：{', '.join(tool_names)}",
                   "details": {"round": round_num, "tools": tool_names}}
            # 推送 tool_call 事件
            for call in planned:
                yield {
                    "type": "tool_call",
                    "tool": call["name"],
                    "arguments": call.get("arguments", {}),
                    "round": round_num,
                }

        # ===== executor =====
        yield {"type": "reasoning_step", "step": "executor", "phase": "start",
               "message": f"执行 {len(planned)} 个工具...",
               "details": {"round": round_num}}
        prev_len = len(state.get("tool_calls", []))
        executor_update = executor_node(state)
        state.update(executor_update)
        # 推送 tool_result 事件（增量）
        new_calls = state.get("tool_calls", [])
        for tc in new_calls[prev_len:]:
            yield {
                "type": "tool_result",
                "tool": tc["tool_name"],
                "result": tc.get("result", {}),
                "error": tc.get("error", ""),
                "round": tc.get("round", round_num),
            }
        yield {"type": "reasoning_step", "step": "executor", "phase": "done",
               "message": f"工具执行完成（{len(new_calls) - prev_len} 个）",
               "details": {"round": round_num}}

        # planner 的 should_continue 决定是否继续循环
        should_continue = state.get("should_continue", False)
        if not should_continue:
            max_rounds = get_max_rounds()
            reason = "max_rounds" if state.get("rounds", 0) >= max_rounds else "info_sufficient"
            msg = (f"已达最大轮次（{max_rounds}），强制结束循环" if reason == "max_rounds"
                   else "信息已充分，开始综合研判")
            yield {"type": "reasoning_step", "step": "planner", "phase": "done",
                   "message": msg,
                   "details": {"should_continue": False, "reason": reason,
                               "round": state.get("rounds", 0)}}
            break
        else:
            yield {"type": "reasoning_step", "step": "planner", "phase": "done",
                   "message": "信息不足，继续调用工具",
                   "details": {"should_continue": True, "round": state.get("rounds", 0)}}

    return False


def _stream_synthesizer_phase(
    state: AgentState,
    user_query: str,
    cancel_event: threading.Event | None = None,
):
    """synthesizer 阶段：流式生成最终回答。

    yields reasoning_step / synth_meta / answer_delta 事件。
    返回 (synth_meta, final_answer)。
    """
    yield {"type": "reasoning_step", "step": "synthesizer", "phase": "start",
           "message": "正在综合研判并生成回答...", "details": {}}
    tool_results = state.get("tool_results", {})
    history = state.get("history", [])
    skill_instructions = state.get("skill_instructions", "")
    synth_meta = None
    final_answer = ""
    for ev in _synth_via_llm_stream(
        user_query, tool_results, history, skill_instructions,
        recalled_context=state.get("recalled_context", ""),
    ):
        # 客户端已断开：停止消费 LLM token 流，提前结束
        if cancel_event is not None and cancel_event.is_set():
            return None, ""
        if ev["type"] == "synth_meta":
            synth_meta = ev["data"]
            yield ev
        elif ev["type"] == "answer_delta":
            yield ev
        elif ev["type"] == "synth_answer_full":
            final_answer = ev["content"]

    yield {"type": "reasoning_step", "step": "synthesizer", "phase": "done",
           "message": "综合研判完成",
           "details": {"warning_level": (synth_meta or {}).get("warning_level", "")}}
    return synth_meta, final_answer


def _maybe_trigger_reflection(state: AgentState, user_query: str, final_answer: str) -> None:
    """自进化：在响应完成后异步触发反思循环（不阻塞响应发送）。

    任何异常均捕获并降级为 debug 日志，确保不影响主流程。
    同时把本次注入的记忆传给反思，评估注入有效性（效果闭环）。
    """
    try:
        from agent.memory import run_reflection_async, should_reflect
        from agent.memory.experience import get_injected_memories
        tool_calls_state = state.get("tool_calls", [])
        tool_errors = [tc.get("error", "") for tc in tool_calls_state if tc.get("error")]
        trigger_reason = should_reflect(
            user_query=user_query,
            final_answer=final_answer,
            tool_calls=tool_calls_state,
            tool_errors=tool_errors,
            rounds=state.get("rounds", 0),
        )
        if trigger_reason:
            run_reflection_async(
                user_query=user_query,
                final_answer=final_answer,
                tool_calls=tool_calls_state,
                tool_errors=tool_errors,
                rounds=state.get("rounds", 0),
                trigger_reason=trigger_reason,
                # 效果闭环：本次注入的记忆（planner 经验 + synthesizer 偏好）
                # 在同一 worker 线程内记录，反思时评估是否需要降权
                injected_memories=get_injected_memories(),
            )
    except Exception as e:
        logger.debug("[stream_v2] 反思触发失败（不影响响应）：%s", e)


def run_graph_agent_stream_v2(
    user_query: str,
    history: list[dict[str, Any]] = None,
    cancel_event: threading.Event | None = None,
) -> Iterator[dict[str, Any]]:
    """流式 Agent（v2）：手动驱动状态机，支持推理过程可视化 + 真流式输出。

    借鉴 agent-service-toolkit（LangGraph 官方推荐模板，4.4k star）的做法：
      stream_mode=["updates", "messages", "custom"], subgraphs=True
      - updates：节点级状态变化（工具调用、路由决策）
      - messages：token 级流式（LLM 逐 token 输出）
      - custom：自定义事件（推理步骤）

    本项目用 OpenAI SDK 直连 DashScope（非 LangChain ChatModel），
    所以不使用 LangGraph 的 stream(messages) 模式，而是手动驱动状态机：
      1. 手动调用各节点函数（保留 LangGraph 图定义用于非流式兼容）
      2. 在节点前后推送 reasoning_step 事件（推理过程可视化）
      3. synthesizer 用 _synth_via_llm_stream（两阶段真流式：Phase1 metadata 非流式 + Phase2 answer stream=True）
      4. direct_chat 用 _direct_chat_stream（LLM stream=True 真 token 流式）

    路由策略（借鉴 OpenAI / Cohere 主流方案）：
      移除独立路由层，由 planner LLM 原生 Function Calling 统一决策。
      第 1 轮 planner 返回空工具调用 → 闲聊分支；否则 → 业务分支。

    事件类型：
      - {"type": "reasoning_step", "step": "planner|executor|synthesizer|direct_chat",
         "phase": "start|thinking|decision|done", "message": "...", "details": {...}}
      - {"type": "tool_call", "tool": "...", "arguments": {...}, "round": N}
      - {"type": "tool_result", "tool": "...", "result": {...}, "error": "", "round": N}
      - {"type": "synth_meta", "data": {warning_level, reasoning, actions}}  # 结构化结论
      - {"type": "answer_delta", "content": "..."}   # token 级流式
      - {"type": "done", "data": {完整响应}}
      - {"type": "error", "message": "..."}
    """
    try:
        # 效果闭环：清空上一请求的注入追踪（thread-local，防止跨请求残留）
        try:
            from agent.memory.experience import clear_injected_tracking
            clear_injected_tracking()
        except Exception:
            pass
        compacted_history = _compact_history_entry(history or [])
        state: AgentState = {
            "user_query": user_query,
            "history": compacted_history,
            "recalled_context": _recall_context_entry(user_query, history or []),
            "rounds": 0,
            "tool_results": {},
            "tool_calls": [],
        }

        # planner → executor 循环（第 1 轮空工具会自动转入闲聊分支）
        went_chitchat = yield from _stream_planner_executor_loop(
            state, user_query, compacted_history, cancel_event,
            raw_history=history or [],
        )
        if went_chitchat:
            return

        # 客户端已断开：跳过 synthesizer 阶段（不再消耗 LLM token）
        if cancel_event is not None and cancel_event.is_set():
            return

        # synthesizer：流式生成最终回答
        synth_meta, final_answer = yield from _stream_synthesizer_phase(
            state, user_query, cancel_event,
        )

        # 客户端断开导致的提前返回：不再产出 done 事件
        if cancel_event is not None and cancel_event.is_set():
            return

        # 效果闭环：注入记忆计数（成功路径）
        try:
            from agent.memory.experience import finalize_injected_tracking
            finalize_injected_tracking(success=True)
        except Exception:
            pass

        # 自进化：异步触发反思循环（不阻塞响应发送）
        _maybe_trigger_reflection(state, user_query, final_answer)

        # 收尾归档：本轮（含工具轨迹）异步追加进所属任务段
        _maybe_archive_round(
            history or [], user_query, final_answer,
            state.get("tool_calls", []),
            (synth_meta or {}).get("warning_level", ""),
        )

        yield {
            "type": "done",
            "data": {
                "answer": final_answer,
                "warning_level": (synth_meta or {}).get("warning_level", ""),
                "reasoning": (synth_meta or {}).get("reasoning", ""),
                "actions": (synth_meta or {}).get("actions", []),
                "citations": (synth_meta or {}).get("citations", []),
                "tool_calls": state.get("tool_calls", []),
                "rounds": state.get("rounds", 0),
                "intent": "agent_task",
            },
        }
    except LLMError as e:
        # 效果闭环：注入记忆计数（失败路径）
        try:
            from agent.memory.experience import finalize_injected_tracking
            finalize_injected_tracking(success=False)
        except Exception:
            pass
        # LLM 分类异常：保留 kind/status_code 传给前端（与非流式接口行为一致）
        logger.warning("[stream_v2] LLM error: %s (kind=%s)", e, e.kind)
        yield {"type": "error", "message": str(e), "kind": e.kind,
               "status_code": e.status_code}
    except Exception as e:
        try:
            from agent.memory.experience import finalize_injected_tracking
            finalize_injected_tracking(success=False)
        except Exception:
            pass
        logger.exception("[stream_v2] Agent 流式运行失败")
        yield {"type": "error", "message": str(e)}
