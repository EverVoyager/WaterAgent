"""LangGraph 构建、同步运行和流式运行入口。"""
import logging
import threading
from collections.abc import Iterator
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent.graph import run_context
from agent.graph.direct_chat_stream import direct_chat_node
from agent.graph.errors import LLMError
from agent.graph.executor import executor_node
from agent.graph.planner import planner_node
from agent.graph.state import AgentState
from agent.graph.streaming_steps import _stream_planner_executor_loop, _stream_synthesizer_phase
from agent.graph.synthesizer_node import (
    synthesizer_node,
)

logger = logging.getLogger(__name__)


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
    # 效果闭环：清空上一请求的注入追踪（thread-local，防止跨请求残留）
    try:
        from agent.memory.experience import clear_injected_tracking
        clear_injected_tracking()
    except Exception:
        pass
    compacted_history = run_context._compact_history_entry(history or [])
    initial_state: AgentState = {
        "user_query": user_query,
        "history": compacted_history,
        "recalled_context": run_context._recall_context_entry(user_query, history or []),
        "rounds": 0,
        "tool_results": {},
        "tool_calls": [],
    }
    final_state = app.invoke(initial_state, config={"recursion_limit": 20})
    # intent 由 planner 间接决定：无工具调用走 direct_chat → chitchat；否则 agent_task
    is_chitchat = not final_state.get("tool_calls") and final_state.get("rounds", 0) <= 1
    intent = "chitchat" if is_chitchat else "agent_task"
    # 效果闭环：注入记忆计数（与流式入口对齐；漏计会导致 hit_count 被低估，
    # Curator 的"零命中超期"剪枝会误删实际高频使用的记忆）
    try:
        from agent.memory.experience import finalize_injected_tracking
        finalize_injected_tracking(success=True)
    except Exception:
        pass
    # 自进化：异步触发反思（format_retry 经 synthesizer_node 写入 final_state）
    run_context._maybe_trigger_reflection(final_state, user_query, final_state.get("final_answer", ""))
    # 收尾归档：本轮（含工具轨迹）异步追加进所属任务段
    run_context._maybe_archive_round(
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
        compacted_history = run_context._compact_history_entry(history or [])
        state: AgentState = {
            "user_query": user_query,
            "history": compacted_history,
            "recalled_context": run_context._recall_context_entry(user_query, history or []),
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
            # planner 第 1 轮已注入过经验，仍需计数收尾（按未成功交付计）
            try:
                from agent.memory.experience import finalize_injected_tracking
                finalize_injected_tracking(success=False)
            except Exception:
                pass
            return

        # synthesizer：流式生成最终回答
        synth_meta, final_answer = yield from _stream_synthesizer_phase(
            state, user_query, cancel_event,
        )

        # 客户端断开导致的提前返回：不再产出 done 事件，
        # 但回答已完整生成，计数/反思/归档等收尾仍执行
        client_gone = cancel_event is not None and cancel_event.is_set()

        # 效果闭环：注入记忆计数（成功路径）
        try:
            from agent.memory.experience import finalize_injected_tracking
            finalize_injected_tracking(success=True)
        except Exception:
            pass

        # 自进化：异步触发反思循环（不阻塞响应发送）
        run_context._maybe_trigger_reflection(
            state, user_query, final_answer,
            format_retry=bool((synth_meta or {}).get("format_retry", False)),
        )

        # 收尾归档：本轮（含工具轨迹）异步追加进所属任务段
        run_context._maybe_archive_round(
            history or [], user_query, final_answer,
            state.get("tool_calls", []),
            (synth_meta or {}).get("warning_level", ""),
        )

        if client_gone:
            return

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
