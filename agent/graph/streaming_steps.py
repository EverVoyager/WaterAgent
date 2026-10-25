"""流式状态机阶段：闲聊分支、工具循环与研判事件。"""
import logging
import threading
from typing import Any

from agent.graph import run_context
from agent.graph.direct_chat_stream import _direct_chat_stream
from agent.graph.executor import executor_node
from agent.graph.planner import planner_node
from agent.graph.planner_context import get_max_rounds
from agent.graph.state import AgentState
from agent.graph.synthesis.streaming import _synth_via_llm_stream

logger = logging.getLogger(__name__)


def _stream_chitchat_branch(
    user_query: str,
    history: list[dict[str, Any]],
    skill_instructions: str = "",
    cancel_event: threading.Event | None = None,
    raw_history: list[dict[str, Any]] | None = None,
    recalled_context: str = "",
    experiences: str = "",
):
    """闲聊分支：流式 LLM 对话。

    yields reasoning_step + answer_delta 事件，最终 yield done 事件。
    raw_history：未压缩的原始 history（收尾归档用，缺省退回 history）。
    experiences：planner round-1 检索的历史经验（方案 A 作答端注入）。
    """
    yield {"type": "reasoning_step", "step": "direct_chat", "phase": "start",
           "message": "正在生成回复...", "details": {}}
    final_answer = ""
    for ev in _direct_chat_stream(
        user_query, history or [], skill_instructions,
        recalled_context=recalled_context,
        experiences=experiences,
    ):
        # 客户端已断开：停止消费 LLM 流，提前结束（planner 注入已发生，仍计数收尾）
        if cancel_event is not None and cancel_event.is_set():
            try:
                from agent.memory.experience import finalize_injected_tracking
                finalize_injected_tracking(success=False)
            except Exception:
                pass
            return
        if ev["type"] == "answer_delta":
            yield ev
        elif ev["type"] == "synth_answer_full":
            final_answer = ev["content"]
    yield {"type": "reasoning_step", "step": "direct_chat", "phase": "done",
           "message": "回复生成完成", "details": {}}
    # 收尾归档：本轮异步追加进所属任务段（闲聊无工具轨迹）
    run_context._maybe_archive_round(raw_history or history or [], user_query, final_answer, [])
    # 效果闭环 + 反思：闲聊路径同样收尾（"以后回答简洁点"类偏好多出现在闲聊，
    # 长期记忆写入不依赖 MySQL，不应因走闲聊分支而漏学）
    try:
        from agent.memory.experience import finalize_injected_tracking
        finalize_injected_tracking(success=True)
    except Exception:
        pass
    run_context._maybe_trigger_reflection(
        {"tool_calls": [], "rounds": 1}, user_query, final_answer, format_retry=False,
    )
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
                    experiences=state.get("experiences", ""),
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
        experiences=state.get("experiences", ""),  # 方案 A：作答端消费
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
