"""LangGraph 图构建与运行入口。

从 workflow.py 拆分而来。依赖 state, nodes, synthesizer_node。

提供：
- build_agent_graph()：构建并编译 LangGraph
- run_graph_agent()：同步入口
- run_graph_agent_stream()：旧流式入口
- run_graph_agent_stream_v2()：新流式入口（v2）
- _route_by_intent / _route_after_executor：条件边
"""
import logging
import re
from typing import Any, Dict, List

from langgraph.graph import END, START, StateGraph

from agent.graph.nodes import (
    MAX_ROUNDS,
    _direct_chat_stream,
    direct_chat_node,
    executor_node,
    planner_node,
    router_node,
)
from agent.graph.state import AgentState
from agent.graph.synthesizer_node import (
    _synth_via_llm_stream,
    synthesizer_node,
)

logger = logging.getLogger(__name__)


# ====== 条件边 ======

def _route_by_intent(state: AgentState) -> str:
    """根据意图路由：闲聊走 direct_chat，业务走 planner。"""
    return "direct_chat" if state.get("intent") == "chitchat" else "planner"


def _route_after_executor(state: AgentState) -> str:
    """P4 改进后：基于 planner 设置的 should_continue 决策下一步。

    替代原 _should_continue（基于 reflector 的 sufficient 字段）。
    """
    return "synthesizer" if not state.get("should_continue", False) else "planner"


# ====== 构建图 ======

def build_agent_graph():
    """构建并编译 LangGraph。

    P4 改进后拓扑（reflector 已合并到 planner）：
        START → router ──(chitchat)──→ direct_chat → END
                  │
                  └──(agent_task)──→ planner → executor
                                          ↓
                                   (should_continue?) ─是→ planner (循环)
                                          ↓ 否
                                    synthesizer → END

    planner 同时输出 tool_calls + should_continue，省掉一次 LLM 调用。
    """
    graph = StateGraph(AgentState)

    graph.add_node("router", router_node)
    graph.add_node("direct_chat", direct_chat_node)
    graph.add_node("planner", planner_node)
    graph.add_node("executor", executor_node)
    graph.add_node("synthesizer", synthesizer_node)

    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        _route_by_intent,
        {"direct_chat": "direct_chat", "planner": "planner"},
    )
    graph.add_edge("direct_chat", END)
    graph.add_edge("planner", "executor")
    # P4：executor 后直接基于 should_continue 路由，不再经过 reflector
    graph.add_conditional_edges(
        "executor",
        _route_after_executor,
        {"synthesizer": "synthesizer", "planner": "planner"},
    )
    graph.add_edge("synthesizer", END)

    return graph.compile()


# ====== 运行入口 ======

def run_graph_agent(user_query: str, history: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """运行 LangGraph Agent（非流式，保留兼容）。

    Returns:
        Dict 包含：
          - final_answer: str
          - warning_level: str（闲聊为空）
          - reasoning: str（闲聊为空）
          - actions: list（闲聊为空）
          - tool_calls: list
          - rounds: int
          - intent: str  chitchat / agent_task
    """
    app = build_agent_graph()
    initial_state: AgentState = {
        "user_query": user_query,
        "history": history or [],
        "rounds": 0,
        "tool_results": {},
        "tool_calls": [],
    }
    final_state = app.invoke(initial_state, config={"recursion_limit": 20})
    intent = final_state.get("intent", "agent_task")
    is_chitchat = intent == "chitchat"
    return {
        "final_answer": final_state.get("final_answer", ""),
        "warning_level": "" if is_chitchat else final_state.get("warning_level", "IV"),
        "reasoning": "" if is_chitchat else final_state.get("reasoning", ""),
        "actions": [] if is_chitchat else final_state.get("actions", []),
        "tool_calls": final_state.get("tool_calls", []),
        "rounds": final_state.get("rounds", 0),
        "intent": intent,
    }


def run_graph_agent_stream(user_query: str, history: List[Dict[str, Any]] = None):
    """运行 LangGraph Agent（流式生成器版本）。

    基于 LangGraph 的 stream() 方法，逐节点 yield 事件。
    事件类型：
      - {"type": "node_start", "node": "router"|"planner"|...}
      - {"type": "intent", "intent": "chitchat"|"agent_task"}
      - {"type": "tool_call", "tool": "get_hydrology", "arguments": {...}, "round": 1}
      - {"type": "tool_result", "tool": "get_hydrology", "result": {...}, "round": 1}
      - {"type": "round_end", "round": 1}
      - {"type": "answer_delta", "content": "..."}  # 最终答案分块推送
      - {"type": "done", "data": {完整响应数据}}
      - {"type": "error", "message": "..."}
    """
    try:
        app = build_agent_graph()
        initial_state: AgentState = {
            "user_query": user_query,
            "history": history or [],
            "rounds": 0,
            "tool_results": {},
            "tool_calls": [],
        }

        final_state: Dict[str, Any] = {}
        # 记录上一轮的 tool_calls 长度，用于增量推送
        prev_tool_calls_len = 0

        # LangGraph stream 模式：逐节点输出状态更新
        for chunk in app.stream(
            initial_state,
            config={"recursion_limit": 20},
            stream_mode="updates",
        ):
            # chunk 是 {node_name: state_update} 字典
            for node_name, state_update in chunk.items():
                if not isinstance(state_update, dict):
                    continue
                final_state.update(state_update)

                yield {"type": "node_start", "node": node_name}

                # router 节点：推送意图识别结果
                if node_name == "router" and "intent" in state_update:
                    yield {"type": "intent", "intent": state_update["intent"]}

                # planner 节点：推送本轮规划的工具调用
                if node_name == "planner" and "planned_calls" in state_update:
                    round_num = state_update.get("rounds", 0)
                    for call in state_update["planned_calls"]:
                        yield {
                            "type": "tool_call",
                            "tool": call["name"],
                            "arguments": call.get("arguments", {}),
                            "round": round_num,
                        }

                # executor 节点：推送工具执行结果（增量）
                if node_name == "executor":
                    new_calls = state_update.get("tool_calls", [])
                    for tc in new_calls[prev_tool_calls_len:]:
                        yield {
                            "type": "tool_result",
                            "tool": tc["tool_name"],
                            "result": tc.get("result", {}),
                            "error": tc.get("error", ""),
                            "round": tc.get("round", 1),
                        }
                    prev_tool_calls_len = len(new_calls)

                # P4：不再有 reflector 节点，planner 的 should_continue 决定循环
                # executor 后通过 _route_after_executor 直接路由到 synthesizer 或 planner

        # 最终结果
        intent = final_state.get("intent", "agent_task")
        is_chitchat = intent == "chitchat"
        final_answer = final_state.get("final_answer", "")

        # 分块推送最终答案（模拟打字机效果，按句子/段落切分）
        if final_answer:
            # 按标点切分，保留标点
            chunks = re.split(r"(?<=[。！？\n])", final_answer)
            for c in chunks:
                if c.strip():
                    yield {"type": "answer_delta", "content": c}

        yield {
            "type": "done",
            "data": {
                "answer": final_answer,
                "warning_level": "" if is_chitchat else final_state.get("warning_level", ""),
                "reasoning": "" if is_chitchat else final_state.get("reasoning", ""),
                "actions": [] if is_chitchat else final_state.get("actions", []),
                "tool_calls": final_state.get("tool_calls", []),
                "rounds": final_state.get("rounds", 0),
                "intent": intent,
            },
        }
    except Exception as e:
        logger.exception("[stream] Agent 流式运行失败")
        yield {"type": "error", "message": str(e)}


def run_graph_agent_stream_v2(user_query: str, history: List[Dict[str, Any]] = None):
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
      3. synthesizer 用 _synth_via_llm_stream（细粒度切分 answer）
      4. direct_chat 用 _direct_chat_stream（LLM stream=True 真 token 流式）

    事件类型：
      - {"type": "reasoning_step", "step": "router|planner|executor|reflector|synthesizer|direct_chat",
         "phase": "start|thinking|decision|done", "message": "...", "details": {...}}
      - {"type": "intent", "intent": "chitchat"|"agent_task"}
      - {"type": "tool_call", "tool": "...", "arguments": {...}, "round": N}
      - {"type": "tool_result", "tool": "...", "result": {...}, "error": "", "round": N}
      - {"type": "synth_meta", "data": {warning_level, reasoning, actions}}  # 结构化结论
      - {"type": "answer_delta", "content": "..."}   # token 级流式
      - {"type": "done", "data": {完整响应}}
      - {"type": "error", "message": "..."}
    """
    try:
        state: AgentState = {
            "user_query": user_query,
            "history": history or [],
            "rounds": 0,
            "tool_results": {},
            "tool_calls": [],
        }

        # ===== 1. 路由节点 =====
        yield {"type": "reasoning_step", "step": "router", "phase": "start",
               "message": "正在识别问题意图...", "details": {}}
        router_update = router_node(state)
        state.update(router_update)
        intent = state.get("intent", "agent_task")
        yield {"type": "intent", "intent": intent}
        yield {"type": "reasoning_step", "step": "router", "phase": "done",
               "message": f"识别为{'闲聊' if intent == 'chitchat' else '业务问题'}",
               "details": {"intent": intent}}

        # ===== 2a. 闲聊分支：流式 LLM 对话 =====
        if intent == "chitchat":
            yield {"type": "reasoning_step", "step": "direct_chat", "phase": "start",
                   "message": "正在生成回复...", "details": {}}
            final_answer = ""
            for ev in _direct_chat_stream(user_query, history or []):
                if ev["type"] == "answer_delta":
                    yield ev
                elif ev["type"] == "synth_answer_full":
                    final_answer = ev["content"]
            yield {"type": "reasoning_step", "step": "direct_chat", "phase": "done",
                   "message": "回复生成完成", "details": {}}
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
            return

        # ===== 2b. 业务分支：planner → executor → reflector 循环 =====
        while True:
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
            if planned:
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

            # ===== P4：planner 的 should_continue 已包含反思判断 =====
            # 不再调用独立 reflector 节点，直接基于 should_continue 决定是否继续循环
            should_continue = state.get("should_continue", False)
            if not should_continue:
                # 信息已充分或达到 MAX_ROUNDS，进入 synthesizer
                reason = "max_rounds" if state.get("rounds", 0) >= MAX_ROUNDS else "info_sufficient"
                msg = (f"已达最大轮次（{MAX_ROUNDS}），强制结束循环" if reason == "max_rounds"
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

        # ===== synthesizer：流式生成最终回答 =====
        yield {"type": "reasoning_step", "step": "synthesizer", "phase": "start",
               "message": "正在综合研判并生成回答...", "details": {}}
        tool_results = state.get("tool_results", {})
        synth_meta = None
        final_answer = ""
        for ev in _synth_via_llm_stream(user_query, tool_results):
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

        # 自进化：在响应完成后异步触发反思循环（不阻塞响应发送）
        try:
            from agent.memory import should_reflect, run_reflection_async
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
                )
        except Exception as e:
            logger.debug("[stream_v2] 反思触发失败（不影响响应）：%s", e)

        yield {
            "type": "done",
            "data": {
                "answer": final_answer,
                "warning_level": (synth_meta or {}).get("warning_level", ""),
                "reasoning": (synth_meta or {}).get("reasoning", ""),
                "actions": (synth_meta or {}).get("actions", []),
                "tool_calls": state.get("tool_calls", []),
                "rounds": state.get("rounds", 0),
                "intent": "agent_task",
            },
        }
    except Exception as e:
        logger.exception("[stream_v2] Agent 流式运行失败")
        yield {"type": "error", "message": str(e)}
