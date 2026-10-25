"""综合研判图节点：把研判结果映射为 AgentState 更新。"""
import logging
from typing import Any

from agent.graph.state import AgentState
from agent.graph.synthesis.generation import _synth_via_llm

logger = logging.getLogger(__name__)


def synthesizer_node(state: AgentState) -> dict[str, Any]:
    """综合研判节点：LLM 综合所有工具结果生成最终回答。

    包含 Citation Grounding：生成后校验引用原文是否真实存在，失败则重生成。
    Skill 机制：从 state 读取匹配到的 Skill 指令注入 synthesizer。
    LLM 调用失败时直接抛错，由上层 API 返回 500 给前端。
    """
    tool_results = state.get("tool_results", {})
    query = state["user_query"]
    history = state.get("history", [])
    skill_instructions = state.get("skill_instructions", "")

    synth, citations = _synth_via_llm(
        query, tool_results, history, skill_instructions,
        recalled_context=state.get("recalled_context", ""),
        experiences=state.get("experiences", ""),  # 方案 A：作答端消费 planner 检索的经验
    )

    logger.info("[synthesizer] LLM synth level=%s citations=%d",
                synth.get("warning_level", ""), len(citations))
    return {
        "warning_level": synth.get("warning_level", ""),
        "reasoning": synth.get("reasoning", ""),
        "actions": synth.get("actions", []),
        "final_answer": synth.get("answer", ""),
        "citations": citations,
        # 结构化输出是否触发过重试（反思 format_error 触发信号，经 state 透传）
        "format_retry": bool(synth.get("_format_retry", False)),
    }
