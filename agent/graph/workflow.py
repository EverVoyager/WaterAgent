"""LangGraph 状态机：防汛预警 Agent 主流程（入口模块）。

本文件仅为入口，实际实现已按职责拆分到同目录的多个内聚模块：

    state.py            —— 状态定义（AgentState / ToolCallRecord）
    errors.py           —— LLM 异常分类（LLMError / _classify_llm_error）
    cache.py            —— 工具结果缓存（_cached_execute_tool 等）
    llm_helpers.py      —— LLM 调用辅助（_call_llm_json / _stream_llm）
    nodes.py            —— 图节点函数（planner/executor/direct_chat 等）
    synthesizer_node.py —— 综合研判节点（synthesizer_node + _summarize_results 等）
    runner.py           —— 图构建与运行入口（build_agent_graph / run_graph_agent*）

拓扑（无独立路由层，由 planner 的 LLM 原生 Function Calling 统一决策）：
    START → planner ──(第 1 轮无工具调用)──→ direct_chat → END
              │
              └─(有工具调用)→ executor ─(should_continue?)─┬─ 是 → planner（循环，≤ LLM_MAX_TOOL_ROUNDS）
                                                          └─ 否 → synthesizer → END

为保持向后兼容，公共 API 仍可从 `agent.graph.workflow` 直接导入。
"""
from agent.graph.errors import LLMError, _classify_llm_error
from agent.graph.runner import (
    build_agent_graph,
    run_graph_agent,
    run_graph_agent_stream_v2,
)
from agent.graph.state import AgentState, ToolCallRecord

__all__ = [
    "build_agent_graph",
    "run_graph_agent",
    "run_graph_agent_stream_v2",
    "AgentState",
    "ToolCallRecord",
    "LLMError",
    "_classify_llm_error",
]
