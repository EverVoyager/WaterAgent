"""LangGraph 状态机：防汛预警 Agent 主流程（入口模块）。

本文件仅为入口，实际实现已按职责拆分到同目录的多个内聚模块：

    state.py            —— 状态定义（AgentState / ToolCallRecord）
    errors.py           —— LLM 异常分类（LLMError / _classify_llm_error）
    cache.py            —— 工具结果缓存（_cached_execute_tool 等）
    llm_helpers.py      —— LLM 调用辅助（_call_llm_json / _stream_llm）
    nodes.py            —— 图节点函数（router/planner/executor/direct_chat 等）
    synthesizer_node.py —— 综合研判节点（synthesizer_node + _summarize_results 等）
    runner.py           —— 图构建与运行入口（build_agent_graph / run_graph_agent*）

拓扑（P4 改进后，reflector 已合并到 planner）：
    START → router → planner → executor
                       ↑          ↓
                       └── (should_continue?) ─否→ synthesizer → END
                       │
                  (达到 MAX_ROUNDS) ─是→ synthesizer → END

planner 同时输出 tool_calls + should_continue，省掉独立 reflector 节点。

为保持向后兼容，公共 API 仍可从 `agent.graph.workflow` 直接导入。
"""
from agent.graph.errors import LLMError, _classify_llm_error
from agent.graph.nodes import reflector_node  # noqa: F401  (向后兼容：旧图定义可能引用)
from agent.graph.runner import (
    build_agent_graph,
    run_graph_agent,
    run_graph_agent_stream,
    run_graph_agent_stream_v2,
)
from agent.graph.state import AgentState, ToolCallRecord

__all__ = [
    "build_agent_graph",
    "run_graph_agent",
    "run_graph_agent_stream",
    "run_graph_agent_stream_v2",
    "AgentState",
    "ToolCallRecord",
    "LLMError",
    "_classify_llm_error",
    "reflector_node",
]
