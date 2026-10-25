"""LangGraph 状态机：防汛预警 Agent 主流程（入口模块）。

本文件仅为入口，实际实现已按职责拆分到同目录的多个内聚模块：

    state.py            —— 状态定义（AgentState / ToolCallRecord）
    errors.py           —— LLM 异常分类（LLMError / _classify_llm_error）
    cache.py            —— 工具结果缓存（_cached_execute_tool 等）
    planner.py          —— 工具规划节点（上下文构造在 planner_context.py）
    executor.py         —— 工具并发执行、数据注入与结果校验
    direct_chat_stream.py —— 同步与流式对话（共用消息构造）
    synthesis/          —— 研判上下文、结构化协议、生成、校验与流式输出
    synthesizer_node.py —— 研判结果到图状态的映射
    stream_output.py    —— 流式思考内容与引用过滤
    runner.py           —— 图构建与运行入口（流式阶段在 streaming_steps.py）
    run_context.py      —— 请求上下文和归档、反思收尾

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
