"""LangGraph 状态定义：AgentState 与 ToolCallRecord。

从 workflow.py 拆分而来，无内部依赖。
"""
from typing import Any, Dict, List
from typing_extensions import TypedDict


class ToolCallRecord(TypedDict, total=False):
    """单次工具调用记录（与 B 阶段兼容）。"""

    tool_name: str
    arguments: Dict[str, Any]
    result: Dict[str, Any]
    error: str
    round: int


class AgentState(TypedDict, total=False):
    """LangGraph 状态。"""

    user_query: str
    history: List[Dict[str, Any]]
    intent: str                              # 意图：chitchat / agent_task
    rounds: int
    planned_calls: List[Dict[str, Any]]      # 本轮计划 [{"name":..., "arguments":...}]
    tool_results: Dict[str, Any]             # 累积的工具结果 {tool_name_idx: result}
    tool_calls: List[ToolCallRecord]         # 完整调用链
    should_continue: bool                    # planner 判断是否需要继续调工具（P4 合并 reflector）
    warning_level: str                       # Ⅰ/Ⅱ/Ⅲ/Ⅳ
    reasoning: str
    actions: List[str]
    final_answer: str
