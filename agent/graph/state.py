"""LangGraph 状态定义：AgentState 与 ToolCallRecord。

从 workflow.py 拆分而来，无内部依赖。
"""
from typing import Any

from typing_extensions import TypedDict


class ToolCallRecord(TypedDict, total=False):
    """单次工具调用记录（与 B 阶段兼容）。"""

    tool_name: str
    arguments: dict[str, Any]
    result: dict[str, Any]
    error: str
    round: int


class AgentState(TypedDict, total=False):
    """LangGraph 状态。"""

    user_query: str
    history: list[dict[str, Any]]
    intent: str                              # 意图：chitchat / agent_task
    rounds: int
    planned_calls: list[dict[str, Any]]      # 本轮计划 [{"name":..., "arguments":...}]
    tool_results: dict[str, Any]             # 累积的工具结果 {tool_name_idx: result}
    tool_calls: list[ToolCallRecord]         # 完整调用链
    should_continue: bool                    # planner 判断是否需要继续调工具（P4 合并 reflector）
    warning_level: str                       # Ⅰ/Ⅱ/Ⅲ/Ⅳ
    reasoning: str
    actions: list[str]
    final_answer: str
    citations: list[dict[str, Any]]          # Citation Grounding 引用列表（已校验）
    # Skill 机制（借鉴 Claude Skills）：匹配到的技能指令 + 工具子集
    skill_name: str                          # 匹配到的 Skill 名（未匹配为空）
    skill_instructions: str                  # 匹配到的 Skill 行为指令
    skill_tool_names: list[str]              # Skill 限制的工具子集（空 = 不限制）
    # 第 1 轮注入的上下文段落，跨轮原样保留（KV Cache 前缀"只增不改"：
    # 后续轮次 user 消息保留这些段落，前缀缓存才能延伸）
    experiences: str                         # 历史经验（成功工具模式 + 失败教训）
    history_context: str                     # 历史对话摘要（压缩后的早轮 + 近轮原文）
