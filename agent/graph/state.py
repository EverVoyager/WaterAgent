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
    tc_id: str  # 原生 FC 序列的 tool_call id（planner 消息配对用）


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
    # 后续轮次 user 消息保留这些段落，前缀缓存才能跨轮延伸）
    experiences: str                         # 历史经验（成功工具模式 + 失败教训）
    history_context: str                     # 历史对话摘要（压缩后的早轮 + 近轮原文）
    # 按需还原的历史任务段全文（入口按 query-段匹配加载，请求内不变；
    # 注入 user 消息末尾，KV Cache 前缀友好）
    recalled_context: str                    # 命中段全文（含工具数据），无命中为空
    # planner 原生 FC 消息序列（user → assistant(tool_calls) → tool(结果) → …），
    # 请求内只追加不重写；assistant 消息携带 reasoning_content 原样回传
    fc_messages: list[dict[str, Any]]
