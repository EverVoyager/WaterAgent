"""Agent 运行时：多轮工具调用循环。"""
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

from openai import OpenAI

from agent.tools.mock_executor import execute_tool
from agent.tools.schemas import build_openai_tools
from app.core.llm import get_default_system_prompt, get_llm_client, get_llm_config

logger = logging.getLogger(__name__)


@dataclass
class ToolCallRecord:
    """单次工具调用记录。"""

    tool_name: str
    arguments: Dict[str, Any]
    result: Dict[str, Any]
    error: str = ""


@dataclass
class AgentRunResult:
    """Agent 单次运行结果。"""

    answer: str                       # LLM 最终自然语言回答
    tool_calls: List[ToolCallRecord]  # 工具调用链路
    rounds: int                       # 实际执行轮次
    raw_messages: List[Dict[str, Any]] = field(default_factory=list)


def run_agent(
    user_query: str,
    system_prompt: str = "",
    history: List[Dict[str, Any]] = None,
) -> AgentRunResult:
    """运行 Agent 工具调用循环。

    Args:
        user_query: 用户问题
        system_prompt: 自定义系统提示词，为空则用默认
        history: 历史对话消息，格式同 OpenAI

    Returns:
        AgentRunResult: 包含最终回答与工具调用链路
    """
    settings_llm = get_llm_config()
    client: OpenAI = get_llm_client()
    tools = build_openai_tools()

    sys_msg = system_prompt or get_default_system_prompt()
    messages: List[Dict[str, Any]] = [{"role": "system", "content": sys_msg}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_query})

    tool_call_records: List[ToolCallRecord] = []
    max_rounds = settings_llm["max_tool_rounds"]
    rounds = 0

    while rounds < max_rounds:
        rounds += 1
        logger.info("Agent round %d/%d, messages=%d", rounds, max_rounds, len(messages))

        resp = client.chat.completions.create(
            model=settings_llm["model"],
            messages=messages,
            tools=tools,
            temperature=settings_llm["temperature"],
            max_tokens=settings_llm["max_tokens"],
        )
        msg = resp.choices[0].message

        # 将 assistant 消息（可能含 tool_calls）加入上下文
        assistant_msg: Dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        messages.append(assistant_msg)

        # 无工具调用 → 终止循环
        if not msg.tool_calls:
            logger.info("Agent finished without tool calls at round %d", rounds)
            return AgentRunResult(
                answer=msg.content or "",
                tool_calls=tool_call_records,
                rounds=rounds,
                raw_messages=messages,
            )

        # 执行每个工具调用，结果作为 tool 消息回填
        for tc in msg.tool_calls:
            tool_name = tc.function.name
            try:
                arguments = json.loads(tc.function.arguments) if tc.function.arguments else {}
            except json.JSONDecodeError as e:
                logger.warning("Invalid tool arguments JSON: %s", e)
                arguments = {}

            logger.info("Tool call: %s args=%s", tool_name, arguments)
            try:
                result = execute_tool(tool_name, arguments)
                error = ""
            except Exception as e:
                logger.exception("Tool execution failed: %s", tool_name)
                result = {}
                error = str(e)

            tool_call_records.append(
                ToolCallRecord(
                    tool_name=tool_name,
                    arguments=arguments,
                    result=result,
                    error=error,
                )
            )

            # 回填 tool 消息
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(
                    {"error": error} if error else result,
                    ensure_ascii=False,
                ),
            })

    # 达到最大轮次仍未结束
    logger.warning("Agent reached max rounds %d, forcing stop", max_rounds)
    return AgentRunResult(
        answer="(已达最大工具调用轮次，未能给出最终回答)",
        tool_calls=tool_call_records,
        rounds=rounds,
        raw_messages=messages,
    )
