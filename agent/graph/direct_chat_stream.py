"""闲聊节点流式生成器。

从 nodes.py 拆分而来，独立维护 LLM stream=True 的 token 级流式输出逻辑，
包括 Qwen3 思考内容剥离（与 llm_helpers._stream_llm 保持一致的状态机）。
"""
import logging
from typing import Any

from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    RateLimitError,
)

from agent.graph.errors import _classify_llm_error
from agent.prompts import DIRECT_CHAT_PROMPT
from agent.utils import CitationMarkerFilter
from app.core.llm import LLM_TIMEOUTS, get_llm_client, get_llm_config

logger = logging.getLogger(__name__)


def _direct_chat_stream(
    query: str,
    history: list[dict[str, Any]],
    skill_instructions: str = "",
):
    """流式版本的闲聊生成器。使用 LLM stream=True，逐 token yield answer_delta。

    借鉴 LangChain astream_events 的 on_chat_model_stream 事件思路。
    LLM 调用失败时抛 LLMError。
    """
    settings = get_llm_config()
    client = get_llm_client().with_options(timeout=LLM_TIMEOUTS["chat"])

    # 注入已启用 Skill 元信息（name + description）作为上下文
    # 借鉴 Claude 原生 Skills：元数据始终可见，LLM 自然能回答"你有哪些技能"
    # 不使用"当用户询问X时..."硬编码规则（已废弃，改用 list_skills 工具 + 元信息上下文）
    system_content = DIRECT_CHAT_PROMPT
    # 长期记忆常驻注入（用户手册 + Agent 自动积累，双层文件）
    try:
        from agent.memory import build_longterm_section
        system_content += build_longterm_section()
    except Exception as e:
        logger.debug("[direct_chat_stream] 注入长期记忆失败（不影响主流程）：%s", e)
    try:
        from agent.skills import get_enabled_skills_brief
        skills_brief = get_enabled_skills_brief()
        if skills_brief:
            system_content += (
                "\n\n# 已启用技能（Skills）\n以下技能当前处于启用状态：\n"
                + skills_brief
                + "\n当用户询问你有哪些技能/能力时，逐项列出上方清单中的技能名称和用途；"
                "不要遗漏，也不要编造清单之外的能力。\n"
            )
    except Exception as e:
        logger.debug("[direct_chat_stream] 注入 Skill 元信息失败（不影响主流程）：%s", e)

    # 若规划阶段匹配到了 Skill，注入其行为指令作为概念/知识问答的领域依据。
    if skill_instructions:
        system_content += (
            "\n\n=== 当前激活的 Skill 行为指令 ===\n"
            + skill_instructions
            + "\n=== Skill 指令结束 ===\n"
            + "请在上方指令指导下回答用户问题。若属于概念解释类问题，直接解释，"
            "不要调用工具，也不要给出需要实时数据支撑的结论。\n"
        )

    messages = [{"role": "system", "content": system_content}]
    # 压缩过的 history（首条 system 摘要）整体已受 token 预算控制，全量使用；
    # 未压缩的 history 截断到最近 3 轮（6 条）避免 token 超限
    is_compacted = (
        bool(history)
        and history[0].get("role") == "system"
        and "[历史对话摘要]" in history[0].get("content", "")
    )
    history_slice = history if is_compacted else history[-6:]
    for m in history_slice:
        messages.append({"role": m.get("role", "user"), "content": m.get("content", "")})
    messages.append({"role": "user", "content": query})

    try:
        stream = client.chat.completions.create(
            model=settings["model"],
            messages=messages,
            temperature=0.7,
            max_tokens=4096,
            stream=True,
        )
    except (APITimeoutError, RateLimitError, APIConnectionError, APIError) as e:
        logger.exception("[direct_chat_stream] LLM 流式调用失败 (%s)", type(e).__name__)
        raise _classify_llm_error(e) from e
    except Exception as e:
        logger.exception("[direct_chat_stream] LLM 未知异常")
        raise _classify_llm_error(e) from e

    # Qwen3 思考内容剥离：流式需缓冲检测 <think>...</think> 块
    # 与 llm_helpers._stream_llm 保持一致的状态机逻辑
    # 仅取 delta.content，忽略 delta.reasoning_content（推理过程不流式推给前端）
    # 引用标记过滤：闲聊路径无联网搜索，answer 中的 [N] 一律剥离（防模型凭空编"参考文献"）
    cite_filter = CitationMarkerFilter(valid_ref_ids=None)
    filtered_answer: list[str] = []
    buffer = ""
    in_think = False
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        content = getattr(delta, "content", None)
        if not content:
            continue
        buffer += content
        output = ""
        while buffer:
            if in_think:
                end = buffer.find("</think>")
                if end == -1:
                    # think 块未结束，继续缓冲
                    break
                # 跳过 think 内容及结束标签
                buffer = buffer[end + len("</think>"):]
                while buffer and buffer[0] in " \n\t":
                    buffer = buffer[1:]
                in_think = False
            else:
                start = buffer.find("<think>")
                if start == -1:
                    # 没有 think 标签，但保留末尾可能是 "<think" 前缀的部分
                    partial = 0
                    for plen in range(min(len(buffer), len("<think>")), 0, -1):
                        if buffer.endswith("<think>"[:plen]):
                            partial = plen
                            break
                    output += buffer[:len(buffer) - partial]
                    buffer = buffer[len(buffer) - partial:]
                    break
                output += buffer[:start]
                buffer = buffer[start + len("<think>"):]
                in_think = True
        if output:
            emitted = cite_filter.feed(output)
            if emitted:
                filtered_answer.append(emitted)
                yield {"type": "answer_delta", "content": emitted}

    tail = cite_filter.flush()
    if tail:
        filtered_answer.append(tail)

    # 流结束时若 think 块仍未闭合（异常中断），丢弃残留 buffer
    # 推送完整 answer 供 done 事件使用（已过滤 think）
    yield {"type": "synth_answer_full", "content": "".join(filtered_answer)}


# 显式导出（strip_think 用于 import 兼容性，避免外部依赖 nodes.strip_think）
__all__ = ["_direct_chat_stream"]
