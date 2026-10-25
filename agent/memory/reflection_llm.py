"""反思与语义记忆压缩的 LLM 调用。"""
import json
import logging
from typing import Any

from agent.prompts.reflection import (
    COMPACT_SYSTEM_PROMPT as _COMPACT_SYSTEM_PROMPT,
)
from agent.prompts.reflection import (
    REFLECTION_SYSTEM_PROMPT as _REFLECTION_SYSTEM_PROMPT,
)
from agent.utils import parse_json_from_llm
from app.core.llm import LLM_TIMEOUTS, get_llm_client, get_llm_config, strip_think

logger = logging.getLogger(__name__)

# 反思输出 token 预算：思考型模型的 <think> 块会消耗大量预算
_REFLECTION_MAX_TOKENS = 4096


# ============ 记忆压缩（语义记忆 LLM 合并，Curator 复用）===========


def _llm_compact_semantic(
    memories: list[dict[str, Any]],
) -> list[dict[str, Any]] | None:
    """调用 LLM 生成语义记忆合并方案。失败返回 None。"""
    if len(memories) < 2:
        return None

    settings = get_llm_config()
    client = get_llm_client().with_options(timeout=LLM_TIMEOUTS["reflector"])

    input_memories = [
        {
            "id": m["id"],
            "content": (m.get("title", "") + "：" + m.get("content", ""))[:200],
            "updated_at": str(m.get("updated_at", "")),
        }
        for m in memories
    ]
    user_prompt = (
        f"记忆类型：semantic\n"
        f"记忆列表（共 {len(memories)} 条，按时间正序）：\n"
        f"{json.dumps(input_memories, ensure_ascii=False, indent=2)}\n\n"
        f"请给出合并方案。"
    )

    try:
        resp = client.chat.completions.create(
            model=settings["model"],
            messages=[
                {"role": "system", "content": _COMPACT_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=_REFLECTION_MAX_TOKENS,
        )
    except Exception as e:
        logger.warning("[compact] LLM 调用失败：%s", e)
        return None

    content = strip_think((resp.choices[0].message.content or "").strip())
    plan = parse_json_from_llm(content)
    if not isinstance(plan, list):
        logger.warning("[compact] LLM 返回非 JSON list | content=%s", content[:200])
        return None
    return plan


def _generate_reflection(reflection_input: dict[str, Any]) -> dict[str, Any] | None:
    """调用 LLM 生成反思。失败时返回 None（不抛错，反思失败不影响主流程）。"""
    settings = get_llm_config()
    # reflector 专属超时（90s）：思考型模型 <think> 耗时长，但必须有线程挂死防线
    client = get_llm_client().with_options(timeout=LLM_TIMEOUTS["reflector"])

    # 移除过长的字段避免上下文溢出
    input_truncated = {
        "user_query": reflection_input["user_query"][:500],
        "tool_calls": reflection_input["tool_calls"][:1500],
        "tool_errors": reflection_input["tool_errors"][:300],
        "final_answer": reflection_input["final_answer"][:800],
        "rounds": reflection_input["rounds"],
        "trigger_reason": reflection_input["trigger_reason"],
        "format_retry": reflection_input["format_retry"],
        # 本次注入的记忆（效果闭环：被注入后仍被纠正 → demote 降权）
        "injected_memories": (reflection_input.get("injected_memories") or [])[:10],
    }

    user_prompt = f"对话摘要：\n{json.dumps(input_truncated, ensure_ascii=False, indent=2)}"

    try:
        resp = client.chat.completions.create(
            model=settings["model"],
            messages=[
                {"role": "system", "content": _REFLECTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=_REFLECTION_MAX_TOKENS,
        )
    except Exception as e:
        logger.warning("[reflection] LLM 调用失败：%s", e)
        return None

    content = strip_think((resp.choices[0].message.content or "").strip())
    result = parse_json_from_llm(content)
    if result is None:
        logger.warning("[reflection] LLM 返回非 JSON | content=%s", content[:200])
    return result
