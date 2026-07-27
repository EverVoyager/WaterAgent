"""异步反思循环（Hermes 范式核心）。

触发条件（满足任一即触发）：
1. user_correction — 用户明确纠正（"不对"/"应该是"/"错了"等）
2. tool_failure — 工具调用失败后重试成功
3. format_error — LLM 输出格式错误重试
4. multi_round — 多轮（>=2）才解决的问题
5. explicit_feedback — 用户给出偏好（"以后..."）

反思流程：
1. 判断是否值得反思（should_reflect）
2. 用轻量 LLM 生成反思内容（结构化 JSON）
3. 提取经验写入 MySQL（长期记忆 + 技能记忆）
4. 记录到反思日志（审计）

异步执行：用 ThreadPoolExecutor 后台运行，不阻塞响应。
"""
import json
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from agent.memory.memory_store import (
    MemoryType,
    get_memory_store,
    is_memory_enabled,
)
from app.core.llm import get_llm_client, get_llm_config

logger = logging.getLogger(__name__)

# 反思专用线程池（与工具执行池隔离，避免互相阻塞）
_REFLECT_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="reflect")

# 触发关键词（用户纠正/反馈）
_CORRECTION_KEYWORDS = {"不对", "错了", "应该是", "不是", "不对吧", "搞错了", "修正"}
_FEEDBACK_KEYWORDS = {"以后", "下次", "请记住", "建议", "希望", "偏好", "不要", "不要用"}


def should_reflect(
    user_query: str,
    final_answer: str,
    tool_calls: List[Dict[str, Any]],
    tool_errors: List[str],
    rounds: int,
    format_retry: bool = False,
) -> Optional[str]:
    """判断是否应触发反思。返回触发原因（None 表示不反思）。

    Args:
        user_query: 用户原始查询
        final_answer: 最终回答
        tool_calls: 工具调用记录
        tool_errors: 工具错误列表（非空的 error 字段）
        rounds: 规划轮次
        format_retry: 是否经历了格式错误重试

    Returns:
        触发原因字符串（如 "user_correction"），None 表示不反思
    """
    # 1. 用户明确纠正
    if any(kw in user_query for kw in _CORRECTION_KEYWORDS):
        return "user_correction"
    # 2. 用户反馈偏好
    if any(kw in user_query for kw in _FEEDBACK_KEYWORDS):
        return "explicit_feedback"
    # 3. 工具调用失败
    if tool_errors:
        return "tool_failure"
    # 4. 格式错误重试
    if format_retry:
        return "format_error"
    # 5. 多轮才解决（说明问题有难度，值得反思）
    if rounds >= 2 and tool_calls:
        return "multi_round"
    return None


def run_reflection_async(
    user_query: str,
    final_answer: str,
    tool_calls: List[Dict[str, Any]],
    tool_errors: List[str],
    rounds: int,
    trigger_reason: str,
    format_retry: bool = False,
) -> None:
    """异步执行反思循环（fire-and-forget，不阻塞调用方）。

    在 done 事件后调用，反思结果写入 MySQL。
    """
    if not is_memory_enabled():
        logger.debug("[reflection] 记忆模块未启用，跳过反思")
        return

    # 提交到线程池异步执行
    _REFLECT_EXECUTOR.submit(
        _run_reflection_sync,
        user_query=user_query,
        final_answer=final_answer,
        tool_calls=tool_calls,
        tool_errors=tool_errors,
        rounds=rounds,
        trigger_reason=trigger_reason,
        format_retry=format_retry,
    )
    logger.info("[reflection] 已提交异步反思任务 reason=%s", trigger_reason)


def _run_reflection_sync(
    user_query: str,
    final_answer: str,
    tool_calls: List[Dict[str, Any]],
    tool_errors: List[str],
    rounds: int,
    trigger_reason: str,
    format_retry: bool,
) -> None:
    """反思循环同步实现（在线程池中执行）。"""
    try:
        store = get_memory_store()
        if not store.enabled:
            return

        # 构造反思输入
        tool_summary = _summarize_tool_calls(tool_calls)
        reflection_input = {
            "user_query": user_query,
            "tool_calls": tool_summary,
            "tool_errors": tool_errors,
            "final_answer": final_answer,
            "rounds": rounds,
            "trigger_reason": trigger_reason,
            "format_retry": format_retry,
        }

        # LLM 生成反思（用 default 超时配置，避免阻塞太久）
        reflection = _generate_reflection(reflection_input)
        if not reflection:
            logger.warning("[reflection] LLM 未生成有效反思")
            return

        memories_created = 0

        # 1. 提取长期记忆（用户偏好/纠正/领域知识）
        for mem in reflection.get("memories", []):
            mem_type_str = mem.get("type", "")
            content = mem.get("content", "").strip()
            if not content or not mem_type_str:
                continue
            try:
                mem_type = MemoryType(mem_type_str)
            except ValueError:
                continue
            tags = mem.get("tags", [])
            context = {
                "trigger": trigger_reason,
                "query": user_query[:200],
                "rounds": rounds,
            }
            store.add_memory(mem_type, content, context=context, tags=tags)
            memories_created += 1

        # 2. 提取技能记忆（成功的工具调用模式）
        if reflection.get("skill_worthy") and tool_calls:
            query_pattern = reflection.get("query_pattern", user_query[:100])
            store.add_skill(
                query_pattern=query_pattern,
                tool_calls=[
                    {"name": tc.get("tool_name", ""), "arguments": tc.get("arguments", {})}
                    for tc in tool_calls
                ],
                success=not tool_errors,
                rounds_used=rounds,
            )

        # 3. 记录反思日志（审计）
        reflection_text = reflection.get("reflection", "")
        store.add_reflection(
            user_query=user_query,
            trigger_reason=trigger_reason,
            tool_calls_summary=tool_summary,
            final_answer=final_answer,
            reflection_text=reflection_text,
            memories_created=memories_created,
        )

        # 4. 触发记忆压缩：对本次新增记忆的类型做 LLM 语义合并
        #    避免"新记忆直接覆盖旧记忆"，改为 LLM 判断一致/冲突/可整合
        if memories_created > 0:
            _trigger_compact_for_types(store, reflection.get("memories", []))

        logger.info(
            "[reflection] 反思完成 reason=%s memories=%d skill=%s",
            trigger_reason, memories_created, reflection.get("skill_worthy", False),
        )
    except Exception as e:
        logger.warning("[reflection] 反思失败：%s\n%s", e, traceback.format_exc())


def _summarize_tool_calls(tool_calls: List[Dict[str, Any]]) -> str:
    """摘要工具调用记录为可读字符串。"""
    if not tool_calls:
        return "(无工具调用)"
    parts = []
    for i, tc in enumerate(tool_calls, 1):
        name = tc.get("tool_name", "")
        args = tc.get("arguments", {})
        error = tc.get("error", "")
        if error:
            parts.append(f"  {i}. {name}({args}) ❌ {error[:80]}")
        else:
            result_keys = list((tc.get("result") or {}).keys())[:5]
            parts.append(f"  {i}. {name}({args}) ✓ keys={result_keys}")
    return "\n".join(parts)


# ============ 记忆压缩（LLM 驱动的语义合并）============

_COMPACT_SYSTEM_PROMPT = """你是 Agent 的记忆压缩模块。给定同一类型的记忆列表（按时间正序，旧的在前，新的在后），请判断它们之间的关系并给出合并方案。

判断规则：
1. 语义完全一致（表达相同事实/规则）→ action="replace"，content 用最新的那条
2. 内容冲突（相互矛盾，如"警戒水位 377.5m" vs "警戒水位 378m"）→ action="replace"，content 用最新的那条（保留新记忆）
3. 不冲突但可整合（相关但补充，如"get_weather 返回降水" + "get_weather 不返回气温"）→ action="merge"，content 写整合后的一句话
4. 完全无关（不同主题）→ action="keep"，原样保留

输出严格的 JSON 数组，每个元素对应一条"最终保留的记忆"：
[
  {
    "action": "keep" | "merge" | "replace",
    "source_ids": [原记忆 id 列表],
    "content": "action=keep 时留空；merge/replace 时填写最终 content",
    "tags": ["可选标签"]
  }
]

要求：
- source_ids 必须覆盖所有输入记忆（每条原记忆只能出现在一个 group 中）
- action=keep 时 content 可留空（保留原记忆不动）
- merge/replace 时 content 必须是中文一句话，简洁准确
- 优先合并明显相关的记忆，减少记忆总数"""


def _llm_compact_memories(
    memory_type: str,
    memories: List[Dict[str, Any]],
) -> Optional[List[Dict[str, Any]]]:
    """调用 LLM 生成记忆合并方案。失败返回 None。"""
    if len(memories) < 2:
        return None

    settings = get_llm_config()
    client = get_llm_client().with_options(timeout=None)

    # 构造输入（精简字段避免上下文溢出）
    input_memories = [
        {
            "id": m["id"],
            "content": m.get("content", "")[:200],
            "updated_at": str(m.get("updated_at", "")),
        }
        for m in memories
    ]
    user_prompt = (
        f"记忆类型：{memory_type}\n"
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
            max_tokens=800,
        )
    except Exception as e:
        logger.warning("[compact] LLM 调用失败：%s", e)
        return None

    content = (resp.choices[0].message.content or "").strip()
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()
        if content.endswith("```"):
            content = content[:-3].strip()

    try:
        plan = json.loads(content)
        if not isinstance(plan, list):
            return None
        return plan
    except json.JSONDecodeError as e:
        logger.warning("[compact] LLM 返回非 JSON：%s | content=%s", e, content[:200])
        return None


def _trigger_compact_for_types(
    store: "MemoryStore",
    memories_from_reflection: List[Dict[str, Any]],
) -> None:
    """对本次反思涉及的记忆类型触发压缩。"""
    # 收集本次新增的记忆类型
    types_to_compact = set()
    for mem in memories_from_reflection:
        mem_type_str = mem.get("type", "")
        if mem_type_str:
            try:
                types_to_compact.add(MemoryType(mem_type_str))
            except ValueError:
                continue

    for mem_type in types_to_compact:
        try:
            deleted = store.compact_memories(mem_type, _llm_compact_memories)
            if deleted > 0:
                logger.info("[compact] type=%s 压缩删除 %d 条旧记忆", mem_type.value, deleted)
        except Exception as e:
            logger.debug("[compact] type=%s 压缩失败（不影响主流程）：%s", mem_type.value, e)


_REFLECTION_SYSTEM_PROMPT = """你是防汛预警 Agent 的反思模块。基于本次对话过程，提取值得长期记住的经验。

输入是 JSON 格式的对话摘要（user_query、tool_calls、tool_errors、final_answer 等）。
请分析以下问题：
1. 是否有用户偏好需要记住？（如"不要用 emoji"、"输出要简洁"）
2. 是否有领域知识需要记住？（如某站水位阈值、某工具的参数用法）
3. 是否有工具失败教训需要记住？（如某站无数据、某参数格式要求）
4. 本次工具调用序列是否值得作为"技能"复用？（同类问题下次直接套用）

输出严格的 JSON：
{
  "reflection": "一句话总结本次反思（中文）",
  "memories": [
    {
      "type": "user_preference|user_correction|domain_knowledge|tool_failure|format_learning",
      "content": "具体记忆内容（中文，一句话）",
      "tags": ["可选标签1", "标签2"]
    }
  ],
  "skill_worthy": true/false,
  "query_pattern": "如果 skill_worthy=true，给出查询模式（如'水情查询'）"
}

注意：
- memories 为空数组表示无值得记住的经验（这很正常，不要硬凑）
- 只记录真正有价值的经验，避免噪音
- tool_failure 类型应包含"什么参数下会失败"的具体信息
- domain_knowledge 应包含具体数值/规则（如"龙门站警戒水位 377.5m"）"""


def _generate_reflection(reflection_input: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """调用 LLM 生成反思。失败时返回 None（不抛错，反思失败不影响主流程）。"""
    settings = get_llm_config()
    client = get_llm_client().with_options(timeout=None)  # 反思不需要超时压力

    # 移除过长的字段避免上下文溢出
    input_truncated = {
        "user_query": reflection_input["user_query"][:500],
        "tool_calls": reflection_input["tool_calls"][:1500],
        "tool_errors": reflection_input["tool_errors"][:300],
        "final_answer": reflection_input["final_answer"][:800],
        "rounds": reflection_input["rounds"],
        "trigger_reason": reflection_input["trigger_reason"],
        "format_retry": reflection_input["format_retry"],
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
            max_tokens=800,
        )
    except Exception as e:
        logger.warning("[reflection] LLM 调用失败：%s", e)
        return None

    content = (resp.choices[0].message.content or "").strip()
    # 兼容 ```json ``` 包裹
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()
        if content.endswith("```"):
            content = content[:-3].strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        logger.warning("[reflection] LLM 返回非 JSON：%s | content=%s", e, content[:200])
        return None
