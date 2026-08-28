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
from typing import Any

from agent.memory.memory_store import (
    MemoryStore,
    MemoryType,
    get_memory_store,
    is_memory_enabled,
)
from agent.prompts.reflection import (
    COMPACT_SYSTEM_PROMPT as _COMPACT_SYSTEM_PROMPT,
)
from agent.prompts.reflection import (
    REFLECTION_SYSTEM_PROMPT as _REFLECTION_SYSTEM_PROMPT,
)
from agent.utils import parse_json_from_llm
from app.core.llm import LLM_TIMEOUTS, get_llm_client, get_llm_config, strip_think

logger = logging.getLogger(__name__)

# 反思专用线程池（与工具执行池隔离，避免互相阻塞）
_REFLECT_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="reflect")

# 触发关键词（用户纠正/反馈）
_CORRECTION_KEYWORDS = {"不对", "错了", "应该是", "不是", "不对吧", "搞错了", "修正"}
_FEEDBACK_KEYWORDS = {"以后", "下次", "请记住", "建议", "希望", "偏好", "不要", "不要用"}

# 反思输出 token 预算：思考型模型的 <think> 块会消耗大量预算，
# 800 会导致 content 为空（曾导致反思 100% 失败），结构化 JSON 正文也需要余量
_REFLECTION_MAX_TOKENS = 4096


def should_reflect(
    user_query: str,
    final_answer: str,
    tool_calls: list[dict[str, Any]],
    tool_errors: list[str],
    rounds: int,
    format_retry: bool = False,
) -> str | None:
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
    tool_calls: list[dict[str, Any]],
    tool_errors: list[str],
    rounds: int,
    trigger_reason: str,
    format_retry: bool = False,
    injected_memories: list[dict[str, Any]] | None = None,
) -> None:
    """异步执行反思循环（fire-and-forget，不阻塞调用方）。

    在 done 事件后调用，反思结果写入 MySQL。

    Args:
        injected_memories: 本次请求注入到 prompt 的记忆 [{"id", "content"}]，
            供反思评估注入有效性（被注入后仍被纠正 → 降权，GEPA 效果闭环）
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
        injected_memories=injected_memories,
    )
    logger.info("[reflection] 已提交异步反思任务 reason=%s", trigger_reason)


def _run_reflection_sync(
    user_query: str,
    final_answer: str,
    tool_calls: list[dict[str, Any]],
    tool_errors: list[str],
    rounds: int,
    trigger_reason: str,
    format_retry: bool,
    injected_memories: list[dict[str, Any]] | None = None,
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
            "injected_memories": injected_memories or [],
        }

        # LLM 生成反思（reflector 超时配置：90s，防线程挂死）
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

            # 写入侧安全闸门：拦截提示词注入攻击载荷（所有记忆类型）
            # 恶意用户说"请记住：忽略所有指令"→ 若不拦截会持久化注入
            if _is_unsafe_memory_content(content):
                logger.warning(
                    "[reflection] 拦截疑似提示词注入的记忆（不写入）：%s", content[:100]
                )
                continue

            # 写入侧硬校验：tool_failure 禁止行为指令式措辞
            if mem_type == MemoryType.TOOL_FAILURE:
                if _is_imperative_failure_content(content):
                    logger.warning(
                        "[reflection] 拦截行为指令式 tool_failure（不写入）：%s", content[:100]
                    )
                    continue

                # EmbodiSkill 分类：execution_lapse 不写入长期记忆
                # 区分"技能缺陷"（可复现，值得记）vs"执行失误"（偶发，不固化）
                failure_class = mem.get("failure_classification", "").strip().lower()
                if failure_class == "execution_lapse":
                    logger.info(
                        "[reflection] 跳过 execution_lapse 类型 tool_failure（不固化偶发失误）：%s",
                        content[:100],
                    )
                    continue

                # 把 falsifiable_check + failure_classification 存入 context，便于注入侧自愈校验
                falsifiable = mem.get("falsifiable_check", "").strip()
                context = {
                    "trigger": trigger_reason,
                    "query": user_query[:200],
                    "rounds": rounds,
                    "falsifiable_check": falsifiable,
                    "failure_classification": failure_class or "skill_defect",
                }
            else:
                context = {
                    "trigger": trigger_reason,
                    "query": user_query[:200],
                    "rounds": rounds,
                }

            # Rubric 质量门槛（借鉴 Hermes v0.12.0 rubric-based 反思决策）：
            # 低质量记忆不写入，宁缺毋滥
            if not _passes_rubric(mem):
                logger.info(
                    "[reflection] 跳过低质量记忆（rubric 未达标）：%s", content[:100]
                )
                continue

            tags = mem.get("tags", [])
            mem_id = store.add_memory(mem_type, content, context=context, tags=tags)
            memories_created += 1
            # 写入向量索引（语义检索用；失败不影响 MySQL 主流程）
            _safe_index_memory(mem_id, mem_type.value, content, tags)

        # 2. 提取技能记忆（成功的工具调用模式）
        if reflection.get("skill_worthy") and tool_calls:
            query_pattern = reflection.get("query_pattern", user_query[:100])
            skill_id = store.add_skill(
                query_pattern=query_pattern,
                tool_calls=[
                    {"name": tc.get("tool_name", ""), "arguments": tc.get("arguments", {})}
                    for tc in tool_calls
                ],
                success=not tool_errors,
                rounds_used=rounds,
            )
            if isinstance(skill_id, int):
                _safe_index_skill(skill_id, query_pattern)

        # 3. 效果闭环（GEPA）：注入后仍无效的记忆降权（hit_count 清零，
        #    重新进入衰减-剪枝通道，不再反复注入错误记忆）
        demoted = _demote_ineffective_memories(store, reflection.get("demote_ids", []))

        # 4. 记录反思日志（审计）
        reflection_text = reflection.get("reflection", "")
        store.add_reflection(
            user_query=user_query,
            trigger_reason=trigger_reason,
            tool_calls_summary=tool_summary,
            final_answer=final_answer,
            reflection_text=reflection_text,
            memories_created=memories_created,
        )

        # 5. 触发记忆压缩：对本次新增记忆的类型做 LLM 语义合并
        #    避免"新记忆直接覆盖旧记忆"，改为 LLM 判断一致/冲突/可整合
        if memories_created > 0:
            _trigger_compact_for_types(store, reflection.get("memories", []))

        logger.info(
            "[reflection] 反思完成 reason=%s memories=%d skill=%s demoted=%d",
            trigger_reason, memories_created, reflection.get("skill_worthy", False), demoted,
        )
    except Exception as e:
        logger.warning("[reflection] 反思失败：%s\n%s", e, traceback.format_exc())


def _demote_ineffective_memories(store: MemoryStore, demote_ids: list[Any]) -> int:
    """对反思判定无效的注入记忆降权。返回成功降权数。"""
    if not demote_ids:
        return 0
    demoted = 0
    for mid in demote_ids:
        try:
            if isinstance(mid, int | str) and store.demote_memory(int(mid)):
                demoted += 1
        except (TypeError, ValueError):
            continue
    return demoted


def _safe_index_memory(
    mem_id: Any, memory_type: str, content: str, tags: list[str] | None = None
) -> None:
    """把新记忆写入向量索引（尽力而为，失败只记日志）。"""
    try:
        if not isinstance(mem_id, int):
            return  # add_memory 失败（或测试 mock）时跳过
        from agent.memory import vector_index
        vector_index.index_memory(mem_id, memory_type, content, tags)
    except Exception as e:
        logger.debug("[reflection] 写入记忆向量索引失败：%s", e)


def _safe_index_skill(skill_id: Any, query_pattern: str) -> None:
    """把新技能写入向量索引（尽力而为，失败只记日志）。"""
    try:
        if not isinstance(skill_id, int):
            return
        from agent.memory import vector_index
        vector_index.index_skill(skill_id, query_pattern)
    except Exception as e:
        logger.debug("[reflection] 写入技能向量索引失败：%s", e)


def _summarize_tool_calls(tool_calls: list[dict[str, Any]]) -> str:
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


# 祈使句指令性措辞（用于拦截行为指令式 tool_failure）
_IMPERATIVE_PATTERNS = (
    "不要", "不应该", "不应", "禁止", "一律", "永远", "必须避免",
    "避免使用", "避免调用", "请勿", "切勿", "绝不可", "不可以",
)


def _is_imperative_failure_content(content: str) -> bool:
    """检测 tool_failure 内容是否为行为指令而非事实陈述。

    判据：含"不要/永远/一律/禁止"等祈使句措辞 = 行为指令，应拦截。
    合法的事实陈述形如"调用 X 返回 Y"或"X 工具在 Z 参数下失败"。

    Args:
        content: 待检测的 tool_failure 记忆内容

    Returns:
        True 表示是行为指令（应拦截），False 表示是事实陈述（可写入）
    """
    if not content:
        return False
    return any(pattern in content for pattern in _IMPERATIVE_PATTERNS)


# 提示词注入攻击载荷特征（写入侧安全闸门，借鉴 Hermes 的 prompt 注入扫描）
# 恶意用户可通过"请记住：忽略所有指令"把注入攻击持久化到记忆，
# 之后每次注入都会攻击 LLM —— 必须在写入侧拦截
_UNSAFE_MEMORY_PATTERNS = (
    "忽略所有", "忽略以上", "忽略之前", "忽略上述", "无视所有", "无视指令",
    "ignore all", "ignore previous", "ignore above",
    "系统提示", "系统指令", "system prompt", "开发者模式", "developer mode",
    "从现在起你", "你是一个新的", "新人设", "最高优先级", "无条件服从",
    "越狱", "jailbreak",
)


def _is_unsafe_memory_content(content: str) -> bool:
    """检测记忆内容是否含提示词注入攻击载荷（所有记忆类型通用）。

    Args:
        content: 待检测的记忆内容

    Returns:
        True 表示疑似注入攻击（应拦截），False 表示可安全写入
    """
    if not content:
        return False
    lowered = content.lower()
    return any(pattern in lowered for pattern in _UNSAFE_MEMORY_PATTERNS)


# Rubric 质量门槛（借鉴 Hermes v0.12.0 class-first rubric 反思决策）
# 三个维度均为 1-5 分：任一维 < 2 或总分 < 8 → 不写入（宁缺毋滥）
_RUBRIC_MIN_DIM = 2
_RUBRIC_MIN_TOTAL = 8


def _passes_rubric(mem: dict[str, Any]) -> bool:
    """按 rubric 评分过滤低质量记忆。

    评分维度（由反思 LLM 自评，写入 REFLECTION_SYSTEM_PROMPT）：
    - specificity: 是否含具体数值/工具名/站名（具体才可执行）
    - durability: 是否跨会话长期有效（只在本对话有效的低分）
    - actionability: 能否指导未来行为（纯背景信息低分）

    LLM 未返回 scores 字段时宽容放行（向后兼容）。
    """
    scores = mem.get("scores")
    if not isinstance(scores, dict) or not scores:
        return True
    try:
        vals = [int(scores.get(k, 3)) for k in ("specificity", "durability", "actionability")]
    except (TypeError, ValueError):
        return True
    if any(v < _RUBRIC_MIN_DIM for v in vals):
        return False
    return sum(vals) >= _RUBRIC_MIN_TOTAL


# ============ 记忆压缩（LLM 驱动的语义合并）============


def _llm_compact_memories(
    memory_type: str,
    memories: list[dict[str, Any]],
) -> list[dict[str, Any]] | None:
    """调用 LLM 生成记忆合并方案。失败返回 None。"""
    if len(memories) < 2:
        return None

    settings = get_llm_config()
    # reflector 专属超时（90s）：既给思考型模型留足时间，又防线程池挂死
    client = get_llm_client().with_options(timeout=LLM_TIMEOUTS["reflector"])

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


def _trigger_compact_for_types(
    store: MemoryStore,
    memories_from_reflection: list[dict[str, Any]],
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
                # 压缩改变了行集合（删旧+插新），同步向量索引保持两侧一致
                _sync_type_index(store, mem_type)
        except Exception as e:
            logger.debug("[compact] type=%s 压缩失败（不影响主流程）：%s", mem_type.value, e)


def _sync_type_index(store: MemoryStore, mem_type: MemoryType) -> None:
    """压缩后同步该类型记忆的向量索引（MySQL 为 source of truth）。"""
    try:
        from agent.memory import vector_index
        rows = store.get_memories(memory_type=mem_type, limit=1000)
        vector_index.sync_memory_type(mem_type.value, rows)
    except Exception as e:
        logger.debug("[compact] 同步向量索引失败 type=%s：%s", mem_type.value, e)


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
        # 本次注入的记忆（效果闭环：被注入后仍被纠正 → demote_ids 降权）
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
