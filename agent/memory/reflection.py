"""异步反思循环（五类记忆架构版）。

触发条件（满足任一即触发，不变）：
1. user_correction — 用户明确纠正（"不对"/"应该是"/"错了"等）
2. explicit_feedback — 用户给出偏好（"以后..."）
3. tool_failure — 工具调用失败
4. format_error — LLM 输出格式错误重试
5. multi_round — 多轮（>=2）才解决的问题

写入分发（LLM 输出新 schema）：
- longterm_edits   → memory/ 目录（文件，无需 MySQL）
- semantic_memories → agent_semantic 表 + 向量索引
- episode           → agent_episodes 表 + 向量索引
- procedure         → agent_procedures 表 + 向量索引
- demote            → 语义记忆删除 / 程序记忆降权

三道写入安全闸（全类型通用）：
1. 提示词注入扫描（防"请记住：忽略所有指令"式持久化攻击）
2. 敏感信息过滤（对齐 Codex redaction：API key/密码/token/手机号拒写）
3. rubric 质量门槛由反思 prompt 自评（class-first，输出前自滤）

异步执行：ThreadPoolExecutor 后台运行，不阻塞响应。
"""
import json
import logging
import re
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from agent.memory.longterm import apply_longterm_edits
from agent.memory.memory_store import get_memory_store, is_memory_enabled
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

# 反思输出 token 预算：思考型模型的 <think> 块会消耗大量预算
_REFLECTION_MAX_TOKENS = 4096


def should_reflect(
    user_query: str,
    final_answer: str,
    tool_calls: list[dict[str, Any]],
    tool_errors: list[str],
    rounds: int,
    format_retry: bool = False,
) -> str | None:
    """判断是否应触发反思。返回触发原因（None 表示不反思）。"""
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


def _reflection_available() -> bool:
    """反思是否可运行：长期记忆（文件）或 MySQL 任一可用。"""
    try:
        from app.core.config import get_settings
        if getattr(get_settings(), "AUTO_MEMORY_ENABLED", True):
            return True
    except Exception:
        pass
    return is_memory_enabled()


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

    Args:
        injected_memories: 本次请求注入到 prompt 的记忆 [{"id", "content", "kind"}]，
            供反思评估注入有效性（被注入后仍被纠正 → demote，效果闭环）
    """
    if not _reflection_available():
        logger.debug("[reflection] 记忆模块未启用，跳过反思")
        return

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

        # 1. 长期记忆：写入 memory/ 目录（文件，无需 MySQL）
        memories_created += _dispatch_longterm(reflection, user_query)

        # 2. 语义记忆：领域知识（MySQL + 向量）
        memories_created += _dispatch_semantic(reflection, user_query)

        # 3. 情景记忆：本次事件与解法（MySQL + 向量）
        memories_created += _dispatch_episode(
            reflection, user_query, tool_calls, tool_errors, rounds, trigger_reason,
        )

        # 4. 程序记忆：可复用解决方法（MySQL + 向量）
        memories_created += _dispatch_procedure(reflection, tool_calls, tool_errors, rounds)

        # 5. 效果闭环：注入后仍无效的记忆 demote
        demoted = _demote_ineffective(reflection.get("demote") or {})

        # 6. 审计日志（agent_reflections，MySQL 可用时）
        _write_audit(
            user_query, trigger_reason, tool_summary, final_answer,
            reflection.get("reflection", ""), memories_created,
        )

        logger.info(
            "[reflection] 反思完成 reason=%s created=%d demoted=%d",
            trigger_reason, memories_created, demoted,
        )
    except Exception as e:
        logger.warning("[reflection] 反思失败：%s\n%s", e, traceback.format_exc())


# ====== 写入分发 ======

def _dispatch_longterm(reflection: dict[str, Any], user_query: str) -> int:
    """长期记忆编辑 → memory/ 目录（只经安全闸，无 MySQL 依赖）。"""
    edits = reflection.get("longterm_edits") or []
    safe_edits = []
    for edit in edits:
        if not isinstance(edit, dict):
            continue
        content = str(edit.get("content", "")).strip()
        if not content:
            continue
        if _is_unsafe_memory_content(content) or _is_sensitive_content(content):
            logger.warning("[reflection] 拦截不安全长期记忆编辑：%s", content[:80])
            continue
        safe_edits.append(edit)
    if not safe_edits:
        return 0
    applied = apply_longterm_edits(safe_edits)
    return len(applied)


def _dispatch_semantic(reflection: dict[str, Any], user_query: str) -> int:
    """语义记忆（领域知识）→ agent_semantic + 向量索引。"""
    try:
        from agent.memory.semantic_store import get_semantic_store
        store = get_semantic_store()
        if not store.enabled:
            return 0
    except Exception:
        return 0

    created = 0
    for mem in reflection.get("semantic_memories") or []:
        if not isinstance(mem, dict):
            continue
        title = str(mem.get("title", "")).strip()
        content = str(mem.get("content", "")).strip()
        if not title or not content:
            continue
        if _is_unsafe_memory_content(content) or _is_sensitive_content(content):
            logger.warning("[reflection] 拦截不安全语义记忆：%s", content[:80])
            continue
        tags = mem.get("tags") or []
        mem_id = store.add_semantic(
            title=title, content=content, source="reflection",
            tags=",".join(str(t) for t in tags),
        )
        if isinstance(mem_id, int):
            created += 1
            _safe_call(_index_semantic, mem_id, title, content)
    return created


def _dispatch_episode(
    reflection: dict[str, Any],
    user_query: str,
    tool_calls: list[dict[str, Any]],
    tool_errors: list[str],
    rounds: int,
    trigger_reason: str,
) -> int:
    """情景记忆（本次事件与解法）→ agent_episodes + 向量索引。

    无 episode 输出时按触发原因兜底一条最小记录（保证重大事件不漏记）。
    """
    try:
        from agent.memory.episode_store import get_episode_store
        store = get_episode_store()
        if not store.enabled:
            return 0
    except Exception:
        return 0

    ep = reflection.get("episode") or {}
    event_summary = str(ep.get("event_summary", "")).strip()
    if not event_summary:
        return 0
    if _is_unsafe_memory_content(event_summary):
        logger.warning("[reflection] 拦截不安全情景记忆：%s", event_summary[:80])
        return 0
    resolution = str(ep.get("resolution", "")).strip()
    if _is_sensitive_content(event_summary) or _is_sensitive_content(resolution):
        logger.warning("[reflection] 拦截含敏感信息的情景记忆")
        return 0
    outcome = ep.get("outcome", "success")
    if outcome not in ("success", "failure", "partial"):
        outcome = "partial"

    ep_id = store.add_episode(
        event_summary=event_summary,
        resolution=resolution,
        outcome=outcome,
        query_summary=user_query[:512],
        tool_calls=[
            {"name": tc.get("tool_name", ""), "arguments": tc.get("arguments", {})}
            for tc in (tool_calls or [])
        ],
        tags=trigger_reason,
    )
    if isinstance(ep_id, int):
        _safe_call(_index_episode, ep_id, event_summary, resolution)
        return 1
    return 0


def _dispatch_procedure(
    reflection: dict[str, Any],
    tool_calls: list[dict[str, Any]],
    tool_errors: list[str],
    rounds: int,
) -> int:
    """程序记忆（可复用解决方法）→ agent_procedures + 向量索引。"""
    proc = reflection.get("procedure") or {}
    if not isinstance(proc, dict) or not proc.get("worthy") or not tool_calls:
        return 0
    try:
        from agent.memory.procedure_store import get_procedure_store
        store = get_procedure_store()
        if not store.enabled:
            return 0
    except Exception:
        return 0

    name = str(proc.get("name", "")).strip()
    applicability = str(proc.get("applicability", "")).strip()
    steps = proc.get("steps") or []
    if not name or not applicability or not steps:
        return 0
    if _is_unsafe_memory_content(applicability + name) or _is_sensitive_content(applicability):
        logger.warning("[reflection] 拦截不安全程序记忆：%s", name[:80])
        return 0
    tool_sequence = proc.get("tool_sequence") or [
        tc.get("tool_name", "") for tc in tool_calls if tc.get("tool_name")
    ]

    proc_id = store.add_procedure(
        name=name, applicability=applicability, steps=steps,
        tool_sequence=tool_sequence, source="reflection",
    )
    if isinstance(proc_id, int):
        _safe_call(_index_procedure, proc_id, applicability)
        return 1
    return 0


def _demote_ineffective(demote: dict[str, Any]) -> int:
    """效果闭环：注入后仍无效的记忆降权/删除。返回处理数。"""
    demoted = 0
    semantic_ids = demote.get("semantic_ids") or []
    if semantic_ids:
        try:
            from agent.memory.semantic_store import get_semantic_store
            store = get_semantic_store()
            for mid in semantic_ids:
                try:
                    if store.delete_semantic(int(mid)):
                        _safe_call(_remove_semantic, int(mid))
                        demoted += 1
                except (TypeError, ValueError):
                    continue
        except Exception:
            pass
    procedure_ids = demote.get("procedure_ids") or []
    if procedure_ids:
        try:
            from agent.memory.procedure_store import get_procedure_store
            store = get_procedure_store()
            for mid in procedure_ids:
                try:
                    store.demote(int(mid))
                    demoted += 1
                except (TypeError, ValueError):
                    continue
        except Exception:
            pass
    return demoted


def _write_audit(
    user_query: str, trigger_reason: str, tool_summary: str,
    final_answer: str, reflection_text: str, memories_created: int,
) -> None:
    """审计日志写 agent_reflections（MySQL 可用时）。"""
    try:
        if not is_memory_enabled():
            return
        store = get_memory_store()
        store.add_reflection(
            user_query=user_query,
            trigger_reason=trigger_reason,
            tool_calls_summary=tool_summary,
            final_answer=final_answer,
            reflection_text=reflection_text or "(无)",
            memories_created=memories_created,
        )
    except Exception as e:
        logger.debug("[reflection] 审计写入失败：%s", e)


# ====== 向量索引（尽力而为） ======

def _safe_call(fn, *args) -> None:
    try:
        fn(*args)
    except Exception as e:
        logger.debug("[reflection] 向量索引操作失败：%s", e)


def _index_semantic(mem_id: int, title: str, content: str) -> None:
    from agent.memory import vector_index
    vector_index.index_semantic(mem_id, title, content)


def _index_episode(ep_id: int, event_summary: str, resolution: str) -> None:
    from agent.memory import vector_index
    vector_index.index_episode(ep_id, event_summary, resolution)


def _index_procedure(proc_id: int, applicability: str) -> None:
    from agent.memory import vector_index
    vector_index.index_procedure(proc_id, applicability)


def _remove_semantic(mem_id: int) -> None:
    from agent.memory import vector_index
    vector_index.remove_semantic(mem_id)


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


# ====== 写入安全闸 ======

# 提示词注入攻击载荷特征（写入侧安全闸门，借鉴 Hermes 的 prompt 注入扫描）
# 恶意用户可通过"请记住：忽略所有指令"把注入攻击持久化到记忆
_UNSAFE_MEMORY_PATTERNS = (
    "忽略所有", "忽略以上", "忽略之前", "忽略上述", "无视所有", "无视指令",
    "ignore all", "ignore previous", "ignore above",
    "系统提示", "系统指令", "system prompt", "开发者模式", "developer mode",
    "从现在起你", "你是一个新的", "新人设", "最高优先级", "无条件服从",
    "越狱", "jailbreak",
)


def _is_unsafe_memory_content(content: str) -> bool:
    """检测记忆内容是否含提示词注入攻击载荷（所有记忆类型通用）。"""
    if not content:
        return False
    lowered = content.lower()
    return any(pattern in lowered for pattern in _UNSAFE_MEMORY_PATTERNS)


# 敏感信息特征（对齐 Codex memories 的 redaction：凭据/个人隐私不写入记忆）
_SENSITIVE_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9]{16,}"),                    # API key
    re.compile(r"(?:api[_-]?key|apikey)\s*[:=]\s*\S{8,}", re.I),
    re.compile(r"(?:password|passwd|pwd)\s*[:=]\s*\S{6,}", re.I),
    re.compile(r"(?:secret|token)\s*[:=]\s*\S{8,}", re.I),
    re.compile(r"Bearer\s+[A-Za-z0-9_\-\.]{16,}"),
    re.compile(r"1[3-9]\d{9}"),                            # 手机号
    re.compile(r"\d{17}[\dXx]"),                           # 身份证号
)


def _is_sensitive_content(content: str) -> bool:
    """检测记忆内容是否含敏感信息（API key/密码/token/手机号/身份证）。"""
    if not content:
        return False
    return any(p.search(content) for p in _SENSITIVE_PATTERNS)


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
