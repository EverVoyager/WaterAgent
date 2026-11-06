"""反思结果写入：四类记忆分发和效果闭环。"""
import logging
from typing import Any

from agent.memory.longterm_files import apply_longterm_edits
from agent.memory.memory_store import get_memory_store, is_memory_enabled
from agent.memory.safety import _gate_violation

logger = logging.getLogger(__name__)


# ====== 写入分发 ======

def _dispatch_longterm(reflection: dict[str, Any], user_query: str,
                       trigger_reason: str) -> int:
    """长期记忆编辑 → memory/ 目录（只经安全闸，无 MySQL 依赖）。

    仅在用户明确纠正/反馈偏好时写入：长期记忆是"用户设定层"，multi_round /
    tool_failure 等反思里 LLM 自行推测的"偏好"不可靠（宁少勿滥），也不是
    用户确认过的内容。
    """
    if trigger_reason not in ("user_correction", "explicit_feedback"):
        logger.debug("[reflection] 触发原因 %s 不写长期记忆（仅用户纠正/反馈可写）",
                     trigger_reason)
        return 0
    edits = reflection.get("longterm_edits") or []
    safe_edits = []
    for edit in edits:
        if not isinstance(edit, dict):
            continue
        content = str(edit.get("content", "")).strip()
        if not content:
            continue
        violation = _gate_violation(content)
        if violation:
            logger.warning("[reflection] 拦截不安全长期记忆编辑（%s）：%s", violation, content[:80])
            continue
        safe_edits.append(edit)
    if not safe_edits:
        return 0
    applied = apply_longterm_edits(safe_edits)
    return len(applied)


def _semantic_candidate_violation(mem: dict[str, Any], sources: dict[str, str]) -> str | None:
    """校验可执行的准入条件；知识分类和证据是否蕴含结论仍由反思模型判断。"""
    kind = mem.get("knowledge_type")
    if kind not in ("tool_contract", "project_rule", "domain_fact"):
        return "非允许的知识类型"
    reuse_when = mem.get("reuse_when")
    if not isinstance(reuse_when, str) or not reuse_when.strip():
        return "缺少具体复用场景"
    evidence = mem.get("evidence")
    if not isinstance(evidence, dict):
        return "缺少来源证据"
    source_id, quote = evidence.get("source_id"), evidence.get("quote")
    if not isinstance(source_id, str) or source_id not in sources:
        return "引用了未提供的来源"
    if source_id == "user" and kind != "project_rule":
        return "用户陈述只能支持项目约定"
    # shortcut: 原文匹配只证明引用存在，需要严格事实证明时增加结论与证据的蕴含校验。
    if not isinstance(quote, str) or len(quote.strip()) < 8 or quote not in sources[source_id]:
        return "证据过短或不属于来源原文"
    return _gate_violation(reuse_when, quote)


def _dispatch_semantic(reflection: dict[str, Any], semantic_sources: dict[str, str],
                       source_ref: str = "") -> int:
    """语义记忆通过类型/场景/原文证据校验后写入 Qdrant。"""
    try:
        from agent.memory.semantic_store import get_semantic_store
        store = get_semantic_store()
        if not store.enabled:
            return 0
    except Exception:
        return 0

    created = 0
    candidates = reflection.get("semantic_memories") or []
    if not isinstance(candidates, list):
        return 0
    for mem in candidates:
        if not isinstance(mem, dict):
            continue
        title, content = mem.get("title"), mem.get("content")
        if not isinstance(title, str) or not isinstance(content, str):
            continue
        title, content = title.strip(), content.strip()
        if not title or not content:
            continue
        violation = (_gate_violation(f"{title}。{content}")
                     or _semantic_candidate_violation(mem, semantic_sources))
        if violation:
            logger.info("[reflection] 跳过语义记忆：%s", violation.split(":", 1)[0])
            continue
        reuse_when = mem["reuse_when"].strip()
        evidence = [{"source_id": mem["evidence"]["source_id"],
                     "quote": mem["evidence"]["quote"], "source_ref": source_ref or None}]
        if _semantic_duplicate(store, title, content, reuse_when, evidence):
            logger.info("[reflection] 语义记忆正文及条件相同，跳过：%s", title[:60])
            continue
        mem_id = store.add_semantic(
            title=title, content=content, reuse_when=reuse_when, evidence=evidence,
        )
        if type(mem_id) in (int, str):
            created += 1
    return created


def _semantic_duplicate(store: Any, title: str, content: str,
                        reuse_when: str, evidence: list[dict]) -> bool:
    """仅去除正文和适用条件完全一致的重复，数字或条件不同均保留。"""
    try:
        recent = store.list_semantic(limit=50)
        return any(
            r.get("title", "").strip() == title
            and r.get("content", "").strip() == content
            and r.get("reuse_when") == reuse_when
            and r.get("evidence") == evidence
            for r in recent
        )
    except Exception:
        return False


def _dispatch_episode(
    reflection: dict[str, Any],
    source_ref: str = "",
) -> int:
    """本次事件与解法写入 Qdrant，来源由服务器附加。"""
    try:
        from agent.memory.episode_store import get_episode_store
        store = get_episode_store()
        if not store.enabled:
            return 0
    except Exception:
        return 0

    ep = reflection.get("episode") or {}
    if not isinstance(ep, dict):
        return 0
    event_summary = str(ep.get("event_summary", "")).strip()
    if not event_summary:
        return 0
    resolution = str(ep.get("resolution", "")).strip()
    violation = _gate_violation(event_summary, resolution)
    if violation:
        logger.warning("[reflection] 拦截不安全情景记忆（%s）：%s", violation, event_summary[:80])
        return 0
    outcome = ep.get("outcome", "partial")
    if outcome not in ("success", "failure", "partial"):
        outcome = "partial"

    ep_id = store.add_episode(
        event_summary=event_summary,
        resolution=resolution,
        outcome=outcome,
        source_refs=[{"source_ref": source_ref}] if source_ref else [],
        event_id=source_ref or None,
    )
    return int(type(ep_id) in (int, str))


def _dispatch_procedure(
    reflection: dict[str, Any],
    tool_calls: list[dict[str, Any]],
    source_ref: str = "",
) -> int:
    """程序记忆（可复用解决方法）写入 Qdrant。"""
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
    steps_text = "。".join(str(s) for s in steps)
    violation = _gate_violation(f"{name}。{applicability}", steps_text)
    if violation:
        logger.warning("[reflection] 拦截不安全程序记忆（%s）：%s", violation, name[:80])
        return 0
    proc_id = store.add_procedure(
        name=name, applicability=applicability, steps=steps,
        source_refs=[{"source_ref": source_ref}] if source_ref else [],
    )
    return int(type(proc_id) in (int, str))


def _demote_ineffective(
    demote: dict[str, Any],
    injected_memories: list[dict[str, Any]] | None = None,
    review_context: dict[str, Any] | None = None,
) -> int:
    """语义冲突仅记录待复核；程序记忆仍可停用。返回实际停用数。

    只允许处理本次请求真实注入过的 id（injected_memories）：demote 的 id
    来自 LLM 输出，幻觉 id 若不校验会误删任意无关记忆且不可恢复。
    """
    if not isinstance(demote, dict):
        return 0

    def _valid_ids(raw: list[Any], allowed: set[int | str]) -> list[int | str]:
        from agent.memory.vector_index import point_id

        if not isinstance(raw, list):
            return []
        out = []
        for i in raw:
            try:
                v = point_id(i)
            except (TypeError, ValueError):
                continue
            if v in allowed and v not in out:
                out.append(v)
        return out

    injected = injected_memories or []
    allowed_semantic = {it["id"] for it in injected if it.get("kind") == "semantic"}
    allowed_procedure = {it["id"] for it in injected if it.get("kind") == "procedure"}

    demoted = 0
    semantic_ids = _valid_ids(demote.get("semantic_ids") or [], allowed_semantic)
    if semantic_ids:
        if review_context is not None:
            review_context["semantic_review_ids"] = semantic_ids
        logger.info("[reflection] 保留疑似冲突的语义记忆，待复核 id=%s", semantic_ids)
    procedure_ids = _valid_ids(demote.get("procedure_ids") or [], allowed_procedure)
    if procedure_ids:
        try:
            from agent.memory.procedure_store import get_procedure_store
            store = get_procedure_store()
            for mid in procedure_ids:
                store.demote(mid)
                demoted += 1
        except Exception:
            pass
    return demoted


def _write_audit(
    user_query: str, trigger_reason: str, tool_summary: str,
    final_answer: str, reflection_text: str, memories_created: int,
    context: dict[str, Any] | None = None,
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
            context=context,
        )
    except Exception as e:
        logger.debug("[reflection] 审计写入失败：%s", e)
