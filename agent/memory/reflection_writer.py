"""反思结果写入：四类记忆分发、索引更新和效果闭环。"""
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
        violation = _gate_violation(f"{title}。{content}")
        if violation:
            logger.warning("[reflection] 拦截不安全语义记忆（%s）：%s", violation, content[:80])
            continue
        # 写入查重：新知识基本已被近期记忆覆盖 → 跳过（近重复只靠每周压缩兜底太慢）
        if _semantic_duplicate(store, title, content):
            logger.info("[reflection] 语义记忆与近期条目近重复，跳过：%s", title[:60])
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


def _semantic_duplicate(store: Any, title: str, content: str,
                        threshold: float = 0.8) -> bool:
    """新语义记忆的二元组被任一近期条目覆盖超阈值 → 判为近重复。"""
    from agent.utils import text_coverage

    try:
        recent = store.list_semantic(limit=50)
        new_text = f"{title}。{content}"
        return any(
            text_coverage(new_text,
                          f"{r.get('title', '')}。{r.get('content', '')}") >= threshold
            for r in recent
        )
    except Exception:
        return False


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
    resolution = str(ep.get("resolution", "")).strip()
    violation = _gate_violation(event_summary, resolution)
    if violation:
        logger.warning("[reflection] 拦截不安全情景记忆（%s）：%s", violation, event_summary[:80])
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
    steps_text = "。".join(str(s) for s in steps)
    violation = _gate_violation(f"{name}。{applicability}", steps_text)
    if violation:
        logger.warning("[reflection] 拦截不安全程序记忆（%s）：%s", violation, name[:80])
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


def _demote_ineffective(
    demote: dict[str, Any],
    injected_memories: list[dict[str, Any]] | None = None,
) -> int:
    """效果闭环：注入后仍无效的记忆降权/删除。返回处理数。

    只允许处理本次请求真实注入过的 id（injected_memories）：demote 的 id
    来自 LLM 输出，幻觉 id 若不校验会误删任意无关记忆且不可恢复。
    """
    def _valid_ids(raw: list[Any], allowed: set[int]) -> list[int]:
        out = []
        for i in raw:
            try:
                v = int(i)
            except (TypeError, ValueError):
                continue
            if v in allowed:
                out.append(v)
        return out

    injected = injected_memories or []
    allowed_semantic = {it["id"] for it in injected if it.get("kind") == "semantic"}
    allowed_procedure = {it["id"] for it in injected if it.get("kind") == "procedure"}

    demoted = 0
    semantic_ids = _valid_ids(demote.get("semantic_ids") or [], allowed_semantic)
    if semantic_ids:
        try:
            from agent.memory.semantic_store import get_semantic_store
            store = get_semantic_store()
            for mid in semantic_ids:
                if store.delete_semantic(mid):
                    _safe_call(_remove_semantic, mid)
                    demoted += 1
        except Exception:
            pass
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
