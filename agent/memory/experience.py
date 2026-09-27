"""记忆注入聚合（五类记忆架构版）。

注入体系：
- 长期记忆（文件）：system prompt 常驻（longterm.build_longterm_section，三处注入）
- 会话记忆：context_compact（现状不动）
- 情景记忆：planner 第 1 轮「历史类似情形」top-2（向量，降级时间倒序）
- 程序记忆：planner 第 1 轮「推荐方法」top-2（向量，降级时间倒序）
- 语义记忆：synthesizer「领域知识」top-3（向量，降级时间倒序）

三态注入约定（与 vector_index 一致）：
- None = 索引不可用 → 降级时间倒序（保底注入）
- []   = 索引可用但无相关 → 不注入（防无关记忆污染）
- 非空 = 按相关性注入

效果闭环（thread-local）：
- 注入时记录 id（反思时评估有效性 → demote）
- 请求完成后 finalize_injected_tracking(success) 计数
  （语义记忆 hit_count++ / 程序记忆 use_count、success_count++）
"""
import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

# 注入条数上限
_EPISODE_CAP = 2
_PROCEDURE_CAP = 2
_SEMANTIC_CAP = 3


def _date_of(val: Any) -> str:
    """datetime/str -> 'YYYY-MM-DD'（供注入标注时效，未知返回空串）。"""
    s = str(val or "")
    return s[:10] if len(s) >= 10 and s[4] == "-" and s[7] == "-" else ""


def _passes_read_gate(*texts: str) -> bool:
    """读时安全闸：注入前校验候选记忆内容。

    与写入侧同一套闸（注入载荷/敏感信息/领域事实）。领域事实闸对照的是
    当前 WARNING_THRESHOLDS——写入后阈值若调整，旧断言在此暴露，不再注入。
    闸自身故障时放行（降级为信任写入侧校验，不阻塞主流程）。
    """
    try:
        from agent.memory.reflection import check_memory_safety
        return check_memory_safety(*texts) is None
    except Exception:
        return True


# ============ 注入追踪（效果闭环）============

_local = threading.local()


def _record_injected(kind: str, mem_id: int, content: str) -> None:
    """记录一条被注入 prompt 的记忆（供反思评估有效性 + 完成后计数）。"""
    if not isinstance(mem_id, int) or not content:
        return
    items = getattr(_local, "injected_memories", None)
    if items is None:
        items = []
        _local.injected_memories = items
    if not any(it["id"] == mem_id for it in items):
        items.append({"id": mem_id, "content": content[:80], "kind": kind})


def clear_injected_tracking() -> None:
    """清空注入追踪（每次请求开始时调用，防止跨请求残留）。"""
    _local.injected_memories = []


def get_injected_memories() -> list[dict[str, Any]]:
    """本次请求注入的记忆清单（反思 demote 判定用）。"""
    return getattr(_local, "injected_memories", []) or []


def finalize_injected_tracking(success: bool) -> None:
    """请求完成后计数（runner 在 done/error 后调用）。

    语义记忆 hit_count++（无论成败，命中即统计）；
    程序记忆 use_count++ 且成功时 success_count++。
    """
    injected = get_injected_memories()
    if not injected:
        return
    semantic_ids = [it["id"] for it in injected if it.get("kind") == "semantic"]
    procedure_ids = [it["id"] for it in injected if it.get("kind") == "procedure"]
    try:
        if semantic_ids:
            from agent.memory.semantic_store import get_semantic_store
            store = get_semantic_store()
            for mid in semantic_ids:
                store.increment_hit(mid)
    except Exception as e:
        logger.debug("[experience] 语义命中计数失败：%s", e)
    try:
        if procedure_ids:
            from agent.memory.procedure_store import get_procedure_store
            store = get_procedure_store()
            for pid in procedure_ids:
                store.record_use(pid, success)
    except Exception as e:
        logger.debug("[experience] 程序使用计数失败：%s", e)


# ============ planner 注入：情景 + 程序 ============

def get_relevant_experiences(query: str) -> str:
    """planner 第 1 轮注入：情景记忆（历史类似情形）+ 程序记忆（推荐方法）。

    返回空串表示无可注入经验。文本结构：
      【历史类似情形】...
      【推荐方法】...
    """
    sections: list[str] = []

    episodes = _collect_episodes(query)
    if episodes:
        lines = []
        for ep in episodes:
            outcome_cn = {"success": "顺利解决", "failure": "当时未解决",
                          "partial": "部分解决"}.get(ep.get("outcome", ""), "")
            date = _date_of(ep.get("happened_at"))
            date_tag = f"{date}，" if date else ""
            lines.append(
                f"  {len(lines) + 1}. 曾遇「{ep.get('event_summary', '')}」（{date_tag}{outcome_cn}），"
                f"当时处理：{ep.get('resolution', '') or '（无记录）'}"
            )
        sections.append("【历史类似情形】\n" + "\n".join(lines))

    procedures = _collect_procedures(query)
    if procedures:
        lines = []
        for proc in procedures:
            date = _date_of(proc.get("updated_at") or proc.get("created_at"))
            date_tag = f"（更新 {date}）" if date else ""
            lines.append(
                f"  {len(lines) + 1}. {proc.get('name', '')}{date_tag}"
                f"——适用：{proc.get('applicability', '')}"
            )
        sections.append("【推荐方法】\n" + "\n".join(lines))

    return "\n".join(sections)


def _collect_episodes(query: str) -> list[dict[str, Any]]:
    """情景记忆检索：向量优先，降级时间倒序。"""
    try:
        from agent.memory.episode_store import get_episode_store
        store = get_episode_store()
        if not store.enabled:
            return []
    except Exception:
        return []

    hits = None
    try:
        from agent.memory import vector_index
        hits = vector_index.search_episodes(query, top_k=_EPISODE_CAP)
    except Exception as e:
        logger.debug("[experience] 情景向量检索失败：%s", e)
        hits = None

    if hits is None:
        # 索引不可用 → 降级时间倒序保底
        rows = store.list_episodes(limit=_EPISODE_CAP)
    elif not hits:
        return []  # 索引可用但无相关：不注入
    else:
        rows = store.get_by_ids([h["id"] for h in hits])
        # 按相似度排序（hits 顺序）
        order = {h["id"]: i for i, h in enumerate(hits)}
        rows.sort(key=lambda r: order.get(r["id"], 999))

    rows = [r for r in rows if _episode_readable(r)]
    for r in rows:
        _record_injected("episode", r.get("id", 0), r.get("event_summary", ""))
    return rows


def _episode_readable(ep: dict[str, Any]) -> bool:
    """读时安全闸：未通过的情景记忆跳过注入（不占 top-k 名额也不计数）。"""
    if _passes_read_gate(ep.get("event_summary", ""), ep.get("resolution", "")):
        return True
    logger.warning("[experience] 情景记忆 id=%s 未通过读时校验，跳过注入", ep.get("id"))
    return False


def _collect_procedures(query: str) -> list[dict[str, Any]]:
    """程序记忆检索：向量优先，降级使用次数倒序。"""
    try:
        from agent.memory.procedure_store import get_procedure_store
        store = get_procedure_store()
        if not store.enabled:
            return []
    except Exception:
        return []

    hits = None
    try:
        from agent.memory import vector_index
        hits = vector_index.search_procedures(query, top_k=_PROCEDURE_CAP)
    except Exception as e:
        logger.debug("[experience] 程序向量检索失败：%s", e)
        hits = None

    if hits is None:
        rows = store.list_procedures(limit=_PROCEDURE_CAP, status="active")
    elif not hits:
        return []
    else:
        rows = store.get_by_ids([h["id"] for h in hits])
        order = {h["id"]: i for i, h in enumerate(hits)}
        rows.sort(key=lambda r: order.get(r["id"], 999))

    rows = [r for r in rows if _procedure_readable(r)]
    for r in rows:
        _record_injected("procedure", r.get("id", 0), r.get("applicability", ""))
    return rows


def _procedure_readable(proc: dict[str, Any]) -> bool:
    """读时安全闸：未通过的程序记忆跳过注入。"""
    if _passes_read_gate(proc.get("name", ""), proc.get("applicability", "")):
        return True
    logger.warning("[experience] 程序记忆 id=%s 未通过读时校验，跳过注入", proc.get("id"))
    return False


# ============ synthesizer 注入：语义记忆 ============

def get_semantic_knowledge(query: str | None) -> str:
    """synthesizer 注入：领域知识（语义记忆）top-3。返回空串表示无可注入知识。"""
    try:
        from agent.memory.semantic_store import get_semantic_store
        store = get_semantic_store()
        if not store.enabled:
            return ""
    except Exception:
        return ""

    hits = None
    if query:
        try:
            from agent.memory import vector_index
            hits = vector_index.search_semantic(query, top_k=_SEMANTIC_CAP)
        except Exception as e:
            logger.debug("[experience] 语义向量检索失败：%s", e)
            hits = None

    if hits is None:
        rows = store.list_semantic(limit=_SEMANTIC_CAP)
    elif not hits:
        return ""
    else:
        rows = store.get_by_ids([h["id"] for h in hits])
        order = {h["id"]: i for i, h in enumerate(hits)}
        rows.sort(key=lambda r: order.get(r["id"], 999))

    filtered = []
    for r in rows:
        if _passes_read_gate(r.get("title", ""), r.get("content", "")):
            filtered.append(r)
        else:
            logger.warning("[experience] 语义记忆 id=%s 未通过读时校验，跳过注入", r.get("id"))
    rows = filtered

    if not rows:
        return ""
    for r in rows:
        _record_injected("semantic", r.get("id", 0), r.get("title", ""))
    lines = []
    for i, r in enumerate(rows, 1):
        date = _date_of(r.get("updated_at") or r.get("created_at"))
        date_tag = f"[{date}] " if date else ""
        lines.append(f"  {i}. {date_tag}{r.get('title', '')}：{r.get('content', '')}")
    return "【已积累领域知识】\n" + "\n".join(lines)
