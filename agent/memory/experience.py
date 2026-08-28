"""经验检索与注入（Hermes 范式的应用层）。

将 MySQL 中积累的记忆/技能检索出来，注入到对应节点的 prompt 中：
- planner_node：注入相关技能（成功工具调用模式）+ 工具失败教训
- synthesizer_node：注入用户偏好 + 领域知识

注入策略（P1 改造，借鉴 Generative Agents / Letta / Reflexion）：
- 控制长度（避免 prompt 膨胀）：每次最多 3 条技能 + 3 条偏好
- 语义检索（P2 改造，借鉴 Hermes FTS5 / Letta archival memory）：
  技能与偏好均优先走 Qdrant 向量检索，按与当前 query 的相关性选取，
  替代时间倒序盲注；索引不可用时降级回 LIKE/时间序
- 增量命中计数：被注入的记忆自动 hit_count + 1
- 效果闭环：注入的记忆 id+内容记录到线程本地，
  供反思时评估"注入后是否仍被纠正"（无效记忆降权）
- TTL 过滤：tool_failure 只注入最近 30 天内的记录（recency 衰减）
- 降权威：tool_failure 标题改为"历史故障记录（可能已修复）"
- 自愈校验：注入前检查 tool_failure 是否可证伪地过期，过期则删除
"""
import json
import logging
import re
import threading

from agent.memory.memory_store import (
    MemoryStore,
    MemoryType,
    get_memory_store,
    is_memory_enabled,
)

logger = logging.getLogger(__name__)

# tool_failure TTL：只注入最近 N 天内的故障记录
# 借鉴 Generative Agents 的 recency 因子：时间越久越可能已修复
TOOL_FAILURE_TTL_DAYS = 30

# 语义注入的各类型条数上限（与旧时间序路径保持一致）
_PREF_CAP = 3
_CORRECTION_CAP = 2
_KNOWLEDGE_CAP = 5


# ============ 注入追踪（效果闭环，借鉴 GEPA 行为→评估→优化）============

_local = threading.local()
"""线程本地：本次请求注入了哪些记忆。

Agent 的一次请求（planner→executor→synthesizer）在同一 worker 线程内顺序执行，
反思也在该线程触发，因此 thread-local 可精确传递"本次注入的记忆"。
"""


def _record_injected(mem_id, content: str) -> None:
    """记录一条被注入 prompt 的记忆（供反思评估有效性）。"""
    if not isinstance(mem_id, int) or not content:
        return
    items = getattr(_local, "injected_memories", None)
    if items is None:
        items = []
        _local.injected_memories = items
    if not any(it["id"] == mem_id for it in items):
        items.append({"id": mem_id, "content": content[:80]})


def clear_injected_tracking() -> None:
    """清空注入追踪（每次请求开始时调用，防止跨请求残留）。"""
    _local.injected_memories = []


def get_injected_memories() -> list[dict]:
    """取本次请求已注入的记忆列表 [{"id", "content"}]。"""
    return list(getattr(_local, "injected_memories", None) or [])


# ============ 语义检索（降级链：向量 → LIKE/时间序）============

def _search_skills_semantic(
    store: MemoryStore, query: str, limit: int
) -> tuple[list[dict] | None, bool]:
    """向量检索技能。

    Returns:
        (rows, used_semantic)。rows=None 表示索引不可用（调用方应降级）；
        rows=[] 表示索引可用但无相关技能（不降级，避免注入无关经验）。
    """
    from agent.memory import vector_index
    hits = vector_index.search_skills(query, top_k=limit)
    if hits is None:
        return None, False
    if not hits:
        return [], True
    rows = store.get_skills_by_ids([h["id"] for h in hits])
    by_id = {r["id"]: r for r in rows}
    # 保持语义相关性排序（hits 顺序）
    return [by_id[h["id"]] for h in hits if h["id"] in by_id], True


def get_relevant_experiences(query: str, limit: int = 3) -> str:
    """检索与当前 query 相关的经验，格式化为可注入 prompt 的文本。

    包含：
    - 相关技能（成功的工具调用模式，优先向量语义检索）
    - 工具失败教训（避免重复犯错）

    用于 planner_node 注入。返回空字符串表示无经验或未启用。
    """
    if not is_memory_enabled():
        return ""

    store = get_memory_store()
    parts: list[str] = []

    # 1. 相关技能（成功的工具调用模式）
    #    语义检索优先；索引不可用时降级 LIKE（中文 query split 无效的老问题随之消除）
    try:
        skills, _used_semantic = _search_skills_semantic(store, query, limit)
    except Exception as e:
        logger.debug("[experience] 语义检索技能失败：%s", e)
        skills = None
    if skills is None:
        skills = store.get_relevant_skills(query, limit=limit)
    if skills:
        parts.append("【过往成功经验】")
        for i, s in enumerate(skills, 1):
            tool_calls = s.get("tool_calls", [])
            tool_names = [tc.get("name", "") for tc in tool_calls]
            rounds_used = s.get("rounds_used", 1)
            use_count = s.get("use_count", 1)
            parts.append(
                f"  {i}. 类似查询「{s.get('query_pattern', '')[:40]}」曾用 "
                f"[{', '.join(tool_names)}] 在 {rounds_used} 轮内解决（已复用 {use_count} 次）"
            )

    # 2. 工具失败教训（P1 改造：TTL + 降权威 + 自愈校验）
    try:
        failures = store.get_memories(
            memory_type=MemoryType.TOOL_FAILURE,
            limit=3,
            days_back=TOOL_FAILURE_TTL_DAYS,  # TTL：只取最近 30 天
        )
        if failures:
            # 自愈校验：删除可证伪地过期的记录
            failures = _falsify_tool_failures(store, failures)
            if failures:
                # 降权威：标题从"工具失败教训（避免重复）"改为"历史故障记录（可能已修复）"
                parts.append("【历史故障记录（可能已修复，仅供参考）】")
                parts.append(
                    "  注意：以下为历史故障记录，可能已修复。"
                    "若相关工具当前可用，应正常尝试调用，不要被旧记录阻止。"
                )
                for i, f in enumerate(failures, 1):
                    content = f.get("content", "")[:80]
                    parts.append(f"  {i}. {content}")
                    # 命中计数 +1 + 注入追踪（效果闭环）
                    if f.get("id"):
                        store.increment_hit(f["id"])
                        _record_injected(f["id"], content)
    except Exception as e:
        logger.debug("[experience] 检索失败教训失败：%s", e)

    return "\n".join(parts) if parts else ""


def get_user_preferences(query: str | None = None) -> str:
    """检索用户偏好和领域知识，格式化为可注入 prompt 的文本。

    Args:
        query: 当前用户查询。提供时走向量语义检索（只注入与 query 相关的
               偏好/知识），不提供时回退时间倒序（旧行为，兼容）。

    用于 synthesizer_node 注入。返回空字符串表示无相关偏好或未启用。
    """
    if not is_memory_enabled():
        return ""

    store = get_memory_store()
    parts: list[str] = []

    if query:
        # ===== 语义检索路径：与当前 query 相关的偏好/纠正/知识 =====
        hits = None
        try:
            from agent.memory import vector_index
            hits = vector_index.search_memories(
                query,
                memory_types=[
                    MemoryType.USER_PREFERENCE.value,
                    MemoryType.USER_CORRECTION.value,
                    MemoryType.DOMAIN_KNOWLEDGE.value,
                ],
                top_k=_PREF_CAP + _CORRECTION_CAP + _KNOWLEDGE_CAP,
            )
        except Exception as e:
            logger.debug("[experience] 语义检索偏好失败：%s", e)
            hits = None

        if hits is not None:
            if not hits:
                # 索引可用但无语义相关记忆：不注入（替代旧的时间序盲注，
                # 消除"问太原天气却注入吴堡站水位"）
                return ""
            _format_semantic_preferences(store, hits, parts)
            return "\n".join(parts) if parts else ""
        # hits is None → 索引不可用，降级到下方时间序路径

    # ===== 时间序路径（索引不可用或未传 query 时的降级行为）=====
    _format_preferences_by_time(store, parts)
    return "\n".join(parts) if parts else ""


def _format_semantic_preferences(store: MemoryStore, hits: list[dict], parts: list[str]) -> None:
    """按语义相关性分组格式化偏好/纠正/知识（保持各类型条数上限）。"""
    prefs: list[dict] = []
    corrections: list[dict] = []
    knowledge: list[dict] = []
    for h in hits:
        t = h.get("memory_type", "")
        if t == MemoryType.USER_PREFERENCE.value and len(prefs) < _PREF_CAP:
            prefs.append(h)
        elif t == MemoryType.USER_CORRECTION.value and len(corrections) < _CORRECTION_CAP:
            corrections.append(h)
        elif t == MemoryType.DOMAIN_KNOWLEDGE.value and len(knowledge) < _KNOWLEDGE_CAP:
            knowledge.append(h)

    if prefs:
        parts.append("【用户偏好】")
        for i, p in enumerate(prefs, 1):
            content = (p.get("content") or "")[:80]
            parts.append(f"  {i}. {content}")
            if isinstance(p.get("id"), int):
                store.increment_hit(p["id"])
                _record_injected(p["id"], content)

    if corrections:
        parts.append("【历史纠正（避免再犯）】")
        for i, c in enumerate(corrections, 1):
            content = (c.get("content") or "")[:80]
            parts.append(f"  {i}. {content}")
            if isinstance(c.get("id"), int):
                store.increment_hit(c["id"])
                _record_injected(c["id"], content)

    if knowledge:
        parts.append("【已积累领域知识】")
        for i, k in enumerate(knowledge, 1):
            content = (k.get("content") or "")[:100]
            parts.append(f"  {i}. {content}")
            if isinstance(k.get("id"), int):
                store.increment_hit(k["id"])
                _record_injected(k["id"], content)


def _format_preferences_by_time(store: MemoryStore, parts: list[str]) -> None:
    """时间倒序格式化（索引不可用时的降级行为，与旧版一致）。"""
    # 1. 用户偏好
    try:
        prefs = store.get_memories(memory_type=MemoryType.USER_PREFERENCE, limit=_PREF_CAP)
        if prefs:
            parts.append("【用户偏好】")
            for i, p in enumerate(prefs, 1):
                content = p.get("content", "")[:80]
                parts.append(f"  {i}. {content}")
                if p.get("id"):
                    store.increment_hit(p["id"])
                    _record_injected(p["id"], content)
    except Exception as e:
        logger.debug("[experience] 检索偏好失败：%s", e)

    # 2. 用户纠正（曾被纠正过的错误，避免再犯）
    try:
        corrections = store.get_memories(memory_type=MemoryType.USER_CORRECTION, limit=_CORRECTION_CAP)
        if corrections:
            parts.append("【历史纠正（避免再犯）】")
            for i, c in enumerate(corrections, 1):
                content = c.get("content", "")[:80]
                parts.append(f"  {i}. {content}")
                if c.get("id"):
                    store.increment_hit(c["id"])
                    _record_injected(c["id"], content)
    except Exception as e:
        logger.debug("[experience] 检索纠正失败：%s", e)

    # 3. 领域知识
    try:
        knowledge = store.get_memories(memory_type=MemoryType.DOMAIN_KNOWLEDGE, limit=_KNOWLEDGE_CAP)
        if knowledge:
            parts.append("【已积累领域知识】")
            for i, k in enumerate(knowledge, 1):
                content = k.get("content", "")[:100]
                parts.append(f"  {i}. {content}")
                if k.get("id"):
                    store.increment_hit(k["id"])
                    _record_injected(k["id"], content)
    except Exception as e:
        logger.debug("[experience] 检索领域知识失败：%s", e)


# ============ 自愈校验（P1-③）============

def _falsify_tool_failures(
    store: MemoryStore,
    failures: list[dict],
) -> list[dict]:
    """对 tool_failure 记录做可证伪性校验，删除已过期的记录。

    借鉴 Reflexion 的"反思只描述具体错误"和 Letta 的 self-edit：
    - 若记录声称"工具 X 不存在/Unknown tool"，而 X 现在已注册 → 该记录可证伪地过期，删除
    - 若记录声称"工具 X 返回空/失败"，X 已注册但当前是否可调用无法静态判断 → 保留

    Args:
        store: MemoryStore 实例
        failures: 待校验的 tool_failure 记录列表

    Returns:
        通过校验的记录列表（可能比输入少，过期的已删除）
    """
    if not failures:
        return failures

    try:
        from agent.tools.schemas import TOOL_PARAM_MODELS
        registered_tools: set[str] = set(TOOL_PARAM_MODELS.keys())
    except Exception as e:
        logger.debug("[experience] 无法读取已注册工具列表，跳过自愈校验：%s", e)
        return failures

    # 匹配多种"工具不存在/未注册"的表述模式
    # 工具名通常为 snake_case，正则提取
    unknown_tool_patterns = [
        # "调用 X 返回 Unknown tool" / "调用 X 报 Unknown tool"
        re.compile(
            r"调用\s+([a-z_][a-z0-9_]*)\s+(?:返回|报|抛出|显示).*(?:Unknown\s*tool|不存在|未注册)",
            re.IGNORECASE,
        ),
        # "Unknown tool: X" / "Unknown tool：X"
        re.compile(
            r"Unknown\s*tool[:：]\s*([a-z_][a-z0-9_]*)",
            re.IGNORECASE,
        ),
        # "X 不存在" / "X 未注册" / "X not registered"
        re.compile(
            r"\b([a-z_][a-z0-9_]*)\s*(?:不存在|未注册|not\s+registered)",
            re.IGNORECASE,
        ),
        # falsifiable_check: "检查 X 是否在/注册"
        re.compile(
            r"检查\s+([a-z_][a-z0-9_]*)\s*(?:是否|是不是).*(?:注册|存在|TOOL_PARAM)",
            re.IGNORECASE,
        ),
    ]

    kept: list[dict] = []
    deleted_count = 0

    for f in failures:
        content = f.get("content", "") or ""
        # 也检查 context 中的 falsifiable_check
        context_str = ""
        try:
            ctx = f.get("context")
            if ctx:
                if isinstance(ctx, str):
                    # 尝试解析 JSON，避免 unicode 转义导致中文不匹配
                    try:
                        ctx_parsed = json.loads(ctx)
                        if isinstance(ctx_parsed, dict):
                            context_str = ctx_parsed.get("falsifiable_check", "")
                        else:
                            context_str = ctx
                    except (json.JSONDecodeError, TypeError):
                        context_str = ctx
                elif isinstance(ctx, dict):
                    context_str = ctx.get("falsifiable_check", "")
                else:
                    context_str = json.dumps(ctx, ensure_ascii=False)
        except Exception:
            context_str = ""

        combined = content + " " + str(context_str)

        # 尝试所有模式，提取声称不存在的工具名
        claimed_tool = None
        for pattern in unknown_tool_patterns:
            match = pattern.search(combined)
            if match:
                claimed_tool = match.group(1)
                break

        if claimed_tool and claimed_tool in registered_tools:
            # 可证伪地过期：工具现已注册，故障已修复
            logger.info(
                "[experience] 自愈校验：删除过期 tool_failure id=%s "
                "（工具 %s 现已注册）",
                f.get("id"), claimed_tool,
            )
            if f.get("id"):
                store.delete_memory(f["id"])
                # 同步清理向量索引，防止僵尸点继续被检索命中
                try:
                    from agent.memory import vector_index
                    vector_index.remove_memory(f["id"])
                except Exception as idx_err:
                    logger.debug("[experience] 清理记忆向量索引失败：%s", idx_err)
            deleted_count += 1
            continue

        kept.append(f)

    if deleted_count > 0:
        logger.info(
            "[experience] 自愈校验完成：删除 %d 条过期 tool_failure，保留 %d 条",
            deleted_count, len(kept),
        )

    return kept
