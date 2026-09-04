"""Curator：自治记忆策展（五类记忆架构版，借鉴 Hermes Agent v0.12.0 Curator）。

五步治理（周期后台线程执行，默认 7 天）：
1. 剪枝：僵尸语义记忆（零命中超期）+ 超期情景归档 + deprecated 程序清理
2. 压缩：语义记忆 LLM 合并（高命中受保护门控不参与）
3. 提炼：高复用程序的步骤 LLM 泛化（具体案例 → 通用步骤）
4. 晋升：高复用高质量程序 → 自动生成候选 Skill（enabled=false 人工确认）
5. 对账：三个向量 collection 全量同步 + memory/ 目录索引修复

所有动作失败均降级为日志，不影响 Agent 主流程。
"""
import json
import logging
import threading
import time

from agent.memory.memory_store import get_memory_store, is_memory_enabled

logger = logging.getLogger(__name__)

# 剪枝阈值
PRUNE_STALE_DAYS = 14        # 语义记忆：零命中且超 N 天 → 僵尸
EPISODE_ARCHIVE_DAYS = 90    # 情景记忆：超 N 天归档删除
PRUNE_BATCH_LIMIT = 50

# 后台线程状态（防止重复启动）
_started = False
_start_lock = threading.Lock()


def run_curation_once() -> dict[str, int]:
    """执行一轮完整治理。返回统计信息（也用于测试与手动触发）。"""
    stats = {
        "pruned_semantic": 0, "archived_episodes": 0,
        "compacted": 0, "refined": 0, "promoted": 0,
        "indexed": 0, "repaired_index_lines": 0,
    }

    # ===== 1. 剪枝 =====
    if is_memory_enabled():
        try:
            from agent.memory import vector_index
            from agent.memory.semantic_store import get_semantic_store
            sem_store = get_semantic_store()
            if sem_store.enabled:
                stale_ids = sem_store.get_stale_ids(days=PRUNE_STALE_DAYS, limit=PRUNE_BATCH_LIMIT)
                for mid in stale_ids:
                    if sem_store.delete_semantic(mid):
                        stats["pruned_semantic"] += 1
                        vector_index.remove_semantic(mid)
                # 时效性知识过期清理："截至YYYY-MM-DD"超龄的知识删除（含向量）。
                # 反思写入时效性结论时按约定标注基准日期，超龄即失去参考价值
                try:
                    from agent.memory.experience import is_expired_semantic
                    for row in sem_store.list_semantic(limit=1000):
                        mid = row.get("id")
                        expired = mid and is_expired_semantic(str(row.get("content", "")))
                        if expired and sem_store.delete_semantic(mid):
                            stats["pruned_semantic"] += 1
                            vector_index.remove_semantic(mid)
                except Exception as e:
                    logger.debug("[curator] 时效知识清理失败：%s", e)
        except Exception as e:
            logger.warning("[curator] 语义剪枝失败：%s", e)

        try:
            from agent.memory.episode_store import get_episode_store
            ep_store = get_episode_store()
            if ep_store.enabled:
                stats["archived_episodes"] = ep_store.delete_older_than(days=EPISODE_ARCHIVE_DAYS)
        except Exception as e:
            logger.warning("[curator] 情景归档失败：%s", e)

    # ===== 2. 语义记忆 LLM 压缩 =====
    try:
        stats["compacted"] = _compact_semantic()
    except Exception as e:
        logger.warning("[curator] 压缩失败：%s", e)

    # ===== 3. 程序提炼（具体案例 → 通用步骤）=====
    try:
        stats["refined"] = _refine_procedures()
    except Exception as e:
        logger.warning("[curator] 提炼失败：%s", e)

    # ===== 4. 晋升检查（高复用程序 → 候选 Skill）=====
    try:
        stats["promoted"] = _promote_procedures()
    except Exception as e:
        logger.warning("[curator] 晋升失败：%s", e)

    # ===== 5. 索引对账 + 目录治理 =====
    try:
        stats["indexed"] = _reconcile_indexes()
    except Exception as e:
        logger.warning("[curator] 索引对账失败：%s", e)
    try:
        from agent.memory.longterm import repair_index
        stats["repaired_index_lines"] = repair_index()
    except Exception as e:
        logger.debug("[curator] 目录索引修复失败：%s", e)

    # ===== 治理报告写入审计 =====
    try:
        if is_memory_enabled():
            get_memory_store().add_reflection(
                user_query="(curator) 定期记忆治理",
                trigger_reason="curator",
                reflection_text=json.dumps(stats, ensure_ascii=False),
                memories_created=0,
            )
    except Exception as e:
        logger.debug("[curator] 写治理报告失败：%s", e)

    logger.info("[curator] 治理完成：%s", stats)
    return stats


def _compact_semantic() -> int:
    """语义记忆 LLM 合并：低命中条目送压缩，高命中（>=10）受保护。"""
    from agent.memory.reflection import _llm_compact_semantic
    from agent.memory.semantic_store import get_semantic_store

    store = get_semantic_store()
    if not store.enabled:
        return 0
    memories = store.fetch_for_compact(limit=200)
    if len(memories) < 2:
        return 0
    plan = _llm_compact_semantic(memories)
    if not plan:
        return 0

    deleted_ids: list[int] = []
    created = 0
    for item in plan:
        if not isinstance(item, dict):
            continue
        action = item.get("action")
        source_ids = [int(i) for i in (item.get("source_ids") or [])
                      if str(i).isdigit()]
        content = str(item.get("content", "")).strip()
        if action in ("merge", "replace") and content and source_ids:
            # 整合记忆以最新源条目的 title 为题（plan 未输出 title）
            newest_title = next(
                (m.get("title", "") for m in memories if m["id"] == source_ids[-1]),
                "整合知识",
            )
            new_id = store.add_semantic(
                title=newest_title, content=content, source="curator",
                tags=str(item.get("tags") or ""),
            )
            if isinstance(new_id, int):
                created += 1
                deleted_ids.extend(source_ids)
        elif action == "keep":
            continue
    if deleted_ids:
        store.delete_many(deleted_ids)
        from agent.memory import vector_index
        for mid in deleted_ids:
            vector_index.remove_semantic(mid)
    return created


def _refine_procedures() -> int:
    """程序提炼：use_count>=3 且 refined_count<2 的程序，LLM 把步骤泛化为通用方法。"""
    from agent.memory.procedure_store import get_procedure_store
    from agent.prompts.reflection import COMPACT_SYSTEM_PROMPT as _COMPACT
    from agent.utils import parse_json_from_llm
    from app.core.llm import LLM_TIMEOUTS, get_llm_client, get_llm_config, strip_think

    store = get_procedure_store()
    if not store.enabled:
        return 0
    candidates = store.get_refine_candidates()
    if not candidates:
        return 0

    settings = get_llm_config()
    client = get_llm_client().with_options(timeout=LLM_TIMEOUTS["reflector"])
    refined = 0
    for proc in candidates[:5]:  # 每轮最多提炼 5 个，控成本
        try:
            steps = json.loads(proc["steps_json"] or "[]")
        except (json.JSONDecodeError, TypeError):
            steps = []
        user_prompt = (
            f"以下是 Agent 解决「{proc['name']}」类问题的当前步骤记录（源自具体案例）：\n"
            f"{json.dumps(steps, ensure_ascii=False)}\n"
            f"适用条件：{proc['applicability']}\n\n"
            "请把它提炼为通用的解决步骤（去掉具体案例细节，保留可复用的动作序列）。\n"
            '输出严格 JSON：{"steps": [{"step": 1, "action": "动宾短语", "tool": "工具名或null"}], '
            '"applicability": "泛化后的适用条件"}'
        )
        try:
            resp = client.chat.completions.create(
                model=settings["model"],
                messages=[
                    {"role": "system", "content": _COMPACT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=2048,
            )
            result = parse_json_from_llm(
                strip_think((resp.choices[0].message.content or "").strip()))
            if isinstance(result, dict) and result.get("steps") and store.update_steps(
                proc["id"], result["steps"],
                applicability=str(result.get("applicability", "")) or None,
            ):
                refined += 1
        except Exception as e:
            logger.debug("[curator] 单个程序提炼失败 id=%s：%s", proc["id"], e)
    return refined


def _promote_procedures() -> int:
    """晋升检查：use_count>=5 且 success_rate>=0.8 → 生成候选 Skill（enabled=false）。"""
    from agent.memory.procedure_store import get_procedure_store

    store = get_procedure_store()
    if not store.enabled:
        return 0
    promoted = 0
    for proc in store.get_promote_candidates():
        result = store.promote_to_skill(proc["id"], auto_enable=False)
        if result.get("ok"):
            promoted += 1
            logger.info("[curator] 程序晋升为候选 Skill：%s（待人工启用）",
                        result.get("skill_name"))
    return promoted


def _reconcile_indexes() -> int:
    """三个向量 collection 全量同步（MySQL 为 source of truth，含历史回填）。"""
    from agent.memory import vector_index

    total = 0
    if is_memory_enabled():
        try:
            from agent.memory.semantic_store import get_semantic_store
            rows = get_semantic_store().list_semantic(limit=1000)
            total += vector_index.sync_semantic(rows)
        except Exception as e:
            logger.debug("[curator] 语义索引同步失败：%s", e)
        try:
            from agent.memory.episode_store import get_episode_store
            rows = get_episode_store().list_episodes(limit=1000)
            total += vector_index.sync_episodes(rows)
        except Exception as e:
            logger.debug("[curator] 情景索引同步失败：%s", e)
        try:
            from agent.memory.procedure_store import get_procedure_store
            rows = get_procedure_store().list_procedures(limit=1000, status="active")
            total += vector_index.sync_procedures(rows)
        except Exception as e:
            logger.debug("[curator] 程序索引同步失败：%s", e)
    return total


def start_curator_thread() -> None:
    """启动 Curator 后台线程（幂等，进程内只启动一次）。

    调度：启动后 5 分钟执行首轮（快速回填索引），之后按配置周期执行
    （默认 168h = 7 天，对齐 Hermes Curator 的 7-day cycle）。
    """
    global _started
    from app.core.config import get_settings
    settings = get_settings()
    if not getattr(settings, "CURATOR_ENABLED", True):
        logger.debug("[curator] 已禁用（CURATOR_ENABLED=False）")
        return
    if not settings.SELF_EVOLUTION_ENABLED:
        return

    with _start_lock:
        if _started:
            return
        _started = True

    interval = max(1, int(getattr(settings, "CURATOR_INTERVAL_HOURS", 168))) * 3600
    initial_delay = 300  # 首轮延迟 5 分钟：避开服务启动高峰

    def _loop():
        time.sleep(initial_delay)
        while True:
            try:
                run_curation_once()
            except Exception as e:
                # 单轮失败不终止线程，下个周期重试
                logger.warning("[curator] 治理轮次异常（将继续下轮）：%s", e)
            time.sleep(interval)

    t = threading.Thread(target=_loop, daemon=True, name="memory-curator")
    t.start()
    logger.info(
        "[curator] 后台治理线程已启动：首轮延迟 %ds，周期 %dh",
        initial_delay, interval // 3600,
    )
