"""Curator：自治记忆策展（借鉴 Hermes Agent v0.12.0 Curator）。

Hermes Curator 的做法："runs as a background agent on the gateway's cron
ticker (7-day cycle default). It grades your skill library, consolidates
related skills, prunes dead ones, and writes per-run reports."

本实现对应四个动作（周期后台线程执行，默认 7 天）：
1. 剪枝：删除 hit_count=0 且超过 14 天未被命中的僵尸记忆
2. 压缩：各类型记忆 LLM 语义合并（hit_count >= PROTECTED_MIN_HITS 的
   高价值记忆受保护门控，不参与合并）
3. 索引对账：MySQL ↔ Qdrant 向量索引全量同步（含历史数据回填，
   修复"改造前已存在但未索引"的记忆）
4. 报告：治理结果写入 agent_reflections（trigger_reason="curator"）

所有动作失败均降级为日志，不影响 Agent 主流程。
"""
import json
import logging
import threading
import time

from agent.memory.memory_store import (
    MemoryType,
    get_memory_store,
    is_memory_enabled,
)

logger = logging.getLogger(__name__)

# 剪枝阈值：hit_count=0 且创建超过 N 天 → 僵尸记忆（Hermes: prunes dead ones）
PRUNE_STALE_DAYS = 14
# 单次剪枝上限（避免一次删太多，渐进治理）
PRUNE_BATCH_LIMIT = 50

# 后台线程状态（防止重复启动）
_started = False
_start_lock = threading.Lock()


def run_curation_once() -> dict[str, int]:
    """执行一轮完整治理。返回统计信息（也用于测试与手动触发）。"""
    stats = {"pruned": 0, "compacted": 0, "indexed_memories": 0, "indexed_skills": 0}

    if not is_memory_enabled():
        logger.debug("[curator] 记忆模块未启用，跳过治理")
        return stats

    store = get_memory_store()
    if not store.enabled:
        return stats

    # ===== 1. 剪枝僵尸记忆 =====
    try:
        stale_ids = store.get_stale_memory_ids(days=PRUNE_STALE_DAYS, limit=PRUNE_BATCH_LIMIT)
        for mid in stale_ids:
            if store.delete_memory(mid):
                stats["pruned"] += 1
                try:
                    from agent.memory import vector_index
                    vector_index.remove_memory(mid)
                except Exception as e:
                    logger.debug("[curator] 清理向量索引失败 id=%s：%s", mid, e)
        if stale_ids:
            logger.info("[curator] 剪枝僵尸记忆 %d 条（hit_count=0 且超 %d 天）",
                        stats["pruned"], PRUNE_STALE_DAYS)
    except Exception as e:
        logger.warning("[curator] 剪枝失败：%s", e)

    # ===== 2. 各类型 LLM 压缩（高命中记忆受保护门控）=====
    try:
        from agent.memory.reflection import _llm_compact_memories
        for mem_type in MemoryType:
            try:
                deleted = store.compact_memories(mem_type, _llm_compact_memories)
                if deleted > 0:
                    stats["compacted"] += deleted
                    logger.info("[curator] 压缩 type=%s 删除 %d 条", mem_type.value, deleted)
            except Exception as e:
                logger.debug("[curator] 压缩 type=%s 失败：%s", mem_type.value, e)
    except Exception as e:
        logger.warning("[curator] 加载压缩函数失败：%s", e)

    # ===== 3. 向量索引对账（含历史数据回填）=====
    try:
        from agent.memory import vector_index
        for mem_type in MemoryType:
            rows = store.get_memories(memory_type=mem_type, limit=1000)
            indexed = vector_index.sync_memory_type(mem_type.value, rows)
            if indexed > 0:
                stats["indexed_memories"] += indexed
        skill_rows = store.get_all_skills_for_index(limit=500)
        stats["indexed_skills"] = vector_index.sync_skills(skill_rows)
        logger.info("[curator] 索引对账完成：memories=%d skills=%d",
                    stats["indexed_memories"], stats["indexed_skills"])
    except Exception as e:
        logger.warning("[curator] 索引对账失败：%s", e)

    # ===== 4. 治理报告写入反思日志（审计）=====
    try:
        store.add_reflection(
            user_query="(curator) 定期记忆治理",
            trigger_reason="curator",
            reflection_text=json.dumps(stats, ensure_ascii=False),
            memories_created=0,
        )
    except Exception as e:
        logger.debug("[curator] 写治理报告失败：%s", e)

    logger.info("[curator] 治理完成：%s", stats)
    return stats


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
