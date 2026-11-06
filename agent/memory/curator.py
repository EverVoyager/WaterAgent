"""记忆治理：Qdrant 正文清理/提炼，MySQL 计数和审计；不再覆盖重建主集合。"""
import json
import logging
import threading
import time

from agent.memory.memory_store import get_memory_store, is_memory_enabled

logger = logging.getLogger(__name__)

PRUNE_STALE_DAYS = 14
EPISODE_ARCHIVE_DAYS = 90
PRUNE_BATCH_LIMIT = 50
_started = False
_start_lock = threading.Lock()


def _prune_semantic() -> int:
    from agent.memory.semantic_store import get_semantic_store

    store = get_semantic_store()
    return store.delete_many(store.get_stale_ids(days=PRUNE_STALE_DAYS, limit=PRUNE_BATCH_LIMIT))


def _archive_episodes() -> int:
    from agent.memory.episode_store import get_episode_store

    return len(get_episode_store().delete_older_than(days=EPISODE_ARCHIVE_DAYS))


def run_curation_once() -> dict[str, int]:
    from agent.memory.archive_storage import cleanup_archive
    from agent.memory.longterm import repair_index

    actions = {
        "pruned_semantic": _prune_semantic,
        "archived_episodes": _archive_episodes,
        "archive_cleaned": cleanup_archive,
        "compacted": _compact_semantic,
        "refined": _refine_procedures,
        "promoted": _promote_procedures,
        "repaired_index_lines": repair_index,
    }
    stats = {}
    for name, action in actions.items():
        try:
            stats[name] = action()
        except Exception as e:
            stats[name] = 0
            logger.warning("[curator] %s 未完成：%s", name, type(e).__name__)
    if is_memory_enabled():
        try:
            get_memory_store().add_reflection(
                user_query="(curator) 定期记忆治理", trigger_reason="curator",
                reflection_text=json.dumps(stats, ensure_ascii=False), memories_created=0,
            )
        except Exception as e:
            logger.warning("[curator] 审计未保存：%s", type(e).__name__)
    return stats


def _compact_semantic() -> int:
    """仅合并完整正文/条件相同、有证据且统计可用的重复项。"""
    from agent.memory.reflection_llm import _llm_compact_semantic
    from agent.memory.semantic_store import get_semantic_store
    from agent.memory.usage_store import get_usage_store

    store = get_semantic_store()
    memories = store.fetch_for_compact(limit=200)
    if len(memories) < 2:
        return 0
    plan = _llm_compact_semantic(memories)
    if not plan:
        return 0
    by_id = {m["id"]: m for m in memories}
    seen = set()
    for item in plan:
        if not isinstance(item, dict) or item.get("action") not in ("keep", "merge", "replace"):
            return 0
        ids = item.get("source_ids")
        if (not isinstance(ids, list) or not ids
                or any(type(mid) not in (int, str) or mid not in by_id for mid in ids)
                or len(set(ids)) != len(ids) or seen.intersection(ids)):
            logger.warning("[curator] 无效或重叠的源 ID，保留原记忆")
            return 0
        seen.update(ids)
    if seen != set(by_id):
        return 0
    created = 0
    for item in plan:
        if item["action"] == "keep" or len(item["source_ids"]) < 2:
            continue
        originals = [by_id[mid] for mid in item["source_ids"]]
        first = originals[0]
        if not first.get("reuse_when") or not first.get("evidence"):
            continue
        if item.get("content") != first["content"]:
            continue
        if any(m["title"] != first["title"] or m["content"] != first["content"]
               or m.get("reuse_when") != first["reuse_when"] or not m.get("evidence")
               for m in originals):
            continue
        evidence = []
        for memory in originals:
            for citation in memory["evidence"]:
                if citation not in evidence:
                    evidence.append(citation)
        if len(evidence) > 50:
            continue
        new_id = store.add_semantic(first["title"], first["content"],
                                    reuse_when=first["reuse_when"], evidence=evidence)
        # 先确认正文和合并后的计数保存，再删除源记录；任一失败都保留源记录。
        get_usage_store().import_stats("semantic", new_id, {
            "hit_count": sum(m["hit_count"] for m in originals),
        })
        store.delete_many(item["source_ids"])
        created += 1
    return created


def _refine_procedures() -> int:
    from agent.memory.procedure_store import get_procedure_store
    from agent.utils import parse_json_from_llm
    from app.core.llm import LLM_TIMEOUTS, get_llm_client, get_llm_config, strip_think

    store = get_procedure_store()
    candidates = store.get_refine_candidates()
    if not candidates:
        return 0
    client = get_llm_client().with_options(timeout=LLM_TIMEOUTS["reflector"])
    refined = 0
    for proc in candidates[:5]:
        prompt = (
            f"整理「{proc['name']}」的处理步骤，保留对象、单位、版本与例外，不扩大适用范围。\n"
            f"步骤：{json.dumps(proc['steps'], ensure_ascii=False)}\n"
            f"适用条件：{proc['applicability']}\n"
            '输出 JSON：{"steps":[{"step":1,"action":"动作","tool":null}],"applicability":"适用条件"}'
        )
        try:
            response = client.chat.completions.create(
                model=get_llm_config()["model"],
                messages=[
                    {"role": "system", "content": "整理已有方法，不能编造事实或扩大权限，输出严格 JSON。"},
                    {"role": "user", "content": prompt},
                ], temperature=0.1, max_tokens=2048,
            )
            result = parse_json_from_llm(strip_think(response.choices[0].message.content or ""))
            if (isinstance(result, dict) and result.get("steps")
                    and store.update_steps(proc["id"], result["steps"], result.get("applicability") or None)):
                refined += 1
        except Exception as e:
            logger.warning("[curator] 程序提炼未完成 id=%s：%s", proc["id"], type(e).__name__)
    return refined


def _promote_procedures() -> int:
    from agent.memory.procedure_store import get_procedure_store

    store = get_procedure_store()
    return sum(bool(store.promote_to_skill(proc["id"], auto_enable=False).get("ok"))
               for proc in store.get_promote_candidates())


def start_curator_thread() -> None:
    """启动 Curator 后台线程（幂等，进程内只启动一次）。

    调度：启动后 5 分钟执行首轮，之后按配置周期执行
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
