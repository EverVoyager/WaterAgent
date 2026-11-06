"""情景记忆：完整事件正文和来源引用保存在 Qdrant。"""
from datetime import datetime
from functools import lru_cache
from uuid import uuid4

from agent.memory import vector_index
from agent.memory.usage_store import get_usage_store


class EpisodeStore:
    enabled = True

    def add_episode(self, event_summary: str, resolution: str = "", outcome: str = "partial",
                    happened_at: datetime | str | None = None, source_refs: list[dict] | None = None,
                    event_id: str | None = None) -> int | str:
        stamp = happened_at.isoformat() if isinstance(happened_at, datetime) else happened_at
        return vector_index.put_record("episode", {
            "event_summary": event_summary, "resolution": resolution, "outcome": outcome,
            "happened_at": stamp or vector_index.now_iso(), "source_refs": source_refs or [],
            "event_id": event_id or str(uuid4()),
        })

    def get_by_ids(self, ids: list[int | str]) -> list[dict]:
        return vector_index.get_records("episode", ids)

    def list_episodes(self, limit: int = 100, days_back: int | None = None,
                      outcome: str | None = None) -> list[dict]:
        return vector_index.list_records("episode", limit=limit, days_back=days_back, outcome=outcome)

    def increment_hit(self, memory_id: int | str) -> None:
        get_usage_store().record_hit("episode", memory_id)

    def delete_episode(self, memory_id: int | str) -> bool:
        return vector_index.delete_records("episode", [memory_id]) > 0

    def delete_older_than(self, days: int = 90, limit: int = 200) -> list[int | str]:
        ids = [row["id"] for row in vector_index.iter_records("episode")
               if vector_index.is_older(row["happened_at"], days)][:limit]
        if ids:
            vector_index.delete_records("episode", ids)
        return ids


@lru_cache(maxsize=1)
def get_episode_store() -> EpisodeStore:
    return EpisodeStore()
