"""语义记忆正文仅存 Qdrant，MySQL 仅保存命中统计。"""
from functools import lru_cache

from agent.memory import vector_index
from agent.memory.usage_store import get_usage_store, with_usage


class SemanticStore:
    enabled = True

    def add_semantic(self, title: str, content: str, reuse_when: str = "",
                     evidence: list[dict] | None = None) -> int | str:
        return vector_index.put_record("semantic", {
            "title": title, "content": content, "reuse_when": reuse_when,
            "evidence": evidence or [],
        })

    def list_semantic(self, limit: int = 100, days_back: int | None = None) -> list[dict]:
        return vector_index.list_records("semantic", limit=limit, days_back=days_back)

    def get_by_ids(self, ids: list[int | str]) -> list[dict]:
        return vector_index.get_records("semantic", ids)

    def increment_hit(self, memory_id: int | str) -> None:
        get_usage_store().record_hit("semantic", memory_id)

    def delete_semantic(self, memory_id: int | str) -> bool:
        return vector_index.delete_records("semantic", [memory_id]) > 0

    def get_stale_ids(self, days: int = 14, limit: int = 50) -> list[int | str]:
        rows = with_usage("semantic", list(vector_index.iter_records("semantic")))
        return [row["id"] for row in rows
                if row["hit_count"] == 0 and vector_index.is_older(row["created_at"], days)][:limit]

    def fetch_for_compact(self, limit: int = 200) -> list[dict]:
        rows = with_usage("semantic", list(vector_index.iter_records("semantic")))
        rows = [row for row in rows if row["hit_count"] is not None and row["hit_count"] < 10]
        rows.sort(key=lambda row: row["created_at"])
        return rows[:limit]

    def delete_many(self, ids: list[int | str]) -> int:
        return vector_index.delete_records("semantic", ids)


@lru_cache(maxsize=1)
def get_semantic_store() -> SemanticStore:
    return SemanticStore()


def is_semantic_enabled() -> bool:
    return get_semantic_store().enabled
