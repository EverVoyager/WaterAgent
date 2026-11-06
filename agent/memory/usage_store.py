"""MySQL 仅保存三类记忆的使用统计；故障不会阻断 Qdrant 正文操作。"""
import logging
from functools import lru_cache

from agent.memory.base_store import BaseStore

logger = logging.getLogger(__name__)

_CREATE_USAGE = """
CREATE TABLE IF NOT EXISTS agent_memory_usage (
    memory_type VARCHAR(16) NOT NULL,
    memory_id VARCHAR(36) NOT NULL,
    hit_count BIGINT NOT NULL DEFAULT 0,
    success_count BIGINT NOT NULL DEFAULT 0,
    refined_count INT NOT NULL DEFAULT 0,
    last_hit_at DATETIME NULL,
    PRIMARY KEY (memory_type, memory_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='记忆使用统计，正文保存在Qdrant'
"""


class UsageStore(BaseStore):
    _create_sqls = [_CREATE_USAGE]

    def get_stats(self, kind: str, ids: list[int | str]) -> dict[str, dict] | None:
        if not self.enabled:
            return None
        if not ids:
            return {}
        try:
            self._ensure_tables()
            placeholders = ",".join(["%s"] * len(ids))
            with self._get_conn() as conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT memory_id, hit_count, success_count, refined_count, last_hit_at "
                    f"FROM agent_memory_usage WHERE memory_type = %s AND memory_id IN ({placeholders})",
                    [kind, *(str(mid) for mid in ids)],
                )
                return {row["memory_id"]: row for row in cur.fetchall()}
        except Exception as e:
            logger.warning("[memory-usage] 统计读取不可用：%s", type(e).__name__)
            return None

    def record_hit(self, kind: str, memory_id: int | str, success: bool = False) -> None:
        if not self.enabled:
            return
        try:
            self._ensure_tables()
            with self._get_conn() as conn, conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO agent_memory_usage "
                    "(memory_type, memory_id, hit_count, success_count, last_hit_at) "
                    "VALUES (%s, %s, 1, %s, CURRENT_TIMESTAMP) ON DUPLICATE KEY UPDATE "
                    "hit_count = hit_count + 1, success_count = success_count + VALUES(success_count), "
                    "last_hit_at = CURRENT_TIMESTAMP",
                    (kind, str(memory_id), int(success)),
                )
        except Exception as e:
            logger.warning("[memory-usage] 命中统计未保存：%s", type(e).__name__)

    def record_refinement(self, memory_id: int | str) -> None:
        if not self.enabled:
            return
        try:
            self._ensure_tables()
            with self._get_conn() as conn, conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO agent_memory_usage (memory_type, memory_id, refined_count) "
                    "VALUES ('procedure', %s, 1) ON DUPLICATE KEY UPDATE refined_count = refined_count + 1",
                    (str(memory_id),),
                )
        except Exception as e:
            logger.warning("[memory-usage] 提炼统计未保存：%s", type(e).__name__)

    def import_stats(self, kind: str, memory_id: int | str, stats: dict) -> None:
        """迁移要求确认成功；重跑取较大值，不重置线上累积统计。"""
        self._ensure_tables()
        counts = [max(0, int(stats.get(key) or 0))
                  for key in ("hit_count", "success_count", "refined_count")]
        with self._get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO agent_memory_usage "
                "(memory_type, memory_id, hit_count, success_count, refined_count) "
                "VALUES (%s, %s, %s, %s, %s) ON DUPLICATE KEY UPDATE "
                "hit_count = GREATEST(hit_count, VALUES(hit_count)), "
                "success_count = GREATEST(success_count, VALUES(success_count)), "
                "refined_count = GREATEST(refined_count, VALUES(refined_count))",
                (kind, str(memory_id), *counts),
            )


@lru_cache(maxsize=1)
def get_usage_store() -> UsageStore:
    from app.core.config import get_settings

    s = get_settings()
    return UsageStore(host=s.MYSQL_HOST, port=s.MYSQL_PORT, user=s.MYSQL_USER,
                      password=s.MYSQL_PASSWORD, database=s.MYSQL_DATABASE)


def with_usage(kind: str, rows: list[dict]) -> list[dict]:
    stats = get_usage_store().get_stats(kind, [row["id"] for row in rows])
    for row in rows:
        values = stats.get(str(row["id"]), {}) if stats is not None else {}
        for key in ("hit_count", "success_count", "refined_count"):
            row[key] = int(values.get(key, 0)) if stats is not None else None
    return rows
