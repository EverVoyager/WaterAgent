"""情景记忆存储：某一时间段发生的事件及解决方法（五类记忆之四）。

认知科学情景记忆 = 自传体事件。每条反思轨迹产出一个 episode：
发生了什么（event_summary）+ 当时怎么解决的（resolution）+ 结果（outcome）。
planner 注入"历史类似情形的处理方式"，让 Agent 从过往案例中类推。
向量索引：Qdrant agent_episodes_vec（embed event_summary+resolution）。
"""
import json
import logging
import threading
from datetime import datetime
from typing import Any

from agent.memory.base_store import BaseStore

logger = logging.getLogger(__name__)

_CREATE_EPISODES_TABLE = """
CREATE TABLE IF NOT EXISTS agent_episodes (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    happened_at DATETIME NOT NULL COMMENT '事件发生时间',
    event_summary VARCHAR(512) NOT NULL COMMENT '发生了什么事',
    resolution TEXT COMMENT '当时的解决方法',
    outcome ENUM('success', 'failure', 'partial') NOT NULL DEFAULT 'success' COMMENT '结果',
    query_summary VARCHAR(512) COMMENT '触发事件的原始查询摘要',
    tool_calls_json TEXT COMMENT '当时的工具调用链 JSON',
    tags VARCHAR(256) COMMENT '标签（逗号分隔）',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_happened (happened_at),
    INDEX idx_outcome (outcome)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='情景记忆（事件与解法）'
"""


class EpisodeStore(BaseStore):
    """情景记忆 CRUD + 检索。"""

    _create_sqls = [_CREATE_EPISODES_TABLE]

    def add_episode(self, event_summary: str, resolution: str = "",
                    outcome: str = "success", query_summary: str = "",
                    tool_calls: list[dict[str, Any]] | None = None,
                    tags: str = "", happened_at: datetime | None = None) -> int | None:
        """写入一条情景记忆，返回新 id（失败 None）。"""
        self._ensure_tables()
        try:
            with self._get_conn() as conn, conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO agent_episodes "
                    "(happened_at, event_summary, resolution, outcome, query_summary, "
                    "tool_calls_json, tags) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    (
                        happened_at or datetime.now(),
                        event_summary[:512], resolution, outcome, query_summary[:512],
                        json.dumps(tool_calls, ensure_ascii=False) if tool_calls else None,
                        tags[:256],
                    ),
                )
                return cur.lastrowid
        except Exception as e:
            logger.warning("[episode] 写入失败：%s", e)
            return None

    def get_by_ids(self, ids: list[int]) -> list[dict[str, Any]]:
        if not ids:
            return []
        self._ensure_tables()
        ph = ",".join(["%s"] * len(ids))
        try:
            with self._get_conn() as conn, conn.cursor() as cur:
                cur.execute(
                    f"SELECT * FROM agent_episodes WHERE id IN ({ph})", ids,
                )
                return cur.fetchall()
        except Exception as e:
            logger.warning("[episode] 批量查询失败：%s", e)
            return []

    def list_episodes(self, limit: int = 100, days_back: int | None = None,
                      outcome: str | None = None) -> list[dict[str, Any]]:
        """列表（时间倒序），供治理 API 与 Curator。"""
        self._ensure_tables()
        sql = "SELECT * FROM agent_episodes WHERE 1=1"
        params: list[Any] = []
        if days_back:
            sql += " AND happened_at >= DATE_SUB(NOW(), INTERVAL %s DAY)"
            params.append(days_back)
        if outcome:
            sql += " AND outcome = %s"
            params.append(outcome)
        sql += " ORDER BY happened_at DESC LIMIT %s"
        params.append(limit)
        try:
            with self._get_conn() as conn, conn.cursor() as cur:
                cur.execute(sql, params)
                return cur.fetchall()
        except Exception as e:
            logger.warning("[episode] 列表失败：%s", e)
            return []

    def delete_episode(self, episode_id: int) -> bool:
        self._ensure_tables()
        try:
            with self._get_conn() as conn, conn.cursor() as cur:
                cur.execute("DELETE FROM agent_episodes WHERE id = %s", (episode_id,))
                return cur.rowcount > 0
        except Exception as e:
            logger.warning("[episode] 删除失败：%s", e)
            return False

    def delete_older_than(self, days: int = 90, limit: int = 200) -> int:
        """归档剪枝：删除超期情景（Curator 用），返回删除条数。"""
        self._ensure_tables()
        try:
            with self._get_conn() as conn, conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM agent_episodes WHERE happened_at < "
                    "DATE_SUB(NOW(), INTERVAL %s DAY) LIMIT %s",
                    (days, limit),
                )
                return cur.rowcount
        except Exception as e:
            logger.warning("[episode] 归档失败：%s", e)
            return 0


_store: EpisodeStore | None = None
_store_lock = threading.Lock()


def get_episode_store() -> EpisodeStore:
    """单例 EpisodeStore。"""
    global _store
    with _store_lock:
        if _store is None:
            from app.core.config import get_settings
            s = get_settings()
            _store = EpisodeStore(
                host=s.MYSQL_HOST, port=s.MYSQL_PORT, user=s.MYSQL_USER,
                password=s.MYSQL_PASSWORD, database=s.MYSQL_DATABASE,
            )
        return _store
