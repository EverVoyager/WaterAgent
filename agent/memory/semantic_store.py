"""语义记忆存储：领域知识 / 文档片段（五类记忆之三）。

认知科学语义记忆 = 事实性知识。区别于长期记忆（Agent 核心认知，全量常驻注入）：
语义记忆条目多、需按查询相关性检索注入（synthesizer 用）。
向量索引：Qdrant agent_semantic_vec（embed title+content），不可用时降级时间倒序。
"""
import logging
import threading
from typing import Any

from agent.memory.base_store import BaseStore

logger = logging.getLogger(__name__)

_CREATE_SEMANTIC_TABLE = """
CREATE TABLE IF NOT EXISTS agent_semantic (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    title VARCHAR(256) NOT NULL COMMENT '知识点标题',
    content TEXT NOT NULL COMMENT '知识正文',
    source VARCHAR(32) NOT NULL DEFAULT 'reflection' COMMENT '来源：reflection/curator/manual',
    tags VARCHAR(256) COMMENT '标签（逗号分隔）',
    hit_count INT NOT NULL DEFAULT 0 COMMENT '被注入次数',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_created (created_at),
    INDEX idx_tags (tags(64))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='语义记忆（领域知识）'
"""


class SemanticStore(BaseStore):
    """语义记忆 CRUD + 检索。"""

    _create_sqls = [_CREATE_SEMANTIC_TABLE]

    def add_semantic(self, title: str, content: str, source: str = "reflection",
                     tags: str = "") -> int | None:
        """写入一条语义记忆，返回新 id（失败 None）。"""
        self._ensure_tables()
        try:
            with self._get_conn() as conn, conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO agent_semantic (title, content, source, tags) "
                    "VALUES (%s, %s, %s, %s)",
                    (title[:256], content, source, tags[:256]),
                )
                return cur.lastrowid
        except Exception as e:
            logger.warning("[semantic] 写入失败：%s", e)
            return None

    def list_semantic(self, limit: int = 100, days_back: int | None = None) -> list[dict[str, Any]]:
        """列表（时间倒序）。"""
        self._ensure_tables()
        sql = "SELECT id, title, content, source, tags, hit_count, created_at, updated_at " \
              "FROM agent_semantic"
        params: list[Any] = []
        if days_back:
            sql += " WHERE created_at >= DATE_SUB(NOW(), INTERVAL %s DAY)"
            params.append(days_back)
        sql += " ORDER BY created_at DESC LIMIT %s"
        params.append(limit)
        try:
            with self._get_conn() as conn, conn.cursor() as cur:
                cur.execute(sql, params)
                return cur.fetchall()
        except Exception as e:
            logger.warning("[semantic] 列表失败：%s", e)
            return []

    def get_by_ids(self, ids: list[int]) -> list[dict[str, Any]]:
        if not ids:
            return []
        self._ensure_tables()
        ph = ",".join(["%s"] * len(ids))
        try:
            with self._get_conn() as conn, conn.cursor() as cur:
                cur.execute(
                    f"SELECT id, title, content FROM agent_semantic WHERE id IN ({ph})",
                    ids,
                )
                return cur.fetchall()
        except Exception as e:
            logger.warning("[semantic] 批量查询失败：%s", e)
            return []

    def increment_hit(self, semantic_id: int) -> None:
        try:
            with self._get_conn() as conn, conn.cursor() as cur:
                cur.execute(
                    "UPDATE agent_semantic SET hit_count = hit_count + 1 WHERE id = %s",
                    (semantic_id,),
                )
        except Exception as e:
            logger.debug("[semantic] 命中计数失败：%s", e)

    def delete_semantic(self, semantic_id: int) -> bool:
        self._ensure_tables()
        try:
            with self._get_conn() as conn, conn.cursor() as cur:
                cur.execute("DELETE FROM agent_semantic WHERE id = %s", (semantic_id,))
                return cur.rowcount > 0
        except Exception as e:
            logger.warning("[semantic] 删除失败：%s", e)
            return False

    def get_stale_ids(self, days: int = 14, limit: int = 50) -> list[int]:
        """僵尸语义记忆（零命中且超期），Curator 剪枝用。"""
        self._ensure_tables()
        try:
            with self._get_conn() as conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT id FROM agent_semantic "
                    "WHERE hit_count = 0 AND created_at < DATE_SUB(NOW(), INTERVAL %s DAY) "
                    "ORDER BY created_at LIMIT %s",
                    (days, limit),
                )
                return [r["id"] for r in cur.fetchall()]
        except Exception as e:
            logger.warning("[semantic] 僵尸查询失败：%s", e)
            return []

    # ===== Curator 压缩支持 =====

    def fetch_for_compact(self, limit: int = 200) -> list[dict[str, Any]]:
        self._ensure_tables()
        try:
            with self._get_conn() as conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT id, title, content, tags, hit_count FROM agent_semantic "
                    "WHERE hit_count < 10 ORDER BY created_at LIMIT %s",
                    (limit,),
                )
                return cur.fetchall()
        except Exception as e:
            logger.warning("[semantic] 压缩取数失败：%s", e)
            return []

    def delete_many(self, ids: list[int]) -> int:
        if not ids:
            return 0
        ph = ",".join(["%s"] * len(ids))
        try:
            with self._get_conn() as conn, conn.cursor() as cur:
                cur.execute(f"DELETE FROM agent_semantic WHERE id IN ({ph})", ids)
                return cur.rowcount
        except Exception as e:
            logger.warning("[semantic] 批量删除失败：%s", e)
            return 0


_store: SemanticStore | None = None
_store_lock = threading.Lock()


def get_semantic_store() -> SemanticStore:
    """单例 SemanticStore。"""
    global _store
    with _store_lock:
        if _store is None:
            from app.core.config import get_settings
            s = get_settings()
            _store = SemanticStore(
                host=s.MYSQL_HOST, port=s.MYSQL_PORT, user=s.MYSQL_USER,
                password=s.MYSQL_PASSWORD, database=s.MYSQL_DATABASE,
            )
        return _store


def is_semantic_enabled() -> bool:
    return get_semantic_store().enabled
