"""反思审计日志存储（agent_reflections 表）。

五类记忆架构重构后本模块只剩审计职责：
- 长期记忆 → longterm.py（文件）
- 语义记忆 → semantic_store.py（agent_semantic）
- 情景记忆 → episode_store.py（agent_episodes）
- 程序记忆 → procedure_store.py（agent_procedures）
- 反思审计 → 本模块（agent_reflections，Curator 治理报告也写这里）

旧表 agent_memories / agent_skills（自动习得）已废弃，不再建表与写入。
"""
import logging
import threading

from agent.memory.base_store import BaseStore

logger = logging.getLogger(__name__)

_CREATE_REFLECTIONS_TABLE = """
CREATE TABLE IF NOT EXISTS agent_reflections (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_query TEXT NOT NULL COMMENT '触发反思的用户查询',
    trigger_reason VARCHAR(64) NOT NULL COMMENT '触发原因',
    tool_calls_summary TEXT COMMENT '工具调用摘要',
    final_answer TEXT COMMENT '最终回答',
    reflection_text TEXT NOT NULL COMMENT '反思内容',
    memories_created INT DEFAULT 0 COMMENT '本次反思生成的记忆数',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_reason (trigger_reason),
    INDEX idx_created (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='Agent 反思日志（审计）'
"""


class MemoryStore(BaseStore):
    """反思审计日志（保留类名兼容既有 import）。"""

    _create_sqls = [_CREATE_REFLECTIONS_TABLE]

    def add_reflection(
        self,
        user_query: str,
        trigger_reason: str,
        tool_calls_summary: str = "",
        final_answer: str = "",
        reflection_text: str = "",
        memories_created: int = 0,
    ) -> int | None:
        """写入一条反思审计记录，返回新 id（失败 None）。"""
        self._ensure_tables()
        try:
            with self._get_conn() as conn, conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO agent_reflections "
                    "(user_query, trigger_reason, tool_calls_summary, final_answer, "
                    "reflection_text, memories_created) VALUES (%s, %s, %s, %s, %s, %s)",
                    (
                        user_query, trigger_reason[:64], tool_calls_summary,
                        final_answer, reflection_text, memories_created,
                    ),
                )
                return cur.lastrowid
        except Exception as e:
            logger.warning("[memory] 反思日志写入失败：%s", e)
            return None

    def list_reflections(self, limit: int = 50, days_back: int | None = None) -> list[dict]:
        """反思日志列表（时间倒序，审计 API 用）。"""
        self._ensure_tables()
        sql = ("SELECT id, user_query, trigger_reason, tool_calls_summary, final_answer, "
               "reflection_text, memories_created, created_at FROM agent_reflections")
        params: list = []
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
            logger.warning("[memory] 反思日志查询失败：%s", e)
            return []


_store: MemoryStore | None = None
_store_lock = threading.Lock()


def get_memory_store() -> MemoryStore:
    """单例 MemoryStore（反思审计）。"""
    global _store
    with _store_lock:
        if _store is None:
            from app.core.config import get_settings
            s = get_settings()
            _store = MemoryStore(
                host=s.MYSQL_HOST, port=s.MYSQL_PORT, user=s.MYSQL_USER,
                password=s.MYSQL_PASSWORD, database=s.MYSQL_DATABASE,
            )
        return _store


def is_memory_enabled() -> bool:
    """MySQL 记忆持久化是否启用（反思审计/语义/情景/程序存储依赖）。"""
    return get_memory_store().enabled
