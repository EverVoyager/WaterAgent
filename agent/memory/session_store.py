"""MySQL 会话持久化存储（P1-b）。

双表设计：
1. chat_sessions — 会话元数据（id/title/时间戳）
2. chat_messages — 单条消息（role/content/tool_events/reasoning_steps/response）

硬失败策略：MySQL 不可用时抛 RuntimeError（不降级到 localStorage），
错误直传前端，符合"无降级机制"约束。

设计要点（借鉴 memory_store.py）：
- 连接池：每次操作获取/释放连接
- 自动建表：首次使用时 IF NOT EXISTS
- 线程安全：单例 + init_lock
"""
import json
import logging
import threading
from contextlib import contextmanager, suppress
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)


_CREATE_SESSIONS_TABLE = """
CREATE TABLE IF NOT EXISTS chat_sessions (
    id VARCHAR(32) PRIMARY KEY,
    title VARCHAR(256) NOT NULL DEFAULT '新会话',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_updated (updated_at DESC)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='聊天会话'
"""

_CREATE_MESSAGES_TABLE = """
CREATE TABLE IF NOT EXISTS chat_messages (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(32) NOT NULL,
    seq INT NOT NULL DEFAULT 0 COMMENT '消息在会话中的顺序（0-based）',
    role VARCHAR(16) NOT NULL COMMENT 'user / assistant',
    content TEXT,
    tool_events_json TEXT COMMENT 'ToolEvent[] JSON',
    reasoning_steps_json TEXT COMMENT 'ReasoningStepEntry[] JSON',
    response_json TEXT COMMENT 'AgentQueryResponse JSON',
    thinking TINYINT(1) DEFAULT 0,
    chain_expanded TINYINT(1) DEFAULT 0,
    reasoning_expanded TINYINT(1) DEFAULT 0,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE,
    INDEX idx_session_seq (session_id, seq)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='聊天消息'
"""


class SessionStore:
    """MySQL 会话存储。线程安全（每次操作独立连接）。

    硬失败：MySQL 不可用时抛 RuntimeError，不降级。
    """

    def __init__(self, host: str, port: int, user: str, password: str, database: str):
        self._config = {
            "host": host, "port": port, "user": user,
            "password": password, "database": database,
            "charset": "utf8mb4",
        }
        self._initialized = False
        self._init_lock = threading.Lock()
        self._enabled = bool(password)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @contextmanager
    def _get_conn(self):
        """事务型连接上下文管理器：with 体成功自动 commit，异常 rollback，最后 close。"""
        if not self._enabled:
            raise RuntimeError("SessionStore 未启用（MYSQL_PASSWORD 为空）")
        import pymysql
        import pymysql.cursors
        conn = None
        try:
            conn = pymysql.connect(**self._config, cursorclass=pymysql.cursors.DictCursor)
            yield conn
            conn.commit()
        except Exception:
            if conn:
                with suppress(Exception):
                    conn.rollback()
            raise
        finally:
            if conn:
                conn.close()

    def _ensure_tables(self) -> None:
        """首次使用时自动建表（线程安全，只执行一次）。"""
        if self._initialized:
            return
        with self._init_lock:
            if self._initialized:
                return
            if not self._enabled:
                raise RuntimeError("SessionStore 未启用（MYSQL_PASSWORD 为空），请配置 MySQL 后重启")
            try:
                with self._get_conn() as conn, conn.cursor() as cur:
                    cur.execute(_CREATE_SESSIONS_TABLE)
                    cur.execute(_CREATE_MESSAGES_TABLE)
                logger.info("[session] MySQL 表已就绪（chat_sessions/chat_messages）")
                self._initialized = True
            except Exception as e:
                logger.error("[session] 建表失败：%s", e)
                raise RuntimeError(f"SessionStore 建表失败：{e}") from e

    # ============ 会话 CRUD ============

    def create_session(self, session_id: str, title: str = "新会话") -> None:
        """创建会话。"""
        self._ensure_tables()
        with self._get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO chat_sessions (id, title) VALUES (%s, %s)",
                (session_id, title),
            )

    def list_sessions(self) -> list[dict[str, Any]]:
        """列出所有会话（仅元数据，不含消息）。按 updated_at 降序。"""
        self._ensure_tables()
        with self._get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT id, title, created_at, updated_at "
                "FROM chat_sessions ORDER BY updated_at DESC"
            )
            rows = cur.fetchall()
        return [self._row_to_session_meta(r) for r in rows]

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        """获取单个会话（含所有消息）。不存在返回 None。"""
        self._ensure_tables()
        with self._get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT id, title, created_at, updated_at "
                "FROM chat_sessions WHERE id = %s",
                (session_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            session = self._row_to_session_meta(row)
            cur.execute(
                "SELECT * FROM chat_messages WHERE session_id = %s ORDER BY seq",
                (session_id,),
            )
            msg_rows = cur.fetchall()
            session["messages"] = [self._row_to_message(r) for r in msg_rows]
        return session

    def list_sessions_with_messages(self) -> list[dict[str, Any]]:
        """列出所有会话（含消息）。用于前端启动时全量加载。"""
        self._ensure_tables()
        with self._get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT id, title, created_at, updated_at "
                "FROM chat_sessions ORDER BY updated_at DESC"
            )
            sessions = cur.fetchall()
            result = []
            for s in sessions:
                session = self._row_to_session_meta(s)
                cur.execute(
                    "SELECT * FROM chat_messages WHERE session_id = %s ORDER BY seq",
                    (s["id"],),
                )
                msg_rows = cur.fetchall()
                session["messages"] = [self._row_to_message(r) for r in msg_rows]
                result.append(session)
        return result

    def update_session_title(self, session_id: str, title: str) -> bool:
        """更新会话标题。返回是否成功（会话不存在返回 False）。"""
        self._ensure_tables()
        with self._get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE chat_sessions SET title = %s WHERE id = %s",
                (title, session_id),
            )
            affected = cur.rowcount
        return affected > 0

    def delete_session(self, session_id: str) -> bool:
        """删除会话（级联删除消息）。返回是否成功。"""
        self._ensure_tables()
        with self._get_conn() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM chat_sessions WHERE id = %s", (session_id,))
            affected = cur.rowcount
        return affected > 0

    def sync_session(self, session_id: str, title: str, messages: list[dict[str, Any]]) -> None:
        """全量同步会话：更新标题 + 替换所有消息。

        用于前端 persistActiveSession：流式完成后一次性同步整个会话状态。
        如果会话不存在则自动创建。
        """
        self._ensure_tables()
        with self._get_conn() as conn, conn.cursor() as cur:
            # Upsert session
            cur.execute(
                "INSERT INTO chat_sessions (id, title) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE title = VALUES(title), updated_at = CURRENT_TIMESTAMP",
                (session_id, title),
            )
            # 删除旧消息
            cur.execute("DELETE FROM chat_messages WHERE session_id = %s", (session_id,))
            # 批量插入新消息
            for seq, msg in enumerate(messages):
                cur.execute(
                    "INSERT INTO chat_messages "
                    "(session_id, seq, role, content, tool_events_json, "
                    "reasoning_steps_json, response_json, thinking, "
                    "chain_expanded, reasoning_expanded) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    (
                        session_id,
                        seq,
                        msg.get("role", "user"),
                        msg.get("content", ""),
                        json.dumps(msg["toolEvents"], ensure_ascii=False) if msg.get("toolEvents") else None,
                        json.dumps(msg["reasoningSteps"], ensure_ascii=False) if msg.get("reasoningSteps") else None,
                        json.dumps(msg["response"], ensure_ascii=False) if msg.get("response") else None,
                        1 if msg.get("thinking") else 0,
                        1 if msg.get("chainExpanded") else 0,
                        1 if msg.get("reasoningExpanded") else 0,
                    ),
                )

    # ============ 行映射 ============

    @staticmethod
    def _row_to_session_meta(row: dict[str, Any]) -> dict[str, Any]:
        """数据库行 → 会话元数据 dict（不含 messages）。"""
        return {
            "id": row["id"],
            "title": row["title"],
            "createdAt": int(row["created_at"].timestamp() * 1000) if row.get("created_at") else 0,
            "updatedAt": int(row["updated_at"].timestamp() * 1000) if row.get("updated_at") else 0,
            "messages": [],
        }

    @staticmethod
    def _row_to_message(row: dict[str, Any]) -> dict[str, Any]:
        """数据库行 → 消息 dict（前端 Message 结构）。"""
        msg: dict[str, Any] = {
            "role": row["role"],
            "content": row["content"] or "",
        }
        if row.get("tool_events_json"):
            with suppress(json.JSONDecodeError):
                msg["toolEvents"] = json.loads(row["tool_events_json"])
        if row.get("reasoning_steps_json"):
            with suppress(json.JSONDecodeError):
                msg["reasoningSteps"] = json.loads(row["reasoning_steps_json"])
        if row.get("response_json"):
            with suppress(json.JSONDecodeError):
                msg["response"] = json.loads(row["response_json"])
        if row.get("thinking"):
            msg["thinking"] = bool(row["thinking"])
        if row.get("chain_expanded"):
            msg["chainExpanded"] = bool(row["chain_expanded"])
        if row.get("reasoning_expanded"):
            msg["reasoningExpanded"] = bool(row["reasoning_expanded"])
        return msg


@lru_cache(maxsize=1)
def get_session_store() -> SessionStore:
    """单例 SessionStore。"""
    from app.core.config import get_settings
    s = get_settings()
    return SessionStore(
        host=s.MYSQL_HOST,
        port=s.MYSQL_PORT,
        user=s.MYSQL_USER,
        password=s.MYSQL_PASSWORD,
        database=s.MYSQL_DATABASE,
    )


def is_session_enabled() -> bool:
    """检查会话持久化是否启用（MYSQL_PASSWORD 已配置）。"""
    from app.core.config import get_settings
    s = get_settings()
    return bool(s.MYSQL_PASSWORD)
