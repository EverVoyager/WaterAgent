"""MySQL Skill 存储（P2-b）。

将 Skill 配置从 JSON 文件迁移到 MySQL，支持多实例部署。

表结构：
- skills — Skill 配置（name 主键、description、instructions、tool_names、enabled）

硬失败策略：MySQL 不可用时抛 RuntimeError（不降级到 JSON 文件），
错误直传前端，符合"无降级机制"约束。

设计要点（借鉴 session_store.py / memory_store.py）：
- 连接池：每次操作获取/释放连接
- 自动建表：首次使用时 IF NOT EXISTS
- 线程安全：单例 + init_lock
"""
import contextlib
import json
import logging
import threading
from contextlib import contextmanager, suppress
from functools import lru_cache
from typing import Any

from agent.skills.models import Skill, SkillCreate, SkillUpdate

logger = logging.getLogger(__name__)


_CREATE_SKILLS_TABLE = """
CREATE TABLE IF NOT EXISTS skills (
    id VARCHAR(64) PRIMARY KEY,
    name VARCHAR(64) NOT NULL UNIQUE,
    description TEXT NOT NULL,
    instructions TEXT NOT NULL,
    tool_names_json TEXT COMMENT '工具名列表 JSON',
    enabled TINYINT(1) NOT NULL DEFAULT 1,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_enabled (enabled)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='Agent 技能配置'
"""


class SkillStore:
    """MySQL Skill 存储。线程安全（每次操作独立连接）。

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
            raise RuntimeError("SkillStore 未启用（MYSQL_PASSWORD 为空）")
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
                raise RuntimeError("SkillStore 未启用（MYSQL_PASSWORD 为空），请配置 MySQL 后重启")
            try:
                with self._get_conn() as conn, conn.cursor() as cur:
                    cur.execute(_CREATE_SKILLS_TABLE)
                logger.info("[skills] MySQL 表已就绪（skills）")
                self._initialized = True
            except Exception as e:
                logger.error("[skills] 建表失败：%s", e)
                raise RuntimeError(f"SkillStore 建表失败：{e}") from e

    # ============ Skill CRUD ============

    def list_skills(self, enabled_only: bool = False) -> list[Skill]:
        """列出所有 Skill。"""
        self._ensure_tables()
        with self._get_conn() as conn, conn.cursor() as cur:
            if enabled_only:
                cur.execute(
                    "SELECT id, name, description, instructions, tool_names_json, enabled "
                    "FROM skills WHERE enabled = 1 ORDER BY name"
                )
            else:
                cur.execute(
                    "SELECT id, name, description, instructions, tool_names_json, enabled "
                    "FROM skills ORDER BY name"
                )
            rows = cur.fetchall()
        return [self._row_to_skill(r) for r in rows]

    def get_skill(self, name: str) -> Skill | None:
        """按 name 获取单个 Skill。不存在返回 None。"""
        self._ensure_tables()
        with self._get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT id, name, description, instructions, tool_names_json, enabled "
                "FROM skills WHERE name = %s",
                (name,),
            )
            row = cur.fetchone()
        return self._row_to_skill(row) if row else None

    def create_skill(self, req: SkillCreate) -> Skill:
        """创建 Skill。name 已存在时抛 ValueError。"""
        self._ensure_tables()
        skill = Skill(id=req.name, **req.model_dump())
        with self._get_conn() as conn, conn.cursor() as cur:
            # 查重
            cur.execute("SELECT 1 FROM skills WHERE name = %s", (req.name,))
            if cur.fetchone():
                raise ValueError(f"Skill '{req.name}' 已存在")
            cur.execute(
                "INSERT INTO skills "
                "(id, name, description, instructions, tool_names_json, enabled) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (
                    skill.id,
                    skill.name,
                    skill.description,
                    skill.instructions,
                    json.dumps(skill.tool_names, ensure_ascii=False),
                    1 if skill.enabled else 0,
                ),
            )
        logger.info("[skills] 创建技能: %s", skill.name)
        return skill

    def update_skill(self, name: str, req: SkillUpdate) -> Skill:
        """更新 Skill。不存在时抛 ValueError。不允许修改 name。"""
        self._ensure_tables()
        update_data = req.model_dump(exclude_unset=True, exclude_none=True)
        update_data.pop("name", None)  # name 不可改
        if not update_data:
            # 没有字段需要更新，直接返回当前值
            existing = self.get_skill(name)
            if not existing:
                raise ValueError(f"Skill '{name}' 不存在")
            return existing

        with self._get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT id, name, description, instructions, tool_names_json, enabled "
                "FROM skills WHERE name = %s",
                (name,),
            )
            row = cur.fetchone()
            if not row:
                raise ValueError(f"Skill '{name}' 不存在")
            # 合并更新
            merged = self._row_to_dict(row)
            merged.update(update_data)
            # 重新校验（特别是 tool_names）
            updated = Skill(**merged)
            cur.execute(
                "UPDATE skills SET description = %s, instructions = %s, "
                "tool_names_json = %s, enabled = %s WHERE name = %s",
                (
                    updated.description,
                    updated.instructions,
                    json.dumps(updated.tool_names, ensure_ascii=False),
                    1 if updated.enabled else 0,
                    name,
                ),
            )
        logger.info("[skills] 更新技能: %s", name)
        return updated

    def delete_skill(self, name: str) -> bool:
        """删除 Skill。不存在返回 False。"""
        self._ensure_tables()
        with self._get_conn() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM skills WHERE name = %s", (name,))
            affected = cur.rowcount
        if affected > 0:
            logger.info("[skills] 删除技能: %s", name)
            return True
        return False

    def delete_all(self) -> None:
        """清空所有 Skill（仅供测试使用）。"""
        self._ensure_tables()
        with self._get_conn() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM skills")

    # ============ 行映射 ============

    @staticmethod
    def _row_to_skill(row: dict[str, Any]) -> Skill:
        """数据库行 → Skill 对象。"""
        tool_names = []
        if row.get("tool_names_json"):
            with contextlib.suppress(json.JSONDecodeError):
                tool_names = json.loads(row["tool_names_json"])
        return Skill(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            instructions=row["instructions"],
            tool_names=tool_names,
            enabled=bool(row["enabled"]),
        )

    @staticmethod
    def _row_to_dict(row: dict[str, Any]) -> dict[str, Any]:
        """数据库行 → dict（用于合并更新）。"""
        tool_names = []
        if row.get("tool_names_json"):
            with contextlib.suppress(json.JSONDecodeError):
                tool_names = json.loads(row["tool_names_json"])
        return {
            "id": row["id"],
            "name": row["name"],
            "description": row["description"],
            "instructions": row["instructions"],
            "tool_names": tool_names,
            "enabled": bool(row["enabled"]),
        }


@lru_cache(maxsize=1)
def get_skill_store() -> SkillStore:
    """单例 SkillStore。"""
    from app.core.config import get_settings
    s = get_settings()
    return SkillStore(
        host=s.MYSQL_HOST,
        port=s.MYSQL_PORT,
        user=s.MYSQL_USER,
        password=s.MYSQL_PASSWORD,
        database=s.MYSQL_DATABASE,
    )


def is_skill_store_enabled() -> bool:
    """检查 Skill 存储是否启用（MYSQL_PASSWORD 已配置）。"""
    from app.core.config import get_settings
    s = get_settings()
    return bool(s.MYSQL_PASSWORD)
