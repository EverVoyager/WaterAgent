"""MySQL 持久化记忆存储。

三层记忆表设计：
1. agent_memories — 长期记忆（用户偏好、纠正、领域知识）
2. agent_skills — 技能记忆（query 模式 → 工具组合）
3. agent_reflections — 反思日志（每次反思的完整记录，便于审计）

设计要点：
- 连接池：每次操作获取/释放连接，避免长连接断开
- 自动建表：首次使用时自动创建表（IF NOT EXISTS）
- 降级：MySQL 不可用时返回空结果，不抛错（保持 Agent 主流程可用）
- 时间戳：所有记录带 created_at，便于按时间检索
"""
import json
import logging
import threading
from contextlib import contextmanager
from enum import Enum
from functools import lru_cache
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MemoryType(str, Enum):
    """长期记忆类型。"""
    USER_PREFERENCE = "user_preference"       # 用户偏好（如"不用 emoji"）
    USER_CORRECTION = "user_correction"       # 用户纠正（如"吴堡应该是 900 而非 800"）
    DOMAIN_KNOWLEDGE = "domain_knowledge"     # 领域知识（如"龙门站警戒水位 377.5m"）
    TOOL_FAILURE = "tool_failure"             # 工具失败经验（如"府谷站无数据"）
    FORMAT_LEARNING = "format_learning"       # 输出格式学习


# SQL 建表语句（IF NOT EXISTS 保证幂等）
_CREATE_MEMORIES_TABLE = """
CREATE TABLE IF NOT EXISTS agent_memories (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    memory_type VARCHAR(32) NOT NULL COMMENT '记忆类型',
    content TEXT NOT NULL COMMENT '记忆内容',
    context TEXT COMMENT '上下文（JSON：触发场景、相关 query 等）',
    tags VARCHAR(256) COMMENT '标签（逗号分隔，便于检索）',
    hit_count INT DEFAULT 0 COMMENT '命中次数（被注入到 prompt 的次数）',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_type (memory_type),
    INDEX idx_created (created_at),
    INDEX idx_tags (tags(64))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='Agent 长期记忆'
"""

_CREATE_SKILLS_TABLE = """
CREATE TABLE IF NOT EXISTS agent_skills (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    query_pattern VARCHAR(512) NOT NULL COMMENT '查询模式（如水情查询）',
    query_hash VARCHAR(64) NOT NULL COMMENT '查询指纹（MD5，便于去重）',
    tool_calls_json TEXT NOT NULL COMMENT '工具调用序列 JSON',
    success BOOLEAN NOT NULL DEFAULT TRUE COMMENT '是否成功解决',
    rounds_used INT DEFAULT 1 COMMENT '使用的规划轮次',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_used_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    use_count INT DEFAULT 1 COMMENT '复用次数',
    INDEX idx_pattern (query_pattern(128)),
    INDEX idx_hash (query_hash),
    INDEX idx_last_used (last_used_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='Agent 技能记忆'
"""

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
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='Agent 反思日志'
"""


class MemoryStore:
    """MySQL 记忆存储。线程安全（每次操作独立连接）。

    降级策略：MySQL 不可用时所有方法返回空结果/False，不抛错。
    """

    def __init__(self, host: str, port: int, user: str, password: str, database: str):
        self._config = {
            "host": host, "port": port, "user": user,
            "password": password, "database": database,
            "charset": "utf8mb4",
        }
        self._initialized = False
        self._init_lock = threading.Lock()
        self._enabled = bool(password)  # 密码为空则禁用

    @property
    def enabled(self) -> bool:
        """是否启用（MYSQL_PASSWORD 非空）。"""
        return self._enabled

    @contextmanager
    def _get_conn(self):
        """获取 MySQL 连接（上下文管理器，自动关闭）。"""
        if not self._enabled:
            raise RuntimeError("MemoryStore 未启用（MYSQL_PASSWORD 为空）")
        import pymysql
        import pymysql.cursors
        conn = None
        try:
            conn = pymysql.connect(**self._config)
            yield conn
        except Exception as e:
            logger.warning("[memory] MySQL 操作失败：%s", e)
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
                logger.info("[memory] 记忆存储未启用（MYSQL_PASSWORD 为空）")
                self._initialized = True
                return
            try:
                with self._get_conn() as conn:
                    with conn.cursor() as cur:
                        cur.execute(_CREATE_MEMORIES_TABLE)
                        cur.execute(_CREATE_SKILLS_TABLE)
                        cur.execute(_CREATE_REFLECTIONS_TABLE)
                    conn.commit()
                logger.info("[memory] MySQL 表已就绪（agent_memories/agent_skills/agent_reflections）")
                self._initialized = True
            except Exception as e:
                logger.warning("[memory] 建表失败，记忆存储降级为禁用：%s", e)
                self._enabled = False
                self._initialized = True

    # ============ 长期记忆 ============

    def add_memory(
        self,
        memory_type: MemoryType,
        content: str,
        context: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> Optional[int]:
        """添加长期记忆。返回记忆 ID，失败返回 None。

        注意：此方法直接新增，不做去重。语义合并由 compact_memories() 异步完成，
        会在反思后触发，用 LLM 判断同类型记忆间的关系（一致/冲突/无关），
        整合或替换，避免相似记忆堆积。
        """
        if not self._enabled:
            return None
        self._ensure_tables()
        try:
            with self._get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO agent_memories (memory_type, content, context, tags) "
                        "VALUES (%s, %s, %s, %s)",
                        (
                            memory_type.value,
                            content,
                            json.dumps(context, ensure_ascii=False) if context else None,
                            ",".join(tags) if tags else None,
                        ),
                    )
                    mem_id = cur.lastrowid
                conn.commit()
            logger.info("[memory] 添加记忆 type=%s id=%s content=%s",
                        memory_type.value, mem_id, content[:60])
            return mem_id
        except Exception:
            return None

    def get_memories(
        self,
        memory_type: Optional[MemoryType] = None,
        tags: Optional[List[str]] = None,
        limit: int = 20,
        min_hit: int = 0,
    ) -> List[Dict[str, Any]]:
        """检索长期记忆。按创建时间倒序。"""
        if not self._enabled:
            return []
        self._ensure_tables()
        import pymysql.cursors
        try:
            with self._get_conn() as conn:
                with conn.cursor(pymysql.cursors.DictCursor) as cur:
                    sql = "SELECT id, memory_type, content, context, tags, hit_count, created_at "
                    sql += "FROM agent_memories WHERE 1=1"
                    params: list = []
                    if memory_type:
                        sql += " AND memory_type = %s"
                        params.append(memory_type.value)
                    if tags:
                        # 任意一个 tag 匹配即可
                        tag_conditions = " OR ".join(["tags LIKE %s" for _ in tags])
                        sql += f" AND ({tag_conditions})"
                        params.extend([f"%{t}%" for t in tags])
                    if min_hit > 0:
                        sql += " AND hit_count >= %s"
                        params.append(min_hit)
                    sql += " ORDER BY created_at DESC LIMIT %s"
                    params.append(limit)
                    cur.execute(sql, params)
                    rows = cur.fetchall()
            return rows
        except Exception:
            return []

    def increment_hit(self, memory_id: int) -> None:
        """命中次数 +1（被注入到 prompt 时调用）。"""
        if not self._enabled:
            return
        try:
            with self._get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE agent_memories SET hit_count = hit_count + 1 WHERE id = %s",
                        (memory_id,),
                    )
                conn.commit()
        except Exception:
            pass

    # ============ 技能记忆 ============

    def add_skill(
        self,
        query_pattern: str,
        tool_calls: List[Dict[str, Any]],
        success: bool,
        rounds_used: int = 1,
    ) -> None:
        """记录一次成功的工具调用模式（技能）。"""
        if not self._enabled:
            return
        self._ensure_tables()
        import hashlib
        query_hash = hashlib.md5(query_pattern.encode("utf-8")).hexdigest()
        try:
            with self._get_conn() as conn:
                with conn.cursor() as cur:
                    # 同模式已存在则更新 use_count + last_used_at
                    cur.execute(
                        "SELECT id, use_count FROM agent_skills WHERE query_hash = %s",
                        (query_hash,),
                    )
                    existing = cur.fetchone()
                    if existing:
                        cur.execute(
                            "UPDATE agent_skills SET use_count = use_count + 1, "
                            "last_used_at = NOW() WHERE id = %s",
                            (existing[0],),
                        )
                    else:
                        cur.execute(
                            "INSERT INTO agent_skills "
                            "(query_pattern, query_hash, tool_calls_json, success, rounds_used) "
                            "VALUES (%s, %s, %s, %s, %s)",
                            (
                                query_pattern[:512],
                                query_hash,
                                json.dumps(tool_calls, ensure_ascii=False),
                                success,
                                rounds_used,
                            ),
                        )
                conn.commit()
            logger.info("[memory] 记录技能 pattern=%s success=%s", query_pattern[:40], success)
        except Exception:
            pass

    def get_relevant_skills(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """检索相关技能。当前用 LIKE 模糊匹配（后续可升级为向量检索）。"""
        if not self._enabled:
            return []
        self._ensure_tables()
        import pymysql.cursors
        # 提取 query 的关键词用于匹配
        keywords = [w for w in query.replace("？", "").replace("？", "").split() if len(w) >= 2]
        if not keywords:
            keywords = [query[:4]]
        try:
            with self._get_conn() as conn:
                with conn.cursor(pymysql.cursors.DictCursor) as cur:
                    # 简单关键词匹配（任意一个命中即返回）
                    like_conditions = " OR ".join(["query_pattern LIKE %s" for _ in keywords])
                    sql = (
                        "SELECT query_pattern, tool_calls_json, success, rounds_used, use_count "
                        "FROM agent_skills WHERE success = TRUE AND (" + like_conditions + ") "
                        "ORDER BY use_count DESC, last_used_at DESC LIMIT %s"
                    )
                    params = [f"%{kw}%" for kw in keywords] + [limit]
                    cur.execute(sql, params)
                    rows = cur.fetchall()
            # 反序列化 tool_calls
            for row in rows:
                try:
                    row["tool_calls"] = json.loads(row.pop("tool_calls_json"))
                except (json.JSONDecodeError, KeyError):
                    row["tool_calls"] = []
            return rows
        except Exception:
            return []

    # ============ 反思日志 ============

    def add_reflection(
        self,
        user_query: str,
        trigger_reason: str,
        reflection_text: str,
        tool_calls_summary: str = "",
        final_answer: str = "",
        memories_created: int = 0,
    ) -> None:
        """记录一次反思（审计用）。"""
        if not self._enabled:
            return
        self._ensure_tables()
        try:
            with self._get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO agent_reflections "
                        "(user_query, trigger_reason, tool_calls_summary, final_answer, "
                        " reflection_text, memories_created) "
                        "VALUES (%s, %s, %s, %s, %s, %s)",
                        (
                            user_query[:1000],
                            trigger_reason,
                            tool_calls_summary,
                            final_answer[:2000],
                            reflection_text,
                            memories_created,
                        ),
                    )
                conn.commit()
            logger.info("[memory] 记录反思 reason=%s memories=%d", trigger_reason, memories_created)
        except Exception:
            pass


    def compact_memories(self, memory_type: MemoryType, llm_compact_func) -> int:
        """用 LLM 合并同类型记忆。

        策略（用户要求）：
        - 语义完全一致 → 保留新记忆，删除旧记忆
        - 内容冲突 → 保留新记忆，删除旧记忆
        - 不冲突但可整合 → 合并为一条，删除原记忆
        - 完全无关 → 都保留

        Args:
            memory_type: 要压缩的记忆类型
            llm_compact_func: 可调用对象，接收 List[Dict] 返回 List[Dict]
                              每条 dict: {"content": str, "action": "keep"|"merge"|"replace",
                                          "source_ids": [int], "tags": [str]}

        Returns:
            删除的记忆数（负数表示失败）
        """
        if not self._enabled:
            return -1
        self._ensure_tables()
        import pymysql.cursors
        try:
            with self._get_conn() as conn:
                with conn.cursor(pymysql.cursors.DictCursor) as cur:
                    # 拉取该类型所有记忆（按时间正序，旧的在前）
                    cur.execute(
                        "SELECT id, content, context, tags, hit_count, created_at, updated_at "
                        "FROM agent_memories WHERE memory_type = %s "
                        "ORDER BY updated_at ASC",
                        (memory_type.value,),
                    )
                    memories = cur.fetchall()

            if len(memories) < 2:
                return 0  # 只有一条或没有，无需压缩

            # 调用 LLM 判断记忆间关系
            merged_plan = llm_compact_func(memory_type.value, memories)
            if not merged_plan:
                logger.debug("[compact] LLM 未返回有效合并方案，跳过")
                return 0

            # 执行合并：删除被合并/替换的旧记忆，插入新的整合记忆
            deleted_count = 0
            new_memories_to_insert = []
            keep_ids = set()

            for item in merged_plan:
                action = item.get("action", "keep")
                source_ids = item.get("source_ids", [])
                if action == "keep":
                    # 保留原记忆不动
                    keep_ids.update(source_ids)
                elif action in ("merge", "replace"):
                    # 标记源记忆为待删除，准备插入新整合记忆
                    new_memories_to_insert.append({
                        "content": item.get("content", ""),
                        "tags": item.get("tags", []),
                        "hit_count_sum": sum(
                            (m.get("hit_count", 0) or 0) for m in memories if m["id"] in source_ids
                        ),
                    })

            if not new_memories_to_insert and not keep_ids:
                return 0

            with self._get_conn() as conn:
                with conn.cursor() as cur:
                    # 删除被合并/替换的旧记忆
                    ids_to_delete = set()
                    for item in merged_plan:
                        if item.get("action") in ("merge", "replace"):
                            ids_to_delete.update(item.get("source_ids", []))
                    if ids_to_delete:
                        placeholders = ",".join(["%s"] * len(ids_to_delete))
                        cur.execute(
                            f"DELETE FROM agent_memories WHERE id IN ({placeholders})",
                            tuple(ids_to_delete),
                        )
                        deleted_count = cur.rowcount

                    # 插入新的整合记忆（保留原 hit_count 总和）
                    for new_mem in new_memories_to_insert:
                        cur.execute(
                            "INSERT INTO agent_memories (memory_type, content, tags, hit_count) "
                            "VALUES (%s, %s, %s, %s)",
                            (
                                memory_type.value,
                                new_mem["content"],
                                ",".join(new_mem["tags"]) if new_mem["tags"] else None,
                                new_mem["hit_count_sum"],
                            ),
                        )
                conn.commit()

            logger.info("[compact] type=%s 删除 %d 条，新增 %d 条整合记忆",
                        memory_type.value, deleted_count, len(new_memories_to_insert))
            return deleted_count
        except Exception as e:
            logger.warning("[compact] 压缩失败：%s", e)
            return -1

@lru_cache(maxsize=1)
def get_memory_store() -> MemoryStore:
    """单例 MemoryStore。"""
    from app.core.config import get_settings
    s = get_settings()
    return MemoryStore(
        host=s.MYSQL_HOST,
        port=s.MYSQL_PORT,
        user=s.MYSQL_USER,
        password=s.MYSQL_PASSWORD,
        database=s.MYSQL_DATABASE,
    )


def is_memory_enabled() -> bool:
    """检查记忆模块是否启用（MYSQL_PASSWORD 配置 + SELF_EVOLUTION_ENABLED）。"""
    from app.core.config import get_settings
    s = get_settings()
    return s.SELF_EVOLUTION_ENABLED and bool(s.MYSQL_PASSWORD) and get_memory_store().enabled
