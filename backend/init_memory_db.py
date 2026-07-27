"""MySQL 自进化记忆库初始化脚本。

主动建库建表，便于首次部署时运行。运行后会创建：
- 数据库 water_agent（如不存在）
- 三张表：agent_memories / agent_skills / agent_reflections

用法：
    cd backend
    python init_memory_db.py

前置条件：
    1. MySQL 已安装（5.7+ 或 8.0+）
    2. backend/.env 中 MYSQL_USER/MYSQL_PASSWORD 已配置
       （MYSQL_USER 需有 CREATE DATABASE 权限，通常用 root）
"""
import sys
from pathlib import Path

# 加载项目根目录到 sys.path
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(BACKEND_DIR))

import pymysql
from app.core.config import get_settings


_CREATE_DB_SQL = "CREATE DATABASE IF NOT EXISTS `{db}` DEFAULT CHARSET utf8mb4 COLLATE utf8mb4_unicode_ci"

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


def main() -> int:
    """主入口：建库 + 建表。返回 0 成功，1 失败。"""
    settings = get_settings()

    if not settings.MYSQL_PASSWORD:
        print("[ERROR] MYSQL_PASSWORD 未配置，请先在 backend/.env 中填写 MySQL 密码")
        print("        参考 backend/.env.example 中的 MySQL 配置段")
        return 1

    print(f"[INFO] 连接 MySQL {settings.MYSQL_HOST}:{settings.MYSQL_PORT} as {settings.MYSQL_USER}")

    # 1. 连接 MySQL（不指定 database）创建数据库
    try:
        conn = pymysql.connect(
            host=settings.MYSQL_HOST,
            port=settings.MYSQL_PORT,
            user=settings.MYSQL_USER,
            password=settings.MYSQL_PASSWORD,
            charset="utf8mb4",
        )
    except Exception as e:
        print(f"[ERROR] MySQL 连接失败：{e}")
        print("        请检查：1) MySQL 服务是否启动  2) .env 中的 MYSQL_HOST/PORT/USER/PASSWORD")
        return 1

    try:
        with conn.cursor() as cur:
            db_name = settings.MYSQL_DATABASE
            cur.execute(_CREATE_DB_SQL.format(db=db_name))
            print(f"[OK] 数据库 '{db_name}' 已就绪")
        conn.commit()

        # 2. 切换到目标数据库建表
        conn.select_db(settings.MYSQL_DATABASE)
        with conn.cursor() as cur:
            cur.execute(_CREATE_MEMORIES_TABLE)
            print("[OK] 表 'agent_memories' 已就绪")
            cur.execute(_CREATE_SKILLS_TABLE)
            print("[OK] 表 'agent_skills' 已就绪")
            cur.execute(_CREATE_REFLECTIONS_TABLE)
            print("[OK] 表 'agent_reflections' 已就绪")
        conn.commit()
    finally:
        conn.close()

    print()
    print("[SUCCESS] 自进化记忆库初始化完成")
    print(f"  - 数据库：{settings.MYSQL_DATABASE}")
    print("  - 表：agent_memories / agent_skills / agent_reflections")
    print()
    print("下一步：在 backend/.env 中确认 SELF_EVOLUTION_ENABLED=true")
    print("       启动后端后，每次触发反思条件时会自动写入记忆")
    return 0


if __name__ == "__main__":
    sys.exit(main())
