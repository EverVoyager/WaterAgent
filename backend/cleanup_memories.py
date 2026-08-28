"""清理 agent_memories 表中已积累的重复记忆。

去重策略：对每个 memory_type，按 content 前 12 字符分组，每组只保留 updated_at 最新的一条。
用于一次性修复历史数据（add_memory 方法已升级为自动去重，未来不需要手动清理）。

用法：
    cd backend
    python cleanup_memories.py
"""
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(BACKEND_DIR))

import pymysql

from app.core.config import get_settings

DEDUP_PREFIX_LEN = 12


def main() -> int:
    settings = get_settings()
    if not settings.MYSQL_PASSWORD:
        print("[ERROR] MYSQL_PASSWORD 未配置")
        return 1

    try:
        conn = pymysql.connect(
            host=settings.MYSQL_HOST,
            port=settings.MYSQL_PORT,
            user=settings.MYSQL_USER,
            password=settings.MYSQL_PASSWORD,
            database=settings.MYSQL_DATABASE,
            charset="utf8mb4",
        )
    except Exception as e:
        print(f"[ERROR] MySQL 连接失败：{e}")
        return 1

    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cur:
            # 统计清理前
            cur.execute("SELECT COUNT(*) AS cnt FROM agent_memories")
            before = cur.fetchone()["cnt"]

            # 找出每组重复的记忆（保留 updated_at 最新的一条）
            cur.execute(
                """
                SELECT id, memory_type, LEFT(content, %s) AS prefix, updated_at
                FROM agent_memories
                WHERE id NOT IN (
                    SELECT keep_id FROM (
                        SELECT MAX(id) AS keep_id
                        FROM agent_memories
                        GROUP BY memory_type, LEFT(content, %s)
                    ) AS keep_rows
                )
                """,
                (DEDUP_PREFIX_LEN, DEDUP_PREFIX_LEN),
            )
            duplicates = cur.fetchall()

            if not duplicates:
                print(f"[OK] 无重复记忆（共 {before} 条），无需清理")
                return 0

            print(f"[INFO] 发现 {len(duplicates)} 条重复记忆，开始清理...")

            # 删除重复记忆（保留每组最新的）
            cur.execute(
                """
                DELETE FROM agent_memories
                WHERE id NOT IN (
                    SELECT keep_id FROM (
                        SELECT MAX(id) AS keep_id
                        FROM agent_memories
                        GROUP BY memory_type, LEFT(content, %s)
                    ) AS keep_rows
                )
                """,
                (DEDUP_PREFIX_LEN,),
            )
            deleted = cur.rowcount
        conn.commit()

        with conn.cursor(pymysql.cursors.DictCursor) as cur:
            cur.execute("SELECT COUNT(*) AS cnt FROM agent_memories")
            after = cur.fetchone()["cnt"]

        print(f"[OK] 清理完成：{before} → {after}（删除 {deleted} 条重复）")

        # 显示清理后剩余的记忆
        with conn.cursor(pymysql.cursors.DictCursor) as cur:
            cur.execute(
                "SELECT id, memory_type, content, updated_at "
                "FROM agent_memories ORDER BY memory_type, updated_at DESC"
            )
            rows = cur.fetchall()
        if rows:
            print("\n[INFO] 剩余记忆：")
            current_type = None
            for r in rows:
                if r["memory_type"] != current_type:
                    current_type = r["memory_type"]
                    print(f"\n  [{current_type}]")
                print(f"    id={r['id']} | {r['content'][:80]}")
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
