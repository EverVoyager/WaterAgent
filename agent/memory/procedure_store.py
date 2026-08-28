"""程序记忆存储：长期解决问题累积提炼的通用解决方法（五类记忆之五）。

认知科学程序记忆 = "怎么做"的技能。演进链（借鉴 Codex feedback rules + 本项目打通）：
  反思循环写入具体模式（source=reflection）
    → Curator 周期提炼泛化（refined_count++，steps 从具体案例变通用步骤）
    → 高复用高质量程序晋升候选 Skill（status=promoted，enabled=false 人工确认）

向量索引：Qdrant agent_procedures_vec（embed applicability）。
"""
import json
import logging
import threading
from typing import Any

from agent.memory.base_store import BaseStore

logger = logging.getLogger(__name__)

_CREATE_PROCEDURES_TABLE = """
CREATE TABLE IF NOT EXISTS agent_procedures (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(128) NOT NULL COMMENT '方法名',
    applicability TEXT NOT NULL COMMENT '适用条件描述（语义匹配用）',
    steps_json TEXT NOT NULL COMMENT '通用步骤 [{step, action, tool}]',
    tool_sequence_json TEXT COMMENT '典型工具序列 JSON',
    source VARCHAR(32) NOT NULL DEFAULT 'reflection' COMMENT '来源：reflection/curator/manual',
    use_count INT NOT NULL DEFAULT 0 COMMENT '被注入次数',
    success_count INT NOT NULL DEFAULT 0 COMMENT '注入后请求成功次数',
    refined_count INT NOT NULL DEFAULT 0 COMMENT '被 Curator 提炼次数',
    status ENUM('active', 'promoted', 'deprecated') NOT NULL DEFAULT 'active',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_status (status),
    INDEX idx_use (use_count)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='程序记忆（通用解决方法）'
"""

# Curator 晋升阈值：高复用 + 高成功率才生成候选 Skill
PROMOTE_MIN_USES = 5
PROMOTE_MIN_SUCCESS_RATE = 0.8
# Curator 提炼阈值
REFINE_MIN_USES = 3
REFINE_MAX_REFINED = 2


class ProcedureStore(BaseStore):
    """程序记忆 CRUD + 检索 + 晋升。"""

    _create_sqls = [_CREATE_PROCEDURES_TABLE]

    def add_procedure(self, name: str, applicability: str,
                      steps: list[dict[str, Any]],
                      tool_sequence: list[str] | None = None,
                      source: str = "reflection") -> int | None:
        """写入一条程序记忆，返回新 id（失败 None）。"""
        self._ensure_tables()
        try:
            with self._get_conn() as conn, conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO agent_procedures "
                    "(name, applicability, steps_json, tool_sequence_json, source) "
                    "VALUES (%s, %s, %s, %s, %s)",
                    (
                        name[:128], applicability,
                        json.dumps(steps, ensure_ascii=False),
                        json.dumps(tool_sequence or [], ensure_ascii=False),
                        source,
                    ),
                )
                return cur.lastrowid
        except Exception as e:
            logger.warning("[procedure] 写入失败：%s", e)
            return None

    def get_by_ids(self, ids: list[int]) -> list[dict[str, Any]]:
        if not ids:
            return []
        self._ensure_tables()
        ph = ",".join(["%s"] * len(ids))
        try:
            with self._get_conn() as conn, conn.cursor() as cur:
                cur.execute(
                    f"SELECT * FROM agent_procedures WHERE id IN ({ph}) AND status = 'active'",
                    ids,
                )
                return cur.fetchall()
        except Exception as e:
            logger.warning("[procedure] 批量查询失败：%s", e)
            return []

    def list_procedures(self, limit: int = 100, status: str | None = None) -> list[dict[str, Any]]:
        self._ensure_tables()
        sql = "SELECT * FROM agent_procedures"
        params: list[Any] = []
        if status:
            sql += " WHERE status = %s"
            params.append(status)
        sql += " ORDER BY use_count DESC, updated_at DESC LIMIT %s"
        params.append(limit)
        try:
            with self._get_conn() as conn, conn.cursor() as cur:
                cur.execute(sql, params)
                return cur.fetchall()
        except Exception as e:
            logger.warning("[procedure] 列表失败：%s", e)
            return []

    def get_procedure(self, procedure_id: int) -> dict[str, Any] | None:
        self._ensure_tables()
        try:
            with self._get_conn() as conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM agent_procedures WHERE id = %s", (procedure_id,),
                )
                return cur.fetchone()
        except Exception as e:
            logger.warning("[procedure] 查询失败：%s", e)
            return None

    def record_use(self, procedure_id: int, success: bool) -> None:
        """被注入时计数（planner 注入后调用）。"""
        try:
            with self._get_conn() as conn, conn.cursor() as cur:
                cur.execute(
                    "UPDATE agent_procedures SET use_count = use_count + 1, "
                    "success_count = success_count + %s WHERE id = %s",
                    (1 if success else 0, procedure_id),
                )
        except Exception as e:
            logger.debug("[procedure] 计数失败：%s", e)

    def update_steps(self, procedure_id: int, steps: list[dict[str, Any]],
                     applicability: str | None = None) -> bool:
        """Curator 提炼：替换为泛化后的通用步骤。"""
        try:
            with self._get_conn() as conn, conn.cursor() as cur:
                if applicability:
                    cur.execute(
                        "UPDATE agent_procedures SET steps_json = %s, applicability = %s, "
                        "refined_count = refined_count + 1 WHERE id = %s",
                        (json.dumps(steps, ensure_ascii=False), applicability, procedure_id),
                    )
                else:
                    cur.execute(
                        "UPDATE agent_procedures SET steps_json = %s, "
                        "refined_count = refined_count + 1 WHERE id = %s",
                        (json.dumps(steps, ensure_ascii=False), procedure_id),
                    )
                return cur.rowcount > 0
        except Exception as e:
            logger.warning("[procedure] 提炼更新失败：%s", e)
            return False

    def get_refine_candidates(self) -> list[dict[str, Any]]:
        """Curator 提炼候选：有复用且提炼次数不足。"""
        self._ensure_tables()
        try:
            with self._get_conn() as conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM agent_procedures WHERE status = 'active' "
                    "AND use_count >= %s AND refined_count < %s LIMIT 20",
                    (REFINE_MIN_USES, REFINE_MAX_REFINED),
                )
                return cur.fetchall()
        except Exception as e:
            logger.warning("[procedure] 提炼候选查询失败：%s", e)
            return []

    def get_promote_candidates(self) -> list[dict[str, Any]]:
        """Curator 晋升候选：高复用 + 高成功率的 active 程序。"""
        self._ensure_tables()
        try:
            with self._get_conn() as conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM agent_procedures WHERE status = 'active' "
                    "AND use_count >= %s "
                    "AND success_count / use_count >= %s LIMIT 10",
                    (PROMOTE_MIN_USES, PROMOTE_MIN_SUCCESS_RATE),
                )
                return cur.fetchall()
        except Exception as e:
            logger.warning("[procedure] 晋升候选查询失败：%s", e)
            return []

    def mark_promoted(self, procedure_id: int) -> None:
        try:
            with self._get_conn() as conn, conn.cursor() as cur:
                cur.execute(
                    "UPDATE agent_procedures SET status = 'promoted' WHERE id = %s",
                    (procedure_id,),
                )
        except Exception as e:
            logger.warning("[procedure] 标记晋升失败：%s", e)

    def demote(self, procedure_id: int) -> None:
        """反思判定无效 → 降权（deprecated，不再注入）。"""
        try:
            with self._get_conn() as conn, conn.cursor() as cur:
                cur.execute(
                    "UPDATE agent_procedures SET status = 'deprecated' WHERE id = %s",
                    (procedure_id,),
                )
        except Exception as e:
            logger.debug("[procedure] 降权失败：%s", e)

    def delete_procedure(self, procedure_id: int) -> bool:
        self._ensure_tables()
        try:
            with self._get_conn() as conn, conn.cursor() as cur:
                cur.execute("DELETE FROM agent_procedures WHERE id = %s", (procedure_id,))
                return cur.rowcount > 0
        except Exception as e:
            logger.warning("[procedure] 删除失败：%s", e)
            return False

    # ===== 晋升为 Skill =====

    def promote_to_skill(self, procedure_id: int, auto_enable: bool = False) -> dict[str, Any]:
        """把程序记忆晋升为候选 Skill（默认 enabled=false 待人工确认）。

        Returns:
            {"ok": bool, "skill_name": str, "reason": str}
        """
        proc = self.get_procedure(procedure_id)
        if not proc:
            return {"ok": False, "skill_name": "", "reason": f"程序记忆 {procedure_id} 不存在"}
        if proc["status"] == "promoted":
            return {"ok": False, "skill_name": proc.get("name", ""), "reason": "已晋升过"}

        try:
            steps = json.loads(proc.get("steps_json") or "[]")
            tool_seq = json.loads(proc.get("tool_sequence_json") or "[]")
        except (json.JSONDecodeError, TypeError):
            steps, tool_seq = [], []

        # 步骤渲染为指令文本
        lines = [f"你是{proc['name']}场景的处理专家，按以下步骤执行："]
        for i, s in enumerate(steps, 1):
            tool = f"（调用 {s['tool']}）" if isinstance(s, dict) and s.get("tool") else ""
            action = s.get("action", "") if isinstance(s, dict) else str(s)
            lines.append(f"{i}. {action}{tool}")
        lines.append(f"适用条件：{proc['applicability']}")
        instructions = "\n".join(lines)

        skill_name = self._to_snake_name(proc["name"])
        description = (f"【程序记忆晋升】{proc['applicability'][:200]} "
                       f"（源自 {proc.get('use_count', 0)} 次实践经验）")

        try:
            from agent.skills import create_skill
            from agent.skills.models import SkillCreate
            create_skill(SkillCreate(
                name=skill_name,
                description=description,
                instructions=instructions,
                tool_names=sorted(set(tool_seq)),
                enabled=auto_enable,  # 默认 false：人工在 SkillsView 确认启用
            ))
        except ValueError as e:
            # 同名冲突 → 视为已晋升
            self.mark_promoted(procedure_id)
            return {"ok": False, "skill_name": skill_name, "reason": f"晋升冲突：{e}"}
        except Exception as e:
            return {"ok": False, "skill_name": skill_name, "reason": f"创建 Skill 失败：{e}"}

        self.mark_promoted(procedure_id)
        logger.info("[procedure] 晋升为候选 Skill：%s（enabled=%s）", skill_name, auto_enable)
        return {"ok": True, "skill_name": skill_name,
                "reason": "已生成候选 Skill，请在技能管理页确认启用"}

    @staticmethod
    def _to_snake_name(name: str) -> str:
        """中文/任意名称 → snake_case 合法 Skill 名。"""
        import re
        # 非字母数字下划线全部替换为 _，压缩连续 _，确保字母开头
        s = re.sub(r"[^a-zA-Z0-9_]+", "_", name).strip("_").lower()
        if not s or not s[0].isalpha():
            s = "proc_" + (s or str(abs(hash(name)) % 10000))
        return s[:60]


_store: ProcedureStore | None = None
_store_lock = threading.Lock()


def get_procedure_store() -> ProcedureStore:
    """单例 ProcedureStore。"""
    global _store
    with _store_lock:
        if _store is None:
            from app.core.config import get_settings
            s = get_settings()
            _store = ProcedureStore(
                host=s.MYSQL_HOST, port=s.MYSQL_PORT, user=s.MYSQL_USER,
                password=s.MYSQL_PASSWORD, database=s.MYSQL_DATABASE,
            )
        return _store
