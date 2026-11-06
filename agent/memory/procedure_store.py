"""程序正文保存在 Qdrant，使用/提炼计数保存在 MySQL。"""
import hashlib
import re
from functools import lru_cache

from agent.memory import vector_index
from agent.memory.usage_store import get_usage_store, with_usage

PROMOTE_MIN_USES = 5
PROMOTE_MIN_SUCCESS_RATE = 0.8
REFINE_MIN_USES = 3
REFINE_MAX_REFINED = 2


class ProcedureStore:
    enabled = True

    def add_procedure(self, name: str, applicability: str, steps: list[dict],
                      source_refs: list[dict] | None = None) -> int | str:
        return vector_index.put_record("procedure", {
            "name": name, "applicability": applicability, "steps": steps,
            "source_refs": source_refs or [], "status": "active",
        })

    def get_by_ids(self, ids: list[int | str]) -> list[dict]:
        return [row for row in vector_index.get_records("procedure", ids) if row["status"] == "active"]

    def list_procedures(self, limit: int = 100, status: str | None = None) -> list[dict]:
        return vector_index.list_records("procedure", limit=limit, status=status)

    def get_procedure(self, memory_id: int | str) -> dict | None:
        rows = vector_index.get_records("procedure", [memory_id])
        return rows[0] if rows else None

    def record_use(self, memory_id: int | str, success: bool) -> None:
        get_usage_store().record_hit("procedure", memory_id, success)

    def update_steps(self, memory_id: int | str, steps: list[dict],
                     applicability: str | None = None) -> bool:
        changes = {"steps": steps}
        if applicability:
            changes["applicability"] = applicability
        updated = vector_index.update_record("procedure", memory_id, changes)
        if updated:
            get_usage_store().record_refinement(memory_id)
        return updated

    def get_refine_candidates(self) -> list[dict]:
        rows = with_usage("procedure", self.list_procedures(limit=10_000, status="active"))
        return [row for row in rows if row["hit_count"] is not None
                and row["hit_count"] >= REFINE_MIN_USES
                and row["refined_count"] < REFINE_MAX_REFINED][:20]

    def get_promote_candidates(self) -> list[dict]:
        rows = with_usage("procedure", self.list_procedures(limit=10_000, status="active"))
        return [row for row in rows if row["hit_count"] is not None
                and row["hit_count"] >= PROMOTE_MIN_USES
                and row["success_count"] / row["hit_count"] >= PROMOTE_MIN_SUCCESS_RATE][:10]

    def mark_promoted(self, memory_id: int | str) -> None:
        vector_index.update_record("procedure", memory_id, {"status": "promoted"})

    def demote(self, memory_id: int | str) -> None:
        vector_index.update_record("procedure", memory_id, {"status": "deprecated"})

    def delete_procedure(self, memory_id: int | str) -> bool:
        return vector_index.delete_records("procedure", [memory_id]) > 0

    def promote_to_skill(self, memory_id: int | str, auto_enable: bool = False) -> dict:
        proc = self.get_procedure(memory_id)
        if not proc:
            return {"ok": False, "skill_name": "", "reason": f"程序记忆 {memory_id} 不存在"}
        if proc["status"] != "active":
            return {"ok": False, "skill_name": "", "reason": "仅可晋升可用的程序记忆"}
        steps = proc["steps"]
        tools = sorted({step["tool"] for step in steps if step.get("tool")})
        if not tools:
            return {"ok": False, "skill_name": "", "reason": "步骤未声明工具，无法生成受限工具白名单"}
        lines = [f"你是{proc['name']}场景的处理专家，按以下步骤执行："]
        for i, step in enumerate(steps, 1):
            tool = f"（调用 {step['tool']}）" if step.get("tool") else ""
            lines.append(f"{i}. {step['action']}{tool}")
        lines.append(f"适用条件：{proc['applicability']}")
        skill_name = self._to_snake_name(proc["name"])
        from agent.skills import create_skill
        from agent.skills.models import SkillCreate

        try:
            create_skill(SkillCreate(
                name=skill_name, description=f"【程序记忆晋升】{proc['applicability'][:200]}",
                instructions="\n".join(lines),
                tool_names=tools,
                enabled=auto_enable,
            ))
        except ValueError as e:
            return {"ok": False, "skill_name": skill_name, "reason": f"晋升冲突：{e}"}
        except Exception as e:
            return {"ok": False, "skill_name": skill_name, "reason": f"创建 Skill 失败：{type(e).__name__}"}
        try:
            self.mark_promoted(memory_id)
        except Exception:
            return {"ok": False, "skill_name": skill_name,
                    "reason": f"候选 Skill {skill_name} 已创建，程序状态更新失败，请核查后处理"}
        return {"ok": True, "skill_name": skill_name, "reason": "已生成候选 Skill，请确认启用"}

    @staticmethod
    def _to_snake_name(name: str) -> str:
        value = re.sub(r"[^a-zA-Z0-9_]+", "_", name).strip("_").lower()
        if len(value) < 2 or not value[0].isalpha():
            digest = hashlib.md5(name.encode("utf-8")).hexdigest()[:8]
            value = "proc_" + (value or digest)
        return value[:60]


@lru_cache(maxsize=1)
def get_procedure_store() -> ProcedureStore:
    return ProcedureStore()
