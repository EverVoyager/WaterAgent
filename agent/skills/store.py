"""Skill 存储 CRUD（MySQL 后端，P2-b 迁移自 JSON 文件）。

存储位置：MySQL skills 表（由 SkillStore 管理）
硬失败策略：MySQL 不可用时抛 RuntimeError，不降级到 JSON 文件。

函数 API 保持不变，向后兼容：
    from agent.skills.store import list_skills, get_skill, create_skill, ...
"""
import logging

from agent.skills.models import Skill, SkillCreate, SkillUpdate
from agent.skills.skill_store import get_skill_store

logger = logging.getLogger(__name__)


def list_skills(enabled_only: bool = False) -> list[Skill]:
    """列出所有 Skill。

    Args:
        enabled_only: 仅返回 enabled=True 的 Skill
    """
    return get_skill_store().list_skills(enabled_only)


def get_skill(name: str) -> Skill | None:
    """按 name 获取单个 Skill。不存在返回 None。"""
    return get_skill_store().get_skill(name)


def create_skill(req: SkillCreate) -> Skill:
    """创建 Skill。name 已存在时抛 ValueError。"""
    return get_skill_store().create_skill(req)


def update_skill(name: str, req: SkillUpdate) -> Skill:
    """更新 Skill。不存在时抛 ValueError。不允许修改 name。"""
    return get_skill_store().update_skill(name, req)


def delete_skill(name: str) -> bool:
    """删除 Skill。不存在返回 False。"""
    return get_skill_store().delete_skill(name)


def get_enabled_skills_brief() -> str:
    """返回已启用 Skill 的摘要文本（name + description），供注入 LLM system prompt。

    让 LLM 始终知道自己有哪些 Skill，从而能回答"你有哪些技能"类问题。
    只含元信息（name + description），不含完整 instructions（按需加载）。

    Returns:
        格式化的 Skill 列表文本；无启用 Skill 时返回空字符串。

    Note:
        按 name 排序输出。该文本注入 system prompt，顺序必须确定，
        否则数据库返回顺序变化会破坏 LLM 请求的前缀缓存
        （KV Cache 要求静态前缀逐字一致）。
    """
    skills = list_skills(enabled_only=True)
    if not skills:
        return ""
    lines = []
    for s in sorted(skills, key=lambda s: s.name):
        tools_hint = f"（工具: {', '.join(s.tool_names)}）" if s.tool_names else ""
        lines.append(f"- {s.name}: {s.description}{tools_hint}")
    return "\n".join(lines)
