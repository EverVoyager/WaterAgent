"""Agent Skill 模块（借鉴 Claude Skills 架构）。

Skill = 可配置的能力包，包含：
- name: 技能名（唯一标识）
- description: 触发条件描述（Agent 据此判断是否加载该 Skill）
- instructions: 行为指令（加载后注入 planner/synthesizer 的 system prompt）
- tool_names: 工具子集（限制该 Skill 可用的内置工具）
- enabled: 启用/禁用

与 Tool 的区别：
- Tool 是具体的函数调用（get_hydrology 等）
- Skill 是更上层的能力包，可以组合多个工具 + 自定义指令 + 触发条件

用户可通过 API/前端创建、编辑、启停 Skill，无需改代码。
"""
from agent.skills.importer import (
    ImportResult,
    import_skill_from_md,
    import_skill_from_zip,
)
from agent.skills.matcher import get_active_skill_instructions, match_skill
from agent.skills.models import Skill, SkillCreate, SkillUpdate
from agent.skills.store import (
    create_skill,
    delete_skill,
    get_enabled_skills_brief,
    get_skill,
    list_skills,
    update_skill,
)

__all__ = [
    "Skill",
    "SkillCreate",
    "SkillUpdate",
    "create_skill",
    "delete_skill",
    "get_enabled_skills_brief",
    "get_skill",
    "list_skills",
    "update_skill",
    "match_skill",
    "get_active_skill_instructions",
    "ImportResult",
    "import_skill_from_md",
    "import_skill_from_zip",
]
