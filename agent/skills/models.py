"""Skill 数据模型（Pydantic）。

参考 Claude Skills 的 SKILL.md 结构：
- name: 技能名（snake_case，唯一）
- description: 触发条件（Agent 扫描此字段判断是否加载）
- instructions: 完整行为指令（加载后注入 prompt）
- tool_names: 可用工具子集（空列表 = 不限制，用全部内置工具）
- enabled: 是否启用
"""
from typing import ClassVar

from pydantic import BaseModel, Field, field_validator


class SkillBase(BaseModel):
    """Skill 基础字段。"""

    name: str = Field(
        ...,
        min_length=2,
        max_length=64,
        description="技能名（snake_case，唯一标识）",
        examples=["flood_dispatch_analysis"],
    )
    description: str = Field(
        ...,
        min_length=10,
        max_length=1024,
        description="触发条件描述：什么场景下应该启用此技能",
        examples=["水库防洪调度研判，根据入库流量、水位、泄洪能力给出调度建议"],
    )
    instructions: str = Field(
        ...,
        min_length=10,
        max_length=50000,
        description="行为指令：启用后注入 planner/synthesizer 的完整指令",
        examples=[
            "你是水库防洪调度专家。工作流程：\n"
            "1. 调用 get_hydrology 获取实时水情\n"
            "2. 基于调度规程研判泄洪方案\n"
            "3. 输出推荐泄洪流量、闸门开启方案、风险提示\n"
            "约束：必须引用《水库防洪调度规程》具体条款"
        ],
    )
    tool_names: list[str] = Field(
        default_factory=list,
        description="可用工具子集（空列表 = 不限制，使用全部内置工具）",
        examples=[["get_hydrology", "search_regulation"]],
    )
    enabled: bool = Field(True, description="是否启用")


class SkillCreate(SkillBase):
    """创建 Skill 请求。"""

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """name 必须 snake_case（字母/数字/下划线，首字符为字母）。"""
        v = v.strip()
        if not v[0].isalpha():
            raise ValueError("name 首字符必须为字母")
        if not all(c.isalnum() or c == "_" for c in v):
            raise ValueError("name 只能包含字母、数字、下划线")
        return v


class SkillUpdate(BaseModel):
    """更新 Skill 请求（所有字段可选）。"""

    name: str | None = None
    description: str | None = None
    instructions: str | None = None
    tool_names: list[str] | None = None
    enabled: bool | None = None


class Skill(SkillBase):
    """完整 Skill（含系统字段）。"""

    id: str = Field(..., description="技能 ID（等于 name，作为主键）")

    # 内置工具白名单（用于校验 tool_names）
    BUILTIN_TOOLS: ClassVar[set[str]] = {
        "get_weather",
        "get_hydrology",
        "predict_runoff",
        "query_gis_terrain",
        "search_regulation",
        "web_search",
        "generate_plan",
        "list_skills",
    }

    @field_validator("tool_names")
    @classmethod
    def validate_tool_names(cls, v: list[str]) -> list[str]:
        """校验工具名是否在内置工具白名单内。"""
        invalid = set(v) - cls.BUILTIN_TOOLS
        if invalid:
            raise ValueError(
                f"未知工具名: {invalid}。可用工具: {sorted(cls.BUILTIN_TOOLS)}"
            )
        return v
