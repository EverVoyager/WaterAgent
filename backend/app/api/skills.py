"""Skill 管理 API（借鉴 Claude Skills 架构）。

用户可通过此接口创建、编辑、启停 Skill，无需改代码。
Agent 运行时会自动匹配并加载启用的 Skill。
"""
import logging

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from agent.skills import (
    Skill,
    SkillCreate,
    SkillUpdate,
    create_skill,
    delete_skill,
    get_skill,
    import_skill_from_md,
    import_skill_from_zip,
    list_skills,
    update_skill,
)
from agent.skills.importer import MAX_UPLOAD_SIZE, VALID_STRATEGIES
from agent.skills.matcher import invalidate_cache

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/skills", tags=["skills"])

# 内置工具列表（供前端渲染工具子集选择）
BUILTIN_TOOLS = [
    {"name": "get_weather", "description": "查询天气预报（降雨量、温度）"},
    {"name": "get_hydrology", "description": "查询水文站实时水情（水位、流量）"},
    {"name": "predict_runoff", "description": "径流流量预测（SCS-CN 模型）"},
    {"name": "query_gis_terrain", "description": "GIS 地形河床分析（坡度、断面、淹没）"},
    {"name": "search_regulation", "description": "法规政策检索（RAG）"},
    {"name": "web_search", "description": "联网搜索最新信息"},
    {"name": "generate_plan", "description": "生成应急预案方案"},
]


class SkillResponse(BaseModel):
    """Skill 响应模型。"""

    id: str = Field(..., description="技能 ID（等于 name）")
    name: str = Field(..., description="技能名")
    description: str = Field(..., description="触发条件描述")
    instructions: str = Field(..., description="行为指令")
    tool_names: list[str] = Field(default_factory=list, description="工具子集")
    enabled: bool = Field(..., description="是否启用")


class ImportResultResponse(BaseModel):
    """Skill 包导入结果响应。"""

    skill: SkillResponse | None = Field(None, description="导入后的 Skill")
    action: str = Field(..., description="导入动作: created/overwritten/renamed")
    original_name: str = Field(..., description="原始技能名")
    final_name: str = Field(..., description="最终技能名（rename 后可能不同）")
    warnings: list[str] = Field(default_factory=list, description="告警信息（如工具被过滤）")


def _to_response(skill: Skill) -> SkillResponse:
    return SkillResponse(
        id=skill.id,
        name=skill.name,
        description=skill.description,
        instructions=skill.instructions,
        tool_names=skill.tool_names,
        enabled=skill.enabled,
    )


@router.get("/tools", response_model=list[dict])
def get_builtin_tools():
    """获取内置工具列表（供前端渲染工具子集选择）。"""
    return BUILTIN_TOOLS


@router.get("", response_model=list[SkillResponse])
def list_all_skills(enabled_only: bool = False):
    """列出所有 Skill。"""
    skills = list_skills(enabled_only=enabled_only)
    return [_to_response(s) for s in skills]


@router.get("/{skill_name}", response_model=SkillResponse)
def get_one_skill(skill_name: str):
    """获取单个 Skill 详情。"""
    skill = get_skill(skill_name)
    if skill is None:
        raise HTTPException(status_code=404, detail=f"Skill '{skill_name}' 不存在")
    return _to_response(skill)


@router.post("", response_model=SkillResponse, status_code=201)
def create_one_skill(req: SkillCreate):
    """创建 Skill。"""
    try:
        skill = create_skill(req)
        invalidate_cache()
        return _to_response(skill)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/import", response_model=ImportResultResponse)
async def import_skill_package(
    file: UploadFile = File(..., description="Skill 包文件（.zip/.skill/.md）"),
    conflict_strategy: str = Form(
        "cancel",
        description="冲突策略: overwrite(覆盖) / rename(重命名) / cancel(取消)",
    ),
):
    """导入 Skill 包。

    支持两种格式：
    - .zip / .skill 压缩包（内含 SKILL.md）
    - 单个 .md 文件（SKILL.md 内容）

    兼容 Claude Skills 开放标准（YAML frontmatter + Markdown 正文）。
    """
    if conflict_strategy not in VALID_STRATEGIES:
        raise HTTPException(
            status_code=400,
            detail=f"无效的冲突策略: {conflict_strategy}，可选: {sorted(VALID_STRATEGIES)}",
        )

    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"文件过大: {len(content)} bytes，超过限制 {MAX_UPLOAD_SIZE} bytes",
        )

    filename = file.filename or ""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    try:
        if ext in ("zip", "skill"):
            result = import_skill_from_zip(content, conflict_strategy=conflict_strategy)
        elif ext == "md":
            text = content.decode("utf-8")
            result = import_skill_from_md(text, conflict_strategy=conflict_strategy)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的文件类型: .{ext}，仅支持 .zip / .skill / .md",
            )
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except UnicodeDecodeError as e:
        raise HTTPException(
            status_code=400, detail=f"文件编码错误（需 UTF-8）: {e}"
        ) from e

    return ImportResultResponse(
        skill=_to_response(result.skill) if result.skill else None,
        action=result.action,
        original_name=result.original_name,
        final_name=result.final_name,
        warnings=result.warnings,
    )


@router.put("/{skill_name}", response_model=SkillResponse)
def update_one_skill(skill_name: str, req: SkillUpdate):
    """更新 Skill（name 不可改）。"""
    try:
        skill = update_skill(skill_name, req)
        invalidate_cache()
        return _to_response(skill)
    except ValueError as e:
        if "不存在" in str(e):
            raise HTTPException(status_code=404, detail=str(e)) from e
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.patch("/{skill_name}/toggle", response_model=SkillResponse)
def toggle_skill(skill_name: str):
    """启用/禁用 Skill（快捷开关）。"""
    skill = get_skill(skill_name)
    if skill is None:
        raise HTTPException(status_code=404, detail=f"Skill '{skill_name}' 不存在")
    new_enabled = not skill.enabled
    updated = update_skill(skill_name, SkillUpdate(enabled=new_enabled))
    invalidate_cache()
    return _to_response(updated)


@router.delete("/{skill_name}", status_code=204)
def delete_one_skill(skill_name: str):
    """删除 Skill。"""
    if not delete_skill(skill_name):
        raise HTTPException(status_code=404, detail=f"Skill '{skill_name}' 不存在")
    invalidate_cache()
