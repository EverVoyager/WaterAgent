"""防汛工具集 Schema 定义（Pydantic）。

每个工具的参数用 Pydantic 模型描述，便于：
1. 自动生成 OpenAI Function Calling 兼容的 JSON Schema
2. 在 mock 执行器中校验入参
3. 在 LangGraph 节点中复用
"""
from typing import Literal

from pydantic import BaseModel, Field

# ====== 工具入参模型 ======

class GetWeatherParams(BaseModel):
    """查询指定区域未来 N 小时的天气。"""

    location: str = Field(..., description="地点名称，如 '吕梁市'、'吴堡水文站'")
    hours: int = Field(6, ge=1, le=168, description="预测时长（小时），范围 1-168")


class GetHydrologyParams(BaseModel):
    """查询水文站实时水情。"""

    station: str = Field(..., description="水文站名称，如 '吴堡'、'龙门'、'府谷'")
    metric: Literal["water_level", "flow", "both"] = Field(
        "both", description="查询指标：water_level=水位, flow=流量, both=两者"
    )


class PredictRunoffParams(BaseModel):
    """调用径流流量预测 API。"""

    station: str = Field(..., description="预测断面对应水文站，如 '吴堡'")
    lead_time_hours: int = Field(24, ge=1, le=168, description="预见期（小时）")
    # 可选：累计降雨量 mm（由 workflow 从 get_weather 结果自动注入；LLM 也可显式传入）
    rainfall_mm: float | None = Field(
        None, ge=0, description="累计降雨量(mm)，用于驱动 SCS-CN 模型；缺失时用默认值"
    )
    # 可选：逐小时降雨序列（由 workflow 从 get_weather 自动注入）
    rainfall_series: list[dict] | None = Field(
        None, description="逐小时降雨序列 [{time, rainfall_mm}]，由系统自动注入"
    )


class QueryGisTerrainParams(BaseModel):
    """查询 GIS 地形河床信息。"""

    bbox: str | None = Field(
        None,
        description="查询范围 bbox，格式 'minx,miny,maxx,maxy'，未提供时默认吴堡断面",
    )
    analysis_type: Literal["slope", "channel_cross_section", "inundation", "all"] = Field(
        "all", description="分析类型：坡度/河床断面/淹没范围/全部"
    )


class SearchRegulationParams(BaseModel):
    """检索防汛相关法规政策。"""

    query: str = Field(..., description="检索关键词，如 '黄河防汛条例'、'转移预案'")
    top_k: int = Field(3, ge=1, le=10, description="返回前 K 条")


class WebSearchParams(BaseModel):
    """联网搜索最新信息。"""

    query: str = Field(..., description="搜索关键词，如 '黄河防汛最新政策'、'吴堡水文站最新水情'")
    max_results: int = Field(5, ge=1, le=10, description="最大返回结果数")


class GeneratePlanParams(BaseModel):
    """生成应急预案。"""

    warning_level: Literal["I", "II", "III", "IV"] = Field(
        ..., description="预警等级：I=红,II=橙,III=黄,IV=蓝"
    )
    affected_area: str = Field(..., description="受影响区域，如 '吕梁市临县'")
    population_at_risk: int = Field(..., ge=0, description="受威胁人口数")


class ListSkillsParams(BaseModel):
    """列出当前已启用的所有技能（Skill）。

    对标 MCP tools/list 的发现机制：让 LLM 自主决定何时查询自身能力，
    而非通过硬编码 prompt 规则触发。用户询问"你有哪些技能/能力/Skill"、
    或需要判断是否有合适技能处理当前任务时调用此工具。
    """

    include_instructions: bool = Field(
        False,
        description="是否包含每个技能的完整指令文本。默认 False 只返回元信息（省 token）；"
        "需要预览技能详细行为时设为 True。",
    )


# ====== 工具描述常量 ======

TOOL_DESCRIPTIONS = {
    "get_weather": "查询黄河吕梁段指定地点未来若干小时的天气预报，包括降雨量、温度等。",
    "get_hydrology": "查询指定水文站的实时水情数据，包括水位、流量。",
    "predict_runoff": "调用径流流量预测 API，对未来时段的径流过程进行预测。",
    "query_gis_terrain": "查询黄河吕梁段 GIS 地形河床信息，包括坡度、河床断面、淹没范围。",
    "search_regulation": "检索防汛相关法规政策、应急预案条款。",
    "web_search": "联网搜索最新信息（新闻、政策、水情等），返回网页标题、摘要和链接。用于获取工具无法提供的最新信息。",
    "generate_plan": "根据预警等级生成具体的应急预案方案（含动作、责任人、时限）。",
    "list_skills": (
        "列出当前已启用的所有技能（Skill）的名称、用途和允许使用的工具范围。"
        "当用户询问'你有哪些技能/能力/Skill'、或需要判断是否有合适技能处理当前任务时调用此工具。"
        "返回 JSON 数组，每个元素含 name/description/tool_names/enabled 字段。"
        "可选参数 include_instructions=true 可同时返回完整指令文本。"
    ),
}

# 工具名 → 参数模型 的映射
TOOL_PARAM_MODELS = {
    "get_weather": GetWeatherParams,
    "get_hydrology": GetHydrologyParams,
    "predict_runoff": PredictRunoffParams,
    "query_gis_terrain": QueryGisTerrainParams,
    "search_regulation": SearchRegulationParams,
    "web_search": WebSearchParams,
    "generate_plan": GeneratePlanParams,
    "list_skills": ListSkillsParams,
}


def build_openai_tools(tool_names: list[str] | None = None) -> list[dict]:
    """生成 OpenAI Function Calling 兼容的 tools 列表。

    Args:
        tool_names: 可选，仅包含指定工具子集。None 或空列表 = 全部工具。
                    用于 Skill 机制限制某技能可用的工具范围。
    """
    # 确定要包含的工具名集合
    if tool_names:
        # 过滤掉不在 TOOL_PARAM_MODELS 中的无效工具名
        names = [n for n in tool_names if n in TOOL_PARAM_MODELS]
    else:
        names = list(TOOL_PARAM_MODELS.keys())

    tools = []
    for name in names:
        model = TOOL_PARAM_MODELS[name]
        # Pydantic v2 schema 生成
        schema = model.model_json_schema()
        # 移除 pydantic 自动加的 title，使结构更接近 OpenAI 规范
        params_schema = {
            "type": schema.get("type", "object"),
            "properties": schema.get("properties", {}),
        }
        if "required" in schema:
            params_schema["required"] = schema["required"]
        tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": TOOL_DESCRIPTIONS[name],
                "parameters": params_schema,
            },
        })
    return tools
