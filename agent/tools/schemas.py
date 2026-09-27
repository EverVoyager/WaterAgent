"""防汛工具集 Schema 定义（Pydantic）。

每个工具的参数用 Pydantic 模型描述，便于：
1. 自动生成 OpenAI Function Calling 兼容的 JSON Schema
2. 在 mock 执行器中校验入参
3. 在 LangGraph 节点中复用

返回值模型（TOOL_RESULT_MODELS）用于 executor 出口的运行时校验：
工具实现漂移（字段改名/类型变化/缺失）在进入 AgentState 前被拦截，
而非等到 synthesizer 规则引擎 KeyError 或下游静默拿到坏数据。
"""
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

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


class ReadMemoryTopicParams(BaseModel):
    """读取 Agent 长期记忆中指定主题的完整内容。"""

    topic: str = Field(
        ...,
        max_length=64,
        description="主题名，须与 system prompt 长期记忆索引中的条目一致（如 'user-prefs'、'constraints'）",
    )


class ReadSessionArchiveParams(BaseModel):
    """读取历史任务段的存档全文（含当时的工具调用数据）。"""

    file: str = Field(
        ...,
        description="存档文件名，形如 'a1b2c3d4e5f67890.md'（16 位十六进制指纹）。"
        "来自历史对话上下文中任务段摘要末尾的 [存档] 行，只能使用上下文中"
        "出现过的文件名，不要自行构造或猜测。",
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
    "read_memory_topic": (
        "读取 Agent 长期记忆中指定主题的完整内容（含 created/updated 日期）。"
        "system prompt 的'长期记忆'段出于 token 预算只注入索引和与当前问题相关的主题；"
        "当索引中某主题与当前任务相关但未展开、且其细节可能影响回答时，"
        "用主题名调用本工具读取全文。主题不存在时返回 found=false。"
    ),
    "read_session_archive": (
        "读取已压缩历史任务段的存档全文（含当时的工具调用轨迹与预警等级）。"
        "历史对话上下文中的任务段摘要末尾标注了 [存档] 文件名；当用户追问"
        "早前任务的细节（如'之前查到的水位是多少''上次定的什么等级'）而摘要"
        "中的关键数据不足以回答时，用该文件名调用本工具获取完整原文。"
        "仅在摘要信息不够时使用：摘要已足够或与历史任务无关时不要调用。"
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
    "read_memory_topic": ReadMemoryTopicParams,
    "read_session_archive": ReadSessionArchiveParams,
}


# ====== 工具返回值模型（executor 出口运行时校验用） ======
#
# 必填字段 = 下游真正依赖的字段（规则引擎 compute_warning_level、跨工具
# 数据流注入 _collect_weather_context、引用核验 _check_citations 直接读取的键）；
# 随入参变化的字段（metric / analysis_type 分支）与时间戳一律 Optional，
# 但出现时校验类型。Pydantic 默认忽略多余字段——评估 overrides 注入的
# 附加键不受影响。

class GetWeatherResult(BaseModel):
    location: str
    hours: int
    total_rainfall_mm: float
    series: list[dict]
    max_hourly_rainfall_mm: float | None = None
    fetched_at: str | None = None


class GetHydrologyResult(BaseModel):
    station: str
    water_level_m: float | None = None
    warning_level_m: float | None = None
    guaranteed_level_m: float | None = None
    flow_m3_s: float | None = None
    warning_flow_m3_s: float | None = None
    fetched_at: str | None = None


class PredictRunoffResult(BaseModel):
    station: str
    lead_time_hours: int
    peak_flow_m3_s: float
    series: list[dict]
    peak_time: str | None = None
    model: str | None = None
    predicted_at: str | None = None


class QueryGisTerrainResult(BaseModel):
    bbox: str
    analysis_type: str
    slope: dict | None = None
    channel_cross_section: dict | None = None
    inundation: dict | None = None
    analyzed_at: str | None = None


class SearchRegulationResult(BaseModel):
    query: str
    top_k: int
    hits: list[dict]
    searched_at: str | None = None


class WebSearchResult(BaseModel):
    query: str
    results: list[dict]
    result_count: int
    searched_at: str | None = None


class GeneratePlanResult(BaseModel):
    warning_level: Literal["I", "II", "III", "IV"]
    actions: list[str]
    level_description: str
    affected_area: str
    population_at_risk: int
    generated_at: str | None = None


class ListSkillsResult(BaseModel):
    skills: list[dict]
    total: int
    queried_at: str | None = None


class ReadSessionArchiveResult(BaseModel):
    file: str
    content: str
    truncated: bool = False
    total_chars: int | None = None
    read_at: str | None = None
    source: str | None = None


class ReadMemoryTopicResult(BaseModel):
    topic: str
    found: bool
    content: str
    hint: str | None = None


TOOL_RESULT_MODELS = {
    "get_weather": GetWeatherResult,
    "get_hydrology": GetHydrologyResult,
    "predict_runoff": PredictRunoffResult,
    "query_gis_terrain": QueryGisTerrainResult,
    "search_regulation": SearchRegulationResult,
    "web_search": WebSearchResult,
    "generate_plan": GeneratePlanResult,
    "list_skills": ListSkillsResult,
    "read_session_archive": ReadSessionArchiveResult,
    "read_memory_topic": ReadMemoryTopicResult,
}


def validate_tool_result(tool_name: str, result: object) -> str:
    """校验工具返回值结构（executor 出口闸）。

    Returns:
        空串 = 合法（或该工具无返回值模型）；非空 = 违规描述，调用方应把
        本次调用记为 error、清空 result，阻止坏数据进入 AgentState。
    """
    model = TOOL_RESULT_MODELS.get(tool_name)
    if model is None:
        return ""
    if not isinstance(result, dict):
        return f"result 应为 dict，实际 {type(result).__name__}"
    try:
        model(**result)
    except ValidationError as e:
        details = "; ".join(
            f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}"
            for err in e.errors()[:3]
        )
        return details
    return ""


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
