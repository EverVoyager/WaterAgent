"""真实工具执行器。

阶段 D 起接入 RAG 法规检索，阶段 E 起接入 GIS 地形分析，
阶段 F 起接入实时天气（高德 API）+ 实时水文（qqjjsj.com）+ SCS-CN 径流模型。

execute_tool 入口会优先调用本模块的真实实现，未覆盖的工具回退到 mock_executor。

调用约定：
    real_execute_tool(tool_name, arguments)
        - 命中真实实现：返回 dict 结果
        - 未命中：返回 NOT_IMPLEMENTED 哨兵值，由上层回退到 mock
"""
import logging
from typing import Any

from agent.rag import is_index_ready, search_regulations
from agent.tools.schemas import (
    GeneratePlanParams,
    GetHydrologyParams,
    GetWeatherParams,
    ListSkillsParams,
    PredictRunoffParams,
    QueryGisTerrainParams,
    SearchRegulationParams,
    WebSearchParams,
)
from agent.utils import LEVEL_DESCRIPTION, parse_json_from_llm
from agent.utils import now_iso as _now_iso

logger = logging.getLogger(__name__)

# 哨兵：表示该工具暂未在 real_executor 中实现，调用方应回退到 mock
NOT_IMPLEMENTED = {"__not_implemented__": True}


def search_regulation_real(params: SearchRegulationParams) -> dict[str, Any]:
    """真实法规检索：基于 Qdrant + DashScope embedding 的 RAG。

    若索引未构建，抛出 RuntimeError 让上层降级到 mock。
    """
    if not is_index_ready():
        raise RuntimeError("Qdrant 索引未构建，请先运行 backend/build_vector_store.py")

    hits = search_regulations(query=params.query, top_k=params.top_k)

    if not hits:
        logger.info("[real_executor] search_regulation 无命中: query=%s", params.query[:60])

    return {
        "query": params.query,
        "top_k": params.top_k,
        "hits": hits,
        "hit_count": len(hits),
        "searched_at": _now_iso(),
        "source": "rag_qdrant",  # 标识数据来源，便于调试
    }


def query_gis_terrain_real(params: QueryGisTerrainParams) -> dict[str, Any]:
    """真实 GIS 地形分析：基于 rasterio + SRTM DEM。

    若 DEM 数据未构建，抛出 RuntimeError 让上层降级到 mock。
    """
    from agent.gis import TerrainAnalyzer, load_study_dem
    from agent.gis.dem_loader import is_dem_ready

    if not is_dem_ready():
        raise RuntimeError(
            "SRTM DEM 数据未构建，请先运行 backend/build_terrain_data.py"
        )

    # 解析 bbox（字符串 "minx,miny,maxx,maxy" → tuple）
    bbox: tuple | None = None
    if params.bbox:
        try:
            parts = [float(x.strip()) for x in params.bbox.split(",")]
            if len(parts) != 4:
                raise ValueError("bbox 必须是 4 个数字")
            bbox = tuple(parts)
        except ValueError as e:
            raise ValueError(f"无效的 bbox 格式 '{params.bbox}': {e}") from e

    # 加载 DEM（按 bbox 裁剪，None 表示加载全部）
    dem = load_study_dem(bbox=bbox)

    # 分析
    analyzer = TerrainAnalyzer(dem)
    result = analyzer.analyze_all(analysis_type=params.analysis_type)
    result_dict = result.to_dict()

    # 标记数据来源
    result_dict["source"] = "gis_srtm_rasterio"
    result_dict["analysis_type"] = params.analysis_type
    result_dict["dem_shape"] = list(dem.shape)
    result_dict["dem_resolution_m"] = [round(dem.resolution_m[0], 2), round(dem.resolution_m[1], 2)]

    return result_dict


def get_weather_real(params: GetWeatherParams) -> dict[str, Any]:
    """真实天气查询：高德天气 API。

    若未配置 AMAP_API_KEY 或调用失败，抛 RuntimeError 让上层降级到 mock。
    """
    from agent.data.weather import fetch_weather

    return fetch_weather(location=params.location, hours=params.hours)


def get_hydrology_real(params: GetHydrologyParams) -> dict[str, Any]:
    """真实水文查询：qqjjsj.com 实时水情爬虫。

    数据源不可用时抛 RuntimeError 让上层降级到 mock。
    """
    from agent.data.hydrology import fetch_hydrology

    return fetch_hydrology(station=params.station, metric=params.metric)


def web_search_real(params: WebSearchParams) -> dict[str, Any]:
    """真实联网搜索：Tavily API。

    未配置 TAVILY_API_KEY 时抛 RuntimeError 让上层降级到 mock。
    """
    from agent.data.web_search import search_web

    return search_web(query=params.query, max_results=params.max_results)


def predict_runoff_real(params: PredictRunoffParams) -> dict[str, Any]:
    """真实径流预测：SCS-CN 降雨-径流模型。

    纯本地计算（无外部依赖），不会失败。
    降雨量优先用 params.rainfall_mm（由 workflow 从 get_weather 结果注入），
    缺失时用经验默认值 30mm（中雨级别）。
    """
    from agent.hydrology.scs_cn import predict_runoff_scs

    rainfall_mm = params.rainfall_mm if params.rainfall_mm is not None else 30.0
    if params.rainfall_series:
        return predict_runoff_scs(
            station=params.station,
            rainfall_mm=rainfall_mm,
            lead_time_hours=params.lead_time_hours,
            rainfall_series=params.rainfall_series,
        )
    return predict_runoff_scs(
        station=params.station,
        rainfall_mm=rainfall_mm,
        lead_time_hours=params.lead_time_hours,
    )


def _retrieve_regulation_context(warning_level: str) -> str:
    """检索与预警等级相关的法规条款，返回 LLM 可读的上下文字符串。

    RAG 未就绪或检索失败时返回 "(未检索到相关法规条款)"。
    """
    from agent.rag import is_index_ready, search_regulations

    default = "(未检索到相关法规条款)"
    try:
        if not is_index_ready():
            return default
        query = f"{LEVEL_DESCRIPTION[warning_level]} 应急响应 转移 预案"
        hits = search_regulations(query=query, top_k=3)
        if not hits:
            return default
        reg_parts = []
        for i, h in enumerate(hits, 1):
            reg_parts.append(
                f"({i}) {h.get('title', '')} {h.get('article', '')}: "
                f"{(h.get('content', '') or '')[:200]}"
            )
        return "\n".join(reg_parts)
    except Exception as e:
        logger.warning("[generate_plan_real] 法规检索失败，降级为无法规上下文: %s", e)
        return default


def _build_plan_prompts(
    params: GeneratePlanParams,
    regulation_context: str,
) -> tuple[str, str]:
    """构造生成预案的 system / user prompt。"""
    from agent.prompts.generate_plan import GENERATE_PLAN_SYSTEM_PROMPT
    system_prompt = GENERATE_PLAN_SYSTEM_PROMPT
    user_prompt = (
        f"预警等级：{params.warning_level} ({LEVEL_DESCRIPTION[params.warning_level]})\n"
        f"受影响区域：{params.affected_area}\n"
        f"受威胁人口：{params.population_at_risk}\n\n"
        f"相关法规条款：\n{regulation_context}\n\n"
        f"请生成具体的应急预案，返回 JSON 对象：\n"
        f'{{"warning_level": "{params.warning_level}", '
        f'"actions": ["具体措施1", "措施2", ...], '
        f'"responsible_department": "主要责任部门", '
        f'"time_limit_hours": 数字, '
        f'"resource_requirements": "物资队伍需求摘要"}}'
    )
    return system_prompt, user_prompt


def _call_plan_llm(system_prompt: str, user_prompt: str) -> dict[str, Any]:
    """调用 LLM 生成预案并解析 JSON。失败时抛 LLMError。"""
    from openai import (
        APIConnectionError,
        APIError,
        APITimeoutError,
        RateLimitError,
    )

    from agent.graph.workflow import LLMError, _classify_llm_error
    from app.core.llm import LLM_TIMEOUTS, extract_content, get_llm_client, get_llm_config

    settings = get_llm_config()
    client = get_llm_client().with_options(timeout=LLM_TIMEOUTS["synthesizer"])

    try:
        resp = client.chat.completions.create(
            model=settings["model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=4096,
        )
    except (APITimeoutError, RateLimitError, APIConnectionError, APIError) as e:
        logger.exception("[generate_plan_real] LLM 调用失败 (%s)", type(e).__name__)
        raise _classify_llm_error(e) from e
    except Exception as e:
        logger.exception("[generate_plan_real] LLM 未知异常")
        raise _classify_llm_error(e) from e

    # extract_content 仅取 message.content（不回退 reasoning_content）
    content = extract_content(resp.choices[0].message)
    plan = parse_json_from_llm(content)
    if plan is None:
        logger.error("[generate_plan_real] LLM 返回非 JSON: %s", content[:200])
        raise LLMError("format_error", "generate_plan LLM 返回非 JSON", status_code=502)
    return plan


def generate_plan_real(params: GeneratePlanParams) -> dict[str, Any]:
    """M10：真实应急预案生成：基于预警等级 + 法规条款 + 受影响区域由 LLM 生成。

    与 mock 的硬编码查表不同，真实实现：
    1. 调用 search_regulation 检索相关法规条款
    2. 把等级 + 区域 + 人口 + 法规条款组合为 LLM prompt
    3. LLM 生成结构化预案（含具体动作、责任人、时限）

    LLM 调用失败时抛 LLMError。
    """
    regulation_context = _retrieve_regulation_context(params.warning_level)
    system_prompt, user_prompt = _build_plan_prompts(params, regulation_context)
    plan = _call_plan_llm(system_prompt, user_prompt)

    # 补充元数据
    plan["warning_level"] = params.warning_level
    plan["level_description"] = LEVEL_DESCRIPTION[params.warning_level]
    plan["affected_area"] = params.affected_area
    plan["population_at_risk"] = params.population_at_risk
    plan["generated_at"] = _now_iso()
    plan["source"] = "llm_generated"
    return plan


def list_skills_real(params: ListSkillsParams) -> dict[str, Any]:
    """列出当前已启用的所有技能（Skill）。

    对标 MCP tools/list 发现机制：让 LLM 通过工具调用自主获取能力清单，
    而非通过硬编码 prompt 规则触发。

    Args:
        params: include_instructions=True 时返回完整指令文本，否则只返回元信息。

    Returns:
        {
            "skills": [{name, description, tool_names, enabled, (instructions)}],
            "total": int,
            "queried_at": ISO 时间,
            "source": "skills_store"
        }
    """
    from agent.skills import list_skills

    skills = list_skills(enabled_only=True)
    items = []
    for s in skills:
        item = {
            "name": s.name,
            "description": s.description,
            "tool_names": s.tool_names,
            "enabled": s.enabled,
        }
        if params.include_instructions:
            item["instructions"] = s.instructions
        items.append(item)

    return {
        "skills": items,
        "total": len(items),
        "queried_at": _now_iso(),
        "source": "skills_store",
    }


# 真实实现的工具映射表
_REAL_IMPLEMENTATIONS = {
    "search_regulation": search_regulation_real,
    "query_gis_terrain": query_gis_terrain_real,
    "get_weather": get_weather_real,
    "get_hydrology": get_hydrology_real,
    "web_search": web_search_real,
    "predict_runoff": predict_runoff_real,
    "generate_plan": generate_plan_real,  # M10：从 mock 升级为 LLM 生成
    "list_skills": list_skills_real,  # 技能发现工具（对标 MCP tools/list）
}


def real_execute_tool(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """真实工具执行入口。

    Returns:
        工具结果 dict。
        若工具未在 real 层实现，返回 NOT_IMPLEMENTED（由上层回退到 mock）。
    """
    if tool_name not in _REAL_IMPLEMENTATIONS:
        return NOT_IMPLEMENTED

    from agent.tools.schemas import TOOL_PARAM_MODELS
    param_model = TOOL_PARAM_MODELS[tool_name]
    try:
        params = param_model(**arguments)
    except Exception as e:
        raise ValueError(f"Invalid arguments for {tool_name}: {e}") from e

    return _REAL_IMPLEMENTATIONS[tool_name](params)


def is_tool_real_implemented(tool_name: str) -> bool:
    """该工具是否已在 real_executor 中实现。"""
    return tool_name in _REAL_IMPLEMENTATIONS
