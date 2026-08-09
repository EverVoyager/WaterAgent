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
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from agent.rag import is_index_ready, search_regulations
from agent.tools.schemas import (
    GeneratePlanParams,
    GetHydrologyParams,
    GetWeatherParams,
    PredictRunoffParams,
    QueryGisTerrainParams,
    SearchRegulationParams,
)

logger = logging.getLogger(__name__)

# 哨兵：表示该工具暂未在 real_executor 中实现，调用方应回退到 mock
NOT_IMPLEMENTED = {"__not_implemented__": True}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def search_regulation_real(params: SearchRegulationParams) -> Dict[str, Any]:
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


def query_gis_terrain_real(params: QueryGisTerrainParams) -> Dict[str, Any]:
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
    bbox: Optional[tuple] = None
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


def get_weather_real(params: GetWeatherParams) -> Dict[str, Any]:
    """真实天气查询：高德天气 API。

    若未配置 AMAP_API_KEY 或调用失败，抛 RuntimeError 让上层降级到 mock。
    """
    from agent.data.weather import fetch_weather

    return fetch_weather(location=params.location, hours=params.hours)


def get_hydrology_real(params: GetHydrologyParams) -> Dict[str, Any]:
    """真实水文查询：qqjjsj.com 实时水情爬虫。

    数据源不可用时抛 RuntimeError 让上层降级到 mock。
    """
    from agent.data.hydrology import fetch_hydrology

    return fetch_hydrology(station=params.station, metric=params.metric)


def predict_runoff_real(params: PredictRunoffParams) -> Dict[str, Any]:
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


def generate_plan_real(params: GeneratePlanParams) -> Dict[str, Any]:
    """M10：真实应急预案生成：基于预警等级 + 法规条款 + 受影响区域由 LLM 生成。

    与 mock 的硬编码查表不同，真实实现：
    1. 调用 search_regulation 检索相关法规条款
    2. 把等级 + 区域 + 人口 + 法规条款组合为 LLM prompt
    3. LLM 生成结构化预案（含具体动作、责任人、时限）

    LLM 调用失败时抛 LLMError。
    """
    import json
    import logging
    from agent.rag import is_index_ready, search_regulations
    from app.core.llm import LLM_TIMEOUTS, get_llm_client, get_llm_config, strip_think
    from agent.graph.workflow import LLMError, _classify_llm_error
    from openai import (
        APIConnectionError, APITimeoutError, APIError, RateLimitError,
    )

    logger = logging.getLogger(__name__)

    level_desc = {
        "I": "Ⅰ级（红色）特别重大",
        "II": "Ⅱ级（橙色）重大",
        "III": "Ⅲ级（黄色）较大",
        "IV": "Ⅳ级（蓝色）一般",
    }

    # 步骤 1：检索相关法规条款（若 RAG 就绪）
    regulation_context = "(未检索到相关法规条款)"
    try:
        if is_index_ready():
            query = f"{level_desc[params.warning_level]} 应急响应 转移 预案"
            hits = search_regulations(query=query, top_k=3)
            if hits:
                reg_parts = []
                for i, h in enumerate(hits, 1):
                    reg_parts.append(
                        f"({i}) {h.get('title', '')} {h.get('article', '')}: "
                        f"{(h.get('content', '') or '')[:200]}"
                    )
                regulation_context = "\n".join(reg_parts)
    except Exception as e:
        logger.warning("[generate_plan_real] 法规检索失败，降级为无法规上下文: %s", e)

    # 步骤 2：LLM 生成结构化预案
    settings = get_llm_config()
    client = get_llm_client().with_options(timeout=LLM_TIMEOUTS["synthesizer"])

    system_prompt = (
        "你是黄河吕梁段防汛预案生成模块。基于预警等级、受影响区域、人口和法规条款，"
        "生成具体的应急预案。\n"
        "要求：\n"
        "1. 动作要具体可执行（如'调集抢险队伍 200 人'而非'调集队伍'）\n"
        "2. 责任部门明确（市防指/县防指/乡镇政府）\n"
        "3. 时限清晰（如'12小时内完成转移'）\n"
        "4. 措施数量 3-6 条，按执行顺序排列\n"
        "5. 严格依据提供的法规条款，不编造法规\n"
    )
    user_prompt = (
        f"预警等级：{params.warning_level} ({level_desc[params.warning_level]})\n"
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

    try:
        resp = client.chat.completions.create(
            model=settings["model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=1024,
        )
    except (APITimeoutError, RateLimitError, APIConnectionError, APIError) as e:
        logger.exception("[generate_plan_real] LLM 调用失败 (%s)", type(e).__name__)
        raise _classify_llm_error(e) from e
    except Exception as e:
        logger.exception("[generate_plan_real] LLM 未知异常")
        raise _classify_llm_error(e) from e

    content = strip_think((resp.choices[0].message.content or "").strip())
    # 兼容 ```json ``` 包裹
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()
        if content.endswith("```"):
            content = content[:-3].strip()

    try:
        plan = json.loads(content)
    except json.JSONDecodeError as e:
        logger.error("[generate_plan_real] LLM 返回非 JSON: %s", content[:200])
        # 格式错误时抛 LLMError 让上层降级到 mock
        raise LLMError("format_error", f"generate_plan LLM 返回非 JSON：{e}", status_code=502) from e

    # 补充元数据
    plan["warning_level"] = params.warning_level
    plan["level_description"] = level_desc[params.warning_level]
    plan["affected_area"] = params.affected_area
    plan["population_at_risk"] = params.population_at_risk
    plan["generated_at"] = _now_iso()
    plan["source"] = "llm_generated"
    return plan


# 真实实现的工具映射表
_REAL_IMPLEMENTATIONS = {
    "search_regulation": search_regulation_real,
    "query_gis_terrain": query_gis_terrain_real,
    "get_weather": get_weather_real,
    "get_hydrology": get_hydrology_real,
    "predict_runoff": predict_runoff_real,
    "generate_plan": generate_plan_real,  # M10：从 mock 升级为 LLM 生成
}


def real_execute_tool(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
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
