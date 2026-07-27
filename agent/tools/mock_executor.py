"""Mock 工具执行器。

阶段 B 用模拟数据先跑通 Agent 链路；阶段 C/D/E 接入真实 API 时，
只需替换 executor 内部实现，调用方保持不变。

训练侧扩展（Task 3）：新增可选 `overrides`/`seed` 参数，默认 None 走原逻辑；
真实实现分支不受影响，overrides/seed 仅作用于 mock 回放。
"""
import random
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

from agent.tools.schemas import (
    TOOL_PARAM_MODELS,
    GeneratePlanParams,
    GetHydrologyParams,
    GetWeatherParams,
    PredictRunoffParams,
    QueryGisTerrainParams,
    SearchRegulationParams,
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ====== 各工具的 mock 实现 ======

def _mock_get_weather(params: GetWeatherParams, overrides: dict | None = None) -> Dict[str, Any]:
    """模拟天气查询：生成未来 N 小时降雨数据。"""
    series = []
    base_time = datetime.now(timezone.utc)
    # 模拟一次降雨过程：前半段小雨，中段大雨，后段转晴
    for i in range(params.hours):
        if i < params.hours // 3:
            rain = round(random.uniform(0.5, 2.5), 1)
        elif i < params.hours * 2 // 3:
            rain = round(random.uniform(5.0, 15.0), 1)
        else:
            rain = round(random.uniform(0.0, 1.0), 1)
        series.append({
            "time": (base_time + timedelta(hours=i)).isoformat(),
            "rainfall_mm": rain,
            "temperature_c": round(random.uniform(18, 28), 1),
            "wind_speed_ms": round(random.uniform(2, 8), 1),
        })
    total_rain = round(sum(s["rainfall_mm"] for s in series), 1)
    result = {
        "location": params.location,
        "hours": params.hours,
        "total_rainfall_mm": total_rain,
        "max_hourly_rainfall_mm": max(s["rainfall_mm"] for s in series),
        "series": series,
        "fetched_at": _now_iso(),
    }
    if overrides:
        result.update(overrides)
    return result


def _mock_get_hydrology(params: GetHydrologyParams, overrides: dict | None = None) -> Dict[str, Any]:
    """模拟水文站实时水情。"""
    station_data = {
        "吴堡": {"base_level": 640.5, "base_flow": 1200},
        "龙门": {"base_level": 382.3, "base_flow": 2400},
        "府谷": {"base_level": 810.2, "base_flow": 850},
    }
    base = station_data.get(params.station, {"base_level": 500.0, "base_flow": 1000})
    result = {
        "station": params.station,
        "fetched_at": _now_iso(),
    }
    if params.metric in ("water_level", "both"):
        result["water_level_m"] = round(base["base_level"] + random.uniform(-0.5, 2.5), 2)
        result["warning_level_m"] = round(base["base_level"] + 2.0, 2)
        result["guaranteed_level_m"] = round(base["base_level"] + 3.5, 2)
    if params.metric in ("flow", "both"):
        result["flow_m3_s"] = round(base["base_flow"] * random.uniform(1.0, 2.5), 0)
        result["warning_flow_m3_s"] = round(base["base_flow"] * 2.0, 0)
    if overrides:
        result.update(overrides)
    return result


def _mock_predict_runoff(params: PredictRunoffParams, overrides: dict | None = None) -> Dict[str, Any]:
    """模拟径流流量预测 API 返回。

    overrides 特殊处理：peak_flow_m3_s 需在 series 生成前生效（驱动曲线形状），
    其余键在结果组装后直接覆盖。
    """
    ov = overrides or {}
    peak_flow = round(float(ov.get("peak_flow_m3_s", random.uniform(3000, 8000))), 0)
    series = []
    base_time = datetime.now(timezone.utc)
    for i in range(0, params.lead_time_hours + 1, 3):
        # 简化的钟形曲线
        ratio = 1.0 - abs(i - params.lead_time_hours / 2) / (params.lead_time_hours / 2)
        flow = round(peak_flow * max(ratio, 0.3), 0)
        series.append({
            "time": (base_time + timedelta(hours=i)).isoformat(),
            "predicted_flow_m3_s": flow,
        })
    result = {
        "station": params.station,
        "lead_time_hours": params.lead_time_hours,
        "peak_flow_m3_s": peak_flow,
        "peak_time": series[len(series) // 2]["time"] if series else None,
        "series": series,
        "model": "mock-lstm-v0.1",
        "predicted_at": _now_iso(),
    }
    for k, v in ov.items():
        if k != "peak_flow_m3_s":  # peak 已用于曲线，其余键直接覆盖
            result[k] = v
    return result


def _mock_query_gis_terrain(params: QueryGisTerrainParams, overrides: dict | None = None) -> Dict[str, Any]:
    """模拟 GIS 地形分析。"""
    result = {
        "bbox": params.bbox or "110.7,37.4,111.2,37.8",  # 默认吴堡附近
        "analysis_type": params.analysis_type,
        "analyzed_at": _now_iso(),
    }
    if params.analysis_type in ("slope", "all"):
        result["slope"] = {
            "mean_degree": round(random.uniform(5, 20), 2),
            "max_degree": round(random.uniform(30, 60), 2),
            "high_risk_area_km2": round(random.uniform(10, 80), 1),
        }
    if params.analysis_type in ("channel_cross_section", "all"):
        result["channel_cross_section"] = {
            "width_m": round(random.uniform(200, 500), 1),
            "max_depth_m": round(random.uniform(3, 8), 1),
            "avg_depth_m": round(random.uniform(1.5, 4), 1),
        }
    if params.analysis_type in ("inundation", "all"):
        result["inundation"] = {
            "inundated_area_km2": round(random.uniform(5, 50), 1),
            "affected_villages": random.randint(3, 15),
        }
    if overrides:
        result.update(overrides)
    return result


def _mock_search_regulation(params: SearchRegulationParams, overrides: dict | None = None) -> Dict[str, Any]:
    """模拟法规检索。"""
    docs = [
        {
            "title": "中华人民共和国防洪法",
            "article": "第四十一条",
            "content": "当江河、湖泊水位接近保证水位或者安全流量时，有关县级以上人民政府防汛指挥机构可以宣布进入紧急防汛期。",
            "score": 0.92,
        },
        {
            "title": "黄河防汛预案",
            "article": "第三章 第十二条",
            "content": "黄河中游出现 5000m³/s 以上洪峰时，应启动Ⅱ级应急响应，组织沿河低洼地区人员转移。",
            "score": 0.88,
        },
        {
            "title": "山西省防汛抗旱应急预案",
            "article": "第五章",
            "content": "吕梁市辖区内黄河干流出现警戒水位以上洪水时，由吕梁市防汛指挥部统一调度。",
            "score": 0.85,
        },
        {
            "title": "水利部水情信息编码标准",
            "article": "附录 A",
            "content": "水文站水情信息应在 10 分钟内上报至省级水情中心。",
            "score": 0.79,
        },
        {
            "title": "黄河水量调度条例",
            "article": "第十八条",
            "content": "汛期水量调度服从防汛指挥机构统一调度。",
            "score": 0.74,
        },
    ]
    result = {
        "query": params.query,
        "top_k": params.top_k,
        "hits": docs[: params.top_k],
        "searched_at": _now_iso(),
    }
    if overrides:
        result.update(overrides)
    return result


def _mock_generate_plan(params: GeneratePlanParams, overrides: dict | None = None) -> Dict[str, Any]:
    """模拟生成应急预案。"""
    level_desc = {
        "I": "Ⅰ级（红色）特别重大",
        "II": "Ⅱ级（橙色）重大",
        "III": "Ⅲ级（黄色）较大",
        "IV": "Ⅳ级（蓝色）一般",
    }
    actions_by_level = {
        "I": [
            "立即启动Ⅰ级应急响应，市防指进入战时状态",
            "组织受威胁群众 24 小时内全部转移至安全区域",
            "调集抢险队伍 500 人、编织袋 10 万条",
            "每隔 1 小时通过广播、短信发布汛情通报",
        ],
        "II": [
            "启动Ⅱ级应急响应，市防指全员到岗",
            "组织危险区域群众 12 小时内转移",
            "调集抢险队伍 200 人、物资一批",
            "每隔 2 小时发布汛情通报",
        ],
        "III": [
            "启动Ⅲ级应急响应，相关县区防指到岗",
            "加强巡堤查险，重点关注低洼地段",
            "前置抢险物资",
        ],
        "IV": [
            "启动Ⅳ级应急响应，加强监测预警",
            "通知沿河乡镇做好转移准备",
        ],
    }
    result = {
        "warning_level": params.warning_level,
        "level_description": level_desc[params.warning_level],
        "affected_area": params.affected_area,
        "population_at_risk": params.population_at_risk,
        "actions": actions_by_level[params.warning_level],
        "responsible_department": "吕梁市防汛抗旱指挥部",
        "time_limit_hours": {"I": 24, "II": 12, "III": 24, "IV": 48}[params.warning_level],
        "generated_at": _now_iso(),
    }
    if overrides:
        result.update(overrides)
    return result


# ====== 执行器入口 ======

_MOCK_IMPLEMENTATIONS = {
    "get_weather": _mock_get_weather,
    "get_hydrology": _mock_get_hydrology,
    "predict_runoff": _mock_predict_runoff,
    "query_gis_terrain": _mock_query_gis_terrain,
    "search_regulation": _mock_search_regulation,
    "generate_plan": _mock_generate_plan,
}


def execute_tool(
    tool_name: str,
    arguments: Dict[str, Any],
    overrides: Dict[str, Any] | None = None,
    seed: int | None = None,
) -> Dict[str, Any]:
    """统一工具执行入口。

    优先调用 real_executor 中的真实实现（如 RAG 法规检索），
    未覆盖或索引未就绪时回退到 mock 实现。

    Args:
        tool_name: 工具名称
        arguments: 工具入参 dict
        overrides: 训练侧确定性回放用，仅作用于 mock 分支；真实实现分支忽略
        seed: 训练侧确定性回放用，设全局 random 种子；真实实现分支忽略

    Returns:
        工具执行结果 dict

    Raises:
        ValueError: 工具名未知或入参不合法
    """
    if tool_name not in _MOCK_IMPLEMENTATIONS:
        raise ValueError(f"Unknown tool: {tool_name}")

    # 训练侧确定性：设全局种子（仅 mock 分支依赖 random）
    if seed is not None:
        random.seed(seed)

    # overrides/seed 是训练侧确定性回放信号：跳过 real 分支，直接走 mock，
    # 避免真实实现（如 qqjjsj.com 爬虫）返回非确定数据覆盖 overrides。
    # 正常运行（overrides=seed=None）仍优先 real_executor。
    if overrides is None and seed is None:
        # 优先尝试真实实现（阶段 D 起：search_regulation 走 RAG）
        try:
            from agent.tools.real_executor import real_execute_tool
            result = real_execute_tool(tool_name, arguments)
            if not result.get("__not_implemented__"):
                return result
        except RuntimeError as e:
            # 真实实现依赖未就绪（如 FAISS 索引未构建），降级到 mock
            import logging
            logging.getLogger(__name__).warning(
                "[executor] %s 真实实现不可用，降级到 mock: %s", tool_name, e,
            )
        except Exception as e:
            import logging
            logging.getLogger(__name__).exception(
                "[executor] %s 真实实现异常，降级到 mock: %s", tool_name, e,
            )

    # 回退到 mock
    param_model = TOOL_PARAM_MODELS[tool_name]
    # Pydantic 校验入参
    try:
        params = param_model(**arguments)
    except Exception as e:
        raise ValueError(f"Invalid arguments for {tool_name}: {e}") from e

    impl = _MOCK_IMPLEMENTATIONS[tool_name]
    mock_result = impl(params, overrides=overrides)
    # 标记数据来源，便于调试
    if isinstance(mock_result, dict):
        mock_result["source"] = "mock"
    return mock_result
