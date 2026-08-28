"""综合研判：基于工具结果计算预警等级。

阶段 C 用规则化的方式做综合研判，确保即使 LLM 失败也能给出可解释的等级。
后续阶段 H 的 GRPO 训练会基于此规则构造 reward。
"""
from typing import Any

from agent.utils import (  # noqa: F401（LEVEL_DESCRIPTION 供测试 re-export）
    LEVEL_DESCRIPTION,
    WARNING_THRESHOLDS,
)


def _extract_flow(tool_results: dict[str, Any]) -> float:
    """提取最大流量 m³/s。"""
    max_flow = 0.0
    for _, val in tool_results.items():
        if not isinstance(val, dict):
            continue
        if "flow_m3_s" in val:
            max_flow = max(max_flow, float(val["flow_m3_s"]))
        if "peak_flow_m3_s" in val:
            max_flow = max(max_flow, float(val["peak_flow_m3_s"]))
        # 径流预测 series
        if "series" in val and isinstance(val["series"], list):
            for item in val["series"]:
                if isinstance(item, dict) and "predicted_flow_m3_s" in item:
                    max_flow = max(max_flow, float(item["predicted_flow_m3_s"]))
    return max_flow


def _extract_rain(tool_results: dict[str, Any]) -> float:
    """提取最大累计降雨量 mm。"""
    max_rain = 0.0
    for _, val in tool_results.items():
        if not isinstance(val, dict):
            continue
        if "total_rainfall_mm" in val:
            max_rain = max(max_rain, float(val["total_rainfall_mm"]))
        if "max_hourly_rainfall_mm" in val:
            max_rain = max(max_rain, float(val["max_hourly_rainfall_mm"]) * 24)  # 粗估24h
    return max_rain


def _extract_water_level_status(tool_results: dict[str, Any]) -> str:
    """判断水位状态：normal / warning / guaranteed。"""
    for _, val in tool_results.items():
        if not isinstance(val, dict):
            continue
        level = val.get("water_level_m")
        warn = val.get("warning_level_m")
        guar = val.get("guaranteed_level_m")
        if level is None:
            continue
        if guar is not None and level >= guar:
            return "guaranteed"
        if warn is not None and level >= warn:
            return "warning"
        return "normal"
    return "unknown"


def compute_warning_level(tool_results: dict[str, Any]) -> tuple[str, str]:
    """根据工具结果综合计算预警等级。

    Returns:
        (level, reasoning) 元组，level 为 I/II/III/IV
    """
    flow = _extract_flow(tool_results)
    rain = _extract_rain(tool_results)
    level_status = _extract_water_level_status(tool_results)

    reasons = []
    if flow > 0:
        reasons.append(f"最大流量 {flow:.0f}m³/s")
    if rain > 0:
        reasons.append(f"24h降雨约 {rain:.1f}mm")
    if level_status != "unknown":
        reasons.append(f"水位状态 {level_status}")

    # 阈值统一引用 WARNING_THRESHOLDS，避免多处硬编码不一致
    f1 = WARNING_THRESHOLDS["flow_level1"]
    f2 = WARNING_THRESHOLDS["flow_level2"]
    f3 = WARNING_THRESHOLDS["flow_level3"]
    r1 = WARNING_THRESHOLDS["rain_level1"]
    r2 = WARNING_THRESHOLDS["rain_level2"]

    # Ⅰ级：流量≥f1 / 水位超保证 / 24h降雨>r1
    if flow >= f1 or level_status == "guaranteed" or rain > r1:
        return "I", "达到Ⅰ级（红色）预警标准：" + "，".join(reasons)
    # Ⅱ级：流量 f2-f1 / 水位超警戒 / 24h降雨 r2-r1
    if f2 <= flow < f1 or level_status == "warning" or r2 <= rain <= r1:
        return "II", "达到Ⅱ级（橙色）预警标准：" + "，".join(reasons)
    # Ⅲ级：流量 f3-f2 / 水位接近警戒
    if f3 <= flow < f2:
        return "III", "达到Ⅲ级（黄色）预警标准：" + "，".join(reasons)
    # Ⅳ级：其他
    return "IV", "当前水情平稳，维持Ⅳ级（蓝色）一般预警：" + "，".join(reasons) if reasons else "暂无足够数据，默认Ⅳ级"


def get_actions_for_level(level: str, area: str = "吕梁市") -> list:
    """根据预警等级返回标准应急措施。"""
    actions = {
        "I": [
            f"立即启动Ⅰ级应急响应，{area}防指进入战时状态",
            "组织受威胁群众 24 小时内全部转移至安全区域",
            "调集抢险队伍 500 人、编织袋 10 万条、冲锋舟 5 艘",
            "每隔 1 小时通过广播、短信发布汛情通报",
            "通知沿河工矿企业停产撤人",
        ],
        "II": [
            f"启动Ⅱ级应急响应，{area}防指全员到岗",
            "组织危险区域群众 12 小时内转移",
            "调集抢险队伍 200 人、物资一批",
            "每隔 2 小时发布汛情通报",
            "加强巡堤查险，重点盯防险工险段",
        ],
        "III": [
            "启动Ⅲ级应急响应，相关县区防指到岗",
            "加强巡堤查险，重点关注低洼地段",
            "前置抢险物资到危险断面",
            "每 4 小时通报水情变化",
        ],
        "IV": [
            "启动Ⅳ级应急响应，加强监测预警",
            "通知沿河乡镇做好转移准备",
            "保持 24 小时值班，密切监视水情变化",
        ],
    }
    return actions.get(level, actions["IV"])
