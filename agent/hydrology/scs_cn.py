"""SCS-CN 降雨-径流经验模型。

经典水文模型，由美国农业部土壤保持局（Soil Conservation Service）提出。
基于降雨量 P 和曲线数 CN 计算 runoff 深度，再结合汇流时间推出洪峰流量 Q_peak。

公式：
    S = 25400 / CN - 254  (mm，CN∈[0,100])
    λ = 0.2 (SCS 经验初损系数)
    Ia = λ * S  (初损)
    if P <= Ia:  Q = 0
    else:        Q = (P - Ia)^2 / (P - Ia + S)

    Q_peak = base_flow + 0.208 * A_eff * Q / Tc  (单位：A_eff km², Q mm, Tc h, Q_peak m³/s)

注意：SCS-CN + 合理化公式原本适用于中小流域（<250 km²）。
黄河吴堡以上集水面积 43 万 km²，单场降雨不可能均匀覆盖全流域，
因此引入"降雨有效面积 A_eff"概念——单场暴雨实际覆盖的子流域面积。
黄河中游暴雨面积一般 5000-50000 km²，默认 10000 km²（约 2.3% 流域面积）。

参考：
- USDA-SCS National Engineering Handbook, Section 4
- 黄河流域水文预报方法研究（黄河水利委员会）
- 黄河中游暴雨洪水特性分析（黄委会水文局）
"""
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# 黄河吕梁段主要水文站流域参数
# area_km2 = 完整集水面积（用于显示）
# effective_area_km2 = 单场降雨有效面积（用于洪峰计算）
STATION_BASIN = {
    "吴堡": {
        "area_km2": 433576,           # 吴堡站集水面积（黄河干流）
        "effective_area_km2": 10000,  # 单场暴雨有效面积
        "cn": 75,                     # 黄土丘陵区 CN
        "tc_hours": 12,               # 汇流时间
        "base_flow_m3_s": 900,
    },
    "龙门": {
        "area_km2": 497552,
        "effective_area_km2": 12000,
        "cn": 75,
        "tc_hours": 14,
        "base_flow_m3_s": 1000,
    },
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def compute_runoff_depth(p_mm: float, cn: float, lambda_coef: float = 0.2) -> float:
    """SCS-CN 公式：降雨 P（mm）→ 径流深 Q（mm）。

    Args:
        p_mm: 降雨量 mm
        cn: Curve Number [0, 100]
        lambda_coef: 初损系数，SCS 标准 0.2

    Returns:
        径流深 mm
    """
    if p_mm <= 0:
        return 0.0
    cn = max(1.0, min(100.0, cn))
    s = 25400.0 / cn - 254.0  # 最大可能滞留量 mm
    ia = lambda_coef * s       # 初损
    if p_mm <= ia:
        return 0.0
    q = (p_mm - ia) ** 2 / (p_mm - ia + s)
    return round(q, 2)


def compute_peak_flow(
    runoff_mm: float,
    area_km2: float,
    tc_hours: float,
    base_flow_m3_s: float = 0,
) -> float:
    """合理化公式推洪峰流量。

    Q_peak = base_flow + 0.208 * A * Q / Tc
    （单位约定：A km², Q mm, Tc h, Q_peak m³/s）

    Args:
        runoff_mm: 径流深 mm
        area_km2: 流域面积 km²
        tc_hours: 汇流时间 h
        base_flow_m3_s: 基流 m³/s

    Returns:
        洪峰流量 m³/s
    """
    if runoff_mm <= 0 or area_km2 <= 0 or tc_hours <= 0:
        return float(base_flow_m3_s)
    # 三角形汇流单位线近似：Q_peak = 0.208 * A * Q / Tc
    # 0.208 是单位换算系数（km²·mm/h → m³/s 的几何系数）
    q_peak = 0.208 * area_km2 * runoff_mm / tc_hours
    return round(q_peak + base_flow_m3_s, 0)


def _generate_flow_series(
    peak_flow: float,
    base_flow: float,
    lead_time_hours: int,
    tc_hours: float,
) -> List[Dict[str, Any]]:
    """生成流量过程线（钟形近似）。

    用 SCS 单位线思想：洪峰在 Tc 附近，前后衰减。
    """
    series = []
    base_time = datetime.now(timezone.utc)
    step = max(1, lead_time_hours // 8)
    for i in range(0, lead_time_hours + 1, step):
        # 钟形分布：peak 在 Tc 处，前后线性衰减
        if i <= tc_hours:
            ratio = i / max(tc_hours, 1)
        else:
            # 退水段：3 倍 Tc 后回到基流
            recession = max(0, 1 - (i - tc_hours) / (2 * tc_hours))
            ratio = recession
        flow = base_flow + (peak_flow - base_flow) * max(ratio, 0.1)
        series.append({
            "time": (base_time + timedelta(hours=i)).isoformat(),
            "predicted_flow_m3_s": round(flow, 0),
            "ratio_of_peak": round(max(ratio, 0.1), 2),
        })
    return series


def predict_runoff_scs(
    station: str,
    rainfall_mm: float,
    lead_time_hours: int = 24,
    rainfall_series: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """SCS-CN 模型预测径流。

    Args:
        station: 水文站名（"吴堡"/"龙门"）
        rainfall_mm: 总降雨量 mm（若提供 rainfall_series 则忽略）
        lead_time_hours: 预见期（小时）
        rainfall_series: 可选，逐小时降雨序列 [{"time", "rainfall_mm"}]

    Returns:
        与 mock_predict_runoff 兼容的 dict：
        {
            "station": str,
            "lead_time_hours": int,
            "peak_flow_m3_s": float,
            "peak_time": iso,
            "total_rainfall_mm": float,
            "runoff_depth_mm": float,
            "curve_number": float,
            "basin_area_km2": float,         # 完整集水面积
            "effective_area_km2": float,     # 降雨有效面积
            "tc_hours": float,
            "series": [{"time", "predicted_flow_m3_s", "ratio_of_peak"}],
            "model": "scs-cn-v0.1",
            "predicted_at": iso,
            "source": "scs_cn_model",
        }

    Raises:
        RuntimeError: 站点不支持
    """
    if station not in STATION_BASIN:
        raise RuntimeError(
            f"不支持的站点：{station}（支持：{list(STATION_BASIN.keys())}）"
        )

    basin = STATION_BASIN[station]
    cn = basin["cn"]
    area = basin["area_km2"]                # 完整集水面积（显示用）
    effective_area = basin["effective_area_km2"]  # 降雨有效面积（计算用）
    tc = basin["tc_hours"]
    base_flow = basin["base_flow_m3_s"]

    # 计算总降雨量
    if rainfall_series:
        total_p = sum(s.get("rainfall_mm", 0) for s in rainfall_series)
    else:
        total_p = float(rainfall_mm)

    # SCS-CN 计算径流深
    runoff_depth = compute_runoff_depth(total_p, cn)

    # 推洪峰流量（用 effective_area 而非完整 area）
    peak_flow = compute_peak_flow(
        runoff_mm=runoff_depth,
        area_km2=effective_area,
        tc_hours=tc,
        base_flow_m3_s=base_flow,
    )

    # 生成过程线
    series = _generate_flow_series(
        peak_flow=peak_flow,
        base_flow=base_flow,
        lead_time_hours=lead_time_hours,
        tc_hours=tc,
    )

    # 洪峰时间（Tc 附近）
    peak_idx = min(len(series) - 1, len(series) // 3)
    peak_time = series[peak_idx]["time"] if series else None

    return {
        "station": station,
        "lead_time_hours": lead_time_hours,
        "total_rainfall_mm": round(total_p, 1),
        "runoff_depth_mm": runoff_depth,
        "curve_number": cn,
        "basin_area_km2": area,
        "effective_area_km2": effective_area,
        "tc_hours": tc,
        "base_flow_m3_s": base_flow,
        "peak_flow_m3_s": peak_flow,
        "peak_time": peak_time,
        "series": series,
        "model": "scs-cn-v0.1",
        "predicted_at": _now_iso(),
        "source": "scs_cn_model",
    }
