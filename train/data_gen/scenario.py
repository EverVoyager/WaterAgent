"""防汛场景生成器：组合维度生成确定性场景，携带等级真值与 mock 覆盖值。

等级真值直接由流量档位决定（与 synthesizer 阈值同源）：
  I 级 >=5000 | II 级 [3000,5000) | III 级 [2000,3000) | IV 级 <2000  m³/s
种子区间约定（保证 SFT / GRPO / 评估零重叠）：
  SFT:   seed in [0, 100_000)
  GRPO:  seed in [100_000, 200_000)
  EVAL:  seed in [200_000, 300_000)

外置化改造（2026-09-14）：档位区间与站点表不再硬编码复制，由
config/thresholds.json 推导（唯一数值来源）——I 档上限 = f1×1.3、
II/III 档上限 = 下一档阈值-1（防扰动跨档），数值与旧硬编码逐位一致。
数据生成为离线确定性流程：import 时取档案快照，进程内不随热换档案
变化（重跑进程即用新档）。
"""
import random
from dataclasses import dataclass, field

from agent.thresholds import get_thresholds

STATIONS = ["吴堡", "龙门", "府谷"]
# 站点基准水位（档案快照，mock 警戒/保证线同源）
_PROFILE = get_thresholds()
STATION_BASE_LEVEL = {
    name: float(p["base_level_m"]) for name, p in _PROFILE.stations.items()
}
QUERY_TYPES = ["single_tool", "multi_tool", "plan_only"]
PERSONAS = ["防汛值班员", "乡镇干部", "沿河企业负责人"]

# 档位区间由阈值档案推导（勿手改——改 config/thresholds.json）
_LEVEL_TO_FLOW_RANGE = {
    level: _PROFILE.flow_range_for(level) for level in ("I", "II", "III", "IV")
}

_QUERY_TEMPLATES = {
    "single_tool": ["{station}站现在水情怎么样？", "查一下{station}水文站的实时流量和水位。"],
    "multi_tool": [
        "{station}站未来24小时有洪水风险吗？需要预警吗？",
        "我是{persona}，{station}站一带在下雨，帮我研判一下防汛形势。",
    ],
    "plan_only": ["{station}站已达{level_cn}预警，请生成{persona}的应急处置预案。"],
}

_LEVEL_CN = {"I": "Ⅰ级", "II": "Ⅱ级", "III": "Ⅲ级", "IV": "Ⅳ级"}


@dataclass
class Scenario:
    scenario_id: str
    station: str
    query: str
    expected_level: str
    tool_overrides: dict = field(default_factory=dict)  # 工具名 -> overrides


def _make_overrides(rng: random.Random, station: str, level: str) -> dict:
    """按等级档位生成各工具 mock 覆盖值（同 rng 保证确定性）。

    警戒/保证线取站点档案绝对值（与旧 base+2.0/+3.5 偏移结果一致）。
    """
    lo, hi = _LEVEL_TO_FLOW_RANGE[level]
    flow = round(rng.uniform(lo, hi), 1)
    st = _PROFILE.station(station)
    base_level = st["base_level_m"]
    warn = round(float(st["warning_level_m"]), 2)
    guar = round(float(st["guaranteed_level_m"]), 2)
    # 水位状态与等级对齐：I 级超保证，II 级超警戒，III/IV 正常
    if level == "I":
        water_level = round(guar + rng.uniform(0.0, 0.5), 2)
    elif level == "II":
        water_level = round(warn + rng.uniform(0.0, 0.4), 2)
    else:
        water_level = round(base_level + rng.uniform(-0.3, 0.5), 2)
    rain = _PROFILE.rain_for_level(level)
    # peak 取 flow 的 1.0-1.1 倍但不越过本档上限 hi，防止跨档改变等级真值
    # （如 II 档 flow=4900 × 1.15 = 5635 ≥ 5000 会被规则引擎误判为 I 级）
    peak = round(min(flow * rng.uniform(1.0, 1.1), hi), 1)
    return {
        "get_weather": {
            "total_rainfall_mm": rain,
            "max_hourly_rainfall_mm": round(rain / 24, 1),
        },
        "get_hydrology": {
            "flow_m3_s": flow,
            "water_level_m": water_level,
            "warning_level_m": warn,
            "guaranteed_level_m": guar,
        },
        "predict_runoff": {"peak_flow_m3_s": peak},
    }


def generate_scenarios(n: int, seed: int) -> list:
    """生成 n 条确定性场景。等级在业务场景内均匀轮换。"""
    rng = random.Random(seed)
    scenarios = []
    levels_cycle = ["I", "II", "III", "IV"]

    for i in range(n):
        level = levels_cycle[i % 4]  # 轮换保证严格均衡
        station = rng.choice(STATIONS)
        persona = rng.choice(PERSONAS)
        qtype = rng.choice(QUERY_TYPES)  # 仅用于选模板，不暴露为字段
        template = rng.choice(_QUERY_TEMPLATES[qtype])
        query = template.format(station=station, persona=persona, level_cn=_LEVEL_CN[level])
        scenarios.append(Scenario(
            scenario_id=f"scn-{seed}-{i}",
            station=station,
            query=query,
            expected_level=level,
            tool_overrides=_make_overrides(rng, station, level),
        ))

    rng.shuffle(scenarios)
    return scenarios


def from_expanded_queries(expanded_queries: list, seed: int) -> list:
    """从扩张后的查询列表创建 Scenario（种子扩张流程用）。

    expanded_queries 是 train.data_gen.query_expander.ExpandedQuery 的列表，
    每个包含 query/station/level/intent 字段。
    """
    rng = random.Random(seed)
    scenarios = []
    for i, eq in enumerate(expanded_queries):
        scenarios.append(Scenario(
            scenario_id=f"scn-{seed}-exp-{i}",
            station=eq.station,
            query=eq.query,
            expected_level=eq.level,
            tool_overrides=_make_overrides(rng, eq.station, eq.level),
        ))
    rng.shuffle(scenarios)
    return scenarios
