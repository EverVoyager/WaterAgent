"""确定性 mock：同 seed+overrides 数值完全一致；覆盖值生效。

注意：mock 结果含真实时间戳（fetched_at / series[].time），确定性只保证
数值字段，比较时剔除时间字段。
"""
from agent.tools.mock_executor import execute_tool


def _strip_times(obj):
    if isinstance(obj, dict):
        return {k: _strip_times(v) for k, v in obj.items()
                if not (k.endswith("_at") or k == "time")}
    if isinstance(obj, list):
        return [_strip_times(x) for x in obj]
    return obj


def test_deterministic_with_seed():
    a = execute_tool("get_hydrology", {"station": "吴堡", "metric": "both"}, seed=42)
    b = execute_tool("get_hydrology", {"station": "吴堡", "metric": "both"}, seed=42)
    assert _strip_times(a) == _strip_times(b)


def test_overrides_inject_values():
    out = execute_tool(
        "get_hydrology",
        {"station": "吴堡", "metric": "both"},
        overrides={"flow_m3_s": 5200.0, "water_level_m": 644.5},
        seed=42,
    )
    assert out["flow_m3_s"] == 5200.0
    assert out["water_level_m"] == 644.5


def test_overrides_none_keeps_existing_signature():
    # 不传新参数 = 现状行为（只断言结构，不断言具体随机值）
    out = execute_tool("get_weather", {"location": "吴堡", "hours": 6})
    assert "series" in out and len(out["series"]) == 6


def test_runoff_peak_override():
    out = execute_tool(
        "predict_runoff",
        {"station": "吴堡", "lead_time_hours": 24},
        overrides={"peak_flow_m3_s": 6100.0},
        seed=7,
    )
    assert out["peak_flow_m3_s"] == 6100.0
    assert max(s["predicted_flow_m3_s"] for s in out["series"]) <= 6100.0
