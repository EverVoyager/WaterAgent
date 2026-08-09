"""场景生成器：确定性、配额均衡、等级真值正确、种子区间隔离。"""
from train.data_gen.scenario import Scenario, generate_scenarios


def test_deterministic_same_seed():
    a = generate_scenarios(n=20, seed=1)
    b = generate_scenarios(n=20, seed=1)
    assert [s.scenario_id for s in a] == [s.scenario_id for s in b]
    assert [s.expected_level for s in a] == [s.expected_level for s in b]


def test_level_quota_balanced():
    scenarios = generate_scenarios(n=400, seed=1)
    for level in ("I", "II", "III", "IV"):
        ratio = sum(1 for s in scenarios if s.expected_level == level) / len(scenarios)
        assert 0.20 <= ratio <= 0.30, f"{level} 占比 {ratio:.2f} 超出 ±5% 容差"


def test_level_truth_matches_thresholds():
    scenarios = generate_scenarios(n=200, seed=2)
    for s in scenarios:
        flow = s.tool_overrides["get_hydrology"]["flow_m3_s"]
        if s.expected_level == "I":
            assert flow >= 5000
        elif s.expected_level == "II":
            assert 3000 <= flow < 5000
        elif s.expected_level == "III":
            assert 2000 <= flow < 3000
        elif s.expected_level == "IV":
            assert flow < 2000


def test_seed_ranges_do_not_overlap():
    train = generate_scenarios(n=50, seed=1000)
    other = generate_scenarios(n=50, seed=101000)
    assert {s.scenario_id for s in train}.isdisjoint({s.scenario_id for s in other})


def test_scenario_fields_complete():
    s = generate_scenarios(n=1, seed=5)[0]
    assert isinstance(s, Scenario)
    assert s.station and s.query
    assert "get_hydrology" in s.tool_overrides
