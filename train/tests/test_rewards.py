"""奖励函数：门控 / 等级 / 工具 / 预案，全对=1.0，格式坏=0。"""
from train.data_gen.hermes_format import make_tool_call_text
from train.data_gen.scenario import generate_scenarios
from train.rewards.composite import compute_reward


def _scn(level="II"):
    # 必须选 multi_tool 场景：其 reference_tools 覆盖三工具，
    # 与 _good_completion 的调用集合一致
    return next(s for s in generate_scenarios(n=50, seed=21)
                if s.expected_level == level and s.query_type == "multi_tool")


def _good_completion(scn):
    flow = scn.tool_overrides["get_hydrology"]["flow_m3_s"]
    station = scn.station
    calls = (
        make_tool_call_text("get_weather", {"location": station, "hours": 24})
        + make_tool_call_text("get_hydrology", {"station": station, "metric": "both"})
        + make_tool_call_text("predict_runoff", {"station": station, "lead_time_hours": 24})
    )
    return (
        f"{calls}\n综上：流量 {flow:.0f}m³/s，发布Ⅱ级（橙色）预警。"
        "\n预案：12 小时内组织危险区域群众转移，调集抢险物资，"
        "吕梁市防汛抗旱指挥部牵头负责。依据《黄河防汛预案》第三章第十二条。"
    )


def test_full_marks():
    scn = _scn("II")
    rag_hits = [{"title": "黄河防汛预案", "article": "第三章 第十二条", "content": "……"}]
    r, parts = compute_reward(_good_completion(scn), scn, rag_hits=rag_hits)
    assert r == 1.0
    assert parts == {"r1": 0.4, "r2": 0.3, "r3": 0.3}


def test_format_gate_zero():
    scn = _scn("II")
    r, parts = compute_reward("<tool_call>{bad json}</tool_call>", scn, rag_hits=[])
    assert r == 0.0 and parts == {}


def test_adjacent_level_partial_credit():
    scn = _scn("II")
    completion = _good_completion(scn).replace("Ⅱ级（橙色）", "Ⅲ级（黄色）")
    r, parts = compute_reward(completion, scn, rag_hits=[{"title": "黄河防汛预案", "article": "第三章 第十二条"}])
    assert parts["r1"] == 0.2  # 相邻一级部分分
    assert 0.0 < r < 1.0


def test_r2_missing_weather_before_runoff():
    scn = _scn("II")
    completion = (
        "<tool_call>\n{\"name\": \"predict_runoff\", \"arguments\": {\"station\": \"吴堡\"}}\n</tool_call>"
        "\n发布Ⅱ级预警。转移群众，调集物资，指挥部负责，12 小时。依据《黄河防汛预案》第三章第十二条。"
    )
    r, parts = compute_reward(completion, scn, rag_hits=[{"title": "黄河防汛预案", "article": "第三章 第十二条"}])
    assert parts["r2"] < 0.3


def test_r3_requires_rag_hit():
    scn = _scn("II")
    r_with, p_with = compute_reward(_good_completion(scn), scn,
                                    rag_hits=[{"title": "黄河防汛预案", "article": "第三章 第十二条"}])
    r_without, p_without = compute_reward(_good_completion(scn), scn, rag_hits=[])
    assert p_with["r3"] == 0.3
    assert p_without["r3"] == 0.15  # 要素在、无 RAG 命中
