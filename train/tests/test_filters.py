"""三道过滤：F1 参数合法 / F2 序列合法 / F3 等级一致（chatty 豁免）。"""
from train.data_gen.filters import FilterResult, filter_trace
from train.data_gen.hermes_format import make_tool_call_text, make_tool_response_text
from train.data_gen.scenario import generate_scenarios


def _scenario():
    return generate_scenarios(n=1, seed=42)[0]


def _trace_with(calls_and_results, final_text):
    msgs = [{"role": "user", "content": "q"}]
    for call, result in calls_and_results:
        msgs.append({"role": "assistant", "content": make_tool_call_text(call[0], call[1])})
        msgs.append({"role": "tool", "content": make_tool_response_text(result)})
    msgs.append({"role": "assistant", "content": final_text})
    return msgs


def _hydro_result(scn):
    return {"station": scn.station, **scn.tool_overrides["get_hydrology"]}


def test_f1_rejects_invalid_params():
    scn = _scenario()
    msgs = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": make_tool_call_text("get_hydrology", {"station": "吴堡", "metric": "非法值"})},
        {"role": "tool", "content": make_tool_response_text(_hydro_result(scn))},
        {"role": "assistant", "content": "Ⅱ级预警"},
    ]
    r = filter_trace(msgs, scn)
    assert r == FilterResult.REJECT_F1


def test_f2_rejects_runoff_before_weather():
    scn = _scenario()
    msgs = _trace_with(
        [(("predict_runoff", {"station": scn.station}), {"peak_flow_m3_s": 4000.0, "series": []})],
        "Ⅱ级预警",
    )
    assert filter_trace(msgs, scn) == FilterResult.REJECT_F2


def test_f2_rejects_unknown_tool():
    scn = _scenario()
    msgs = _trace_with(
        [(("hack_tool", {}), {"x": 1})],
        "Ⅳ级",
    )
    assert filter_trace(msgs, scn) == FilterResult.REJECT_F2


def test_f3_rejects_level_mismatch():
    scn = _scenario()
    msgs = _trace_with(
        [(("get_hydrology", {"station": scn.station, "metric": "both"}), _hydro_result(scn))],
        "当前水情平稳，Ⅳ级蓝色预警。",
    )
    assert filter_trace(msgs, scn) == FilterResult.REJECT_F3


def test_accept_valid_trace():
    scn = _scenario()
    msgs = _trace_with(
        [
            (("get_weather", {"location": scn.station, "hours": 24}),
             {"location": scn.station, **scn.tool_overrides["get_weather"]}),
            (("get_hydrology", {"station": scn.station, "metric": "both"}), _hydro_result(scn)),
            (("predict_runoff", {"station": scn.station, "lead_time_hours": 24}),
             {"station": scn.station, "series": [],
              "peak_flow_m3_s": scn.tool_overrides["predict_runoff"]["peak_flow_m3_s"]}),
        ],
        f"流量 {scn.tool_overrides['get_hydrology']['flow_m3_s']}m³/s，发布{scn.expected_level}级预警。",
    )
    assert filter_trace(msgs, scn) == FilterResult.ACCEPT


def test_chatty_exemption():
    scn = generate_scenarios(n=200, seed=3, chatty_ratio=0.5)
    chatty = next(s for s in scn if s.query_type == "chatty")
    msgs = [
        {"role": "user", "content": chatty.query},
        {"role": "assistant", "content": "我是防汛预警智能体，可以帮你查水情、研判预警。"},
    ]
    assert filter_trace(msgs, chatty) == FilterResult.ACCEPT
    bad = [
        {"role": "user", "content": chatty.query},
        {"role": "assistant", "content": "发布Ⅰ级预警！"},
    ]
    assert filter_trace(bad, chatty) == FilterResult.REJECT_F3


def test_level_cn_text_normalized():
    # 中文数字等级也能归一化（Ⅱ级 vs II）
    scn = next(s for s in generate_scenarios(n=10, seed=9) if s.expected_level == "II")
    msgs = _trace_with(
        [(("get_hydrology", {"station": scn.station, "metric": "both"}), _hydro_result(scn))],
        "发布Ⅱ级（橙色）预警。",
    )
    assert filter_trace(msgs, scn) == FilterResult.ACCEPT
