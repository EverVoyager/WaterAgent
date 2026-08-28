"""Hermes 轨迹序列化/解析互逆 + 非法输入拒绝。"""

from train.data_gen.hermes_format import (
    extract_tool_calls,
    make_tool_call_text,
    parse_final_answer,
    parse_trace,
    round_trip_ok,
)


def test_make_and_extract_tool_call_roundtrip():
    text = make_tool_call_text("get_hydrology", {"station": "吴堡", "metric": "both"})
    calls = extract_tool_calls(text)
    assert calls == [{"name": "get_hydrology", "arguments": {"station": "吴堡", "metric": "both"}}]


def test_extract_multiple_tool_calls():
    text = (
        make_tool_call_text("get_weather", {"location": "吴堡", "hours": 24})
        + make_tool_call_text("get_hydrology", {"station": "吴堡", "metric": "both"})
    )
    assert [c["name"] for c in extract_tool_calls(text)] == ["get_weather", "get_hydrology"]


def test_extract_rejects_bad_json():
    assert extract_tool_calls("<tool_call>\n{not json}\n</tool_call>") == []


def test_parse_trace_rejects_non_json_tool_response():
    msgs = [
        {"role": "user", "content": "查水情"},
        {"role": "assistant", "content": make_tool_call_text("get_hydrology", {"station": "吴堡"})},
        {"role": "tool", "content": "不是JSON"},
    ]
    assert parse_trace(msgs) is None


def test_parse_final_answer_level():
    msgs = [{"role": "assistant", "content": "……综上，发布Ⅱ级（橙色）预警……"}]
    assert parse_final_answer(msgs) == "II"


def test_round_trip_ok():
    msgs = [
        {"role": "user", "content": "吴堡水情？"},
        {"role": "assistant", "content": make_tool_call_text("get_hydrology", {"station": "吴堡"})},
        {"role": "tool", "content": '{"station": "吴堡", "flow_m3_s": 3250.0}'},
        {"role": "assistant", "content": "吴堡站流量 3250m³/s，达到Ⅱ级预警。"},
    ]
    assert round_trip_ok(msgs)
