"""教师合成：轨迹拼装、断点续传、轮次上限。"""
import json
from pathlib import Path
from unittest.mock import MagicMock

from train.data_gen.scenario import generate_scenarios
from train.data_gen.teacher import synthesize_dataset, synthesize_one


def _scenario():
    return generate_scenarios(n=20, seed=7)[0]


def _fc_response(tool_name: str, arguments: dict, call_id: str = "call_1"):
    """构造 OpenAI SDK 风格的 tool_calls 响应对象。"""
    call = MagicMock()
    call.id = call_id
    call.type = "function"
    call.function.name = tool_name
    call.function.arguments = json.dumps(arguments, ensure_ascii=False)
    msg = MagicMock()
    msg.tool_calls = [call]
    msg.content = None
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _text_response(text: str):
    msg = MagicMock()
    msg.tool_calls = None
    msg.content = text
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def test_synthesize_one_builds_trace():
    scn = _scenario()
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        _fc_response("get_weather", {"location": scn.station, "hours": 24}, "call_1"),
        _fc_response("get_hydrology", {"station": scn.station, "metric": "both"}, "call_2"),
        _fc_response("predict_runoff", {"station": scn.station, "lead_time_hours": 24}, "call_3"),
        _text_response("流量 3250m³/s，发布Ⅱ级预警。"),
    ]
    trace = synthesize_one(client, "fake-model", scn, max_rounds=8)
    assert trace is not None
    roles = [m["role"] for m in trace]
    assert roles[0] == "system" and roles[1] == "user"
    assert "assistant" in roles and "tool" in roles
    # mock 覆盖值已注入回放结果（定位含 flow_m3_s 的 hydrology 结果，避免误匹配 weather）
    hydro = next(m for m in trace if m["role"] == "tool" and "flow_m3_s" in m["content"])
    assert str(scn.tool_overrides["get_hydrology"]["flow_m3_s"]) in hydro["content"]


def test_synthesize_one_gives_up_at_max_rounds():
    scn = _scenario()
    client = MagicMock()
    client.chat.completions.create.return_value = _fc_response(
        "get_hydrology", {"station": scn.station, "metric": "both"}
    )
    assert synthesize_one(client, "m", scn, max_rounds=2) is None


def test_resume_skips_completed(tmp_path: Path):
    out = tmp_path / "raw.jsonl"
    scn1, scn2 = generate_scenarios(n=2, seed=11)[:2]
    out.write_text(json.dumps({"scenario_id": scn1.scenario_id, "messages": []}, ensure_ascii=False) + "\n")
    client = MagicMock()
    client.chat.completions.create.return_value = _text_response("Ⅳ级，水情平稳。")
    written = synthesize_dataset(client, "m", [scn1, scn2], out, rpm=10000)
    assert written == 1  # scn1 已存在被跳过
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 2
