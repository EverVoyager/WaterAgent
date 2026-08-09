"""DPO 正负对构建模块测试。"""
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from train.data_gen.dpo_pairs import (
    DPOPair,
    build_dpo_pair,
    write_dpo_jsonl,
    _extract_final_answer,
    _extract_prompt,
)


def _make_record(final_answer: str, scenario_id: str = "scn-test-1") -> dict:
    """构造模拟轨迹记录（含工具调用+最终回答）。"""
    return {
        "scenario_id": scenario_id,
        "level": "II",
        "messages": [
            {"role": "system", "content": "你是防汛预警智能体。"},
            {"role": "user", "content": "龙门站未来24小时有洪水风险吗？"},
            {"role": "assistant", "content": '<tool_call>\n{"name": "get_hydrology", "arguments": {"station": "龙门"}}\n</tool_call>'},
            {"role": "tool", "content": '<tool_response>\n{"flow_m3_s": 3500}\n</tool_response>'},
            {"role": "assistant", "content": final_answer},
        ],
    }


def test_extract_final_answer():
    """从轨迹中提取最后一个不含 tool_call 的 assistant 消息。"""
    rec = _make_record("发布Ⅱ级预警，建议加强巡查。")
    answer = _extract_final_answer(rec["messages"])
    assert answer == "发布Ⅱ级预警，建议加强巡查。"


def test_extract_final_answer_no_tool_call():
    """纯知识问答轨迹（无工具调用）也能提取。"""
    messages = [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "什么是防汛？"},
        {"role": "assistant", "content": "防汛是指..."},
    ]
    assert _extract_final_answer(messages) == "防汛是指..."


def test_extract_prompt():
    """提取 system + user 消息作为 prompt。"""
    rec = _make_record("回答")
    prompt = _extract_prompt(rec["messages"])
    assert len(prompt) == 2
    assert prompt[0]["role"] == "system"
    assert prompt[1]["role"] == "user"


def test_build_dpo_pair_normal():
    """正常高低分配对。"""
    high = _make_record("发布Ⅱ级预警，建议加强巡查。", "scn-1")
    low = _make_record("不知道，自己看吧。", "scn-1")
    pair = build_dpo_pair(high, low)
    assert pair is not None
    assert pair.chosen == "发布Ⅱ级预警，建议加强巡查。"
    assert pair.rejected == "不知道，自己看吧。"
    assert len(pair.prompt) == 2
    assert pair.scenario_id == "scn-1"


def test_build_dpo_pair_same_answer():
    """高低分回答相同，无偏好价值，返回 None。"""
    high = _make_record("相同回答。", "scn-1")
    low = _make_record("相同回答。", "scn-1")
    pair = build_dpo_pair(high, low)
    assert pair is None


def test_build_dpo_pair_empty_answer():
    """无法提取最终回答时返回 None。"""
    high = _make_record("高质量回答", "scn-1")
    # 低分轨迹只有工具调用，无最终回答
    low = {
        "scenario_id": "scn-1",
        "level": "II",
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": '<tool_call>\n{"name": "get_hydrology"}\n</tool_call>'},
        ],
    }
    pair = build_dpo_pair(high, low)
    assert pair is None


def test_write_dpo_jsonl():
    """DPO 对写入 JSONL 格式正确。"""
    pairs = [
        DPOPair(
            prompt=[{"role": "user", "content": "问题1"}],
            chosen="高质量回答1",
            rejected="低质量回答1",
            scenario_id="scn-1",
        ),
        DPOPair(
            prompt=[{"role": "user", "content": "问题2"}],
            chosen="高质量回答2",
            rejected="低质量回答2",
            scenario_id="scn-2",
        ),
    ]
    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "dpo.jsonl"
        written = write_dpo_jsonl(pairs, path)
        assert written == 2
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        rec = json.loads(lines[0])
        assert "prompt" in rec
        assert "chosen" in rec
        assert "rejected" in rec
        assert rec["chosen"][0]["content"] == "高质量回答1"
        assert rec["rejected"][0]["content"] == "低质量回答1"
