"""统计报告：分布与过滤率正确。"""
from train.data_gen.stats import summarize


def test_summarize_counts():
    records = [
        {"level": "I", "query_type": "multi_tool", "messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "<tool_call>\n{\"name\": \"get_hydrology\", \"arguments\": {}}\n</tool_call>"},
            {"role": "tool", "content": "{}"},
            {"role": "assistant", "content": "Ⅰ级"},
        ]},
        {"level": "II", "query_type": "single_tool", "messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "Ⅱ级"},
        ]},
    ]
    rejects = {"reject_f1_params": 1, "reject_f2_sequence": 2, "reject_f3_level": 3}
    s = summarize(records, rejects, total_scenarios=8)
    assert s["accepted"] == 2
    assert s["level_dist"] == {"I": 1, "II": 1}
    assert s["reject_dist"]["reject_f3_level"] == 3
    assert s["accept_rate"] == 0.25
    assert s["avg_rounds"] == 1.5  # (2 assistant 段 + 1) / 2
