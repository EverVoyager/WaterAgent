"""数据集统计报告（FR-D6）。"""
from collections import Counter


def summarize(records: list, reject_counter: dict, total_scenarios: int) -> dict:
    accepted = len(records)
    level_dist = Counter(r["level"] for r in records if r.get("level"))
    qtype_dist = Counter(r["query_type"] for r in records)
    rounds = [
        sum(1 for m in r["messages"] if m["role"] == "assistant") for r in records
    ]
    return {
        "total_scenarios": total_scenarios,
        "accepted": accepted,
        "accept_rate": round(accepted / max(total_scenarios, 1), 4),
        "level_dist": dict(level_dist),
        "query_type_dist": dict(qtype_dist),
        "reject_dist": dict(reject_counter),
        "avg_rounds": round(sum(rounds) / max(len(rounds), 1), 2),
    }


def print_report(summary: dict) -> str:
    lines = ["# 数据集统计报告", ""]
    for k, v in summary.items():
        lines.append(f"- **{k}**: {v}")
    text = "\n".join(lines)
    print(text)
    return text
