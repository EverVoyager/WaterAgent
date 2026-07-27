"""评估对比报告（Markdown，AC-2/AC-3 证据）。"""


def render_report(results: dict) -> str:
    lines = [
        "# 三模型对比评估报告", "",
        "| 模型 | 等级准确率 | reward | r1 | r2 | r3 | 工具成功率 |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, m in results.items():
        lines.append(
            f"| {name} | {m.get('level_acc', 0.0):.2f} | {m.get('reward', 0.0):.2f} | "
            f"{m.get('r1', 0.0):.2f} | {m.get('r2', 0.0):.2f} | {m.get('r3', 0.0):.2f} | "
            f"{m.get('tool_ok', 0.0):.2f} |"
        )
    if "sft" in results and "sft_grpo" in results:
        delta = results["sft_grpo"]["level_acc"] - results["sft"]["level_acc"]
        lines += ["", f"**GRPO 相对 SFT 等级准确率提升：{delta:+.2f}**（验收目标 ≥ +0.05）"]
    return "\n".join(lines)
