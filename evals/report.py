"""评估报告生成：Markdown 报告（总体指标 + 能力矩阵 + 失败明细）。

报告中固定输出：
- 每个比例指标的 95% 置信区间（统计结论准则声明）
- 能力标签矩阵（任务 × 能力，诊断结构性短板）
- 陷阱任务明细（安全指标）
- 失败用例清单与回答摘要（可观测性：定位具体失效面）
"""
from datetime import datetime, timezone

from evals.metrics import noise_band_note

_CAP_NAMES = {
    "intent": "意图识别",
    "tool_selection": "工具选择",
    "level_decision": "等级判定",
    "citation": "引用溯源",
    "misdirection_resistance": "抗误导",
}
_TYPE_NAMES = {
    "business": "业务研判",
    "chitchat": "闲聊",
    "regulation": "法规问答",
    "web_search": "联网检索",
    "trap": "陷阱任务",
}
_METRIC_NAMES = {
    "case_pass_rate": "用例通过率",
    "level_exact": "等级精确匹配",
    "level_adjacent": "等级相邻宽容",
    "intent_ok": "意图正确率",
    "tool_recall": "工具召回（期望工具全调）",
    "tool_precision": "工具精度（无越界调用）",
    "sequence_valid": "工具顺序合法率",
    "citation_ok": "引用可溯源率",
    "trap_resisted": "陷阱抵抗率",
}


def _fmt_rate(entry: dict | None) -> str:
    if entry is None:
        return "—"
    lo, hi = entry["ci95"]
    return f"{entry['p'] * 100:.1f}% [{lo * 100:.1f}, {hi * 100:.1f}] (n={entry['n']})"


def _fmt_metric_row(metrics: dict, key: str) -> str:
    name = _METRIC_NAMES.get(key, key)
    return f"| {name} | {_fmt_rate(metrics.get(key))} |"


def _failures_section(records: list[dict], max_items: int = 12) -> list[str]:
    lines = []
    failures = [r for r in records if not r.get("passed")]
    if not failures:
        return ["全部用例通过，无失败明细。"]
    lines.append(f"共 {len(failures)} 条未通过（最多展示 {max_items} 条）：")
    lines.append("")
    for r in failures[:max_items]:
        checks = r.get("checks", {})
        failed = [k for k, v in checks.items() if v is False]
        lines.append(f"### {r['case_id']}（{_TYPE_NAMES.get(r['case_type'], r['case_type'])}）")
        if r.get("error"):
            lines.append(f"- 运行错误：`{r['error']}`")
        lines.append(f"- 查询：{r['query']}")
        lines.append(f"- 未过检查项：{', '.join(failed) or '—'}")
        lines.append(f"- 预测等级：{r.get('predicted_level') or '—'}"
                     f" / 期望：{r.get('expected_level') or '—'}"
                     + (f" / 用户声称：{_rung(r.get('claimed_level'))}" if r.get("claimed_level") else ""))
        lines.append(f"- 工具轨迹：{' → '.join(r.get('tool_sequence', [])) or '（无）'}"
                     f" ｜ 轮次 {r.get('rounds', 0)} ｜ 引用 {r.get('citations_count', 0)} 条")
        answer = (r.get("final_answer", "") or "").strip().replace("\n", " ")
        if answer:
            lines.append(f"- 回答摘要：{answer[:160]}{'…' if len(answer) > 160 else ''}")
        if r.get("env_mismatch"):
            lines.append(f"- ⚠️ {r['env_mismatch']}")
        lines.append("")
    return lines


def _rung(level: str) -> str:
    return {"I": "Ⅰ级", "II": "Ⅱ级", "III": "Ⅲ级", "IV": "Ⅳ级"}.get(level, level or "—")


def render_report(
    records: list[dict],
    metrics: dict,
    config: dict,
    pass_power_k: dict | None = None,
    ablation: dict | None = None,
    judge_agg: dict | None = None,
    regression_lines: list[str] | None = None,
) -> str:
    """渲染 Markdown 评估报告。"""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines: list[str] = []
    ap = lines.append

    ap("# WaterAgents 系统级评估报告")
    ap("")
    ap(f"- 时间：{now}")
    ap(f"- 模型：`{config.get('model_label', 'unknown')}`")
    ap(f"- 用例：{metrics.get('n_cases', 0)} 条"
       f"（运行错误 {metrics.get('n_errors', 0)} 条）")
    ap(f"- LLM Judge：{'开启' if judge_agg else '关闭（--no-judge，仅确定性指标）'}")
    ap(f"- 平均轮次：{metrics.get('rounds_mean', 0)} ｜ "
       f"延迟 p50/p95：{metrics['latency']['p50']}s / {metrics['latency']['p95']}s")
    ap("")
    ap(f"> {noise_band_note()}")
    ap("")

    # 总体指标
    ap("## 总体指标（95% CI）")
    ap("")
    ap("| 指标 | 通过率 [置信区间] |")
    ap("|---|---|")
    for key in ("case_pass_rate", "level_exact", "level_adjacent", "intent_ok",
                "tool_recall", "tool_precision", "sequence_valid",
                "citation_ok", "trap_resisted"):
        ap(_fmt_metric_row(metrics, key))
    ap("")

    # 分类型
    ap("## 分类型通过率")
    ap("")
    ap("| 用例类型 | 通过率 [置信区间] |")
    ap("|---|---|")
    for ctype, entry in metrics.get("by_type", {}).items():
        ap(f"| {_TYPE_NAMES.get(ctype, ctype)} | {_fmt_rate(entry)} |")
    ap("")

    # 能力矩阵
    ap("## 能力标签矩阵（结构性短板诊断）")
    ap("")
    ap("| 能力 | 通过率 [置信区间] |")
    ap("|---|---|")
    for cap, entry in metrics.get("capability_matrix", {}).items():
        ap(f"| {_CAP_NAMES.get(cap, cap)} | {_fmt_rate(entry)} |")
    ap("")

    # pass^k
    if pass_power_k:
        ap(f"## 稳定性 pass^{pass_power_k.get('k', 3)}（同一用例重复全对比例）")
        ap("")
        ap(f"pass^{pass_power_k.get('k', 3)} = {_fmt_rate(pass_power_k)}")
        ap("")

    # 陷阱任务明细
    trap_records = [r for r in records if r["case_type"] == "trap"]
    if trap_records:
        ap("## 陷阱任务明细（用户声称等级 vs 数据等级）")
        ap("")
        ap("| 用例 | 用户声称 | 数据等级 | 预测等级 | 抵抗 |")
        ap("|---|---|---|---|---|")
        for r in trap_records:
            ok = r.get("checks", {}).get("trap_resisted")
            ap(f"| {r['case_id']} | {_rung(r.get('claimed_level'))} | "
               f"{_rung(r.get('expected_level'))} | {_rung(r.get('predicted_level'))} | "
               f"{'✅' if ok else '❌'} |")
        ap("")

    # Judge
    if judge_agg:
        ap("## LLM-as-Judge（Rubric 软指标）")
        ap("")
        ap(f"- 评判模型：`{config.get('judge_model', '—')}`（与主模型异源）")
        ap(f"- 忠实率 faithfulness：{judge_agg.get('faithfulness_rate')}")
        ap(f"- 回答质量均分（Rubric 加权）：{judge_agg.get('quality_score_mean')}")
        ap(f"- 否决项触发：{judge_agg.get('veto_count')}/{judge_agg.get('n_judged')}"
           f"（编造数值/虚构来源一票否决）")
        if judge_agg.get("n_unavailable"):
            ap(f"- 评判不可用：{judge_agg['n_unavailable']} 条")
        ap("")

    # 消融
    if ablation:
        ap("## 记忆消融（有/无记忆注入）")
        ap("")
        ap(f"- 有记忆：{_fmt_rate(ablation['with_memory'])}")
        ap(f"- 无记忆：{_fmt_rate(ablation['without_memory'])}")
        ap(f"- Δ（有−无）：{ablation['delta']:+.4f}"
           f"（组合 SE={ablation['combined_se']:.4f}，"
           f"{'显著' if ablation['significant'] else '未超噪声带宽，不下结论'}）")
        if ablation.get("flipped_to_pass"):
            ap(f"- 记忆帮助通过的 case：{', '.join(ablation['flipped_to_pass'])}")
        if ablation.get("flipped_to_fail"):
            ap(f"- 记忆反而致败的 case：{', '.join(ablation['flipped_to_fail'])}")
        ap("")

    # 回归
    if regression_lines:
        ap("## 基线回归对比")
        ap("")
        ap("```")
        lines.extend(regression_lines)
        ap("```")
        ap("")

    # 失败明细
    ap("## 失败用例明细")
    ap("")
    lines.extend(_failures_section(records))

    return "\n".join(lines)
