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
    "memory_recall": "记忆召回",
    "needle_retention": "针保留（压缩后）",
}
_TYPE_NAMES = {
    "business": "业务研判",
    "chitchat": "闲聊",
    "regulation": "法规问答",
    "web_search": "联网检索",
    "trap": "陷阱任务",
    "memory": "记忆召回",
    "compression": "压缩等价性",
    "tool_edge": "工具边界",
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
    "needle_found": "针保留率（答案含关键事实）",
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


# ====== 量化声明表（experiments 结果 → 可对外引用的声明行） ======

def _fmt_contrast_rate(entry: dict | None) -> str:
    return _fmt_rate(entry) if entry else "—"


def _fmt_pct(v: float | None) -> str:
    return "—" if v is None else f"{v * 100:.1f}%"


def experiment_claims(name: str, result: dict) -> list[dict]:
    """把单个实验结果归一为声明行（量化声明表的数据来源）。

    每行字段：mechanism/metric/baseline/treated/delta/relative/significant/n/extras。
    方向约定：treated=机制开启版，baseline=对照版；compression 方向特殊
    （机制=压缩，声明的是保留率与 token 节省的权衡），单独措辞。
    """
    claims: list[dict] = []
    if name == "memory":
        c = result["contrast"]
        claims.append({
            "mechanism": "记忆注入（脚本化）", "metric": "用例通过率",
            "baseline": _fmt_contrast_rate(c["baseline_rate"]),
            "treated": _fmt_contrast_rate(c["treated_rate"]),
            "delta": c["delta"], "relative": c["relative_lift"],
            "significant": c["significant"], "n": c["n_cases"],
            "extras": "子类见明细",
        })
        nc = result.get("needle_contrast")
        if nc:
            claims.append({
                "mechanism": "记忆注入（脚本化）", "metric": "针召回率（答案含记忆事实）",
                "baseline": _fmt_contrast_rate(nc["baseline_rate"]),
                "treated": _fmt_contrast_rate(nc["treated_rate"]),
                "delta": nc["delta"], "relative": nc["relative_lift"],
                "significant": nc["significant"], "n": nc["n_cases"], "extras": "",
            })
    elif name == "compression":
        c = result["retention_contrast"]
        savings = result.get("token_savings", {})
        claims.append({
            "mechanism": "上下文压缩", "metric": "针保留率",
            "baseline": _fmt_contrast_rate(c["baseline_rate"]),
            "treated": _fmt_contrast_rate(c["treated_rate"]),
            "delta": c["delta"], "relative": c["relative_lift"],
            "significant": c["significant"], "n": c["n_cases"],
            "extras": f"历史 token 均省 {savings.get('mean_saved_pct', 0):.1f}%",
        })
    elif name == "self_evolution":
        c = result["final_contrast"]
        curve_e = result.get("learning_curve_experimental", [])
        curve_c = result.get("learning_curve_control", [])
        claims.append({
            "mechanism": "反思写回（自进化）", "metric": "末轮用例通过率",
            "baseline": _fmt_contrast_rate(c["baseline_rate"]),
            "treated": _fmt_contrast_rate(c["treated_rate"]),
            "delta": c["delta"], "relative": c["relative_lift"],
            "significant": c["significant"], "n": c["n_cases"],
            "extras": (f"学习曲线 实验 {[f'{p:.0%}' for p in curve_e]} vs "
                    f"对照 {[f'{p:.0%}' for p in curve_c]}"),
        })
    elif name == "kv_cache":
        claims.append({
            "mechanism": "KV 前缀冻结", "metric": "planner 节点前缀命中率",
            "baseline": _fmt_pct(result.get("planner_hit_broken")),
            "treated": _fmt_pct(result.get("planner_hit_frozen")),
            "delta": result.get("planner_hit_delta"), "relative": None,
            "significant": None, "n": result.get("frozen", {}).get("nodes", {})
            .get("planner", {}).get("calls", 0),
            "extras": "对照=前缀破坏（nonce）；命中来自 cached_tokens 观测",
        })
    elif name == "model_ladder":
        for step in result.get("step_contrasts", []):
            c = step["contrast"]
            lc = step.get("level_exact_contrast") or {}
            claims.append({
                "mechanism": f"{step['from']} → {step['to']}", "metric": "用例通过率",
                "baseline": _fmt_contrast_rate(c["baseline_rate"]),
                "treated": _fmt_contrast_rate(c["treated_rate"]),
                "delta": c["delta"], "relative": c["relative_lift"],
                "significant": c["significant"], "n": c["n_cases"],
                "extras": (f"等级准确率 {_fmt_contrast_rate(lc.get('baseline_rate'))}"
                        f" → {_fmt_contrast_rate(lc.get('treated_rate'))}"),
            })
    return claims


def render_claims_section(experiments: dict[str, dict]) -> list[str]:
    """量化声明表：所有实验的"基线 → 机制"对照（书/论文可直接引用的格式）。"""
    lines: list[str] = []
    rows: list[dict] = []
    for name, result in experiments.items():
        rows.extend(experiment_claims(name, result))
    if not rows:
        return lines
    lines.append("## 量化声明表（基线 → 机制）")
    lines.append("")
    lines.append("| 机制 | 指标 | 基线 | +机制 | Δ | 相对提升 | 显著性 | n | 副指标 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        delta = "—" if r["delta"] is None else f"{r['delta'] * 100:+.1f} pp"
        relative = "—" if r["relative"] is None else f"{r['relative'] * 100:+.1f}%"
        if r["significant"] is None:
            sig = "—"
        else:
            sig = "显著" if r["significant"] else "未超噪声带宽"
        lines.append(
            f"| {r['mechanism']} | {r['metric']} | {r['baseline']} | {r['treated']} "
            f"| {delta} | {relative} | {sig} | {r['n']} | {r['extras']} |"
        )
    lines.append("")
    return lines


def _experiment_details(experiments: dict[str, dict]) -> list[str]:
    """实验专属明细段（学习曲线/KV 节点表/阶梯表/记忆子类）。"""
    lines: list[str] = []
    for name, result in experiments.items():
        if name == "memory" and result.get("by_subtype"):
            lines.append("### 记忆实验分子类")
            lines.append("")
            lines.append("| 子类 | 基线 | +记忆 | Δ | n |")
            lines.append("|---|---|---|---|---|")
            subtype_names = {"fact": "跨会话事实", "update": "知识更新", "temporal": "时间推理"}
            for subtype, c in result["by_subtype"].items():
                delta = "—" if c["delta"] is None else f"{c['delta'] * 100:+.1f} pp"
                lines.append(
                    f"| {subtype_names.get(subtype, subtype)} "
                    f"| {_fmt_contrast_rate(c['baseline_rate'])} "
                    f"| {_fmt_contrast_rate(c['treated_rate'])} | {delta} | {c['n_cases']} |"
                )
            lines.append("")
        elif name == "self_evolution":
            lines.append("### 自进化学习曲线（Reflexion 式）")
            lines.append("")
            lines.append("| 轮次 | 实验组（开） | 对照组（关） |")
            lines.append("|---|---|---|")
            for i, (e, c) in enumerate(zip(
                    result.get("learning_curve_experimental", []),
                    result.get("learning_curve_control", []), strict=False), 1):
                lines.append(f"| 第 {i} 轮 | {_fmt_pct(e)} | {_fmt_pct(c)} |")
            lines.append("")
            if result.get("note"):
                lines.append(f"> {result['note']}")
                lines.append("")
        elif name == "kv_cache":
            lines.append("### KV Cache 分节点命中率（冻结 vs 破坏）")
            lines.append("")
            lines.append("| 节点 | 冻结命中率 | 破坏命中率 | prompt tokens（冻结） |")
            lines.append("|---|---|---|---|")
            for node, s in result.get("frozen", {}).get("nodes", {}).items():
                broken_rate = result.get("broken", {}).get("nodes", {}).get(node, {})
                lines.append(
                    f"| {node} | {_fmt_pct(s['hit_rate'])} "
                    f"| {_fmt_pct(broken_rate.get('hit_rate'))} | {s['prompt_tokens']} |"
                )
            lines.append("")
        elif name == "model_ladder":
            lines.append("### 训练阶梯逐级明细")
            lines.append("")
            lines.append("| checkpoint | 用例通过率 | 等级准确率 | 工具召回 |")
            lines.append("|---|---|---|---|")
            for rung in result.get("rungs", []):
                lines.append(
                    f"| {rung['model']} | {_fmt_contrast_rate(rung.get('case_pass_rate'))} "
                    f"| {_fmt_contrast_rate(rung.get('level_exact'))} "
                    f"| {_fmt_contrast_rate(rung.get('tool_recall'))} |"
                )
            lines.append("")
    return lines



def render_report(
    records: list[dict],
    metrics: dict,
    config: dict,
    pass_power_k: dict | None = None,
    ablation: dict | None = None,
    judge_agg: dict | None = None,
    regression_lines: list[str] | None = None,
    experiments: dict[str, dict] | None = None,
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
    if config.get("experiment"):
        ap(f"- 实验：`{config['experiment']}` ｜ 复现：`{config.get('reproduce_cmd', '—')}`")
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
                "citation_ok", "trap_resisted", "needle_found"):
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

    # 量化实验（声明表 + 专属明细）
    if experiments:
        lines.extend(render_claims_section(experiments))
        details = _experiment_details(experiments)
        if details:
            ap("## 实验明细")
            ap("")
            lines.extend(details)

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
