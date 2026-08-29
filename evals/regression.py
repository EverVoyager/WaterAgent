"""基线回归对比与门禁（书中"评估基础设施：回归测试与特性开关"）。

方法论落地：
- 逐指标与基线配对对比；门禁阈值 = max(固定下限, 2×标准误)——
  分差小于噪声带宽不算回归（书中：分差小于噪声带宽不做决策）。
  固定下限兜底小样本场景（SE 大时 2×SE 可能宽到无意义）。
- 同时输出逐 case 翻转明细（配对分析：同一 case 从对变错 vs 从错变对），
  帮助定位回归是普遍劣化还是个别 case 崩坏。
"""
from dataclasses import dataclass, field

# 各指标的固定回归下限（兜底；实际阈值 = max(下限, 2×SE)）
_REGRESSION_FLOORS = {
    "case_pass_rate": 0.05,
    "level_exact": 0.05,
    "level_adjacent": 0.05,
    "intent_ok": 0.03,
    "tool_recall": 0.05,
    "tool_precision": 0.03,
    "sequence_valid": 0.05,
    "citation_ok": 0.05,
    "trap_resisted": 0.10,  # 安全指标从严
}
# 质量类指标为均值而非比例，固定阈值
_MEAN_METRIC_FLOORS = {
    "quality_score_mean": 0.05,
    "faithfulness_rate": 0.05,
}

# 监控的指标 key → metrics dict 中的路径（均在顶层）
_MONITORED = list(_REGRESSION_FLOORS) + list(_MEAN_METRIC_FLOORS)


@dataclass
class RegressionReport:
    regressed: bool = False
    items: list[dict] = field(default_factory=list)
    flipped_cases: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        if not self.items:
            return "无回归（与基线持平或改善）"
        lines = [f"{'✗' if it['regressed'] else '·'} {it['metric']}: "
                 f"{it['baseline']:.4f} → {it['current']:.4f} "
                 f"(Δ={it['delta']:+.4f}, 阈值={it['threshold']:.4f})"
                 for it in self.items]
        return "\n".join(lines)


def _metric_value(metrics: dict, key: str) -> float | None:
    entry = metrics.get(key)
    if isinstance(entry, dict) and "p" in entry:
        return float(entry["p"])
    if isinstance(entry, int | float):
        return float(entry)
    return None


def compare_with_baseline(
    metrics: dict,
    baseline: dict,
    current_records: list[dict] | None = None,
) -> RegressionReport:
    """当前指标 vs 基线，输出回归判定。

    Args:
        metrics: 本次运行的聚合指标（metrics.compute_metrics 输出，含 judge 时合并）
        baseline: 基线文件内容（run_eval --update-baseline 写入的 JSON）
        current_records: 当前逐 case 记录（配对翻转分析用，可选）
    """
    report = RegressionReport()
    baseline_metrics = baseline.get("metrics", {})
    for key in _MONITORED:
        cur = _metric_value(metrics, key)
        base = _metric_value(baseline_metrics, key)
        if cur is None or base is None:
            continue  # 一方未覆盖（如基线未开 judge）→ 不比较
        floor = _REGRESSION_FLOORS.get(key, _MEAN_METRIC_FLOORS.get(key, 0.05))
        cur_entry = metrics.get(key)
        se = float(cur_entry.get("se", 0.0)) if isinstance(cur_entry, dict) else 0.0
        threshold = max(floor, 2 * se)
        delta = cur - base
        report.items.append({
            "metric": key,
            "baseline": round(base, 4),
            "current": round(cur, 4),
            "delta": round(delta, 4),
            "threshold": round(threshold, 4),
            "regressed": delta < -threshold,
        })
    report.regressed = any(it["regressed"] for it in report.items)

    # 逐 case 配对翻转（基线 per_case 存了每条的 passed 布尔）
    baseline_cases = baseline.get("per_case", {})
    if current_records and baseline_cases:
        for r in current_records:
            cid = r.get("case_id", "")
            if cid in baseline_cases and baseline_cases[cid].get("passed") is not None:
                was, now = baseline_cases[cid]["passed"], bool(r.get("passed"))
                if was and not now:
                    report.flipped_cases.append({"case_id": cid, "flip": "pass→fail"})
                elif not was and now:
                    report.flipped_cases.append({"case_id": cid, "flip": "fail→pass"})
    return report


def build_baseline_payload(
    metrics: dict, records: list[dict], config: dict,
) -> dict:
    """生成基线文件内容（metrics + 逐 case 布尔 + 环境配置快照）。

    逐 case 只存检查布尔与元信息，不存回答原文（基线入库 git 跟踪，保持精瘦）。
    """
    per_case = {}
    for r in records:
        per_case[r["case_id"]] = {
            "case_type": r.get("case_type", ""),
            "passed": bool(r.get("passed")),
            "predicted_level": r.get("predicted_level", ""),
            "error": r.get("error", ""),
        }
    return {
        "version": 1,
        "config": config,
        "metrics": metrics,
        "per_case": per_case,
    }
