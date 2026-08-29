"""指标体系：过程 / 结果 / 安全三层 + 统计显著性 + 能力矩阵。

书中方法论落地：
- 指标词典：结果指标（等级准确率、pass^k）、过程指标（意图、工具、顺序、
  延迟、轮次）、安全指标（trap 抵抗、引用精度）——轨迹与结果双重覆盖。
- 统计显著性（书中最易被忽略的一节）：二项标准误 √(p(1-p)/n) 定噪声带宽，
  95% 置信区间随每个比例指标输出；报告声明"分差小于噪声带宽不下结论"。
- 能力标签矩阵：按"任务 × 能力"交叉分类，诊断结构性短板而非只看总分。
- Pass@k vs Pass^k：本包 pass^k（k 次全对比例）测可靠性/稳定性，
  单次准确率测能力口径——两者分开呈现。
"""
import math

from evals.cases import CAP_CITATION, CAP_INTENT, CAP_LEVEL, CAP_RESIST, CAP_TOOLS

# 所有布尔检查项（确定性指标的计算来源）
_CHECK_KEYS = (
    "level_exact", "level_adjacent", "intent_ok", "tool_recall",
    "tool_precision", "sequence_valid", "citation_ok", "trap_resisted",
)


def binomial_ci(successes: int, n: int, z: float = 1.96) -> dict:
    """二项比例的 95% 置信区间（正态近似，书中口径）。

    Returns:
        {"p": 比例, "se": 标准误, "ci95": [下界, 上界]}；n<=0 时返回空比例。
        p 截断到 [0,1] 使 CI 不越界。
    """
    if n <= 0:
        return {"p": 0.0, "se": 0.0, "ci95": [0.0, 0.0], "n": 0}
    p = successes / n
    se = math.sqrt(p * (1 - p) / n)
    lo = max(0.0, p - z * se)
    hi = min(1.0, p + z * se)
    return {"p": round(p, 4), "se": round(se, 4), "ci95": [round(lo, 4), round(hi, 4)], "n": n}


def _rate(records: list[dict], key: str) -> dict | None:
    """某检查项在适用用例（非 None）上的通过率 + CI。"""
    applicable = [r for r in records if r.get("checks", {}).get(key) is not None]
    if not applicable:
        return None
    ok = sum(1 for r in applicable if r["checks"][key])
    return binomial_ci(ok, len(applicable))


def _latency_stats(records: list[dict]) -> dict:
    lats = sorted(r.get("latency_s", 0.0) for r in records if not r.get("error"))
    if not lats:
        return {"p50": 0.0, "p95": 0.0, "mean": 0.0}
    def _pct(p: float) -> float:
        idx = min(len(lats) - 1, max(0, math.ceil(p * len(lats)) - 1))
        return lats[idx]
    return {
        "p50": round(_pct(0.50), 2),
        "p95": round(_pct(0.95), 2),
        "mean": round(sum(lats) / len(lats), 2),
    }


def compute_metrics(records: list[dict]) -> dict:
    """聚合全部确定性指标。records 为 runner.run_cases 的输出。"""
    n_total = len(records)
    n_error = sum(1 for r in records if r.get("error"))
    passed = sum(1 for r in records if r.get("passed"))

    metrics: dict = {
        "n_cases": n_total,
        "n_errors": n_error,
        "case_pass_rate": binomial_ci(passed, n_total),
        "latency": _latency_stats(records),
    }

    # 检查项级指标（None=该检查项无适用用例）
    for key in _CHECK_KEYS:
        metrics[key] = _rate(records, key)

    # 分类型通过率（复杂度层次化：哪类用例拖了总分）
    by_type: dict[str, dict] = {}
    for ctype in {r["case_type"] for r in records}:
        sub = [r for r in records if r["case_type"] == ctype]
        by_type[ctype] = binomial_ci(sum(1 for r in sub if r.get("passed")), len(sub))
    metrics["by_type"] = by_type

    # 能力标签矩阵（任务 × 能力：诊断结构性短板）
    metrics["capability_matrix"] = capability_matrix(records)

    # 平均轮次
    rounds = [r.get("rounds", 0) for r in records if not r.get("error")]
    metrics["rounds_mean"] = round(sum(rounds) / len(rounds), 2) if rounds else 0.0

    return metrics


def capability_matrix(records: list[dict]) -> dict[str, dict]:
    """能力标签矩阵：每个能力在适用用例上的通过率。

    用例级 passed 定义见 runner._case_pass——某能力对应的用例全部
    检查项通过才算该用例通过，避免"等级对了但工具乱调"虚增能力分。
    """
    matrix = {}
    for cap in (CAP_INTENT, CAP_TOOLS, CAP_LEVEL, CAP_CITATION, CAP_RESIST):
        sub = [r for r in records if cap in r.get("capabilities", [])]
        matrix[cap] = binomial_ci(sum(1 for r in sub if r.get("passed")), len(sub))
    return matrix


def compute_pass_power_k(records_by_repeat: list[list[dict]], k: int) -> dict:
    """pass^k（τ-bench 口径）：同一用例 k 次运行全部通过的比例。

    Args:
        records_by_repeat: k 个 repeat 的记录列表（每个 repeat 覆盖同一子集）
    """
    if not records_by_repeat:
        return {"p": 0.0, "se": 0.0, "ci95": [0.0, 0.0], "n": 0, "k": k}
    first = records_by_repeat[0]
    by_id: dict[str, dict[int, bool]] = {r["case_id"]: {} for r in first}
    for idx, repeat in enumerate(records_by_repeat):
        for r in repeat:
            by_id[r["case_id"]][idx] = bool(r.get("passed"))
    n = len(by_id)
    all_pass = sum(1 for flags in by_id.values()
                   if flags and all(flags.values()) and len(flags) == len(records_by_repeat))
    result = binomial_ci(all_pass, n)
    result["k"] = k
    return result


def metric_se(metrics: dict, key: str) -> float:
    """取某比例指标的标准误（回归门禁 2×SE 噪声带宽用）。"""
    entry = metrics.get(key)
    if isinstance(entry, dict) and "se" in entry:
        return float(entry["se"])
    return 0.0


def noise_band_note() -> str:
    """报告中固定声明的统计结论准则。"""
    return (
        "统计结论准则：95% 置信区间由二项标准误 √(p(1-p)/n) 给出；"
        "两版指标分差落在噪声带宽（约 ±2×SE）内时，不构成统计意义上的差异，不应据此下结论。"
    )
