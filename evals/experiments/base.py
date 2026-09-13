"""机制消融框架：ablation.py（记忆专用）的通用化。

书中方法论落地（消融实验一节的通用化）：
- 表现差异要先能归因：run_toggle_ablation 关闭"恰好一个机制"跑对照，
  其余全等，两版通过率之差即该机制的净贡献。
- 开关两类：env 型（settings 布尔字段，如 SELF_EVOLUTION_ENABLED）、
  patch 型（包命名空间函数替换，节点内运行时 import，patch 生效）。
- 显著性口径与 metrics.py 一致：|Δ| > 2×组合标准误才算显著，
  落在噪声带宽内的分差不写入对外声明。

对外的量化数字统一结构（report.py 的量化声明表直接消费）：
    基线率（CI95） → 机制率（CI95）｜Δ 绝对｜相对提升｜显著性｜n｜翻转明细
"""
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

from evals.metrics import binomial_ci

TOGGLE_ENV = "env"    # settings 属性覆盖
TOGGLE_PATCH = "patch"  # 包命名空间函数替换


@dataclass(frozen=True)
class Toggle:
    """一个可关闭（或改写）的机制入口。

    kind="env":  target 为 settings 属性名，off_value 为关闭值（False/0/""）
    kind="patch": target 为 "package.module.func" 全限定名；
        off_fn 优先（整体替换，如恒等函数）；否则 off_value 作为替换函数的
        固定返回值（mock.patch(return_value=...) 语义）
    """

    kind: str
    target: str
    off_value: Any = False
    off_fn: Any = None

    def __post_init__(self) -> None:
        if self.kind not in (TOGGLE_ENV, TOGGLE_PATCH):
            raise ValueError(f"未知开关类型: {self.kind}（应为 env/patch）")


@contextmanager
def toggles_applied(toggles: list[Toggle]):
    """临时应用一组开关（异常路径同样恢复，settings 实例属性原样写回）。"""
    if not toggles:
        yield
        return
    with ExitStack() as stack:
        for t in toggles:
            if t.kind == TOGGLE_ENV:
                from app.core.config import get_settings

                settings = get_settings()
                if not hasattr(settings, t.target):
                    raise AttributeError(
                        f"settings 无属性 {t.target}——env 开关目标不存在"
                    )
                stack.enter_context(
                    patch.object(settings, t.target, t.off_value)
                )
            else:
                if t.off_fn is not None:
                    stack.enter_context(patch(t.target, new=t.off_fn))
                else:
                    stack.enter_context(patch(t.target, return_value=t.off_value))
        yield


def _rate_on(records: list[dict], check_key: str | None) -> dict | None:
    """records 在 check_key 检查项（None=用例级 passed）上的通过率 + CI。"""
    if check_key is None:
        if not records:
            return None
        return binomial_ci(sum(1 for r in records if r.get("passed")), len(records))
    applicable = [
        r for r in records if r.get("checks", {}).get(check_key) is not None
    ]
    if not applicable:
        return None
    ok = sum(1 for r in applicable if r["checks"][check_key])
    return binomial_ci(ok, len(applicable))


def summarize_contrast(
    records_treated: list[dict],
    records_baseline: list[dict],
    check_key: str | None = None,
) -> dict:
    """两版记录的对照汇总（量化声明表的通用结构）。

    treated = 机制开启版；baseline = 机制关闭版。
    check_key 指定时按该检查项统计（如 needle_found），否则按用例级 passed。
    """
    treated_rate = _rate_on(records_treated, check_key)
    baseline_rate = _rate_on(records_baseline, check_key)

    delta: float | None = None
    relative_lift: float | None = None
    combined_se = 0.0
    significant: bool | None = None
    if treated_rate and baseline_rate and treated_rate["n"] > 0:
        delta = round(treated_rate["p"] - baseline_rate["p"], 4)
        combined_se = round(
            (treated_rate["se"] ** 2 + baseline_rate["se"] ** 2) ** 0.5, 4
        )
        significant = abs(delta) > 2 * combined_se if combined_se > 0 else None
        if baseline_rate["p"] > 0:
            # 相对提升只在基线非零时有意义（Mem0/AWM 的 relative 口径）
            relative_lift = round(delta / baseline_rate["p"], 4)

    by_id_treated = {r["case_id"]: r for r in records_treated}
    by_id_baseline = {r["case_id"]: r for r in records_baseline}

    def _ok(r: dict) -> bool:
        if check_key is None:
            return bool(r.get("passed"))
        return bool(r.get("checks", {}).get(check_key))

    flipped_to_pass = sorted(
        cid for cid, r in by_id_treated.items()
        if cid in by_id_baseline and _ok(r) and not _ok(by_id_baseline[cid])
    )
    flipped_to_fail = sorted(
        cid for cid, r in by_id_treated.items()
        if cid in by_id_baseline and not _ok(r) and _ok(by_id_baseline[cid])
    )

    return {
        "check_key": check_key,
        "n_cases": len(records_treated),
        "treated_rate": treated_rate,
        "baseline_rate": baseline_rate,
        "delta": delta,
        "relative_lift": relative_lift,
        "combined_se": combined_se,
        "significant": significant,
        "flipped_to_pass": flipped_to_pass,
        "flipped_to_fail": flipped_to_fail,
    }


def run_toggle_ablation(
    cases: list,
    run_fn,
    off_toggles: list[Toggle],
    model_label: str = "",
    check_key: str | None = None,
) -> dict:
    """同一批用例跑"机制开 vs 机制关"两遍并对照。

    Args:
        cases: 评估用例列表
        run_fn: 执行函数（签名 (cases, model_label) -> records），
            可复用 runner.run_cases，也可注入缓存版避免重复 LLM 调用
        off_toggles: "机制关"版要应用的开关（patch/改 settings）
        check_key: 对照统计的检查项（None=用例级 passed）
    """
    treated = run_fn(cases, model_label)
    baseline: list[dict] = []
    with toggles_applied(off_toggles):
        baseline = run_fn(cases, model_label)
    return summarize_contrast(treated, baseline, check_key=check_key)
