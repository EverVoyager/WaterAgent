"""自进化/反思学习曲线实验：自进化开/关 × 多轮迭代（Reflexion 式）。

设计（与记忆增益实验的区别）：
- 记忆实验测"注入内容在场 vs 不在场"的单步增益（内容受控）；
- 本实验测"系统自己写回的经验能否随迭代累积"——
  同批用例反复运行 T 轮，实验组每轮结束后触发反思写回
  （SELF_EVOLUTION_ENABLED=True，需 MySQL/Qdrant 可用，否则写回
  降级不持久化、曲线预期走平——报告需注明前置条件），
  对照组全程关闭。第 t 轮通过率序列即学习曲线。

顺序敏感性：先跑对照组（关闭），再跑实验组——实验组写回的记忆
会留在库里，避免污染对照；这也意味着本实验对库有副作用，
生产环境跑完建议清理（报告注明）。

声明形如："反思写回后第 2 轮通过率 A% → B%（对照同轮 C%，
Δ 显著）"；曲线本身即 Reflexion 报告中的 learning curve。
"""
import logging

from evals.experiments.base import (
    Toggle,
    summarize_contrast,
    toggles_applied,
)
from evals.runner import run_cases

logger = logging.getLogger(__name__)

_EVO_ON = Toggle(kind="env", target="SELF_EVOLUTION_ENABLED", off_value=True)
_EVO_OFF = Toggle(kind="env", target="SELF_EVOLUTION_ENABLED", off_value=False)


def run_self_evolution_experiment(
    cases: list,
    iterations: int = 3,
    model_label: str = "",
) -> dict:
    """自进化学习曲线：对照组（关）先跑 T 轮，实验组（开）再跑 T 轮。"""
    control_curves: list[list[dict]] = []
    experimental_curves: list[list[dict]] = []

    logger.info("[exp-evo] 对照组（自进化关）%d 轮 × %d cases", iterations, len(cases))
    with toggles_applied([_EVO_OFF]):
        for it in range(iterations):
            control_curves.append(run_cases(cases, model_label=model_label))
            logger.info("[exp-evo] 对照组第 %d/%d 轮完成", it + 1, iterations)

    logger.info("[exp-evo] 实验组（自进化开）%d 轮", iterations)
    with toggles_applied([_EVO_ON]):
        for it in range(iterations):
            experimental_curves.append(run_cases(cases, model_label=model_label))
            logger.info("[exp-evo] 实验组第 %d/%d 轮完成（反思已写回）", it + 1, iterations)

    def _pass_rate(records: list[dict]) -> float:
        if not records:
            return 0.0
        return round(sum(1 for r in records if r.get("passed")) / len(records), 4)

    final_contrast = summarize_contrast(
        experimental_curves[-1] if experimental_curves else [],
        control_curves[-1] if control_curves else [],
    )
    return {
        "experiment": "self_evolution",
        "iterations": iterations,
        "learning_curve_experimental": [
            _pass_rate(records) for records in experimental_curves
        ],
        "learning_curve_control": [_pass_rate(records) for records in control_curves],
        "final_contrast": final_contrast,
        "records_experimental_last": experimental_curves[-1] if experimental_curves else [],
        "records_control_last": control_curves[-1] if control_curves else [],
        "note": (
            "前置条件：MySQL + Qdrant 可用（否则反思写回不持久化，曲线走平）。"
            "实验对记忆库有写副作用，生产库跑完建议清理。"
        ),
    }
