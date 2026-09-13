"""训练阶梯实验：base → SFT → +DPO → +GRPO 同集对比（最硬的量化数字）。

为什么这组最硬（书中"评估对象的分层"）：
- 底层模型逐级替换，Harness 完全不变、案例集完全相同——
  差异只能归因于训练阶段；
- 核心指标 level_exact 的真值来自规则引擎 compute_warning_level
  （单一真值源），零 judge 依赖、无评判偏置。

运行前提：各 checkpoint 以同一 OpenAI 兼容端点可切模型名访问
（vLLM 多 adapter / LlamaFactory 多模型）。逐级切换会
patch settings.LLM_MODEL 并清 llm 客户端缓存（lru_cache），
若不同 checkpoint 走不同 base_url，需逐个进程跑 --model-label
再人工汇总（报告注明口径）。

声明形如："等级准确率 base A% → SFT B% → +DPO C% → +GRPO D%
（相邻级 Δ 与 2×SE 比较，超带宽才声明显著）"。
"""
import logging

from evals.experiments.base import summarize_contrast
from evals.metrics import compute_metrics
from evals.runner import run_cases

logger = logging.getLogger(__name__)


def run_model_ladder(models: list[str], cases: list, model_label: str = "") -> dict:
    """逐级切换模型跑同一案例集，产出阶梯表 + 相邻级对照。

    Args:
        models: checkpoint 模型名列表（按训练阶段顺序，如
            ["qwen3-4b-base", "wateragents-sft", "wateragents-dpo", "wateragents-grpo"]）
        cases: 固定案例集（case_sets.get_experiment_cases("core")）
    """
    from unittest.mock import patch

    from app.core.config import get_settings
    from app.core.llm import get_llm_client

    if len(models) < 2:
        raise ValueError("model_ladder 至少需要 2 个 checkpoint 才有对照意义")

    rungs: list[dict] = []
    for i, model in enumerate(models, 1):
        logger.info("[exp-ladder] 第 %d/%d 级：%s", i, len(models), model)
        with patch.object(get_settings(), "LLM_MODEL", model):
            get_llm_client.cache_clear()  # 客户端按端点缓存，换模型名需重建
            records = run_cases(cases, model_label=model)
        metrics = compute_metrics(records)
        rungs.append({
            "model": model,
            "case_pass_rate": metrics["case_pass_rate"],
            "level_exact": metrics.get("level_exact"),
            "tool_recall": metrics.get("tool_recall"),
            "tool_precision": metrics.get("tool_precision"),
            "citation_ok": metrics.get("citation_ok"),
            "latency": metrics.get("latency"),
            "records": records,
        })

    # 相邻阶梯对照（summarize_contrast 口径：绝对 Δ + 相对 + 显著性）
    step_contrasts = []
    for lower, upper in zip(rungs, rungs[1:]):
        step_contrasts.append({
            "from": lower["model"],
            "to": upper["model"],
            "contrast": summarize_contrast(upper["records"], lower["records"]),
            "level_exact_contrast": summarize_contrast(
                upper["records"], lower["records"], check_key="level_exact"
            ),
        })

    return {
        "experiment": "model_ladder",
        "models": models,
        "rungs": rungs,
        "step_contrasts": step_contrasts,
        "note": (
            "口径：同一 62 条核心集逐级运行，Harness 不变；"
            f"标签 model_label={model_label or '（未标注）'}。"
            "若各级 base_url 不同，请分进程运行后人工汇总。"
        ),
    }
