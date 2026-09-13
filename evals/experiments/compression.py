"""压缩等价性实验：针保留率 + token 节省（LLMLingua 式口径）。

与记忆增益的方向差异（对外声明的措辞要点）：
- 记忆实验：机制带来提升（A% → B%）；
- 压缩实验：机制以极小的保留损失换取 token 大幅节省
  （"保留率 ≥X% 的同时历史 token −Y%"）——基线（不压缩）天然满分保留，
  比的是机制版离满分多远，方向相反，报告措辞不能写成"提升"。

两遍对照：
- 基线（关压缩）：patch runner 入口为恒等函数，全量历史在场，
  针必然可寻（该版针保留率本身就是"历史可用性"的健全性检查）；
- 机制（开压缩）：正常链路，早段折叠为冻结摘要 + 近轮原文，
  针必须经摘要/按需还原存活到答案。

token 节省为离线确定性测量（不经 LLM 回答，压缩本身按需调 LLM 做摘要）。
"""
import logging
from contextlib import contextmanager

from evals.experiments.base import Toggle, summarize_contrast, toggles_applied
from evals.runner import run_case

logger = logging.getLogger(__name__)

# runner 模块顶层 import 了 compact_history 并在 _compact_history_entry 内
# 调用——patch runner 命名空间的入口函数即"压缩关闭"
_COMPACT_OFF_TOGGLE = Toggle(
    kind="patch",
    target="agent.graph.runner._compact_history_entry",
    off_fn=lambda history, *args, **kwargs: history,
)


@contextmanager
def compression_disabled():
    """关闭上下文压缩（入口恒等：全量历史直接进 prompt）。"""
    with toggles_applied([_COMPACT_OFF_TOGGLE]):
        yield


def measure_token_savings(history: list, compact_fn=None) -> dict:
    """离线测量压缩前后的历史 token（确定性，不经 Agent 链路）。

    Args:
        history: 会话历史（list[dict]，role/content）
        compact_fn: 压缩函数（默认真实 compact_history；测试可注入假实现）
    """
    from agent.graph.context_compact import compact_history, estimate_tokens

    fn = compact_fn or compact_history
    before = sum(estimate_tokens(m.get("content", "")) for m in history)
    after_history = fn([dict(m) for m in history])
    after = sum(estimate_tokens(m.get("content", "")) for m in after_history)
    return {
        "tokens_before": before,
        "tokens_after": after,
        "saved_pct": round((1 - after / before) * 100, 2) if before else 0.0,
    }


def run_compression_experiment(cases: list, model_label: str = "") -> dict:
    """压缩等价性：基线（不压缩）vs 机制（压缩）× 针保留 + token 节省。

    Returns:
        retention_contrast: needle_found 检查项上的对照（核心口径）
        pass_contrast:      用例级 passed 的对照（含意图等综合口径）
        token_savings:      每用例离线 token 测量 + 汇总均值
    """
    logger.info("[exp-compression] 基线 pass（压缩关闭，%d cases）", len(cases))
    baseline_records = []
    with compression_disabled():
        for case in cases:
            baseline_records.append(run_case(case, model_label=model_label))

    logger.info("[exp-compression] 机制 pass（压缩开启）")
    treated_records = [
        run_case(case, model_label=model_label) for case in cases
    ]

    savings = {
        c.case_id: measure_token_savings(c.history) for c in cases
    }
    pcts = [s["saved_pct"] for s in savings.values()]
    savings_summary = {
        "mean_saved_pct": round(sum(pcts) / len(pcts), 2) if pcts else 0.0,
        "per_case": savings,
    }

    return {
        "experiment": "compression",
        "retention_contrast": summarize_contrast(
            treated_records, baseline_records, check_key="needle_found"
        ),
        "pass_contrast": summarize_contrast(treated_records, baseline_records),
        "token_savings": savings_summary,
        "records_baseline": baseline_records,
        "records_treated": treated_records,
    }
