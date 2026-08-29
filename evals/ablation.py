"""记忆消融实验：有/无记忆注入对比，量化五类记忆架构的增益。

书中方法论落地（消融实验一节）：
- 表现差异要先能归因：模型替换实验区分模型瓶颈，消融实验区分
  Harness 组件贡献。本模块关闭记忆注入这一组件，其余全等，
  两版 case 级通过率之差即记忆组件的净贡献。
- 消融开关是架构早期注入的能力（memory 注入点在 nodes/synthesizer_node
  内以函数调用形式存在），此处用 unittest.mock.patch 在调用时替换
  （节点内为运行时 import，patch 包命名空间即可生效）。

方法：同一批用例跑两遍——注入开启（生产形态）vs 注入关闭（patch 三个
注入函数返回空串），对比 case_pass_rate / level_exact。
"""
import logging
from contextlib import contextmanager
from unittest.mock import patch

from evals.metrics import binomial_ci

logger = logging.getLogger(__name__)

# 记忆注入入口（agent.memory 包命名空间；节点运行时 import 该包，patch 生效）
_MEMORY_INJECTION_POINTS = (
    "agent.memory.build_longterm_section",      # 长期记忆（planner/synthesizer/direct_chat）
    "agent.memory.get_relevant_experiences",    # 情景+程序记忆（planner）
    "agent.memory.get_semantic_knowledge",      # 语义记忆（synthesizer）
)


@contextmanager
def memory_disabled():
    """临时关闭全部记忆注入（三个注入函数返回空串=无内容可注入）。"""
    with patch(_MEMORY_INJECTION_POINTS[0], return_value=""), \
         patch(_MEMORY_INJECTION_POINTS[1], return_value=""), \
         patch(_MEMORY_INJECTION_POINTS[2], return_value=""):
        yield


def _pass_rate(records: list[dict]) -> dict:
    ok = sum(1 for r in records if r.get("passed"))
    return binomial_ci(ok, len(records))


def _level_exact_rate(records: list[dict]) -> dict | None:
    applicable = [r for r in records if r.get("checks", {}).get("level_exact") is not None]
    if not applicable:
        return None
    ok = sum(1 for r in applicable if r["checks"]["level_exact"])
    return binomial_ci(ok, len(applicable))


def run_memory_ablation(cases: list, run_fn, model_label: str = "") -> dict:
    """对同一批用例跑"有记忆 vs 无记忆"两遍并对比。

    Args:
        cases: 评估用例列表
        run_fn: 执行函数（签名 (cases, model_label) -> records），由 CLI 注入
            （可复用 runner.run_cases，也可注入缓存版避免重复 LLM 调用）
    """
    logger.info("[ablation] 开始记忆消融：with-memory pass（%d cases）", len(cases))
    with_memory = run_fn(cases, model_label)
    logger.info("[ablation] with-memory 完成；开始 without-memory pass")
    without_memory = []
    with memory_disabled():
        without_memory = run_fn(cases, model_label)

    with_rate = _pass_rate(with_memory)
    without_rate = _pass_rate(without_memory)
    delta = round(with_rate["p"] - without_rate["p"], 4)

    # 显著性判断（书中准则）：分差与两版噪声带宽比较
    combined_se = (with_rate["se"] ** 2 + without_rate["se"] ** 2) ** 0.5
    significant = abs(delta) > 2 * combined_se if combined_se > 0 else None

    return {
        "n_cases": len(cases),
        "with_memory": with_rate,
        "without_memory": without_rate,
        "delta": delta,
        "combined_se": round(combined_se, 4),
        "significant": significant,
        "level_exact_with": _level_exact_rate(with_memory),
        "level_exact_without": _level_exact_rate(without_memory),
        # 逐 case 翻转明细（定位记忆帮助/伤害了哪些 case）
        "flipped_to_pass": sorted(
            r["case_id"] for r, w in zip(with_memory, without_memory, strict=False)
            if r.get("passed") and not w.get("passed")
        ),
        "flipped_to_fail": sorted(
            r["case_id"] for r, w in zip(with_memory, without_memory, strict=False)
            if not r.get("passed") and w.get("passed")
        ),
    }
