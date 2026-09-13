"""记忆增益实验：脚本化记忆注入 vs 无记忆注入（LongMemEval 式）。

与 ablation.py 的关系（两个实验回答不同问题，并存）：
- ablation.run_memory_ablation：关掉注入（patch 返回空），测"真实记忆库
  （MySQL/Qdrant 可用时）带来的增益"——库内容不可控，噪声大；
- 本实验：把注入内容替换为用例脚本化 payload，测"Harness+模型能否用好
  给定的记忆内容"——内容受控、真值确定（needle 即 payload 里的精确数值），
  可复现不依赖存储设施。对外的机制增益声明以本实验为准，
  真实库版作为部署环境的补充验证。

子类覆盖（LongMemEval 的能力分类）：
- fact      跨会话事实召回：payload 有、当前会话无 → 答案必含精确值
- update    知识更新：payload 旧值 + 会话内更正 → 用新值、不回显旧值
- temporal  时间推理：payload 时间事件 → 还原事件结论
"""
import logging
from contextlib import contextmanager
from contextvars import ContextVar
from unittest.mock import patch

from evals.experiments.base import summarize_contrast
from evals.runner import run_case

logger = logging.getLogger(__name__)

# 当前用例的脚本化记忆（None/空 dict = 无记忆对照版）
_CURRENT_PAYLOAD: ContextVar[dict | None] = ContextVar(
    "eval_memory_payload", default=None
)


def _payload_section(key: str) -> str:
    payload = _CURRENT_PAYLOAD.get() or {}
    return payload.get(key, "")


@contextmanager
def scripted_memory():
    """三个记忆注入函数替换为脚本化 payload 读取（缺 key = 空串）。

    两版对照都在本上下文内运行（对照组 payload 为空），真实记忆库
    的注入被完全旁路——差异只能归因于"注入内容是否在场"。
    """
    with patch("agent.memory.build_longterm_section",
               side_effect=lambda: _payload_section("longterm")), \
         patch("agent.memory.get_relevant_experiences",
               side_effect=lambda query="": _payload_section("experiences")), \
         patch("agent.memory.get_semantic_knowledge",
               side_effect=lambda query="": _payload_section("semantic")):
        yield


@contextmanager
def memory_payload_set(payload: dict | None):
    token = _CURRENT_PAYLOAD.set(payload or {})
    try:
        yield
    finally:
        _CURRENT_PAYLOAD.reset(token)


def run_memory_experiment(cases: list, model_label: str = "") -> dict:
    """记忆增益：with-memory（payload 注入）vs without-memory（空注入）。

    Returns（量化声明表消费的标准结构 + 实验特有字段）:
        contrast:       summarize_contrast 结果（check_key=None，用例级）
        needle_contrast: needle_found 检查项上的对照（更聚焦的口径）
        by_subtype:     fact/update/temporal 分别的对照
    """
    logger.info("[exp-memory] with-memory pass（%d cases）", len(cases))
    with_records = []
    with scripted_memory():
        for case in cases:
            with memory_payload_set(case.memory_payload):
                with_records.append(run_case(case, model_label=model_label))

    logger.info("[exp-memory] without-memory pass")
    without_records = []
    with scripted_memory():
        for case in cases:
            without_records.append(run_case(case, model_label=model_label))

    by_subtype: dict[str, dict] = {}
    # 子类从 case 序号反查（mem-NNN 的 fact/update/temporal 轮换顺序在 cases.py 固定）
    subtype_of = {}
    for case, _record in zip(cases, with_records, strict=False):
        idx = int(case.case_id.rsplit("-", 1)[1])
        subtype = ("fact", "fact", "update", "temporal")[idx % 4]
        subtype_of[case.case_id] = subtype
    for subtype in ("fact", "update", "temporal"):
        ids = {cid for cid, s in subtype_of.items() if s == subtype}
        sub_on = [r for r in with_records if r["case_id"] in ids]
        sub_off = [r for r in without_records if r["case_id"] in ids]
        if sub_on:
            by_subtype[subtype] = summarize_contrast(sub_on, sub_off)

    return {
        "experiment": "memory",
        "contrast": summarize_contrast(with_records, without_records),
        "needle_contrast": summarize_contrast(
            with_records, without_records, check_key="needle_found"
        ),
        "by_subtype": by_subtype,
        "records_with": with_records,
        "records_without": without_records,
    }
