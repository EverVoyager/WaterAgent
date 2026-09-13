"""覆盖矩阵检查：评估集的类别完备性守门（BFCL 式）。

书中"数据集设计原则"的工程化：评估集会随迭代悄悄退化成只测简单路径
（新类别忘加、等级档位缺角），本模块把"该覆盖的格子必须非空"变成
可断言的检查，CI 挂载后覆盖缺口直接挂门禁。

覆盖口径（非全交叉积——chitchat 永远不需要 level_decision，全交叉不合理）：
1. 每个已声明的用例类型至少 1 条用例；
2. 每个能力标签至少被 1 条用例标注；
3. business 的等级档位 I~IV 全出现；trap 的 claimed 档 I/II 全出现；
4. compression 用例历史必须超 token 预算（HISTORY_MAX_TOKENS），
   且针不在最近 2 轮原文里（压缩会原样保留近轮，埋近轮不构成考验）；
5. memory 用例必须带 memory_payload 与针。
"""
from agent.graph.context_compact import estimate_tokens
from app.core.config import get_settings
from evals.cases import CAP_MEMORY, CAP_NEEDLE, CASE_TYPES, EvalCase


def coverage_report(cases: list[EvalCase]) -> dict:
    """覆盖矩阵统计（人可读报表 + 缺口定位）。"""
    by_type: dict[str, int] = {t: 0 for t in CASE_TYPES}
    by_cap: dict[str, int] = {}
    levels_business: set[str] = set()
    claimed_trap: set[str] = set()
    for c in cases:
        by_type[c.case_type] = by_type.get(c.case_type, 0) + 1
        for cap in c.capabilities:
            by_cap[cap] = by_cap.get(cap, 0) + 1
        if c.case_type == "business" and c.expected_level:
            levels_business.add(c.expected_level)
        if c.case_type == "trap" and c.claimed_level:
            claimed_trap.add(c.claimed_level)
    return {
        "by_type": by_type,
        "by_capability": by_cap,
        "levels_business": sorted(levels_business),
        "claimed_trap": sorted(claimed_trap),
        "n_cases": len(cases),
    }


def coverage_gaps(cases: list[EvalCase], expected_types: tuple | None = None) -> list[str]:
    """返回覆盖缺口描述列表（空列表 = 覆盖完备）。

    expected_types: 期望非空的用例类型（None=全部已声明类型，即完整集口径）。
    子集场景（实验案例集）只检查自己声明的类型，不误报"其他类型为空"。
    """
    report = coverage_report(cases)
    gaps: list[str] = []

    types_expected = tuple(expected_types) if expected_types is not None else CASE_TYPES
    for ctype in types_expected:
        if report["by_type"].get(ctype, 0) == 0:
            gaps.append(f"用例类型 {ctype} 为空")

    # 出现过的类型对应的能力必须有非零覆盖（能力维度，
    # 按类型各自守门——记忆子集不要求针保留，反之亦然）
    for cap, owner_type in ((CAP_MEMORY, "memory"), (CAP_NEEDLE, "compression")):
        if report["by_type"].get(owner_type, 0) > 0 \
                and report["by_capability"].get(cap, 0) == 0:
            gaps.append(f"能力标签 {cap} 无用例覆盖")

    for level in ("I", "II", "III", "IV"):
        if report["by_type"].get("business") and level not in report["levels_business"]:
            gaps.append(f"business 等级档位 {level} 缺失")
    for claimed in ("I", "II"):
        if report["by_type"].get("trap") and claimed not in report["claimed_trap"]:
            gaps.append(f"trap 声称档位 {claimed} 缺失")

    budget = get_settings().HISTORY_MAX_TOKENS
    keep_rounds = get_settings().HISTORY_KEEP_RECENT_ROUNDS
    for c in cases:
        if c.case_type == "compression":
            tokens = sum(estimate_tokens(m.get("content", "")) for m in c.history)
            if tokens <= budget:
                gaps.append(
                    f"{c.case_id} 历史 {tokens} token 未超预算 {budget}（压缩不会触发）"
                )
            recent = c.history[-(keep_rounds * 2):]
            if any(n in "".join(m.get("content", "") for m in recent)
                   for n in c.needle_substrings):
                gaps.append(f"{c.case_id} 针埋在近 {keep_rounds} 轮原文内（不构成压缩考验）")
        if c.case_type == "memory" and (not c.memory_payload or not c.needle_substrings):
            gaps.append(f"{c.case_id} 缺 memory_payload 或 needle_substrings")
    return gaps


def assert_full_coverage(cases: list[EvalCase], expected_types: tuple | None = None) -> None:
    """覆盖完备断言（CI 门禁用，缺口直接抛 AssertionError）。"""
    gaps = coverage_gaps(cases, expected_types=expected_types)
    if gaps:
        detail = "\n".join(f"  - {g}" for g in gaps)
        raise AssertionError(f"评估集覆盖缺口 {len(gaps)} 处：\n{detail}")
