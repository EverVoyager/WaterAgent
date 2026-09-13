"""实验 ↔ 案例集映射：每个量化实验只跑自己的子集。

为什么子集而非全量（书中"评估成本与信度平衡"）：
- 归因干净：记忆实验的对照差异只来自记忆用例，不被业务用例噪声稀释；
- 成本可控：实验普遍两遍起步（对照/机制），记忆学习曲线还要 ×迭代数，
  子集把 API 开销控制在必要用例上；
- 组合固定：实验的案例组合写死在本模块，任何两次运行可比、可复现。

种子区间沿用 [300_000, 400_000)，与训练三段零重叠（assert_seed_isolation）。
"""
from evals.cases import EVAL_SEED_BASE, build_cases

# 各实验的固定案例组合（change 需同步更新 docs/eval-experiments.md 台账，
# 并在报告中作为 config 一部分落盘——组合变更视为新实验，旧数字不迁移）
EXPERIMENT_COMPOSITIONS: dict[str, dict] = {
    "memory": dict(
        desc="记忆增益（脚本化注入 vs 无记忆）",
        n_memory=24, n_compression=0, n_tool_edge=0,
    ),
    "compression": dict(
        desc="压缩等价性（针保留 + token 节省）",
        n_memory=0, n_compression=12, n_tool_edge=0,
    ),
    "self_evolution": dict(
        desc="反思学习曲线（自进化开/关 × 多轮迭代）",
        n_business=20, n_memory=8, n_trap=4,
    ),
    "kv_cache": dict(
        desc="KV 前缀冻结命中率（冻结 vs 破坏前缀）",
        n_business=4,
    ),
    "core": dict(  # model_ladder 的底座：62 条核心集，与基线组合一致
        desc="核心 62 条（训练阶梯与回归门禁共用底座）",
        n_business=30, n_chitchat=10, n_regulation=8,
        n_web_search=8, n_trap=6,
    ),
}


def get_experiment_cases(experiment: str, seed: int = EVAL_SEED_BASE) -> list:
    """取某实验的固定案例集（确定性：同 experiment+seed 结果完全一致）。"""
    if experiment not in EXPERIMENT_COMPOSITIONS:
        raise KeyError(
            f"未知实验: {experiment}（可选: {', '.join(EXPERIMENT_COMPOSITIONS)}）"
        )
    comp = EXPERIMENT_COMPOSITIONS[experiment]
    kwargs = {k: v for k, v in comp.items() if k != "desc"}
    return build_cases(seed=seed, **kwargs)
