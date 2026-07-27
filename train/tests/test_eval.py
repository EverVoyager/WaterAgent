"""评估：种子区间隔离 + 报告渲染。"""
from train.data_gen.scenario import generate_scenarios
from train.eval.report import render_report
from train.eval.run_eval import EVAL_SEED_BASE, build_eval_scenarios


def test_eval_seed_disjoint():
    sft = generate_scenarios(n=50, seed=1000)
    grpo = generate_scenarios(n=50, seed=100_000)
    ev = build_eval_scenarios(n=50)
    sft_ids = {s.scenario_id for s in sft}
    grpo_ids = {s.scenario_id for s in grpo}
    ev_ids = {s.scenario_id for s in ev}
    assert ev_ids.isdisjoint(sft_ids) and ev_ids.isdisjoint(grpo_ids)
    assert EVAL_SEED_BASE == 200_000


def test_render_report_table():
    results = {
        "base": {"level_acc": 0.35, "r1": 0.1, "r2": 0.1, "r3": 0.05, "tool_ok": 0.3},
        "sft": {"level_acc": 0.82, "r1": 0.35, "r2": 0.28, "r3": 0.2, "tool_ok": 0.9},
        "sft_grpo": {"level_acc": 0.91, "r1": 0.39, "r2": 0.3, "r3": 0.27, "tool_ok": 0.95},
    }
    md = render_report(results)
    assert "| base |" in md and "| sft_grpo |" in md
    assert "0.91" in md and "+0.09" in md  # GRPO 相对 SFT 提升
