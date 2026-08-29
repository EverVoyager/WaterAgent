"""系统级评估 CLI 入口。

用法（项目根目录）：
    # 冒烟（8 条，确定性指标）
    python evals/run_eval.py --limit 8

    # 全量（62 条：30 业务 + 10 闲聊 + 8 法规 + 8 联网 + 6 陷阱）
    python evals/run_eval.py

    # 带 LLM Judge + 稳定性 pass^3 + 记忆消融
    python evals/run_eval.py --judge --pass-k --ablation 20

    # 更新基线（评审通过后入库）
    python evals/run_eval.py --update-baseline

评估对象是"模型 + Harness 组合体"：驱动真实 Agent 链路（需配置 LLM_API_KEY），
工具层经回放上下文确定性 mock，无需 MySQL/Qdrant（记忆注入不可用时自动降级）。
退出码：0 正常；1 基线回归门禁触发。
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# 路径引导：项目根（agent/evals 可导入）+ backend（app.* 可导入）
_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
_BACKEND_ROOT = str(Path(_PROJECT_ROOT) / "backend")
for _p in (_PROJECT_ROOT, _BACKEND_ROOT):
    if _p not in sys.path and Path(_p).exists():
        sys.path.insert(0, _p)

# config 的 env_file=".env" 相对 CWD 解析；eval 从项目根运行时需显式加载 backend/.env
# （真实环境变量优先级更高，CI 直接注入 env var 的方式不受影响）
from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(_BACKEND_ROOT) / ".env")

from evals.ablation import run_memory_ablation  # noqa: E402
from evals.cases import EVAL_SEED_BASE, build_cases  # noqa: E402
from evals.judge import aggregate_judge, judge_record  # noqa: E402
from evals.metrics import compute_metrics, compute_pass_power_k  # noqa: E402
from evals.regression import build_baseline_payload, compare_with_baseline  # noqa: E402
from evals.report import render_report  # noqa: E402
from evals.runner import run_case_repeated, run_cases  # noqa: E402

EVALS_DIR = Path(_PROJECT_ROOT) / "evals"
BASELINE_PATH = EVALS_DIR / "baselines" / "baseline.json"
HISTORY_DIR = EVALS_DIR / "history"

_PASS_K_SUBSET = 20  # business 子集抽样规模（pass^k 重复跑成本 k 倍，控制预算）
_PASS_K = 3


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WaterAgents 系统级评估")
    parser.add_argument("--n-business", type=int, default=30)
    parser.add_argument("--n-chitchat", type=int, default=10)
    parser.add_argument("--n-regulation", type=int, default=8)
    parser.add_argument("--n-web-search", type=int, default=8)
    parser.add_argument("--n-trap", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0,
                        help="截断用例总数（冒烟用），0=不截断")
    parser.add_argument("--seed", type=int, default=EVAL_SEED_BASE)
    parser.add_argument("--judge", action="store_true", dest="judge",
                        help="启用 LLM-as-Judge 软指标（默认关闭）")
    parser.add_argument("--no-judge", action="store_false", dest="judge")
    parser.set_defaults(judge=False)
    parser.add_argument("--pass-k", action="store_true",
                        help=f"business 子集 {_PASS_K_SUBSET} 条重复 {_PASS_K} 次，算 pass^{_PASS_K}")
    parser.add_argument("--ablation", type=int, default=0, metavar="N",
                        help="记忆消融：N 条 business 用例有/无记忆注入对比")
    parser.add_argument("--model-label", default="",
                        help="模型标签（默认读 settings.LLM_MODEL），多模型对比用")
    parser.add_argument("--update-baseline", action="store_true",
                        help="把本次结果写入基线（评审通过后执行）")
    parser.add_argument("--baseline", default=str(BASELINE_PATH))
    parser.add_argument("--report", default="",
                        help="报告输出路径（默认 evals/history/eval_<时间戳>.md）")
    return parser.parse_args(argv)


def _model_label(explicit: str) -> str:
    if explicit:
        return explicit
    from app.core.config import get_settings
    return get_settings().LLM_MODEL


def _judge_records(records: list[dict], model_label: str) -> tuple[list[dict | None], str]:
    """对全部记录做 Rubric 评判。返回 (judgments, judge_model)。"""
    from app.core.config import get_settings
    from app.core.llm import get_llm_client
    settings = get_settings()
    judge_model = settings.LLM_JUDGE_MODEL
    client = get_llm_client()
    judgments = []
    for i, record in enumerate(records, 1):
        judgments.append(judge_record(record, client=client, model=judge_model))
        if i % 10 == 0:
            print(f"  [judge] {i}/{len(records)}")
    return judgments, judge_model


def _run_pass_k(cases: list, model_label: str) -> dict:
    """business 子集抽样 × k 次重复 → pass^k。"""
    business = [c for c in cases if c.case_type == "business"][:_PASS_K_SUBSET]
    repeats = [run_case_repeated(c, k=_PASS_K, model_label=model_label)
               for c in business]
    # 转置：repeat i 覆盖全部抽样 case
    records_by_repeat = [
        [repeats[j][i] for j in range(len(repeats))]
        for i in range(_PASS_K)
    ]
    return compute_pass_power_k(records_by_repeat, k=_PASS_K)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    model_label = _model_label(args.model_label)
    print(f"[eval] 模型: {model_label} ｜ judge: {'on' if args.judge else 'off'}")

    # 1. 数据集（种子隔离断言在 build_cases 内）
    cases = build_cases(
        n_business=args.n_business,
        n_chitchat=args.n_chitchat,
        n_regulation=args.n_regulation,
        n_web_search=args.n_web_search,
        n_trap=args.n_trap,
        seed=args.seed,
    )
    if args.limit > 0:
        cases = cases[: args.limit]
    print(f"[eval] 用例: {len(cases)} 条")

    # 2. 运行完整 Agent 链路
    records = run_cases(cases, model_label=model_label)

    config = {
        "model_label": model_label,
        "seed": args.seed,
        "n_business": args.n_business,
        "n_chitchat": args.n_chitchat,
        "n_regulation": args.n_regulation,
        "n_web_search": args.n_web_search,
        "n_trap": args.n_trap,
        "limit": args.limit,
        "judge": args.judge,
    }

    # 3. 确定性指标
    metrics = compute_metrics(records)
    print(f"[eval] 用例通过率: {metrics['case_pass_rate']['p'] * 100:.1f}% "
          f"(CI {metrics['case_pass_rate']['ci95']})")

    # 4. pass^k 稳定性
    pass_power_k = _run_pass_k(cases, model_label) if args.pass_k else None
    if pass_power_k:
        print(f"[eval] pass^{pass_power_k['k']}: {pass_power_k['p'] * 100:.1f}%")

    # 5. LLM Judge
    judge_agg = None
    judge_model = ""
    if args.judge:
        print("[eval] LLM Judge 评判中 ...")
        judgments, judge_model = _judge_records(records, model_label)
        judge_agg = aggregate_judge(judgments)
        config["judge_model"] = judge_model
        # 并入 metrics 供回归门禁监控软指标
        metrics["faithfulness_rate"] = judge_agg.get("faithfulness_rate")
        metrics["quality_score_mean"] = judge_agg.get("quality_score_mean")
        print(f"[eval] judge: faithfulness={judge_agg.get('faithfulness_rate')} "
              f"quality={judge_agg.get('quality_score_mean')} "
              f"veto={judge_agg.get('veto_count')}")

    # 6. 记忆消融
    ablation_result = None
    if args.ablation > 0:
        subset = [c for c in cases if c.case_type == "business"][: args.ablation]
        print(f"[eval] 记忆消融（{len(subset)} 条 × 2 passes）...")
        ablation_result = run_memory_ablation(
            subset,
            run_fn=lambda cs, label: run_cases(cs, model_label=label),
            model_label=model_label,
        )
        print(f"[eval] 消融 Δ={ablation_result['delta']:+.4f}")

    # 7. 基线回归
    regression_lines: list[str] = []
    gate_failed = False
    baseline_path = Path(args.baseline)
    if baseline_path.exists() and not args.update_baseline:
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        report = compare_with_baseline(metrics, baseline, records)
        regression_lines = report.summary().splitlines()
        print("[eval] 基线回归对比:")
        for line in regression_lines:
            print("  " + line)
        composition_diff = _baseline_composition_diff(baseline, config)
        if composition_diff:
            print(f"[eval] ⚠️ 基线组合不一致（{composition_diff}），跳过门禁判定")
        elif report.regressed:
            gate_failed = True
            print("[eval] ✗ 回归门禁触发")
        else:
            print("[eval] ✓ 未触发回归门禁")
    elif args.update_baseline:
        print("[eval] --update-baseline：本次结果将写入基线")
    else:
        print("[eval] 无基线文件，跳过回归对比（首次运行用 --update-baseline 建基线）")

    # 8. 报告落盘
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = Path(args.report) if args.report else HISTORY_DIR / f"eval_{timestamp}.md"
    md = render_report(
        records, metrics, config,
        pass_power_k=pass_power_k,
        ablation=ablation_result,
        judge_agg=judge_agg,
        regression_lines=regression_lines,
    )
    report_path.write_text(md, encoding="utf-8")
    (HISTORY_DIR / f"eval_{timestamp}.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"[eval] 报告: {report_path}")

    # 9. 更新基线
    if args.update_baseline:
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        payload = build_baseline_payload(metrics, records, config)
        baseline_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        print(f"[eval] 基线已更新: {baseline_path}")

    return 1 if gate_failed else 0


def _baseline_composition_diff(baseline: dict, config: dict) -> str:
    """基线与本次运行的用例组合/模型是否一致（不一致则门禁不可比）。"""
    base_cfg = baseline.get("config", {})
    diffs = []
    for key in ("n_business", "n_chitchat", "n_regulation", "n_web_search",
                "n_trap", "limit", "model_label"):
        if base_cfg.get(key) != config.get(key):
            diffs.append(f"{key}: {base_cfg.get(key)}→{config.get(key)}")
    return ", ".join(diffs)


if __name__ == "__main__":
    sys.exit(main())
