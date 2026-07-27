"""离线评估：base / SFT / SFT+GRPO 三版模型 × 300 条 held-out 场景。

经 vLLM OpenAI 兼容服务批量推理（评估前手动起服务，见 docs/implementation-plan.md Task 14），
工具回放确定性 mock，指标复用 train/rewards。
"""
import json
import sys
from pathlib import Path

# 路径引导：项目根（agent/train 可导入）+ backend 目录（app.* 可导入）
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
_BACKEND_ROOT = str(Path(__file__).resolve().parents[2] / "backend")
for _p in (_PROJECT_ROOT, _BACKEND_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from openai import OpenAI  # noqa: E402

from agent.tools.mock_executor import _mock_search_regulation  # noqa: E402
from agent.tools.schemas import SearchRegulationParams  # noqa: E402
from train.data_gen.scenario import generate_scenarios  # noqa: E402
from train.rewards.composite import compute_reward  # noqa: E402

EVAL_SEED_BASE = 200_000


def build_eval_scenarios(n: int = 300) -> list:
    return generate_scenarios(n=n, seed=EVAL_SEED_BASE)


def _extract_level(text: str) -> str | None:
    from train.data_gen.hermes_format import _LEVEL_MAP
    for pattern, level in _LEVEL_MAP:
        if pattern.search(text):
            return level
    return None


def eval_model(client: OpenAI, model: str, scenarios: list, max_rounds: int = 8) -> dict:
    from train.data_gen.teacher import synthesize_one  # 复用多轮合成循环（回放一致）

    n_correct, n_tool_ok, rewards, parts_all = 0, 0, [], []
    for scn in scenarios:
        trace = synthesize_one(client, model, scn, max_rounds=max_rounds)
        if trace is None:
            rewards.append(0.0)
            continue
        final = trace[-1]["content"] if trace[-1]["role"] == "assistant" else ""
        completion = "".join(m["content"] for m in trace if m["role"] == "assistant")
        rag = _mock_search_regulation(SearchRegulationParams(query=scn.query, top_k=3))["hits"]
        r, parts = compute_reward(completion, scn, rag_hits=rag)
        rewards.append(r)
        parts_all.append(parts)
        # 等级准确率：归一化提取后比较（"Ⅱ级" → "II"），而非子串匹配
        if scn.expected_level and _extract_level(final) == scn.expected_level:
            n_correct += 1
        if any(m["role"] == "tool" for m in trace):
            n_tool_ok += 1
    n = max(len(scenarios), 1)

    def _mean(xs: list) -> float:
        return round(sum(xs) / max(len(xs), 1), 4)

    return {
        "level_acc": round(n_correct / n, 4),
        "tool_ok": round(n_tool_ok / n, 4),
        "reward": _mean(rewards),
        "r1": _mean([p.get("r1", 0.0) for p in parts_all]),
        "r2": _mean([p.get("r2", 0.0) for p in parts_all]),
        "r3": _mean([p.get("r3", 0.0) for p in parts_all]),
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8001/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--models", nargs="+", required=True,
                        help="如: Qwen/Qwen2.5-7B-Instruct sft-merged grpo-merged")
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--out", type=Path, default=Path("train/eval/outputs/eval_results.json"))
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    scenarios = build_eval_scenarios(args.n)
    results = {m: eval_model(client, m, scenarios) for m in args.models}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[eval] → {args.out}")


if __name__ == "__main__":
    main()
