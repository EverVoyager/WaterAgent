"""数据集编排入口：场景 → 教师合成 → 三道过滤 → JSONL + train/val 切分 + 统计。

用法：
  python -m train.data_gen.build_dataset --n 5000 --seed 1000 \
      --out train/lora/data/hermes_fc_v1.jsonl --rpm 30
干跑（不调用教师 API，验证编排）：
  python -m train.data_gen.build_dataset --n 20 --seed 1000 --dry-run
"""
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

# 路径引导：项目根（agent/train 可导入）+ backend 目录（app.* 可导入）
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
_BACKEND_ROOT = str(Path(__file__).resolve().parents[2] / "backend")
for _p in (_PROJECT_ROOT, _BACKEND_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from openai import OpenAI  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from train.data_gen.filters import FilterResult, filter_trace  # noqa: E402
from train.data_gen.scenario import generate_scenarios  # noqa: E402
from train.data_gen.stats import print_report, summarize  # noqa: E402
from train.data_gen.teacher import synthesize_dataset  # noqa: E402


def split_train_val(records: list, val_ratio: float, seed: int) -> tuple:
    rng = random.Random(seed)
    by_level: dict[str, list] = {}
    for r in records:
        by_level.setdefault(r["level"] or "chatty", []).append(r)
    train, val = [], []
    for group in by_level.values():
        rng.shuffle(group)
        k = max(1, round(len(group) * val_ratio))
        val.extend(group[:k])
        train.extend(group[k:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=Path("train/lora/data/hermes_fc_v1.jsonl"))
    parser.add_argument("--rpm", type=int, default=30)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--dry-run", action="store_true", help="只生成场景并打印配额，不调 API")
    args = parser.parse_args()

    scenarios = generate_scenarios(n=args.n, seed=args.seed)
    print(f"[build] 场景 {len(scenarios)} 条（seed={args.seed}）")
    if args.dry_run:
        print(Counter(s.expected_level or "chatty" for s in scenarios))
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    raw_path = args.out.with_suffix(".raw.jsonl")

    settings = get_settings()
    client = OpenAI(api_key=settings.LLM_API_KEY, base_url=settings.LLM_BASE_URL)
    written = synthesize_dataset(client, settings.LLM_MODEL, scenarios, raw_path, rpm=args.rpm)
    print(f"[build] 本次合成 {written} 条 → {raw_path}")

    records, rejects = [], Counter()
    for line in raw_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        scn = next(s for s in scenarios if s.scenario_id == rec["scenario_id"])
        result = filter_trace(rec["messages"], scn)
        if result == FilterResult.ACCEPT:
            records.append(rec)
        else:
            rejects[result.value] += 1

    train, val = split_train_val(records, args.val_ratio, args.seed)
    with args.out.open("w", encoding="utf-8") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    val_path = args.out.with_suffix(".val.jsonl")
    with val_path.open("w", encoding="utf-8") as f:
        for r in val:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = summarize(records, dict(rejects), len(scenarios))
    print_report(summary)
    print(f"[build] train={len(train)} → {args.out}  val={len(val)} → {val_path}")


if __name__ == "__main__":
    main()
