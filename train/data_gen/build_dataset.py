"""数据集编排入口：种子扩张 → 双模型蒸馏 → 三层过滤 → DPO + 知识问答混入。

用法：
  python -m train.data_gen.build_dataset --n 5000 --seed 1000 \
      --out train/lora/data/hermes_fc_v1.jsonl --rpm 30
干跑（不调用教师 API，验证编排）：
  python -m train.data_gen.build_dataset --n 20 --seed 1000 --dry-run
跳过 DPO + 评判（快速调试）：
  python -m train.data_gen.build_dataset --n 100 --seed 1000 --no-judge --no-dpo
"""
import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

# 路径引导：项目根（agent/train 可导入）+ backend 目录（app.* 可导入）
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
_BACKEND_ROOT = str(Path(__file__).resolve().parents[2] / "backend")
for _p in (_PROJECT_ROOT, _BACKEND_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# 显式加载 backend/.env（pydantic 的 env_file=".env" 按工作目录查找，从项目根运行时找不到）
from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(_BACKEND_ROOT) / ".env")

from openai import OpenAI  # noqa: E402
from tqdm import tqdm  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from train.data_gen.dpo_pairs import build_dpo_pair, write_dpo_jsonl  # noqa: E402
from train.data_gen.filters import FilterResult, filter_trace  # noqa: E402
from train.data_gen.judge import judge_trace  # noqa: E402
from train.data_gen.knowledge_synthesizer import synthesize_knowledge_dataset  # noqa: E402
from train.data_gen.query_expander import (  # noqa: E402
    expand_from_defaults,
    expand_knowledge_from_defaults,
)
from train.data_gen.scenario import from_expanded_queries  # noqa: E402
from train.data_gen.seed_queries import SeedQuery  # noqa: E402
from train.data_gen.stats import print_report, summarize  # noqa: E402
from train.data_gen.teacher import synthesize_dataset  # noqa: E402


def split_train_val(records: list, val_ratio: float, seed: int) -> tuple:
    """按等级分层切分 train/val，保证各等级比例一致。"""
    rng = random.Random(seed)
    by_level: dict[str, list] = {}
    for r in records:
        by_level.setdefault(r.get("level", ""), []).append(r)
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
    parser.add_argument("--knowledge-ratio", type=float, default=0.05, help="知识问答占比")
    parser.add_argument("--dry-run", action="store_true", help="只生成场景并打印配额，不调 API")
    parser.add_argument("--no-judge", action="store_true", help="跳过 LLM-as-Judge 评判")
    parser.add_argument("--no-dpo", action="store_true", help="跳过 DPO 正负对构建")
    args = parser.parse_args()

    settings = get_settings()
    teacher_client = OpenAI(api_key=settings.LLM_API_KEY, base_url=settings.LLM_BASE_URL)
    judge_client = OpenAI(api_key=settings.LLM_API_KEY, base_url=settings.LLM_BASE_URL)

    # 计算数量分配：95% 业务 + 5% 知识问答
    n_knowledge = round(args.n * args.knowledge_ratio)
    n_biz = args.n - n_knowledge

    # === 阶段1：种子扩张 ===
    print(f"[build] 种子扩张：业务 {n_biz} 条 + 知识 {n_knowledge} 条")
    biz_expanded = expand_from_defaults(n_biz, teacher_client, settings.LLM_MODEL, rpm=args.rpm)
    knowledge_expanded = expand_knowledge_from_defaults(
        n_knowledge, teacher_client, settings.LLM_MODEL, rpm=args.rpm
    )
    print(f"[build] 扩张得到 业务 {len(biz_expanded)} 条 + 知识 {len(knowledge_expanded)} 条")
    if args.dry_run:
        print("业务等级分布:", Counter(eq.level for eq in biz_expanded))
        print("知识问答:", len(knowledge_expanded), "条")
        return

    # === 阶段2：场景参数化（仅业务，知识问答不需要 mock 数据）===
    scenarios = from_expanded_queries(biz_expanded, args.seed)
    print(f"[build] 业务场景 {len(scenarios)} 条")

    # === 阶段3：教师合成（业务 + 知识）===
    args.out.parent.mkdir(parents=True, exist_ok=True)
    raw_path = args.out.with_suffix(".raw.jsonl")
    print(f"[build] 业务教师合成 → {raw_path}")
    written_biz = synthesize_dataset(
        teacher_client, settings.LLM_MODEL, scenarios, raw_path, rpm=args.rpm
    )
    print(f"[build] 业务合成 {written_biz} 条")

    # 知识问答合成（不调工具，直接 LLM 生成回答）
    knowledge_raw_path = args.out.with_suffix(".knowledge.raw.jsonl")
    knowledge_seeds = [
        SeedQuery(query=eq.query, station=eq.station, level=eq.level, intent=eq.intent)
        for eq in knowledge_expanded
    ]
    print(f"[build] 知识问答合成 → {knowledge_raw_path}")
    written_knwl = synthesize_knowledge_dataset(
        teacher_client, settings.LLM_MODEL, knowledge_seeds, knowledge_raw_path, rpm=args.rpm
    )
    print(f"[build] 知识合成 {written_knwl} 条")

    # === 阶段4-5：业务过滤 + 评判 ===
    sft_records = []  # 高分轨迹（业务 + 知识）
    low_score_pool = {}  # scenario_id -> (record, scn) 低分轨迹（DPO 候选）
    rejects = Counter()

    raw_lines = [line for line in raw_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    # 建立 scenario_id -> scenario 索引（O(1) 查找）
    scn_map = {s.scenario_id: s for s in scenarios}
    judge_label = "F1/F2 过滤" if args.no_judge else "F1/F2 + 评判"
    judge_interval = 60.0 / max(args.rpm, 1) if not args.no_judge else 0
    for line in tqdm(raw_lines, desc=judge_label, unit="条",
                     bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"):
        rec = json.loads(line)
        scn = scn_map.get(rec["scenario_id"])
        if scn is None:
            # raw 中存在但 scenarios 中没有（缓存丢失等），跳过
            rejects["orphan_raw"] += 1
            continue
        # F1/F2 硬规则过滤
        result = filter_trace(rec["messages"], scn)
        if result != FilterResult.ACCEPT:
            rejects[result.value] += 1
            continue
        # F3 LLM-as-Judge
        if not args.no_judge:
            t0 = time.time()
            judge_result = judge_trace(
                judge_client, settings.LLM_JUDGE_MODEL, rec["messages"], scn
            )
            # 评判限速（避免触发 qwen-max RPM 限制）
            elapsed = time.time() - t0
            if elapsed < judge_interval:
                time.sleep(judge_interval - elapsed)
            if judge_result is None:
                rejects["judge_failed"] += 1
                continue
            if judge_result.is_sft_quality:
                sft_records.append(rec)
            elif judge_result.is_dpo_negative:
                # 低分轨迹保留，用于 DPO 配对
                low_score_pool[scn.scenario_id] = (rec, scn)
                rejects[f"judge_score_{judge_result.total}"] += 1
            else:
                # score == 1，完全不可用，丢弃
                rejects["judge_score_1"] += 1
        else:
            sft_records.append(rec)

    print(f"[build] 业务 SFT {len(sft_records)} 条，低分池 {len(low_score_pool)} 个")

    # === 阶段6：DPO 正负对构建（低分场景重试合成）===
    dpo_pairs = []
    if not args.no_dpo and low_score_pool:
        retry_scenarios = [s for _, s in low_score_pool.values()]
        retry_raw_path = args.out.with_suffix(".retry.raw.jsonl")
        print(f"[build] DPO 重试合成 {len(retry_scenarios)} 个场景 → {retry_raw_path}")
        synthesize_dataset(
            teacher_client, settings.LLM_MODEL, retry_scenarios, retry_raw_path, rpm=args.rpm
        )

        retry_lines = [line for line in retry_raw_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        for line in tqdm(retry_lines, desc="DPO 评判", unit="条",
                         bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"):
            rec = json.loads(line)
            scn_id = rec["scenario_id"]
            if scn_id not in low_score_pool:
                continue
            low_rec, scn = low_score_pool[scn_id]
            # F1/F2
            result = filter_trace(rec["messages"], scn)
            if result != FilterResult.ACCEPT:
                continue
            # F3
            if not args.no_judge:
                t0 = time.time()
                judge_result = judge_trace(
                    judge_client, settings.LLM_JUDGE_MODEL, rec["messages"], scn
                )
                elapsed = time.time() - t0
                if elapsed < judge_interval:
                    time.sleep(judge_interval - elapsed)
                if not judge_result or not judge_result.is_sft_quality:
                    continue
            # 配对 DPO（chosen=重试高分, rejected=第一轮低分）
            pair = build_dpo_pair(rec, low_rec)
            if pair:
                dpo_pairs.append(pair)
                sft_records.append(rec)  # 重试高分也进 SFT

        # 写 DPO 文件
        if dpo_pairs:
            dpo_path = args.out.with_suffix(".dpo.jsonl")
            written_dpo = write_dpo_jsonl(dpo_pairs, dpo_path)
            print(f"[build] DPO 对 {written_dpo} 对 → {dpo_path}")
        else:
            print("[build] 未生成 DPO 对（重试均未达高分或回答相同）")

    # === 阶段7：知识问答加入 SFT（基本检查后直接纳入）===
    knwl_count = 0
    if knowledge_raw_path.exists():
        for line in knowledge_raw_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            messages = rec.get("messages", [])
            # 基本检查：轨迹含 system+user+assistant 三条以上
            if len(messages) >= 3 and messages[-1].get("role") == "assistant":
                sft_records.append(rec)
                knwl_count += 1
    print(f"[build] 知识问答 SFT {knwl_count} 条")

    # === 阶段8：train/val 切分 ===
    train, val = split_train_val(sft_records, args.val_ratio, args.seed)
    with args.out.open("w", encoding="utf-8") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    val_path = args.out.with_suffix(".val.jsonl")
    with val_path.open("w", encoding="utf-8") as f:
        for r in val:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = summarize(sft_records, dict(rejects), len(scenarios))
    print_report(summary)
    print(f"[build] train={len(train)} → {args.out}  val={len(val)} → {val_path}")
    if dpo_pairs:
        print(f"[build] dpo={len(dpo_pairs)} 对 → {args.out.with_suffix('.dpo.jsonl')}")


if __name__ == "__main__":
    main()
