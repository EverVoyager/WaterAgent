"""记忆检索阈值（MIN_SCORE=0.35）校准脚本。

用分布说话，替代拍脑袋：拉取三个记忆 collection 的全量向量，输出两组相似度分布——
1. 记忆两两相似度（冗余视图）：高分尾部 → 重复/可合并记忆
2. 探针查询 vs 记忆（判别视图）：领域相关探针 vs 无关探针的分数分布，
   建议阈值取"无关探针 P95"（保证 95% 的无关查询被挡在外面）

对照当前 MIN_SCORE 给出结论：阈值过松（无关探针大量漏进）/ 过严（相关探针大量漏召）。

用法（项目根目录，需 Qdrant + embedding 服务可用）：
    python backend/scripts/calibrate_memory_threshold.py [--pairs 2000]
Qdrant 不可用时打印原因并以退出码 1 结束（不做任何写操作，只读）。
"""
import argparse
import random
import sys
from pathlib import Path

# 双根结构：agent/ 在项目根，app/ 在 backend/ 下（对齐 seed_skills.py）
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "backend"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(_PROJECT_ROOT / "backend" / ".env")

from agent.memory import vector_index  # noqa: E402
from app.core.llm import get_qdrant_client  # noqa: E402

# 领域相关探针（应命中记忆库：防汛水情、预警等级、应急处置）
_DOMAIN_PROBES = [
    "吴堡站现在水情怎么样？",
    "龙门站未来24小时有洪水风险吗？",
    "防汛预警等级是怎么划分的？",
    "启动Ⅱ级应急响应需要做什么？",
    "黄河流量多大需要发预警？",
    "24小时降雨多少毫米发红色预警？",
    "沿河企业收到洪水预警该怎么办？",
    "洪峰流量怎么推算？",
]
# 无关探针（应被阈值挡住：日常生活、编程、娱乐等离域查询）
_OFF_DOMAIN_PROBES = [
    "今天股票行情怎么样？",
    "红烧肉怎么做才好吃？",
    "推荐一部周末看的电影",
    "Python 列表推导式怎么写？",
    "北京三日游攻略",
    "小孩发烧38度5怎么办？",
    "笔记本电脑怎么选配置？",
    "最近有什么好看的电视剧？",
]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _percentile(sorted_vals: list[float], p: float) -> float:
    """线性插值百分位（输入需已升序）。"""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    frac = k - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def _fmt_pct(vals: list[float]) -> str:
    s = sorted(vals)
    return (f"P5={_percentile(s, 0.05):.3f}  P50={_percentile(s, 0.50):.3f}  "
            f"P95={_percentile(s, 0.95):.3f}  Max={s[-1] if s else 0:.3f}")


def _load_vectors(collection: str) -> list[list[float]]:
    """滚动拉取 collection 全量向量（只读）。"""
    client = get_qdrant_client()
    vectors, offset = [], None
    while True:
        resp, next_offset = client.scroll(
            collection_name=collection, limit=256,
            with_vectors=True, offset=offset,
        )
        vectors.extend(p.vector for p in resp)
        if next_offset is None:
            break
        offset = next_offset
    return vectors


def calibrate(collection: str, n_pairs: int) -> dict | None:
    """对单个 collection 输出两组分布与建议阈值。向量数为 0 时返回 None。"""
    vectors = _load_vectors(collection)
    if not vectors:
        print(f"  {collection}: 0 个向量（库为空或未同步），跳过")
        return None

    # 1. 两两相似度（随机采样对，上限 n_pairs 对）
    rng = random.Random(42)
    pairwise = []
    if len(vectors) >= 2:
        for _ in range(n_pairs):
            i, j = rng.sample(range(len(vectors)), 2)
            pairwise.append(_cosine(vectors[i], vectors[j]))

    # 2. 探针查询 vs 全量记忆（领域相关 / 无关 各一组分数）
    from agent.rag.embedding import embed_texts
    probe_vecs = embed_texts(_DOMAIN_PROBES + _OFF_DOMAIN_PROBES)
    if probe_vecs is None:
        print("  embedding 服务不可用，无法做探针分布，仅输出两两分布")
        probe_vecs = []
    n_domain = len(_DOMAIN_PROBES)
    # 全配对分数（次要参考：反映"领域记忆与领域查询"的整体亲和度）
    domain_scores, off_scores = [], []
    # 逐探针最佳命中（主口径：检索语义下，一个查询只需一条强匹配即可召回；
    # 无关探针的最佳命中 = 最强漏入，领域探针的最佳命中 = 召回下限）
    domain_best, off_best = [], []
    for k, pv in enumerate(probe_vecs):
        best = -1.0
        for mv in vectors:
            score = _cosine(pv, mv)
            best = max(best, score)
            (domain_scores if k < n_domain else off_scores).append(score)
        (domain_best if k < n_domain else off_best).append(best)

    print(f"  {collection}: {len(vectors)} 向量")
    if pairwise:
        print(f"    两两相似度     {_fmt_pct(pairwise)}")
    if domain_best and off_best:
        off_max = max(off_best)
        dom_p05 = _percentile(sorted(domain_best), 0.05)
        # 建议阈值：实测无关漏入最大值 + 0.02 安全边距，上取到 0.05 步长
        # （贴近下界保召回：漏入已有 demote 效果闭环兜底，过严漏召无兜底）
        suggested = round(off_max + 0.07, 2)
        print(f"    领域探针最佳命中 {_fmt_pct(domain_best)}  ← 召回下限")
        print(f"    无关探针最佳命中 {_fmt_pct(off_best)}  ← 漏入上限")
        print(f"    全配对参考       领域{_fmt_pct(domain_scores)}")
        print(f"                     无关{_fmt_pct(off_scores)}")
        print(f"    建议阈值 ≈ {suggested}（漏入最大值 {off_max:.3f}+边距，"
              f"召回下限 {dom_p05:.3f}）")
        return {"suggested": suggested, "off_max": off_max, "domain_p05": dom_p05}
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="记忆检索阈值分布校准")
    parser.add_argument("--pairs", type=int, default=2000, help="两两相似度采样对数上限")
    args = parser.parse_args()

    collections = [
        ("agent_semantic_vec", vector_index.MIN_SCORE),
        ("agent_episodes_vec", vector_index.MIN_SCORE),
        ("agent_procedures_vec", vector_index.MIN_SCORE),
    ]
    print(f"当前 MIN_SCORE = {vector_index.MIN_SCORE}\n")
    suggestions = {}
    _last_metrics: dict[str, tuple[float, float]] = {}
    for name, current in collections:
        try:
            print(f"[{name}]（当前阈值 {current}）")
            result = calibrate(name, args.pairs)
            if result:
                suggestions[name] = (current, result["suggested"])
                _last_metrics[name] = (result["off_max"], result["domain_p05"])
        except Exception as e:
            print(f"  不可用：{e}")
        print()

    if not suggestions:
        print("没有任何 collection 产出建议（Qdrant 未启动或库为空？docker compose up 后重试）")
        raise SystemExit(1)

    print("== 结论 ==")
    for name, (cur, sug) in suggestions.items():
        off_max, dom_p05 = _last_metrics[name]
        if off_max >= cur:
            verdict = (f"过松：无关探针最高分 {off_max:.3f} 已越过当前阈值，"
                       "无关记忆会漏进 prompt")
        elif dom_p05 <= cur:
            verdict = (f"过严：最弱领域探针的最佳命中 {dom_p05:.3f} 低于当前阈值，"
                       "相关记忆会被漏召")
        else:
            verdict = (f"合理：当前阈值落在判别带内（漏入上限 {off_max:.3f} < "
                       f"{cur} < 召回下限 {dom_p05:.3f}）")
        print(f"  {name}: 当前 {cur} → 建议 {sug}（{verdict}）")


if __name__ == "__main__":
    main()
