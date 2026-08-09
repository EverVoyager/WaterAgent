"""Self-Instruct 查询扩张器。

用 qwen-plus 对种子查询改写扩展，生成大规模多样查询。
去重策略：字符 2-gram Jaccard 相似度 > 0.7 剔除。
"""
import json
import logging
import random
import time
from dataclasses import dataclass, field

from train.data_gen.seed_queries import SeedQuery, get_seeds, get_knowledge_seeds

logger = logging.getLogger(__name__)

# 业务查询扩张提示词
_EXPAND_PROMPT = """你是防汛预警领域的查询生成助手。
请基于以下种子查询，生成 {n} 个语义相似但表达不同的变体查询。

种子查询：{seed}

要求：
1. 保持原查询的核心意图（查水情/综合研判/生成预案）
2. 变换表达方式：换措辞、换视角、换紧急程度、换问法
3. 保持站点名称不变（{station}）
4. 每个变体一行，不要编号
5. 变体应该是自然、多样的中文问句

示例（种子"吴堡站现在水情怎么样？"）：
吴堡水文站目前水位流量是多少？
现在吴堡站的水情数据查一下。
帮我看看吴堡站实时水情。
吴堡站当前水文情况如何？

请生成 {n} 个变体："""

# 知识问答扩张提示词
_KNOWLEDGE_EXPAND_PROMPT = """你是防汛预警领域的查询生成助手。
请基于以下种子问题，生成 {n} 个语义相似但表达不同的变体问题。

种子问题：{seed}

要求：
1. 保持原问题的核心意图（防汛知识问答）
2. 变换表达方式：换措辞、换问法、换角度
3. 每个变体一行，不要编号
4. 变体应该是自然、多样的中文问句

示例（种子"什么是防汛？"）：
防汛具体是指什么？
防汛工作包括哪些内容？
能解释一下防汛的概念吗？
防汛的定义是什么？

请生成 {n} 个变体："""


@dataclass
class ExpandedQuery:
    query: str
    station: str
    level: str
    intent: str
    source_seed: str = ""  # 来源种子查询文本


def _char_bigrams(text: str) -> set:
    """计算字符 2-gram 集合。"""
    text = text.replace(" ", "").replace("？", "?").replace("。", ".")
    return {text[i:i + 2] for i in range(len(text) - 1)} if len(text) > 1 else {text}


def jaccard_similarity(a: str, b: str) -> float:
    """字符 2-gram Jaccard 相似度。"""
    sa, sb = _char_bigrams(a), _char_bigrams(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _deduplicate(queries: list[str], threshold: float = 0.7) -> list[str]:
    """去重：与已保留集合的 Jaccard 相似度 > threshold 则剔除。"""
    kept: list[str] = []
    for q in queries:
        if not q.strip():
            continue
        is_dup = any(jaccard_similarity(q, k) > threshold for k in kept)
        if not is_dup:
            kept.append(q.strip())
    return kept


def expand_one(client, model: str, seed: SeedQuery, n_variants: int = 10,
               temperature: float = 0.8) -> list[str]:
    """对单个种子扩张 n_variants 个变体。返回去重后的变体列表。"""
    # 根据 intent 选择 prompt
    if seed.intent == "knowledge":
        prompt = _KNOWLEDGE_EXPAND_PROMPT.format(n=n_variants, seed=seed.query)
    else:
        prompt = _EXPAND_PROMPT.format(
            n=n_variants, seed=seed.query, station=seed.station
        )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=512,
    )
    text = resp.choices[0].message.content or ""
    # 按行分割，去掉空行和编号前缀
    raw_variants = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # 去掉 "1. " "1、" 等编号前缀
        import re
        line = re.sub(r"^[\d]+[.、）)]\s*", "", line)
        if line:
            raw_variants.append(line)

    # 去重：变体之间 + 与种子的相似度
    variants = _deduplicate(raw_variants)
    variants = [v for v in variants if jaccard_similarity(v, seed.query) < 0.7]
    return variants


def expand(seeds: list[SeedQuery], target_n: int, client, model: str,
           rpm: int = 30, n_variants_per_seed: int = 15,
           max_rounds: int = 5) -> list[ExpandedQuery]:
    """从种子多轮扩张到 target_n 条查询。

    每轮对当前所有查询扩张变体，直到达到 target_n 或 max_rounds 轮。
    去重策略：全局 2-gram Jaccard > 0.7 剔除。
    """
    from tqdm import tqdm

    interval = 60.0 / max(rpm, 1)
    results: list[ExpandedQuery] = []
    seen_queries: set[str] = set()  # 全局去重集

    def _add(query: str, station: str, level: str, intent: str, source: str) -> bool:
        q = query.strip()
        if not q or q in seen_queries:
            return False
        # 与已有查询比对相似度
        if any(jaccard_similarity(q, k) > 0.7 for k in seen_queries):
            return False
        seen_queries.add(q)
        results.append(ExpandedQuery(
            query=q, station=station, level=level,
            intent=intent, source_seed=source,
        ))
        return True

    # 第一轮：种子本身 + 每个种子扩张变体
    current_sources = seeds[:]
    pbar_r1 = tqdm(current_sources, desc="种子扩张 R1", unit="种子",
                   bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
    for i, seed in enumerate(pbar_r1):
        t0 = time.time()
        _add(seed.query, seed.station, seed.level, seed.intent, seed.query)
        try:
            variants = expand_one(client, model, seed, n_variants_per_seed)
        except Exception as e:
            logger.warning("[expander] 种子 %d 扩张失败: %s", i, e)
            variants = []
        for v in variants:
            _add(v, seed.station, seed.level, seed.intent, seed.query)
        pbar_r1.set_postfix_str(f"累计 {len(results)}/{target_n}")
        if len(results) >= target_n:
            break
        time.sleep(max(0.0, interval - (time.time() - t0)))
    pbar_r1.close()

    # 多轮扩张：从已有结果中采样作为新种子继续扩张
    rnd = random.Random(42)
    for round_idx in range(2, max_rounds + 1):
        if len(results) >= target_n:
            break
        # 从已有结果采样作为新种子（采样 30% 或全部，取小）
        sample_size = min(len(results), max(len(seeds), len(results) // 3))
        sampled = rnd.sample(results, sample_size)
        added_this_round = 0
        pbar = tqdm(sampled, desc=f"种子扩张 R{round_idx}", unit="种子",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
        for eq in pbar:
            if len(results) >= target_n:
                break
            t0 = time.time()
            seed = SeedQuery(query=eq.query, station=eq.station, level=eq.level, intent=eq.intent)
            try:
                variants = expand_one(client, model, seed, n_variants_per_seed)
            except Exception as e:
                logger.warning("[expander] R%d '%s' 扩张失败: %s", round_idx, eq.query[:20], e)
                variants = []
            for v in variants:
                if _add(v, eq.station, eq.level, eq.intent, eq.source_seed):
                    added_this_round += 1
            pbar.set_postfix_str(f"新增 {added_this_round} 累计 {len(results)}/{target_n}")
            time.sleep(max(0.0, interval - (time.time() - t0)))
        pbar.close()
        tqdm.write(f"  R{round_idx} 采样 {sample_size} 条 → 新增 {added_this_round}（累计 {len(results)}/{target_n}）")
        if added_this_round == 0:
            logger.warning("[expander] R%d 无新增，停止扩张", round_idx)
            break

    return results[:target_n] if len(results) >= target_n else results


def _cache_path(tag: str, target_n: int) -> "Path":
    """缓存文件路径：train/lora/data/.expanded_{tag}_{n}.json"""
    from pathlib import Path
    # Path(__file__) = train/data_gen/query_expander.py
    # parents[0]=data_gen, parents[1]=train, parents[2]=项目根
    cache_dir = Path(__file__).resolve().parents[1] / "lora" / "data"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f".expanded_{tag}_{target_n}.json"


def _load_cache(tag: str, target_n: int) -> list[ExpandedQuery] | None:
    """读缓存；若数量不足 target_n 的 95% 则返回 None 触发重新扩张。

    允许 ±5% 浮动，避免因差几条就重新调 LLM 扩张（浪费成本）。
    """
    import json as _json
    path = _cache_path(tag, target_n)
    if not path.exists():
        return None
    try:
        data = _json.loads(path.read_text(encoding="utf-8"))
        # 允许 5% 浮动，至少要有 95% 的数量
        if len(data) < int(target_n * 0.95):
            return None
        return [ExpandedQuery(**item) for item in data]
    except Exception:
        return None


def _save_cache(tag: str, target_n: int, items: list[ExpandedQuery]) -> None:
    import json as _json
    path = _cache_path(tag, target_n)
    path.write_text(
        _json.dumps([{
            "query": q.query, "station": q.station,
            "level": q.level, "intent": q.intent, "source_seed": q.source_seed,
        } for q in items], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def expand_from_defaults(target_n: int, client, model: str,
                         rpm: int = 30) -> list[ExpandedQuery]:
    """使用默认业务种子列表扩张查询（带文件缓存）。"""
    cached = _load_cache("biz", target_n)
    if cached is not None:
        logger.info("[expander] 命中缓存：业务 %d 条（跳过 LLM 扩张）", len(cached))
        return cached
    result = expand(get_seeds(), target_n, client, model, rpm)
    _save_cache("biz", target_n, result)
    return result


def expand_knowledge_from_defaults(target_n: int, client, model: str,
                                   rpm: int = 30) -> list[ExpandedQuery]:
    """使用默认知识问答种子列表扩张查询（带文件缓存）。"""
    cached = _load_cache("knwl", target_n)
    if cached is not None:
        logger.info("[expander] 命中缓存：知识 %d 条（跳过 LLM 扩张）", len(cached))
        return cached
    result = expand(get_knowledge_seeds(), target_n, client, model, rpm)
    _save_cache("knwl", target_n, result)
    return result
