"""LLM usage 统计：前缀缓存（KV Cache）命中率观测。

三后端 usage 字段命名兼容（防御式提取，字段缺失或后端不支持时归零不报错）：
- OpenAI 风格 / 阿里云 MaaS / vLLM 新版：usage.prompt_tokens_details.cached_tokens
- DeepSeek 原生：usage.prompt_cache_hit_tokens（命中 token 数）
- vLLM 旧版：usage.cached_tokens

按节点（planner / synthesizer / chat / ...）进程内聚合，周期性输出 INFO 日志，
用于验证前缀稳定化改造的缓存命中效果。无外部依赖，单进程内有效。

可选持久化：设置环境变量 LLM_STATS_FILE=<path> 后，每次调用追加一行 JSONL
（ts/node/prompt_tokens/cached_tokens），进程退出后可用 CLI 聚合出报表：
    python -m app.core.llm_stats --file <path> [--cache-price 0.25]
"""
import argparse
import json
import logging
import os
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# node -> {"calls", "prompt_tokens", "cached_tokens"}
_STATS: dict[str, dict[str, int]] = {}
_STATS_LOCK = threading.Lock()

# 每 N 次调用输出一次汇总日志（避免高频刷屏）
_LOG_EVERY = 10

# 缓存命中 token 的计费折扣（相对全价，用于估算节省；主流厂商 1-4 折，默认 2.5 折）
_DEFAULT_CACHE_PRICE = 0.25


def _extract_cached_tokens(usage) -> tuple[int, int]:
    """从 usage 对象提取 (prompt_tokens, cached_tokens)。

    兼容 OpenAI 对象属性、dict、以及三后端的字段命名差异；
    usage 为 None、字段缺失或类型异常时返回 (0, 0)，绝不抛错——
    观测模块的任何异常都不能影响 LLM 调用主路径。
    """
    if usage is None:
        return 0, 0
    try:
        if isinstance(usage, dict):
            def get(obj, key):
                return obj.get(key)
        else:
            def get(obj, key):
                return getattr(obj, key, None)

        prompt = int(get(usage, "prompt_tokens") or 0)
        cached = 0

        # 1) OpenAI 风格：prompt_tokens_details.cached_tokens
        details = (
            usage.get("prompt_tokens_details")
            if isinstance(usage, dict)
            else getattr(usage, "prompt_tokens_details", None)
        )
        if details is not None:
            cached = int(get(details, "cached_tokens") or 0)

        # 2) DeepSeek 原生：prompt_cache_hit_tokens
        if not cached:
            cached = int(get(usage, "prompt_cache_hit_tokens") or 0)

        # 3) vLLM 旧版：cached_tokens
        if not cached:
            cached = int(get(usage, "cached_tokens") or 0)

        return prompt, cached
    except Exception:
        return 0, 0


def _persist_record(node: str, prompt: int, cached: int) -> None:
    """追加一行 JSONL 到 LLM_STATS_FILE（未设置环境变量则不持久化）。

    持久化失败只记 debug 日志——观测模块任何异常都不能影响 LLM 调用主路径。
    """
    path = os.environ.get("LLM_STATS_FILE", "")
    if not path:
        return
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "ts": round(time.time(), 3),
                "node": node,
                "prompt_tokens": prompt,
                "cached_tokens": cached,
            }, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.debug("[llm-cache] 持久化失败：%s", e)


def record_llm_usage(node: str, usage) -> None:
    """记录一次 LLM 调用的 usage（非流式传 resp.usage，流式传末 chunk 的 usage）。

    Args:
        node: 调用节点标识（planner / synthesizer_phase1 / synthesizer_phase2 / chat ...）
        usage: OpenAI Usage 对象、dict 或 None
    """
    prompt, cached = _extract_cached_tokens(usage)
    with _STATS_LOCK:
        s = _STATS.setdefault(
            node, {"calls": 0, "prompt_tokens": 0, "cached_tokens": 0}
        )
        s["calls"] += 1
        s["prompt_tokens"] += prompt
        s["cached_tokens"] += cached
        calls = s["calls"]

    _persist_record(node, prompt, cached)

    if cached:
        logger.debug(
            "[llm-cache] node=%s prompt_tokens=%d cached_tokens=%d", node, prompt, cached
        )
    if calls % _LOG_EVERY == 0:
        log_cache_summary()


def log_cache_summary() -> None:
    """输出各节点的缓存命中率汇总（INFO）。"""
    with _STATS_LOCK:
        snapshot = {k: dict(v) for k, v in _STATS.items()}
    for node, s in sorted(snapshot.items()):
        rate = (
            s["cached_tokens"] / s["prompt_tokens"] * 100
            if s["prompt_tokens"]
            else 0.0
        )
        logger.info(
            "[llm-cache] node=%s calls=%d prompt_tokens=%d cached_tokens=%d hit_rate=%.1f%%",
            node, s["calls"], s["prompt_tokens"], s["cached_tokens"], rate,
        )


def get_cache_stats() -> dict[str, dict[str, int]]:
    """返回统计快照（测试用）。"""
    with _STATS_LOCK:
        return {k: dict(v) for k, v in _STATS.items()}


def reset_cache_stats() -> None:
    """清空统计（测试用）。"""
    with _STATS_LOCK:
        _STATS.clear()


# ====== 离线聚合报表（JSONL 文件 / 内存快照通用）======

def summarize_records(records: list[dict]) -> dict:
    """聚合 usage 记录为报表。records 各项含 node/prompt_tokens/cached_tokens。

    返回 {"nodes": {node: {...}}, "total": {...}}；
    cost_full = 全价 token 数，cost_discounted = 命中部分按 cache_price 折算后的等效开销，
    saved_pct = 节省比例（衡量前缀缓存收益的核心数字）。
    """
    nodes: dict[str, dict[str, float]] = {}
    for r in records:
        node = str(r.get("node", "unknown"))
        prompt = int(r.get("prompt_tokens") or 0)
        cached = int(r.get("cached_tokens") or 0)
        s = nodes.setdefault(
            node, {"calls": 0, "prompt_tokens": 0, "cached_tokens": 0}
        )
        s["calls"] += 1
        s["prompt_tokens"] += prompt
        s["cached_tokens"] += cached

    def _finalize(s: dict) -> dict:
        prompt = s["prompt_tokens"]
        # 后端异常多报时钳制 cached ≤ prompt（口径统一，报表不出现负开销）
        cached = min(s["cached_tokens"], prompt)
        hit_rate = cached / prompt if prompt else 0.0
        discounted = prompt - cached * (1 - _DEFAULT_CACHE_PRICE)
        return {
            "calls": s["calls"],
            "prompt_tokens": prompt,
            "cached_tokens": cached,
            "hit_rate": round(hit_rate, 4),
            "cost_full": prompt,
            "cost_discounted": round(discounted, 1),
            "saved_pct": round((1 - discounted / prompt) * 100, 2) if prompt else 0.0,
        }

    return {
        "nodes": {k: _finalize(v) for k, v in sorted(nodes.items())},
        "total": _finalize({
            "calls": sum(v["calls"] for v in nodes.values()),
            "prompt_tokens": sum(v["prompt_tokens"] for v in nodes.values()),
            "cached_tokens": sum(v["cached_tokens"] for v in nodes.values()),
        }),
    }


def aggregate_file(path: str | Path) -> dict:
    """从 LLM_STATS_FILE 的 JSONL 聚合报表。"""
    records = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return summarize_records(records)


def format_summary(summary: dict) -> str:
    """把 summarize_records 的结果渲染成文本报表。"""
    lines = [
        f"{'node':<24}{'calls':>8}{'prompt':>12}{'cached':>12}"
        f"{'hit_rate':>10}{'saved%':>9}",
        "-" * 75,
    ]
    for node, s in summary["nodes"].items():
        lines.append(
            f"{node:<24}{s['calls']:>8}{s['prompt_tokens']:>12}{s['cached_tokens']:>12}"
            f"{s['hit_rate'] * 100:>9.1f}%{s['saved_pct']:>8.2f}%"
        )
    t = summary["total"]
    lines.append("-" * 75)
    lines.append(
        f"{'TOTAL':<24}{t['calls']:>8}{t['prompt_tokens']:>12}{t['cached_tokens']:>12}"
        f"{t['hit_rate'] * 100:>9.1f}%{t['saved_pct']:>8.2f}%"
    )
    lines.append(
        f"\n等效 token 开销：全价 {t['cost_full']:,.0f} → "
        f"缓存折扣后 {t['cost_discounted']:,.0f}"
        f"（命中部分按 {_DEFAULT_CACHE_PRICE:.0%} 计费估算）"
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="聚合 LLM_STATS_FILE 的缓存命中率报表")
    parser.add_argument("--file", default=os.environ.get("LLM_STATS_FILE", ""),
                        help="JSONL 统计文件路径（默认取 LLM_STATS_FILE 环境变量）")
    args = parser.parse_args()
    if not args.file or not Path(args.file).exists():
        print(f"统计文件不存在：{args.file or '(未指定)'}")
        print("先设置 LLM_STATS_FILE=<path> 再运行服务，跑完场景后重新执行本命令。")
        raise SystemExit(1)
    print(format_summary(aggregate_file(args.file)))


if __name__ == "__main__":
    main()
