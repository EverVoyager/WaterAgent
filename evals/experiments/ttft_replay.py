"""TTFT 回放测量：前缀冻结 vs 破坏的首 token 延迟对照。

方法（回放而非重跑 Agent）：
- 输入是 kv-cache 实验落盘的真实请求序列（``KV_PROMPT_DUMP`` 产出的
  frozen.jsonl，见 ``prompt_capture``）——请求体与生产逐字节一致，测量
  对象纯粹是服务端行为，不混入 Agent 路由/重试的非确定性；
- 每臂各回放两遍：第 1 遍预热（把静态前缀写入服务端缓存，且两臂
  先后顺序的影响被"都预热"抹平），**取第 2 遍计量**；
- frozen 臂：原样重放——服务端可命中前缀缓存，代表"前缀冻结工程"；
- broken 臂：每次重放前在 planner 系统提示尾部注入逐次变化的 nonce
  （与 kv-cache 实验同款破坏方式），前缀必然 miss，代表"无前缀工程"；
- TTFT = 发出请求到收到首个含内容的流式 chunk 的墙钟时间；max_tokens
  压到 1 只测 prefill + 首 token，解码时长不污染测量。

输出：两臂逐请求 TTFT、P50/P95/均值、降幅；JSON 落盘（含原始数据）。

用法（项目根目录，需 backend/.env 的模型配置）::

    python -m evals.experiments.ttft_replay --dump evals/history/kv_dump_0915_v2 \
        --json-out evals/history/exp_ttft_replay.json

注意：nonce 只注入 planner 请求（与 kv-cache 实验口径一致），synthesizer
等其他节点的请求两臂原样重放——因此对照差异聚焦 planner 节点。
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

_PLANNER_MARK = "工具调用规划模块"


def _load(path: Path) -> list[dict]:
    records = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    if not records:
        raise ValueError(f"{path}: 空文件")
    return records


def _break_prompt(record: dict, nonce: str) -> dict:
    """在 planner 系统提示尾部注入 nonce（前缀破坏，口径同 kv-cache 实验）。"""
    obj = json.loads(record["prompt"])
    for msg in obj.get("messages") or []:
        if (
            msg.get("role") == "system"
            and isinstance(msg.get("content"), str)
            and _PLANNER_MARK in msg["content"]
        ):
            msg["content"] += f"\n<!-- eval-nonce:{nonce} -->"
            break
    obj["messages"][-1]["content"] = (obj["messages"][-1].get("content") or "") + " "
    return obj


def _replay(client, model, record, break_nonce: str | None) -> float:
    """回放单条请求，返回 TTFT（秒）。"""
    if break_nonce:
        payload = _break_prompt(record, break_nonce)
    else:
        payload = json.loads(record["prompt"])
    t0 = time.perf_counter()
    stream = client.chat.completions.create(
        model=model,
        messages=payload["messages"],
        tools=payload.get("tools"),
        max_tokens=1,
        stream=True,
    )
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if getattr(delta, "content", None) or getattr(delta, "reasoning_content", None):
            return time.perf_counter() - t0
    return time.perf_counter() - t0  # 无内容 chunk 的兜底


def _arm(client, model, records, broken: bool) -> list[float]:
    label = "broken" if broken else "frozen"
    for pass_i in ("warmup", "measured"):
        ttfts = []
        for i, rec in enumerate(records):
            nonce = f"{time.time_ns()}" if (broken and pass_i == "measured") else (
                "warmup-fixed" if broken else None)
            t = _replay(client, model, rec, nonce)
            if pass_i == "measured":
                ttfts.append(t)
        print(f"[{label}] {pass_i} 完成，{len(records)} 条请求")
        if pass_i == "measured":
            return ttfts


def _pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, round(p * (len(xs) - 1))))
    return xs[k]


def main() -> None:
    parser = argparse.ArgumentParser(description="TTFT 回放测量（冻结 vs 破坏）")
    parser.add_argument("--dump", required=True, help="kv-cache 落盘目录（含 frozen.jsonl）")
    parser.add_argument("--json-out", default="", help="结果 JSON 输出路径")
    args = parser.parse_args()

    # 路径引导（与 run_eval 一致）：项目根 + backend
    root = Path(__file__).resolve().parents[2]
    for p in (str(root), str(root / "backend")):
        if p not in sys.path:
            sys.path.insert(0, p)
    from dotenv import load_dotenv

    load_dotenv(root / "backend" / ".env")
    from openai import OpenAI

    from app.core.config import get_settings

    settings = get_settings()
    client = OpenAI(api_key=settings.LLM_API_KEY, base_url=settings.LLM_BASE_URL,
                    timeout=120, max_retries=0)
    model = settings.LLM_MODEL

    records = _load(Path(args.dump) / "frozen.jsonl")
    print(f"回放 {len(records)} 条请求 ｜ 模型 {model} ｜ 每臂 预热1遍+计量1遍")

    frozen = _arm(client, model, records, broken=False)
    broken = _arm(client, model, records, broken=True)

    def _stats(xs):
        return {
            "n": len(xs),
            "p50": round(_pct(xs, 0.50), 4),
            "p95": round(_pct(xs, 0.95), 4),
            "mean": round(statistics.mean(xs), 4),
        }

    result = {
        "experiment": "ttft_replay",
        "method": "回放 kv-cache 落盘请求；每臂预热1遍取第2遍；max_tokens=1；"
                  "broken=planner 系统提示尾部逐次 nonce（口径同 kv-cache 实验）",
        "model": model,
        "dump": str(args.dump),
        "frozen": _stats(frozen),
        "broken": _stats(broken),
        "raw": {"frozen": [round(x, 4) for x in frozen],
                "broken": [round(x, 4) for x in broken]},
    }
    d_p50 = result["broken"]["p50"] - result["frozen"]["p50"]
    result["delta_p50_s"] = round(d_p50, 4)
    result["reduction_p50"] = round(
        d_p50 / result["broken"]["p50"], 4) if result["broken"]["p50"] else 0.0

    print(json.dumps({k: v for k, v in result.items() if k != "raw"},
                     ensure_ascii=False, indent=2))
    print(f"\nP50 降幅（broken→frozen）: {result['reduction_p50']:.1%}")
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"结果已写入 {args.json_out}")


if __name__ == "__main__":
    main()
