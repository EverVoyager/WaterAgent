"""静态前缀复用占比：跨调用最长公共前缀（LCP）/ 总 prompt。

口径（与端到端命中率可比，且是它的在界上界）：
- 分母含全部调用（首次冷启动计入——它也是真实成本）；
- 第 i 次调用的可复用前缀 = 与之前任意一次请求序列化的最长公共
  token 前缀。服务端 prefix cache 保留历史全部请求的前缀块，
  取 max 贴近真实命中上界；
- 请求序列化与 ``prompt_capture.serialize_request`` 完全一致
  （tools + messages，sort_keys 紧凑 JSON），字符串相同 ⇔ 前缀相同；
- token 口径优先用本地分词器（--tokenizer，默认仓库内
  models/wateragents-qwen3-4b-v1；推理服务端分词器可能略异，但占比
  在同一分词器内自洽）；分词器不可用时回退字符口径并在输出中标注。

用法::

    python -m evals.experiments.prefix_reuse_ratio --dump <dir|file.jsonl>
    python -m evals.experiments.prefix_reuse_ratio --dump <dir> \\
        --tokenizer models/wateragents-qwen3-4b-v1 --json-out result.json

``--dump`` 是目录时计算其中全部 ``*.jsonl``（frozen / broken 两臂各一行），
是文件时只算该文件。broken 臂的占比应显著低于 frozen——它是采集与
计算链路自带的阴性对照。
"""
import argparse
import json
import sys
from pathlib import Path

DEFAULT_TOKENIZER = "models/wateragents-qwen3-4b-v1"


def _normalize(prompt: str) -> str:
    """把落盘串归一化为服务端渲染顺序（tools 在前、messages 在后）。

    早期版本的 prompt_capture 按字母序序列化（messages 在前），消息追加
    会整体挪动 tools 块，静态 LCP 因此被低估。这里解析后按渲染顺序
    重建，新旧落盘统一口径；已是目标顺序的串归一化后不变。
    """
    try:
        obj = json.loads(prompt)
    except json.JSONDecodeError:
        return prompt
    if not isinstance(obj, dict) or "messages" not in obj:
        return prompt

    def dumps(v) -> str:
        return json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    return '{"tools":' + dumps(obj.get("tools")) + ',"messages":' + dumps(obj["messages"]) + "}"


def _load_prompt_strings(path: Path) -> list[str]:
    """按时间序读取落盘的请求序列化串（保持调用顺序，归一化排序）。"""
    prompts: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            prompt = record.get("prompt")
            if not isinstance(prompt, str):
                raise ValueError(f"{path}: 记录缺少 prompt 字段: {record!r}")
            prompts.append(_normalize(prompt))
    if not prompts:
        raise ValueError(f"{path}: 没有任何记录")
    return prompts


def _load_tokenizer(path: str):
    """本地分词器 → tokenize 函数；不可用返回 None（调用方回退字符口径）。

    只认本地目录（含 tokenizer.json），绝不联网拉取——路径不存在直接
    回退。优先 transformers.AutoTokenizer（local_files_only）；没有
    transformers 时直接用 tokenizers 库加载 tokenizer.json。
    """
    p = Path(path)
    if not (p / "tokenizer.json").is_file():
        print(
            f"[warn] {p} 下没有 tokenizer.json（注意：需从仓库根运行），回退字符口径",
            file=sys.stderr,
        )
        return None
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(
            str(p), trust_remote_code=True, local_files_only=True
        )
        return lambda s: tok(s, add_special_tokens=False)["input_ids"]
    except Exception as exc_transformers:  # noqa: BLE001 - 逐级回退
        try:
            from tokenizers import Tokenizer

            tok = Tokenizer.from_file(str(p / "tokenizer.json"))
            return lambda s: tok.encode(s, add_special_tokens=False).ids
        except Exception as exc_tokenizers:  # noqa: BLE001
            print(
                f"[warn] 分词器加载失败（transformers: {exc_transformers}; "
                f"tokenizers: {exc_tokenizers}），回退字符口径",
                file=sys.stderr,
            )
            return None


def _lcp_len(a, b) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _arm_stats(prompts: list[str], units: list[list] | None, unit_name: str) -> dict:
    """单臂统计。units=None 时按字符计（prompts 自身作为序列）。"""
    seqs = units if units is not None else prompts
    total = sum(len(u) for u in seqs)
    reusable = 0
    for i in range(1, len(seqs)):
        reusable += max(_lcp_len(seqs[i], seqs[j]) for j in range(i))
    total_excl_first = total - len(seqs[0])
    return {
        "calls": len(seqs),
        "unit": unit_name,
        "total": total,
        "reusable": reusable,
        # 含首次冷启动——与端到端命中率（cached/total）可比的口径
        "share_all": round(reusable / total, 4) if total else 0.0,
        # 排除首次调用——提示词设计本身的复用上限
        "share_excl_first": (
            round(reusable / total_excl_first, 4) if total_excl_first else 0.0
        ),
    }


def analyze_file(path: Path, tokenize) -> dict:
    prompts = _load_prompt_strings(path)
    if tokenize is not None:
        token_lists = [tokenize(p) for p in prompts]
        stats = _arm_stats(prompts, token_lists, "tokens")
    else:
        stats = _arm_stats(prompts, None, "chars")
    return {"file": str(path), **stats}


def main() -> None:
    parser = argparse.ArgumentParser(description="静态前缀复用占比计算")
    parser.add_argument("--dump", required=True, help="落盘目录或单个 jsonl 文件")
    parser.add_argument(
        "--tokenizer", default=DEFAULT_TOKENIZER, help="本地分词器路径（tokenizer.json 所在目录）"
    )
    parser.add_argument("--json-out", default="", help="可选：结果 JSON 输出路径")
    args = parser.parse_args()

    target = Path(args.dump)
    if target.is_dir():
        files = sorted(target.glob("*.jsonl"))
        if not files:
            print(f"[error] {target} 下没有 .jsonl 文件", file=sys.stderr)
            sys.exit(1)
    else:
        files = [target]

    tokenize = _load_tokenizer(args.tokenizer)
    results = [analyze_file(f, tokenize) for f in files]

    print("\n=== 静态前缀复用占比（口径：跨调用 LCP / 总 prompt，含首次冷启动）===")
    for r in results:
        print(
            f"{r['file']}\n"
            f"  调用数 {r['calls']} ｜ 口径 {r['unit']} ｜ 总量 {r['total']}"
            f" ｜ 可复用前缀 {r['reusable']}\n"
            f"  占比（含冷启动） = {r['share_all']:.1%} ｜"
            f" 排除首次调用 = {r['share_excl_first']:.1%}"
        )
    if len(results) == 2:
        frozen = next((r for r in results if "frozen" in r["file"]), None)
        broken = next((r for r in results if "broken" in r["file"]), None)
        if frozen and broken:
            delta = frozen["share_all"] - broken["share_all"]
            print(
                f"\n阴性对照：frozen {frozen['share_all']:.1%}"
                f" vs broken {broken['share_all']:.1%}"
                f"（Δ={delta:+.1%}，broken 应显著更低，否则采集/计算链路有问题）"
            )

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump({"tokenizer": args.tokenizer, "results": results}, f,
                      ensure_ascii=False, indent=2)
        print(f"\n结果已写入 {args.json_out}")


if __name__ == "__main__":
    main()
