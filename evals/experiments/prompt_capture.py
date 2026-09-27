"""LLM 请求 prompt 落盘：静态前缀复用占比的采集端。

拦截 OpenAI SDK 的 ``Completions.create``（类级别 patch，同时覆盖
``with_options`` 派生的客户端副本、流式与非流式调用），把每次请求的
tools + messages 按 sort_keys 紧凑 JSON 序列化后逐调用追加落盘。

序列化口径与 ``test_prefix_stability`` 的"逐字节一致"一致：字符串相同
⇔ 服务端看到的请求前缀相同。tools 排在 messages 前（对齐服务端渲染：
工具 schema 属于系统区），messages 只追加时序列化串严格保持前缀性质，
因此对落盘串做离线最长公共前缀（LCP）计算，就是端到端前缀缓存
可复用占比的忠实估计。

用法（不改变实验与生产默认行为，仅在环境变量存在时生效）：

    KV_PROMPT_DUMP=<dir> python evals/run_eval.py --experiment kv-cache

落盘：``<dir>/frozen.jsonl``、``<dir>/broken.jsonl``，每行::

    {"ts": <epoch 秒>, "prompt": "<tools+messages 序列化>"}

供 ``evals.experiments.prefix_reuse_ratio`` 离线计算占比。
"""
import json
import os
import time
from contextlib import contextmanager


def serialize_request(kwargs: dict) -> str:
    """把一次请求序列化为单字符串：tools 在前、messages 在后。

    顺序刻意对齐服务端渲染（工具 schema 属于系统区，渲染在消息之前）：
    只追加 messages 时不移动 tools 块，序列化串的前缀性质与真实
    服务端视图一致。内部字典 sort_keys 保证确定性。
    """
    return (
        '{"tools":'
        + json.dumps(kwargs.get("tools"), ensure_ascii=False, sort_keys=True,
                     separators=(",", ":"))
        + ',"messages":'
        + json.dumps(kwargs.get("messages"), ensure_ascii=False, sort_keys=True,
                     separators=(",", ":"))
        + "}"
    )


@contextmanager
def capture_prompts(path: str | None):
    """拦截所有 Completions.create，把请求 prompt 追加写入 path。

    path 为 None 时是空上下文（零开销、零行为变化）。异常安全：退出
    时无论成败都恢复原方法。
    """
    if not path:
        yield None
        return

    from openai.resources.chat.completions import Completions

    original = Completions.create
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)

    def _captured(self, *args, **kwargs):
        record = {"ts": time.time(), "prompt": serialize_request(kwargs)}
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return original(self, *args, **kwargs)

    Completions.create = _captured
    try:
        yield path
    finally:
        Completions.create = original
