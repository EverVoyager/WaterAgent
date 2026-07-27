"""SFT 数据集预处理：Qwen 聊天模板渲染 + assistant-only loss。

不依赖 tokenizer 自带 {% generation %} 标记（小 tokenizer 没有），
改为分段编码：assistant 段（含 <|im_start|>assistant\n 前缀）计 loss，其余 -100。
"""
from datasets import Dataset

IM_START, IM_END = "<|im_start|>", "<|im_end|>"


def render_messages(messages: list, tokenizer=None) -> str:
    """messages → Qwen 模板文本（与 Qwen2.5 官方 apply_chat_template 等价）。"""
    parts = []
    for m in messages:
        parts.append(f"{IM_START}{m['role']}\n{m['content']}{IM_END}\n")
    return "".join(parts)


def _encode_with_masks(messages: list, tokenizer, max_len: int) -> dict:
    """分段编码，构造 input_ids / labels / attention_mask。"""
    input_ids: list[int] = []
    labels: list[int] = []
    for m in messages:
        seg = f"{IM_START}{m['role']}\n{m['content']}{IM_END}\n"
        ids = tokenizer(seg, add_special_tokens=False)["input_ids"]
        if m["role"] == "assistant":
            seg_labels = list(ids)  # assistant 段计 loss
        else:
            seg_labels = [-100] * len(ids)
        input_ids.extend(ids)
        labels.extend(seg_labels)
    input_ids, labels = input_ids[:max_len], labels[:max_len]
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids),
    }


def build_sft_dataset(records: list, tokenizer, max_len: int = 4096) -> Dataset:
    """JSONL records → HF Dataset（已编码）。"""
    features = [_encode_with_masks(r["messages"], tokenizer, max_len) for r in records]
    return Dataset.from_list(features)


def load_jsonl(path) -> list:
    import json
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
