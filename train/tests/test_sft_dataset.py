"""SFT 预处理：assistant-only labels，非 assistant 段 -100。"""
import pytest

transformers = pytest.importorskip("transformers")
from transformers import AutoTokenizer  # noqa: E402

from train.lora.dataset import build_sft_dataset, render_messages  # noqa: E402


@pytest.fixture(scope="module")
def tokenizer():
    # 轻量 tokenizer 验证逻辑；训练时换成 Qwen2.5-7B-Instruct
    return AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-Qwen2ForCausalLM")


def _record():
    return {"messages": [
        {"role": "system", "content": "你是防汛智能体"},
        {"role": "user", "content": "吴堡水情？"},
        {"role": "assistant", "content": "<tool_call>\n{\"name\": \"get_hydrology\", \"arguments\": {\"station\": \"吴堡\"}}\n</tool_call>"},
        {"role": "tool", "content": "<tool_response>\n{\"flow_m3_s\": 3250.0}\n</tool_response>"},
        {"role": "assistant", "content": "发布Ⅱ级预警。"},
    ]}


def test_render_contains_hermes_tags(tokenizer):
    text = render_messages(_record()["messages"], tokenizer)
    assert "<tool_call>" in text and "<tool_response>" in text
    assert "<|im_start|>" in text


def test_assistant_only_loss(tokenizer):
    ds = build_sft_dataset([_record()], tokenizer, max_len=512)
    sample = ds[0]
    ids, labels = sample["input_ids"], sample["labels"]
    assert len(ids) == len(labels)
    # 有监督 token（labels != -100）非空且占比合理
    supervised = [i for i, lab in zip(ids, labels) if lab != -100]
    assert 0 < len(supervised) < len(ids)
    # 监督区解码后应包含最终等级文本
    text = tokenizer.decode(supervised)
    assert "预警" in text or "<tool_call>" in text
    # system/user 段不监督：首段 labels 全 -100
    assert labels[0] == -100
