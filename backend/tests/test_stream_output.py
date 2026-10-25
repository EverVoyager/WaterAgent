"""同步拆分后的流式过滤共用同一状态机，标签可跨任意 chunk 边界。"""
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.graph.stream_output import stream_answer_events


def _chunk(content):
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=content))], usage=None,
    )


@pytest.mark.parametrize("valid_ids,expected", [({1}, "结论[1]。"), (None, "结论。")])
def test_thinking_and_citations_across_every_chunk_boundary(valid_ids, expected):
    text = "<think>内部推理</think>结论[1][2]。"
    for cut in range(1, len(text)):
        events = list(stream_answer_events(
            [_chunk(text[:cut]), _chunk(text[cut:])], "test", valid_ids,
        ))
        assert events[-1] == {"type": "synth_answer_full", "content": expected}
        assert "".join(e["content"] for e in events[:-1]) == expected


def test_unfinished_thinking_stays_hidden_and_usage_is_recorded():
    usage = SimpleNamespace(prompt_tokens=5)
    stream = [_chunk("答复<th"), _chunk("ink>未结束的内部推理"),
              SimpleNamespace(choices=[], usage=usage)]
    with patch("agent.graph.stream_output.record_llm_usage") as record:
        events = list(stream_answer_events(stream, "chat_stream"))
    assert events[-1]["content"] == "答复"
    record.assert_called_once_with("chat_stream", usage)
