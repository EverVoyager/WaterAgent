"""context_compact 单元测试。

机制已切换为任务段折叠（session_archive.py）：compact_history 只做预算
判断，超预算委托 compact_with_segments。分段/冻结/还原细节见
test_session_archive.py。此处覆盖：
- estimate_tokens：中英文 token 粗估
- compact_history：未超预算零开销 / 超预算委托分段压缩
- is_compacted_history：新旧摘要标记双兼容
- extract_history_context：从压缩 history 提取可读文本
"""
from unittest.mock import patch

from agent.graph.context_compact import (
    compact_history,
    estimate_tokens,
    extract_history_context,
    is_compacted_history,
)

# ====== estimate_tokens 测试 ======

class TestEstimateTokens:
    def test_empty_string(self):
        assert estimate_tokens("") == 0

    def test_none_safe(self):
        # estimate_tokens 内部 `if not text: return 0` 会把 None 当作空值处理
        # 上层 compact_history 调用前已对 history 判空，此处仅验证不抛异常
        assert estimate_tokens(None) == 0

    def test_chinese_estimation(self):
        # 中文按 1.5 字/token：300 字 ≈ 200 token
        assert 150 <= estimate_tokens("汛" * 300) <= 250

    def test_english_estimation(self):
        # 英文/数字按 4 字符/token：400 字符 ≈ 100 token
        assert 80 <= estimate_tokens("a" * 400) <= 120

    def test_mixed_estimation(self):
        tokens = estimate_tokens("吴堡站 flow 537 m3/s")
        assert tokens > 0


# ====== compact_history（预算判断 + 委托）======

class TestCompactHistory:
    def test_under_budget_returns_original(self):
        """未超预算零开销：原样返回（同一对象，不调任何 LLM/分段）。"""
        history = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！"},
        ]
        with patch("agent.memory.session_archive.compact_with_segments") as m:
            result = compact_history(history, max_tokens=4000)
        assert result is history
        m.assert_not_called()

    def test_over_budget_delegates_to_segments(self):
        """超预算委托 compact_with_segments（keep_recent_rounds 透传）。"""
        history = [
            {"role": "user", "content": "水情" * 2000},
            {"role": "assistant", "content": "流量平稳" * 2000},
        ]
        delegated = [{"role": "system", "content": "[历史任务·1] 意图：x"}]
        with patch(
            "agent.memory.session_archive.compact_with_segments",
            return_value=delegated,
        ) as m:
            result = compact_history(history, max_tokens=100, keep_recent_rounds=2)
        assert result == delegated
        m.assert_called_once_with(history, 2)

    def test_empty_history_passthrough(self):
        assert compact_history([]) == []


# ====== is_compacted_history（双标记兼容）======

class TestIsCompactedHistory:
    def test_segment_summary_marker(self):
        # 新机制：任务段摘要（可能多条）
        assert is_compacted_history([
            {"role": "system", "content": "[历史任务·1] 意图：查水情"},
            {"role": "system", "content": "[历史任务·2] 意图：问天气"},
            {"role": "user", "content": "下一个问题"},
        ])

    def test_legacy_merge_marker(self):
        # 旧机制残留（或历史会话）的合并摘要标记
        assert is_compacted_history([
            {"role": "system", "content": "[历史对话摘要]\n今天聊了水情"},
        ])

    def test_plain_history_false(self):
        assert not is_compacted_history([
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！"},
        ])

    def test_empty_false(self):
        assert not is_compacted_history([])


# ====== extract_history_context ======

class TestExtractHistoryContext:
    def test_renders_summary_and_rounds(self):
        history = [
            {"role": "system", "content": "[历史任务·1] 意图：查水情｜结论：Ⅳ级"},
            {"role": "user", "content": "吴堡站水情如何"},
            {"role": "assistant", "content": "流量 537"},
        ]
        text = extract_history_context(history)
        assert "[历史任务·1]" in text
        assert "用户：吴堡站水情如何" in text
        assert "助手：流量 537" in text

    def test_empty_history(self):
        assert extract_history_context([]) == ""
