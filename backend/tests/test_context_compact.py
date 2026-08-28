"""context_compact 单元测试。

覆盖：
- estimate_tokens：中英文 token 粗估
- compact_history：未超预算 / 超预算触发 LLM 摘要 / LLM 失败降级截断
- _split_recent：保留窗口边界
- extract_history_context：从压缩 history 提取可读文本
- 摘要缓存命中
"""
from unittest.mock import MagicMock, patch

from agent.graph.context_compact import (
    _SUMMARY_CACHE,
    _split_recent,
    _truncate_history,
    compact_history,
    estimate_tokens,
    extract_history_context,
)

# ====== estimate_tokens 测试 ======

class TestEstimateTokens:
    def test_empty_string(self):
        assert estimate_tokens("") == 0

    def test_none_safe(self):
        # estimate_tokens 内部 `if not text: return 0` 会把 None 当作空值处理
        # 上层 compact_history 调用前已对 history 判空，此处仅验证不抛异常
        assert estimate_tokens(None) == 0

    def test_pure_chinese(self):
        # 中文按 1.5 字/token：3 个汉字 = 2 token（+1 上取整）
        tokens = estimate_tokens("黄河水")
        assert tokens == int(3 / 1.5) + 1  # 2 + 1 = 3

    def test_pure_english(self):
        # 英文按 4 字符/token：8 个字符 = 2 token（+1）
        tokens = estimate_tokens("abcdefgh")
        assert tokens == int(8 / 4) + 1  # 2 + 1 = 3

    def test_mixed(self):
        # 2 中文 + 4 英文 = 2/1.5 + 4/4 + 1 = 1 + 1 + 1 = 3
        tokens = estimate_tokens("黄河abcd")
        assert tokens >= 3

    def test_returns_positive_for_any_text(self):
        assert estimate_tokens("x") >= 1


# ====== _split_recent 测试 ======

class TestSplitRecent:
    def test_empty_history(self):
        old, recent = _split_recent([], 2)
        assert old == []
        assert recent == []

    def test_short_history_all_recent(self):
        # 3 条消息，keep_rounds=2（保留 4 条）→ 全部进 recent
        history = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
        ]
        old, recent = _split_recent(history, 2)
        assert old == []
        assert recent == history

    def test_split_correctly(self):
        # 6 条消息，keep_rounds=2（保留 4 条）→ 前 2 进 old，后 4 进 recent
        history = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "q3"},
            {"role": "assistant", "content": "a3"},
        ]
        old, recent = _split_recent(history, 2)
        assert len(old) == 2
        assert len(recent) == 4
        assert old[0]["content"] == "q1"
        assert recent[-1]["content"] == "a3"


# ====== compact_history 测试 ======

class TestCompactHistory:
    def setup_method(self):
        # 每个测试前清空摘要缓存
        _SUMMARY_CACHE.clear()

    def test_empty_history_returns_empty(self):
        assert compact_history([]) == []

    def test_short_history_no_compaction(self):
        # 未超预算：原样返回（同一对象引用）
        history = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好，有什么可以帮您？"},
        ]
        result = compact_history(history, max_tokens=10000)
        assert result is history  # 未触发压缩，返回原对象

    def test_long_history_triggers_llm_summary(self):
        # 超预算：触发 LLM 摘要
        # 构造超 100 token 的 history（max_tokens=100 触发压缩）
        long_text = "黄河吕梁段水位告警，请尽快处置。" * 20  # 约 400 字 ≈ 270 token
        history = [
            {"role": "user", "content": long_text},
            {"role": "assistant", "content": long_text},
            {"role": "user", "content": long_text},
            {"role": "assistant", "content": long_text},
            {"role": "user", "content": "现在情况如何？"},
            {"role": "assistant", "content": "已发布Ⅲ级预警。"},
        ]
        mock_summary = "用户询问黄河水情，已发布Ⅲ级预警。"
        with patch(
            "agent.graph.context_compact._summarize_via_llm",
            return_value=mock_summary,
        ) as mock_llm:
            result = compact_history(history, max_tokens=100, keep_recent_rounds=1)

        # 验证调用了 LLM 摘要
        mock_llm.assert_called_once()
        # 验证返回格式：首条是 system 摘要，后跟 recent
        assert result[0]["role"] == "system"
        assert "[历史对话摘要]" in result[0]["content"]
        assert mock_summary in result[0]["content"]
        # keep_recent_rounds=1 → 保留最后 2 条（1 问 1 答）
        assert len(result) == 3  # 1 摘要 + 2 recent
        assert result[1]["content"] == "现在情况如何？"
        assert result[2]["content"] == "已发布Ⅲ级预警。"

    def test_llm_failure_falls_back_to_truncation(self):
        # LLM 摘要失败：降级为简单截断
        long_text = "黄河吕梁段水位告警，请尽快处置。" * 20
        history = [
            {"role": "user", "content": long_text},
            {"role": "assistant", "content": long_text},
            {"role": "user", "content": "现在情况如何？"},
            {"role": "assistant", "content": "已发布Ⅲ级预警。"},
        ]
        with patch(
            "agent.graph.context_compact._summarize_via_llm",
            return_value="",  # LLM 失败返回空
        ):
            result = compact_history(history, max_tokens=100, keep_recent_rounds=1)

        # 验证首条是 system 摘要（截断内容）
        assert result[0]["role"] == "system"
        assert "[历史对话摘要]" in result[0]["content"]
        # 截断内容包含原 role 标签
        assert "user：" in result[0]["content"]

    def test_keep_rounds_too_large_no_compaction(self):
        # keep_recent_rounds 覆盖全部 history：to_summarize 为空，返回原 history
        history = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
        ]
        # 强制进入超预算分支
        with patch(
            "agent.graph.context_compact.estimate_tokens",
            return_value=99999,
        ):
            result = compact_history(history, max_tokens=100, keep_recent_rounds=10)
        # to_summarize 为空 → 直接返回原 history
        assert result is history

    def test_cache_hit_avoids_duplicate_llm_call(self):
        """相同 history 指纹命中缓存，不重复调 LLM client。

        缓存逻辑在 _summarize_via_llm 内部，需 mock 底层 LLM client 才能验证。
        """
        from agent.graph.context_compact import _summarize_via_llm

        long_text = "黄河吕梁段水位告警，请尽快处置。" * 20
        history_to_summarize = [
            {"role": "user", "content": long_text},
            {"role": "assistant", "content": long_text},
        ]
        # 缓存确保相同 history 指纹不再调 LLM client
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "这是缓存测试摘要"

        mock_client = MagicMock()
        mock_client.with_options.return_value = mock_client
        mock_client.chat.completions.create.return_value = mock_resp

        # patch extract_content 让它直接返回固定字符串（避免依赖 message.content 解析）
        with patch(
            "agent.graph.context_compact.get_llm_client", return_value=mock_client
        ), patch(
            "agent.graph.context_compact.get_llm_config",
            return_value={"model": "test-model", "temperature": 0.1, "max_tokens": 512},
        ), patch(
            "agent.graph.context_compact.extract_content", return_value="这是缓存测试摘要"
        ):
            # 第一次调用：未命中缓存，会调 LLM client
            summary1 = _summarize_via_llm(history_to_summarize)
            assert summary1 == "这是缓存测试摘要"
            assert mock_client.chat.completions.create.call_count == 1
            # 缓存已写入
            assert len(_SUMMARY_CACHE) == 1

            # 第二次相同 history：命中缓存，不再调 LLM client
            summary2 = _summarize_via_llm(history_to_summarize)
            assert summary2 == "这是缓存测试摘要"
            # 调用次数仍为 1（缓存命中）
            assert mock_client.chat.completions.create.call_count == 1


# ====== _truncate_history 测试 ======

class TestTruncateHistory:
    def test_empty_history(self):
        assert _truncate_history([]) == ""

    def test_short_message_kept_intact(self):
        history = [{"role": "user", "content": "短消息"}]
        result = _truncate_history(history, max_chars_per_msg=200)
        assert "短消息" in result

    def test_long_message_truncated(self):
        long_content = "a" * 500
        history = [{"role": "user", "content": long_content}]
        result = _truncate_history(history, max_chars_per_msg=100)
        # 截断后包含省略号
        assert "..." in result
        # 截断后的内容不超过 100 字符（不含 role 前缀和省略号）
        assert "a" * 100 in result


# ====== extract_history_context 测试 ======

class TestExtractHistoryContext:
    def test_empty_history(self):
        assert extract_history_context([]) == ""

    def test_compacted_history_with_summary(self):
        # 压缩过的 history：首条是 system 摘要
        history = [
            {"role": "system", "content": "[历史对话摘要]\n用户询问水情，已Ⅲ级预警。"},
            {"role": "user", "content": "现在情况如何？"},
            {"role": "assistant", "content": "已发布Ⅲ级预警。"},
        ]
        result = extract_history_context(history)
        assert "[历史对话摘要]" in result
        assert "用户：" in result
        assert "助手：" in result

    def test_uncompacted_history(self):
        history = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好，有什么可以帮您？"},
        ]
        result = extract_history_context(history)
        assert "用户：你好" in result
        assert "助手：你好，有什么可以帮您？" in result

    def test_unknown_role_defaults_to_assistant_label(self):
        history = [{"role": "tool", "content": "tool result"}]
        result = extract_history_context(history)
        # 非 system 角色按 user/assistant 标签，未知角色（如 tool）走 else 分支（助手标签）
        assert "助手：tool result" in result
