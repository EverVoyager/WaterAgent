"""llm_stats 前缀缓存命中率观测测试。

验证三后端 usage 字段命名的兼容提取，以及观测模块的绝对防御性
（任何异常输入都归零、绝不影响 LLM 调用主路径）。
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

from app.core.llm_stats import (
    _extract_cached_tokens,
    get_cache_stats,
    record_llm_usage,
    reset_cache_stats,
)


class TestExtractCachedTokens:
    def test_openai_style_object(self):
        """OpenAI 风格：prompt_tokens_details.cached_tokens（MaaS / vLLM 新版）。"""
        usage = SimpleNamespace(
            prompt_tokens=1000,
            prompt_tokens_details=SimpleNamespace(cached_tokens=600),
        )
        assert _extract_cached_tokens(usage) == (1000, 600)

    def test_openai_style_dict(self):
        """dict 形式的同结构。"""
        usage = {
            "prompt_tokens": 1000,
            "prompt_tokens_details": {"cached_tokens": 600},
        }
        assert _extract_cached_tokens(usage) == (1000, 600)

    def test_deepseek_native(self):
        """DeepSeek 原生字段：prompt_cache_hit_tokens。"""
        usage = SimpleNamespace(prompt_tokens=1000, prompt_cache_hit_tokens=400)
        assert _extract_cached_tokens(usage) == (1000, 400)

    def test_vllm_legacy(self):
        """vLLM 旧版字段：顶层 cached_tokens。"""
        usage = SimpleNamespace(prompt_tokens=1000, cached_tokens=300)
        assert _extract_cached_tokens(usage) == (1000, 300)

    def test_none_usage(self):
        assert _extract_cached_tokens(None) == (0, 0)

    def test_missing_fields(self):
        """字段缺失时归零，不抛错。"""
        assert _extract_cached_tokens(SimpleNamespace()) == (0, 0)
        assert _extract_cached_tokens({}) == (0, 0)

    def test_details_without_cached(self):
        """prompt_tokens_details 存在但无 cached_tokens。"""
        usage = SimpleNamespace(
            prompt_tokens=100, prompt_tokens_details=SimpleNamespace()
        )
        assert _extract_cached_tokens(usage) == (100, 0)

    def test_magicmock_never_raises(self):
        """流式循环中 chunk 为 MagicMock 时（现有测试的构造方式），usage 属性是
        自动生成的 MagicMock——提取不得抛错，返回合法二元组即可（观测不影响主路径）。"""
        result = _extract_cached_tokens(MagicMock())
        assert isinstance(result, tuple) and len(result) == 2

    def test_int_conversion_error_never_raises(self):
        """int 转换抛错时兜底返回 (0, 0)。"""

        class Bad:
            def __int__(self):
                raise TypeError("bad int")

        usage = SimpleNamespace(prompt_tokens=Bad(), prompt_tokens_details=None)
        assert _extract_cached_tokens(usage) == (0, 0)


class TestRecordUsage:
    def setup_method(self):
        reset_cache_stats()

    def test_aggregates_by_node(self):
        """按节点聚合 calls / prompt_tokens / cached_tokens。"""
        record_llm_usage(
            "planner",
            SimpleNamespace(
                prompt_tokens=100,
                prompt_tokens_details=SimpleNamespace(cached_tokens=50),
            ),
        )
        record_llm_usage(
            "planner",
            {"prompt_tokens": 200, "prompt_tokens_details": {"cached_tokens": 100}},
        )
        stats = get_cache_stats()
        assert stats["planner"] == {
            "calls": 2,
            "prompt_tokens": 300,
            "cached_tokens": 150,
        }

    def test_garbage_usage_recorded_as_zero(self):
        """异常 usage 记录不抛错（MagicMock 的 __int__ 返回 1，故仅断言不炸）。"""
        record_llm_usage("chat", MagicMock())
        stats = get_cache_stats()
        assert stats["chat"]["calls"] == 1
        assert isinstance(stats["chat"]["prompt_tokens"], int)
        assert isinstance(stats["chat"]["cached_tokens"], int)

    def test_none_usage_counts_call(self):
        record_llm_usage("synthesizer", None)
        assert get_cache_stats()["synthesizer"]["calls"] == 1

    def test_nodes_isolated(self):
        record_llm_usage("planner", {"prompt_tokens": 10})
        record_llm_usage("chat", {"prompt_tokens": 20})
        stats = get_cache_stats()
        assert stats["planner"]["prompt_tokens"] == 10
        assert stats["chat"]["prompt_tokens"] == 20
