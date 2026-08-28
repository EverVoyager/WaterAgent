"""strip_think + _classify_llm_error 单元测试。

strip_think 是前端输出关键路径，bug 会导致 Qwen3 思考内容泄漏到用户界面。
_classify_llm_error 决定 HTTP 状态码和重试策略。
"""
import httpx
from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    RateLimitError,
)

from agent.graph.errors import LLMError, _classify_llm_error
from app.core.llm import extract_content, strip_think

# ====== strip_think 测试 ======

class TestStripThink:
    def test_empty_string(self):
        assert strip_think("") == ""

    def test_none_input(self):
        assert strip_think(None) is None

    def test_no_think_tag_unchanged(self):
        text = "这是普通文本，无思考标签。"
        assert strip_think(text) == text

    def test_plain_text_with_content(self):
        text = "研判结论：Ⅲ级响应"
        assert strip_think(text) == "研判结论：Ⅲ级响应"

    def test_single_think_block_removed(self):
        text = "<think>这是思考内容</think>这是最终答案"
        assert strip_think(text) == "这是最终答案"

    def test_think_block_multiline_removed(self):
        text = "<think>\n这是多行\n思考内容\n</think>\n最终答案"
        assert strip_think(text) == "最终答案"

    def test_think_block_with_special_chars(self):
        text = "<think>包含特殊字符：<>&\"'</think>答案"
        assert strip_think(text) == "答案"

    def test_multiple_think_blocks_all_removed(self):
        text = "<think>思考1</think>答案1<think>思考2</think>答案2"
        assert strip_think(text) == "答案1答案2"

    def test_think_block_at_start_removed(self):
        text = "<think>开头的思考</think>后续内容"
        assert strip_think(text) == "后续内容"

    def test_think_block_at_end_removed(self):
        text = "前面的内容<think>结尾的思考</think>"
        assert strip_think(text) == "前面的内容"

    def test_only_think_block(self):
        text = "<think>只有思考</think>"
        assert strip_think(text) == ""

    def test_unclosed_think_tag_preserved(self):
        # 未闭合的 <think> 标签：正则不匹配，原样返回（lstrip \n）
        text = "<think>未闭合的思考"
        assert strip_think(text) == "<think>未闭合的思考"

    def test_nested_like_think_content(self):
        # think 内容中包含 "think" 字样
        text = "<think>我在think about这个问题</think>结论"
        assert strip_think(text) == "结论"

    def test_leading_newlines_after_think_stripped(self):
        text = "<think>思考</think>\n\n\n答案"
        assert strip_think(text) == "答案"

    def test_unicode_in_think_block(self):
        text = "<think>中文思考内容：吴堡站水情</think>中文答案"
        assert strip_think(text) == "中文答案"

    def test_empty_think_block(self):
        text = "<think></think>答案"
        assert strip_think(text) == "答案"


# ====== extract_content 测试 ======

class TestExtractContent:
    """验证 extract_content 不回退 reasoning_content（推理过程不暴露给用户）。"""

    def _make_msg(self, content=None, reasoning_content=None):
        """构造模拟的 LLM message 对象。"""
        from unittest.mock import MagicMock
        msg = MagicMock()
        if content is not None:
            msg.content = content
        else:
            msg.content = None
        if reasoning_content is not None:
            msg.reasoning_content = reasoning_content
        else:
            msg.reasoning_content = None
        return msg

    def test_normal_content(self):
        msg = self._make_msg(content="Ⅲ级预警", reasoning_content="思考过程")
        assert extract_content(msg) == "Ⅲ级预警"

    def test_content_with_think_tag(self):
        msg = self._make_msg(content="<think>思考</think>答案", reasoning_content="原始思考")
        assert extract_content(msg) == "答案"

    def test_empty_content_with_reasoning_does_not_fallback(self):
        """content 为空时不应回退到 reasoning_content（核心修复点）。"""
        msg = self._make_msg(content="", reasoning_content="这是推理过程，不应返回给用户")
        result = extract_content(msg)
        # 必须返回空字符串，不返回推理过程
        assert result == ""
        assert "推理过程" not in result

    def test_none_content_with_reasoning_does_not_fallback(self):
        """content 为 None 时不应回退。"""
        msg = self._make_msg(content=None, reasoning_content="推理内容")
        assert extract_content(msg) == ""

    def test_empty_content_no_reasoning(self):
        """content 和 reasoning_content 都为空。"""
        msg = self._make_msg(content="", reasoning_content="")
        assert extract_content(msg) == ""

    def test_reasoning_content_never_in_output(self):
        """即使 content 为空且 reasoning_content 包含看似有用的信息，也不应出现在输出中。"""
        sensitive_reasoning = (
            "我们需要理解用户的问题。用户问'你有什么功能'，"
            "这是一个关于我自身能力的询问..."
        )
        msg = self._make_msg(content="", reasoning_content=sensitive_reasoning)
        result = extract_content(msg)
        assert result == ""
        # 确保推理过程的任何片段都不在输出中
        for word in ["理解", "用户的问题", "自身能力", "询问"]:
            assert word not in result


# ====== _classify_llm_error 测试 ======

class TestClassifyLlmError:
    def _make_timeout_error(self):
        # APITimeoutError 需要 request 参数
        request = httpx.Request("POST", "https://api.example.com/v1/chat")
        return APITimeoutError(request=request)

    def _make_connection_error(self):
        request = httpx.Request("POST", "https://api.example.com/v1/chat")
        return APIConnectionError(request=request)

    def _make_rate_limit_error(self):
        # RateLimitError 需要 response
        response = httpx.Response(
            status_code=429,
            request=httpx.Request("POST", "https://api.example.com/v1/chat"),
        )
        return RateLimitError(
            message="Rate limit exceeded",
            response=response,
            body=None,
        )

    def _make_api_error(self):
        # APIError 签名：(message, request, *, body)
        # APIError 实例本身不存储 status_code（_classify_llm_error 用 getattr 兜底为 500）
        request = httpx.Request("POST", "https://api.example.com/v1/chat")
        return APIError(message="API error", request=request, body=None)

    def test_timeout_classified_correctly(self):
        err = self._make_timeout_error()
        result = _classify_llm_error(err)
        assert isinstance(result, LLMError)
        assert result.kind == "timeout"
        assert result.status_code == 504

    def test_rate_limit_classified_correctly(self):
        err = self._make_rate_limit_error()
        result = _classify_llm_error(err)
        assert result.kind == "rate_limit"
        assert result.status_code == 429

    def test_connection_error_classified_correctly(self):
        err = self._make_connection_error()
        result = _classify_llm_error(err)
        assert result.kind == "connection"
        assert result.status_code == 502

    def test_api_error_classified_correctly(self):
        err = self._make_api_error()
        result = _classify_llm_error(err)
        assert result.kind == "api_error"
        # APIError 无 status_code 属性，_classify_llm_error 用 getattr 兜底为 500
        assert result.status_code == 500

    def test_httpx_timeout_classified_as_timeout(self):
        err = httpx.ReadTimeout("read timed out")
        result = _classify_llm_error(err)
        assert result.kind == "timeout"
        assert result.status_code == 504

    def test_generic_exception_classified_as_unknown(self):
        err = ValueError("some value error")
        result = _classify_llm_error(err)
        assert result.kind == "unknown"
        assert result.status_code == 500

    def test_runtime_error_classified_as_unknown(self):
        err = RuntimeError("runtime issue")
        result = _classify_llm_error(err)
        assert result.kind == "unknown"
        assert result.status_code == 500

    def test_llm_error_message_contains_kind_prefix(self):
        err = self._make_timeout_error()
        result = _classify_llm_error(err)
        assert str(result).startswith("[timeout]")

    def test_llm_error_is_runtime_error_subclass(self):
        err = ValueError("test")
        result = _classify_llm_error(err)
        # LLMError 继承 RuntimeError，RuntimeError 继承 Exception
        assert isinstance(result, RuntimeError)
        assert isinstance(result, Exception)
