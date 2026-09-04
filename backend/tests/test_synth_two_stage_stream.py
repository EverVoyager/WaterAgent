"""两阶段真流式 synthesizer 测试（P1-a）。

验证 _synth_via_llm_stream 的两阶段架构：
- Phase 1：非流式获取 metadata（_SYNTH_META_SCHEMA，无 answer）
- Phase 2：stream=True 逐 token 生成 answer（真流式，非伪切分）

关键验证点：
1. synth_meta 事件在 answer_delta 之前 yield
2. answer_delta 是真 token 流式（来自 LLM stream，非字符串切分）
3. <think> 块在流式中被剥离
4. synth_answer_full 包含过滤 think 后的完整 answer
"""
import json
from unittest.mock import MagicMock, patch

from openai import APIError

from agent.graph.synthesizer_node import (
    _SYNTH_META_SCHEMA,
    _build_synth_messages,
    _build_synth_system_content,
    _call_synth_with_fallback,
    _reset_response_format_memory,
    _stream_answer_via_llm,
    _synth_metadata_via_llm,
    _synth_via_llm_stream,
)

# ====== Mock 工厂 ======

def _make_mock_chat_response(content: str):
    """构造非流式 chat.completions.create 的 mock response。"""
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


def _make_mock_stream_chunks(tokens: list):
    """构造流式 chat.completions.create(stream=True) 的 mock chunk 列表。"""
    chunks = []
    for tok in tokens:
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = tok
        chunks.append(chunk)
    # 结束 chunk（无 content）
    end_chunk = MagicMock()
    end_chunk.choices = []
    chunks.append(end_chunk)
    return chunks


# ====== _build_synth_system_content 测试 ======

class TestBuildSynthSystemContent:
    def test_base_prompt_present(self):
        """基础 SYNTHESIZER_PROMPT 必须存在。"""
        content = _build_synth_system_content()
        assert len(content) > 0

    def test_citation_guidance_appended(self):
        """citation guidance 必须追加到末尾。"""
        content = _build_synth_system_content()
        assert "引用规范" in content or "citations" in content.lower()

    def test_skill_instructions_appended(self):
        """skill_instructions 参数应追加到 system content。"""
        content = _build_synth_system_content(skill_instructions="TEST_SKILL_INSTRUCTIONS")
        assert "TEST_SKILL_INSTRUCTIONS" in content
        assert "Skill 行为指令" in content

    def test_answer_only_prefix_alignment(self):
        """Phase 2（answer_only=True）system = Phase 1 system + 追加指令块。

        KV Cache 前缀对齐：Phase 1 内容必须是 Phase 2 的严格前缀，
        追加块显式声明阶段切换（覆盖前文"仅返回 JSON"的阶段一要求）。
        """
        phase1 = _build_synth_system_content()
        phase2 = _build_synth_system_content(answer_only=True)
        # Phase 1 system 是 Phase 2 的严格前缀（非空、真追加）
        assert phase2.startswith(phase1)
        assert len(phase2) > len(phase1)
        # 追加块包含纯文本回答要求与阶段切换声明
        assert "不要输出 JSON" in phase2
        assert "第二阶段" in phase2
        assert "仅适用于第一阶段" in phase2


# ====== _build_synth_messages 测试 ======

class TestBuildSynthMessages:
    def test_returns_messages_and_registry(self):
        """返回 (messages, source_registry) 元组。"""
        messages, registry = _build_synth_messages(
            "查询吴堡水情",
            {"get_hydrology_0": {"station": "吴堡", "flow_m3_s": 3000}},
        )
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "吴堡" in messages[1]["content"]

    def test_extra_context_appended_to_user(self):
        """extra_context 应追加到 user 消息末尾。"""
        messages, _ = _build_synth_messages(
            "查询",
            {},
            extra_context="EXTRA_CONTEXT_MARKER",
        )
        assert "EXTRA_CONTEXT_MARKER" in messages[1]["content"]

    def test_web_search_enters_registry(self):
        """web_search 结果进入 source_registry（可引用）。"""
        tool_results = {
            "web_search_0": {
                "results": [
                    {"title": "新闻1", "snippet": "黄河汛情", "url": "http://example.com"}
                ],
            },
        }
        _, registry = _build_synth_messages("查询", tool_results)
        assert len(registry) >= 1
        entry = list(registry.values())[0]
        assert entry["source_type"] == "web_search"
        assert entry["url"] == "http://example.com"


# ====== _synth_metadata_via_llm (Phase 1) 测试 ======

class TestSynthMetadataViaLlm:
    """Phase 1：非流式获取 metadata（不含 answer）。"""

    def test_returns_metadata_without_answer(self):
        """Phase 1 返回的 metadata 不应包含 answer 字段（由 Phase 2 生成）。"""
        mock_resp = _make_mock_chat_response(
            json.dumps({
                "warning_level": "III",
                "reasoning": "流量接近警戒",
                "actions": ["加强巡查"],
                "citations": [],
            })
        )
        with patch("agent.graph.synthesizer_node._call_synth_with_fallback", return_value=mock_resp), \
             patch("agent.graph.synthesizer_node.get_llm_config", return_value={"model": "test"}), \
             patch("agent.graph.synthesizer_node.get_llm_client") as mock_client:
            mock_client.return_value.with_options.return_value = MagicMock()
            metadata, citations = _synth_metadata_via_llm("查询", {})
            assert metadata["warning_level"] == "III"
            assert metadata["reasoning"] == "流量接近警戒"
            assert metadata["actions"] == ["加强巡查"]
            assert "answer" not in metadata or metadata.get("answer") is None

    def test_uses_meta_schema(self):
        """Phase 1 应使用 _SYNTH_META_SCHEMA（不含 answer）。"""
        mock_resp = _make_mock_chat_response(
            json.dumps({"warning_level": "", "reasoning": "", "actions": [], "citations": []})
        )
        with patch("agent.graph.synthesizer_node._call_synth_with_fallback", return_value=mock_resp) as mock_call, \
             patch("agent.graph.synthesizer_node.get_llm_config", return_value={"model": "test"}), \
             patch("agent.graph.synthesizer_node.get_llm_client") as mock_client:
            mock_client.return_value.with_options.return_value = MagicMock()
            _synth_metadata_via_llm("查询", {})
            # 验证传入了 schema=_SYNTH_META_SCHEMA
            call_kwargs = mock_call.call_args
            assert call_kwargs.kwargs.get("schema") is _SYNTH_META_SCHEMA


# ====== _stream_answer_via_llm (Phase 2) 测试 ======

class TestStreamAnswerViaLlm:
    """Phase 2：stream=True 真 token 流式。"""

    def test_yields_answer_delta_events(self):
        """Phase 2 yield answer_delta 事件，content 来自 LLM stream。"""
        tokens = ["黄河", "吴堡站", "当前", "流量", "3000m³/s"]
        mock_stream = _make_mock_stream_chunks(tokens)
        with patch("agent.graph.synthesizer_node.get_llm_config", return_value={"model": "test"}), \
             patch("agent.graph.synthesizer_node.get_llm_client") as mock_client:
            mock_client.return_value.with_options.return_value.chat.completions.create.return_value = iter(mock_stream)
            events = list(_stream_answer_via_llm(
                "查询", {}, {"warning_level": "III", "reasoning": "", "actions": []}
            ))
            deltas = [e for e in events if e["type"] == "answer_delta"]
            assert len(deltas) == len(tokens)
            combined = "".join(e["content"] for e in deltas)
            assert combined == "黄河吴堡站当前流量3000m³/s"

    def test_yields_synth_answer_full_at_end(self):
        """Phase 2 最后 yield synth_answer_full 事件。"""
        tokens = ["回答", "内容"]
        mock_stream = _make_mock_stream_chunks(tokens)
        with patch("agent.graph.synthesizer_node.get_llm_config", return_value={"model": "test"}), \
             patch("agent.graph.synthesizer_node.get_llm_client") as mock_client:
            mock_client.return_value.with_options.return_value.chat.completions.create.return_value = iter(mock_stream)
            events = list(_stream_answer_via_llm(
                "查询", {}, {"warning_level": "", "reasoning": "", "actions": []}
            ))
            assert events[-1]["type"] == "synth_answer_full"
            assert events[-1]["content"] == "回答内容"

    def test_think_block_stripped_from_stream(self):
        """<think> 块在流式中应被剥离，不推送给前端。"""
        # 模拟 LLM 先输出 <think>推理</think> 再输出正文
        tokens = ["<think>", "内部推理", "</think>", "正文内容"]
        mock_stream = _make_mock_stream_chunks(tokens)
        with patch("agent.graph.synthesizer_node.get_llm_config", return_value={"model": "test"}), \
             patch("agent.graph.synthesizer_node.get_llm_client") as mock_client:
            mock_client.return_value.with_options.return_value.chat.completions.create.return_value = iter(mock_stream)
            events = list(_stream_answer_via_llm(
                "查询", {}, {"warning_level": "", "reasoning": "", "actions": []}
            ))
            deltas = [e for e in events if e["type"] == "answer_delta"]
            combined = "".join(e["content"] for e in deltas)
            # think 内容不应出现
            assert "内部推理" not in combined
            assert "<think>" not in combined
            assert combined == "正文内容"

    def test_uses_stream_true(self):
        """Phase 2 LLM 调用必须 stream=True。"""
        tokens = ["test"]
        mock_stream = _make_mock_stream_chunks(tokens)
        with patch("agent.graph.synthesizer_node.get_llm_config", return_value={"model": "test"}), \
             patch("agent.graph.synthesizer_node.get_llm_client") as mock_client:
            mock_create = mock_client.return_value.with_options.return_value.chat.completions.create
            mock_create.return_value = iter(mock_stream)
            list(_stream_answer_via_llm(
                "查询", {}, {"warning_level": "", "reasoning": "", "actions": []}
            ))
            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs.get("stream") is True

    def test_metadata_injected_into_prompt(self):
        """Phase 1 的 metadata 应注入 Phase 2 的 user 消息。"""
        tokens = ["answer"]
        mock_stream = _make_mock_stream_chunks(tokens)
        with patch("agent.graph.synthesizer_node.get_llm_config", return_value={"model": "test"}), \
             patch("agent.graph.synthesizer_node.get_llm_client") as mock_client:
            mock_create = mock_client.return_value.with_options.return_value.chat.completions.create
            mock_create.return_value = iter(mock_stream)
            list(_stream_answer_via_llm(
                "查询",
                {},
                {"warning_level": "II", "reasoning": "TEST_REASONING", "actions": ["TEST_ACTION"]},
            ))
            call_kwargs = mock_create.call_args.kwargs
            user_msg = call_kwargs["messages"][1]["content"]
            assert "II" in user_msg
            assert "TEST_REASONING" in user_msg
            assert "TEST_ACTION" in user_msg

    def test_phase2_system_prompt_is_plain_text(self):
        """Phase 2 的 system prompt 应为 Phase 1 前缀 + 第二阶段纯文本指令。"""
        tokens = ["answer"]
        mock_stream = _make_mock_stream_chunks(tokens)
        with patch("agent.graph.synthesizer_node.get_llm_config", return_value={"model": "test"}), \
             patch("agent.graph.synthesizer_node.get_llm_client") as mock_client:
            mock_create = mock_client.return_value.with_options.return_value.chat.completions.create
            mock_create.return_value = iter(mock_stream)
            list(_stream_answer_via_llm(
                "查询", {}, {"warning_level": "", "reasoning": "", "actions": []}
            ))
            call_kwargs = mock_create.call_args.kwargs
            system_msg = call_kwargs["messages"][0]["content"]
            # 追加块要求纯文本回答，并显式声明阶段切换
            assert "不要输出 JSON" in system_msg
            assert "第二阶段" in system_msg


# ====== _synth_via_llm_stream 集成测试 ======


# ====== response_format 降级记忆测试 ======

def _make_400_unavailable() -> APIError:
    """构造 DeepSeek 风格的 400：response_format 不可用。"""
    err = APIError.__new__(APIError)
    err.message = "This response_format type is unavailable now"
    err.status_code = 400
    return err


class TestResponseFormatDowngradeMemory:
    """json_schema 400 探测一次后记住，本进程后续直接从 json_object 开始。"""

    def setup_method(self):
        _reset_response_format_memory()

    def teardown_method(self):
        _reset_response_format_memory()

    @staticmethod
    def _client_400_on_json_schema_then_ok():
        """json_schema 抛 400，其余格式成功的 mock client。"""
        client = MagicMock()
        ok = MagicMock()
        ok.usage = None
        client.chat.completions.create.side_effect = (
            lambda **kw: (_ for _ in ()).throw(_make_400_unavailable())
            if (kw.get("response_format") or {}).get("type") == "json_schema"
            else ok
        )
        return client

    def test_400_then_downgrade_then_remember(self):
        """首次：json_schema 400 → json_object 成功；再次：直接 json_object（不再 400）。"""
        client = self._client_400_on_json_schema_then_ok()
        create = client.chat.completions.create

        _call_synth_with_fallback(client, "m", [{"role": "user", "content": "x"}])
        assert create.call_count == 2  # json_schema(400) + json_object(ok)

        create.reset_mock()
        _call_synth_with_fallback(client, "m", [{"role": "user", "content": "x"}])
        assert create.call_count == 1  # 记忆生效，跳过 json_schema
        fmt = create.call_args.kwargs.get("response_format")
        assert fmt is not None and fmt.get("type") == "json_object"

    def test_json_schema_success_keeps_strongest(self):
        """端点支持 json_schema 时不降级，后续继续用 json_schema。"""
        client = MagicMock()
        ok = MagicMock()
        ok.usage = None
        client.chat.completions.create.return_value = ok

        _call_synth_with_fallback(client, "m", [{"role": "user", "content": "x"}])
        _call_synth_with_fallback(client, "m", [{"role": "user", "content": "x"}])

        assert client.chat.completions.create.call_count == 2
        for call in client.chat.completions.create.call_args_list:
            fmt = call.kwargs.get("response_format")
            assert fmt is not None and fmt.get("type") == "json_schema"


class TestSynthViaLlmStreamIntegration:
    """两阶段集成：synth_meta → answer_delta → synth_answer_full。"""

    def test_event_order_meta_before_delta(self):
        """synth_meta 必须在 answer_delta 之前 yield。"""
        mock_meta_resp = _make_mock_chat_response(
            json.dumps({"warning_level": "III", "reasoning": "test", "actions": [], "citations": []})
        )
        tokens = ["回答"]
        mock_stream = _make_mock_stream_chunks(tokens)
        with patch("agent.graph.synthesizer_node._call_synth_with_fallback", return_value=mock_meta_resp), \
             patch("agent.graph.synthesizer_node.get_llm_config", return_value={"model": "test"}), \
             patch("agent.graph.synthesizer_node.get_llm_client") as mock_client:
            mock_client.return_value.with_options.return_value.chat.completions.create.return_value = iter(mock_stream)
            events = list(_synth_via_llm_stream("查询", {}))
            types = [e["type"] for e in events]
            meta_idx = types.index("synth_meta")
            delta_idx = types.index("answer_delta")
            assert meta_idx < delta_idx

    def test_synth_answer_full_is_last(self):
        """synth_answer_full 必须是最后一个事件。"""
        mock_meta_resp = _make_mock_chat_response(
            json.dumps({"warning_level": "", "reasoning": "", "actions": [], "citations": []})
        )
        tokens = ["A", "B", "C"]
        mock_stream = _make_mock_stream_chunks(tokens)
        with patch("agent.graph.synthesizer_node._call_synth_with_fallback", return_value=mock_meta_resp), \
             patch("agent.graph.synthesizer_node.get_llm_config", return_value={"model": "test"}), \
             patch("agent.graph.synthesizer_node.get_llm_client") as mock_client:
            mock_client.return_value.with_options.return_value.chat.completions.create.return_value = iter(mock_stream)
            events = list(_synth_via_llm_stream("查询", {}))
            assert events[-1]["type"] == "synth_answer_full"
            assert events[-1]["content"] == "ABC"

    def test_true_streaming_not_string_split(self):
        """验证是真 token 流式：answer_delta 的 content 来自 LLM stream 的 token，
        而非对完整 answer 字符串做切分。

        判据：如果 LLM 返回 3 个 token ['AAA', 'BBB', 'CCC']，
        真流式应 yield 3 个 answer_delta；伪切分（按 3 字切）会 yield 1 个。
        """
        mock_meta_resp = _make_mock_chat_response(
            json.dumps({"warning_level": "", "reasoning": "", "actions": [], "citations": []})
        )
        tokens = ["AAA", "BBB", "CCC"]
        mock_stream = _make_mock_stream_chunks(tokens)
        with patch("agent.graph.synthesizer_node._call_synth_with_fallback", return_value=mock_meta_resp), \
             patch("agent.graph.synthesizer_node.get_llm_config", return_value={"model": "test"}), \
             patch("agent.graph.synthesizer_node.get_llm_client") as mock_client:
            mock_client.return_value.with_options.return_value.chat.completions.create.return_value = iter(mock_stream)
            events = list(_synth_via_llm_stream("查询", {}))
            deltas = [e for e in events if e["type"] == "answer_delta"]
            # 真 token 流式：3 个 token → 3 个 delta
            assert len(deltas) == 3
            assert deltas[0]["content"] == "AAA"
            assert deltas[1]["content"] == "BBB"
            assert deltas[2]["content"] == "CCC"
