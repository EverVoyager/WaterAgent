"""synthesizer 结构化输出流派一（tool calling 强制 schema）测试。

降级链第一档为 forced tool_choice 的工具通道（对齐 Pydantic AI Tool Output /
LangChain function_calling 方法），400 时降级 json_object（进程内记忆）。
"""
import json
from unittest.mock import MagicMock

from openai import APIError

from agent.graph.synthesizer_node import (
    _SYNTH_META_SCHEMA,
    _append_structured_retry_feedback,
    _call_synth_with_fallback,
    _extract_structured_content,
    _reset_response_format_memory,
    _tool_def_from_schema,
)


def _tool_call_msg(args: dict, name="submit_meta", tc_id="call_1", content=None):
    """构造走工具通道的 assistant message mock（tool_calls 为真实 list）。"""
    msg = MagicMock()
    msg.content = content
    msg.reasoning_content = "RC_TEXT"
    tc = MagicMock()
    tc.id = tc_id
    tc.function.name = name
    tc.function.arguments = json.dumps(args, ensure_ascii=False)
    msg.tool_calls = [tc]
    return msg


def _make_400(msg="bad") -> APIError:
    err = APIError.__new__(APIError)
    err.message = msg
    err.status_code = 400
    err.args = (msg,)  # str(e) 取 args（__new__ 绕过 __init__ 时需手动设置）
    return err


class TestToolDef:
    def test_response_schema_converted_to_tool(self):
        tool = _tool_def_from_schema(_SYNTH_META_SCHEMA)
        fn = tool[0]["function"]
        assert tool[0]["type"] == "function"
        assert fn["name"] == "synth_meta"
        assert "warning_level" in fn["parameters"]["properties"]


class TestExtractStructuredContent:
    def test_tool_args_preferred(self):
        msg = _tool_call_msg({"warning_level": "III"}, content="正文里的废话")
        assert json.loads(_extract_structured_content(msg))["warning_level"] == "III"

    def test_content_fallback_when_no_tool_calls(self):
        msg = MagicMock()
        msg.tool_calls = None
        msg.content = '{"warning_level": "IV"}'
        assert json.loads(_extract_structured_content(msg))["warning_level"] == "IV"

    def test_magicmock_tool_calls_falls_to_content(self):
        """测试 mock 的 tool_calls 是 MagicMock（非 list）→ 走 content 分支。"""
        msg = MagicMock()
        msg.content = '{"a": 1}'
        assert json.loads(_extract_structured_content(msg)) == {"a": 1}


class TestToolCallingTier:
    def setup_method(self):
        _reset_response_format_memory()

    def teardown_method(self):
        _reset_response_format_memory()

    def test_first_tier_uses_tool_channel(self):
        """首档走工具通道（tools + tool_choice=required），而非 response_format。"""
        ok = MagicMock()
        ok.usage = None
        client = MagicMock()
        client.chat.completions.create.return_value = ok

        _call_synth_with_fallback(client, "m", [{"role": "user", "content": "x"}])

        kw = client.chat.completions.create.call_args.kwargs
        assert kw.get("tools") and kw["tools"][0]["function"]["name"] == "synth_result"
        assert kw.get("tool_choice") == "required"
        assert "response_format" not in kw

    def test_thinking_mode_adapts_to_auto_with_instruction(self):
        """思考模式 400（不支持 forced tool_choice）→ 自动降 auto 并附提交指令。"""
        import agent.graph.synthesizer_node as sn

        ok = MagicMock()
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            _make_400("Thinking mode does not support this tool_choice"),
            ok,  # auto 重试成功
        ]
        resp = sn._call_tool_channel(client, "m", [{"role": "user", "content": "x"}], {})
        assert resp is ok
        assert client.chat.completions.create.call_count == 2
        kw = client.chat.completions.create.call_args.kwargs
        assert kw["tool_choice"] == "auto"
        # 末尾附了提交指令（不污染原 messages）
        assert kw["messages"][-1]["role"] == "user"
        assert "提交本次结构化结果" in kw["messages"][-1]["content"]
        assert len(kw["messages"]) == 2
        assert sn._TOOL_CHOICE_MODE == "auto"

    def test_400_downgrades_and_remembered(self):
        """tool 通道 400（tools 不支持）→ json_object 成功；再次调用直接从 json_object 开始。"""
        ok = MagicMock()
        ok.usage = None
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            _make_400("tools are not supported on this endpoint"),  # 第一档 400
            ok,                                                       # json_object 成功
        ]
        _call_synth_with_fallback(client, "m", [{"role": "user", "content": "x"}])
        assert client.chat.completions.create.call_count == 2

        client.chat.completions.create.reset_mock()
        client.chat.completions.create.side_effect = None
        client.chat.completions.create.return_value = ok
        _call_synth_with_fallback(client, "m", [{"role": "user", "content": "x"}])
        assert client.chat.completions.create.call_count == 1  # 记忆生效
        kw = client.chat.completions.create.call_args.kwargs
        assert kw.get("response_format") == {"type": "json_object"}
        assert "tools" not in kw

    def test_tool_success_sets_memory_true(self):
        ok = MagicMock()
        ok.usage = None
        client = MagicMock()
        client.chat.completions.create.return_value = ok
        _call_synth_with_fallback(client, "m", [{"role": "user", "content": "x"}])
        client.chat.completions.create.reset_mock()
        _call_synth_with_fallback(client, "m", [{"role": "user", "content": "x"}])
        # 仍走 tool 通道（未被降级）
        assert "tools" in client.chat.completions.create.call_args.kwargs


class TestRetryFeedbackPairing:
    def test_tool_mode_pairs_assistant_and_tool_messages(self):
        """工具通道校验失败：assistant(tool_calls+rc) 与 tool(validation_error) 配对。"""
        msg = _tool_call_msg({"warning_level": "WRONG"})
        messages = []
        _append_structured_retry_feedback(messages, msg, "等级不一致，请修正")
        a, t = messages
        assert a["role"] == "assistant"
        assert a["reasoning_content"] == "RC_TEXT"
        assert a["tool_calls"][0]["id"] == "call_1"
        assert t["role"] == "tool" and t["tool_call_id"] == "call_1"
        assert "等级不一致" in t["content"]

    def test_text_mode_keeps_legacy_behavior(self):
        """正文响应：assistant 正文 + user 反馈（向后兼容）。"""
        msg = MagicMock()
        msg.tool_calls = None
        msg.content = "上次的输出"
        messages = []
        _append_structured_retry_feedback(messages, msg, "请修正")
        assert [m["role"] for m in messages] == ["assistant", "user"]
        assert messages[1]["content"] == "请修正"
