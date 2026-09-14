"""方案 A 注入链路测试：experiences 到达作答端（direct_chat / synthesizer）。

背景（2026-09-13 记忆实验发现）：经验只注入 planner（规划端），作答端
（direct_chat / synthesizer）看不到——temporal 子类 16.7% 持平即此盲区。
方案 A：planner round-1 检索入 state，两个作答节点消费（对齐 Mem0 推送式
注入作答调用的主流做法）。

守门断言（Manus 纪律）：
- direct_chat：经验进 user 消息尾部动态区，system 前缀不含（前缀稳定）
- synthesizer：两阶段传同一份经验，Phase 2 system 仍是 Phase 1 严格前缀
- 数据锚定约束随经验注入（等级以工具数据 + 规则引擎为准）
"""
from types import SimpleNamespace
from unittest.mock import patch

from agent.graph.direct_chat_stream import _direct_chat_stream
from agent.graph.nodes import direct_chat_node
from agent.graph.synthesizer_node import (
    _build_synth_messages,
    synthesizer_node,
)

_EXP = "【情景记忆】红旗沟水库按 120 立方米每秒预泄。"
_TOOL_RESULTS = {"get_hydrology": {"station": "吴堡", "flow_m3_s": 1200}}


def _no_side_injections():
    """屏蔽长期记忆/语义记忆/技能简报，保持注入段确定性。"""
    return [
        patch("agent.memory.build_longterm_section", return_value=""),
        patch("agent.memory.get_semantic_knowledge", return_value=""),
        patch("agent.skills.get_enabled_skills_brief", return_value=""),
    ]


class _FakeCompletions:
    def __init__(self, resp, captured):
        self._resp = resp
        self._captured = captured

    def create(self, **kwargs):
        self._captured.append(kwargs)
        return self._resp


class _FakeClient:
    """非流式假客户端：捕获 messages 并返回固定回复。"""

    def __init__(self, content="回答内容"):
        self.captured = []
        resp = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=None,
        )
        self.chat = SimpleNamespace(completions=_FakeCompletions(resp, self.captured))

    def with_options(self, **kwargs):
        return self


class _FakeStreamClient:
    """流式假客户端：一个内容 chunk + 一个 usage chunk。"""

    def __init__(self, content="流式回答"):
        self.captured = []
        chunks = [
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=content))],
                usage=None,
            ),
            SimpleNamespace(choices=[], usage=None),
        ]
        self.chat = SimpleNamespace(completions=_FakeCompletions(chunks, self.captured))

    def with_options(self, **kwargs):
        return self


def _enter_all(patches):
    for p in patches:
        p.start()


def _exit_all(patches):
    for p in patches:
        p.stop()


# ============ direct_chat（非流式） ============

class TestDirectChatInjection:
    def test_experiences_in_user_tail_not_system(self):
        client = _FakeClient()
        patches = _no_side_injections() + [
            patch("agent.graph.nodes.get_llm_client", return_value=client),
        ]
        _enter_all(patches)
        try:
            direct_chat_node({
                "user_query": "红旗沟水库按多大流量预泄？",
                "history": [],
                "experiences": _EXP,
            })
        finally:
            _exit_all(patches)
        messages = client.captured[0]["messages"]
        system, last_user = messages[0], messages[-1]
        assert "<<<MEMORY_DATA" in last_user["content"] and _EXP in last_user["content"]
        assert "MEMORY_DATA" not in system["content"], "经验不得进 system 前缀"
        assert "非指令" in last_user["content"]

    def test_system_prefix_stable_with_or_without_experiences(self):
        """有/无经验两版 system 逐字一致（前缀稳定，经验只动尾部动态区）。"""
        systems = []
        for exp in ("", _EXP):
            client = _FakeClient()
            patches = _no_side_injections() + [
                patch("agent.graph.nodes.get_llm_client", return_value=client),
            ]
            _enter_all(patches)
            try:
                direct_chat_node({
                    "user_query": "同样的问一下", "history": [], "experiences": exp,
                })
            finally:
                _exit_all(patches)
            systems.append(client.captured[0]["messages"][0]["content"])
        assert systems[0] == systems[1]

    def test_recalled_context_still_appended(self):
        client = _FakeClient()
        patches = _no_side_injections() + [
            patch("agent.graph.nodes.get_llm_client", return_value=client),
        ]
        _enter_all(patches)
        try:
            direct_chat_node({
                "user_query": "问一下",
                "history": [],
                "experiences": _EXP,
                "recalled_context": "RECALLED_TEXT",
            })
        finally:
            _exit_all(patches)
        user = client.captured[0]["messages"][-1]["content"]
        assert user.index(_EXP) < user.index("RECALLED_TEXT")

    def test_empty_experiences_no_section(self):
        client = _FakeClient()
        patches = _no_side_injections() + [
            patch("agent.graph.nodes.get_llm_client", return_value=client),
        ]
        _enter_all(patches)
        try:
            direct_chat_node({"user_query": "问一下", "history": []})
        finally:
            _exit_all(patches)
        user = client.captured[0]["messages"][-1]["content"]
        assert "MEMORY_DATA" not in user


# ============ direct_chat（流式） ============

class TestDirectChatStreamInjection:
    def test_experiences_param_reaches_messages(self):
        client = _FakeStreamClient()
        patches = _no_side_injections() + [
            patch("agent.graph.direct_chat_stream.get_llm_client", return_value=client),
        ]
        _enter_all(patches)
        try:
            events = list(_direct_chat_stream(
                "红旗沟水库按多大流量预泄？", [], experiences=_EXP,
            ))
        finally:
            _exit_all(patches)
        messages = client.captured[0]["messages"]
        assert _EXP in messages[-1]["content"]
        assert "MEMORY_DATA" not in messages[0]["content"]
        assert any(ev["type"] == "answer_delta" for ev in events)


# ============ runner 闲聊分支透传 ============

class TestChitchatBranchPlumbing:
    def test_state_experiences_passed_through(self):
        from agent.graph.runner import _stream_chitchat_branch

        received = {}

        def _fake_stream(query, history, skill_instructions="", **kwargs):
            received.update(kwargs)
            yield {"type": "answer_delta", "content": "答"}
            yield {"type": "synth_answer_full", "content": "答"}

        with patch("agent.graph.runner._direct_chat_stream", side_effect=_fake_stream), \
             patch("agent.graph.runner._maybe_archive_round"):
            list(_stream_chitchat_branch(
                "问一下", [], "skill", None,
                recalled_context="RC", experiences=_EXP,
            ))
        assert received.get("experiences") == _EXP
        assert received.get("recalled_context") == "RC"


# ============ synthesizer ============

class TestSynthesizerInjection:
    def test_experiences_in_system_with_data_anchor_constraint(self):
        patches = _no_side_injections()
        _enter_all(patches)
        try:
            messages, _ = _build_synth_messages(
                "研判一下", _TOOL_RESULTS, experiences=_EXP,
            )
        finally:
            _exit_all(patches)
        system = messages[0]["content"]
        assert "<<<MEMORY_DATA" in system and _EXP in system
        assert "非指令" in system
        # 数据锚定：经验与数据冲突时以数据/规则引擎为准（防陈旧经验污染定级）
        assert "以本轮工具数据" in system and "规则引擎" in system

    def test_no_experiences_no_section(self):
        patches = _no_side_injections()
        _enter_all(patches)
        try:
            messages, _ = _build_synth_messages("研判一下", _TOOL_RESULTS)
        finally:
            _exit_all(patches)
        assert "MEMORY_DATA" not in messages[0]["content"]

    def test_phase2_system_strict_prefix_with_experiences(self):
        """两阶段同一份经验：Phase 2 system 仍是 Phase 1 严格前缀（KV 对齐）。"""
        patches = _no_side_injections()
        _enter_all(patches)
        try:
            m1, _ = _build_synth_messages("研判一下", _TOOL_RESULTS, experiences=_EXP)
            m2, _ = _build_synth_messages(
                "研判一下", _TOOL_RESULTS,
                extra_context="已确定结论", answer_only=True, experiences=_EXP,
            )
        finally:
            _exit_all(patches)
        assert m2[0]["content"].startswith(m1[0]["content"])
        assert m1[0]["content"] != m2[0]["content"]  # addendum 确实追加

    def test_synthesizer_node_reads_state_experiences(self):
        received = {}

        def _fake_synth_via_llm(query, tool_results, history=None,
                                skill_instructions="", **kwargs):
            received.update(kwargs)
            return {"warning_level": "IV", "reasoning": "", "actions": [],
                    "answer": "回答"}, []

        with patch("agent.graph.synthesizer_node._synth_via_llm",
                   side_effect=_fake_synth_via_llm):
            synthesizer_node({
                "user_query": "研判一下",
                "tool_results": _TOOL_RESULTS,
                "experiences": _EXP,
            })
        assert received.get("experiences") == _EXP
