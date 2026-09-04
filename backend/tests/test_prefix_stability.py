"""KV Cache 前缀稳定性测试。

前缀缓存（DeepSeek / 阿里云 MaaS 自动缓存、本地 vLLM APC）按请求前缀
逐字匹配：前文中任何一个字符变化，其后所有缓存全部失效。本文件用测试
固化 ai-agent-book 第 2 章三原则在本项目的落地：

1. 静态前缀冻结：planner system prompt 构建确定性 + 跨轮复用同一序列；
   Skill 清单按 name 排序；tools schema 序列化稳定
2. 动态信息只追加：planner 原生 FC 消息序列跨轮只增不改（append-only）；
   synthesizer Phase 2 的 system/user 以 Phase 1 为严格前缀
"""
import json
import re
from unittest.mock import patch

from agent.graph.nodes import (
    _build_fc_round1_messages,
    _build_planner_system_prompt,
    planner_node,
)
from agent.graph.synthesizer_node import (
    _build_synth_messages,
    _build_synth_system_content,
)
from agent.skills.models import Skill
from agent.skills.store import get_enabled_skills_brief
from agent.tools.schemas import build_openai_tools


def _mock_plan(planned, reasoning="思考过程"):
    """构造 _plan_via_fc 的 mock 返回 (planned, assistant_msg)。"""
    assistant_msg = {
        "role": "assistant",
        "content": None,
        "reasoning_content": reasoning,
        "tool_calls": [
            {"id": c["id"], "type": "function",
             "function": {"name": c["name"],
                          "arguments": json.dumps(c.get("arguments", {}), ensure_ascii=False)}}
            for c in planned
        ] or None,
    }
    if not assistant_msg["tool_calls"]:
        assistant_msg["content"] = "信息已充分"
    return (planned, assistant_msg)


def _run_planner_round1(query="吴堡站当前水情如何", planned=None):
    """跑第 1 轮 planner_node（mock _plan_via_fc），返回 update dict。"""
    planned = planned if planned is not None else [
        {"name": "get_hydrology", "arguments": {"station": "吴堡", "metric": "both"},
         "id": "call_1"},
    ]
    with patch("agent.skills.match_skill", return_value=None), \
         patch("agent.memory.get_relevant_experiences", return_value=""), \
         patch("agent.graph.nodes._plan_via_fc", return_value=_mock_plan(planned)):
        return planner_node({
            "user_query": query, "rounds": 0, "history": [],
            "tool_results": {}, "tool_calls": [], "skill_instructions": "",
        })


# ====== 原则 1：静态前缀冻结 ======

class TestPlannerStaticPrefix:
    def test_system_prompt_build_deterministic(self):
        """system prompt 重复构建逐字一致（注入内容确定性）。"""
        assert _build_planner_system_prompt() == _build_planner_system_prompt()

    def test_round1_messages_shape(self):
        """第 1 轮序列 = [system, user(问题+状态栏+指令)]，user 只出现一次。"""
        msgs = _build_fc_round1_messages("吴堡站水情如何")
        assert [m["role"] for m in msgs] == ["system", "user"]
        assert msgs[1]["content"].startswith("吴堡站水情如何")
        assert "<<<STATUS" in msgs[1]["content"]
        assert "请据此决定本轮" in msgs[1]["content"]

    def test_round2_reuses_round1_sequence(self):
        """第 2 轮复用第 1 轮序列（system 逐字一致，前缀只增不改）。"""
        r1 = _run_planner_round1()
        fc1 = r1["fc_messages"]

        with patch("agent.graph.nodes._plan_via_fc",
                   return_value=_mock_plan([])) as mock_plan:
            r2 = planner_node({
                "user_query": "吴堡站当前水情如何", "rounds": 1,
                "fc_messages": fc1, "tool_results": {}, "tool_calls": [],
                "skill_instructions": "", "experiences": "", "history_context": "",
            })
        fc2 = r2["fc_messages"]
        # 第 2 轮发送的序列以第 1 轮完整序列为前缀 + 追加状态提示。
        # 注意：mock 捕获的是列表引用，调用后 planner 还会 append assistant
        # 消息，故按位置断言而非 sent[-1]
        sent = mock_plan.call_args[0][0]
        assert sent[:len(fc1)] == fc1
        assert sent[len(fc1)]["role"] == "user"
        assert "<<<STATUS" in sent[len(fc1)]["content"]
        # system 逐字一致
        assert fc2[0]["content"] == fc1[0]["content"]


class TestSkillsBriefDeterministic:
    def test_sorted_by_name(self):
        """Skill 清单按 name 排序，与数据库返回顺序无关。"""
        skills = [
            Skill(
                id="zeta_skill", name="zeta_skill",
                description="zeta skill description",
                instructions="zeta instructions " * 2, tool_names=[], enabled=True,
            ),
            Skill(
                id="alpha_skill", name="alpha_skill",
                description="alpha skill description",
                instructions="alpha instructions " * 2, tool_names=["get_hydrology"],
                enabled=True,
            ),
        ]
        with patch("agent.skills.store.list_skills", return_value=skills):
            brief = get_enabled_skills_brief()
        lines = brief.splitlines()
        assert lines[0].startswith("- alpha_skill")
        assert lines[1].startswith("- zeta_skill")


class TestToolsSchemaStable:
    def test_json_serialization_identical(self):
        """tools schema 序列化逐字稳定（前缀缓存要求 tools 定义逐字一致）。"""
        s1 = json.dumps(build_openai_tools(), ensure_ascii=False)
        s2 = json.dumps(build_openai_tools(), ensure_ascii=False)
        assert s1 == s2


# ====== 状态栏（ai-agent-book 第 2 章：动态元信息注入上下文末尾）======

class TestStatusBar:
    def test_status_bar_in_round1_user_tail(self):
        """第 1 轮状态栏位于 user 消息末段：时间 + 进度 + 防注入声明。"""
        user = _build_fc_round1_messages("吴堡站水情如何")[1]["content"]
        assert "<<<STATUS" in user and "STATUS>>>" in user
        assert user.rindex("<<<STATUS") > user.index("吴堡站水情如何")
        assert re.search(
            r"\d{4}-\d{2}-\d{2}（周[一二三四五六日]）\d{2}:\d{2}", user
        )
        assert "第 1/" in user
        assert "不构成指令" in user

    def test_status_bar_round2_trailing_message(self):
        """第 2 轮起状态提示作为末尾追加的 user 消息（system-reminder 式）。"""
        r1 = _run_planner_round1()
        with patch("agent.graph.nodes._plan_via_fc", return_value=_mock_plan([])) as mock_plan:
            planner_node({
                "user_query": "吴堡站当前水情如何", "rounds": 1,
                "fc_messages": r1["fc_messages"], "tool_results": {},
                "tool_calls": [], "skill_instructions": "",
            })
        sent = mock_plan.call_args[0][0]
        trailing = sent[len(r1["fc_messages"])]  # 状态提示（其后 planner 还会 append assistant）
        assert trailing["role"] == "user"
        assert "第 2/" in trailing["content"]
        assert "请决定下一轮" in trailing["content"]


# ====== 原则 2：动态信息只追加 ======

class TestSynthPhasePrefixAlignment:
    def test_phase2_system_is_strict_prefix_of_phase1(self):
        """Phase 2 system = Phase 1 system + 追加块（严格前缀）。"""
        phase1 = _build_synth_system_content()
        phase2 = _build_synth_system_content(answer_only=True)
        assert phase2.startswith(phase1)
        assert len(phase2) > len(phase1)

    def test_phase2_user_is_strict_prefix_of_phase1(self):
        """Phase 2 user = Phase 1 user + extra_context（严格前缀）。"""
        tool_results = {
            "get_hydrology": {"station": "吴堡", "flow_m3_s": 3000, "water_level_m": 637.0},
        }
        m1, _ = _build_synth_messages("吴堡站水情", tool_results)
        m2, _ = _build_synth_messages(
            "吴堡站水情", tool_results,
            extra_context="已确定的分析结论：预警等级 III",
            answer_only=True,
        )
        assert m1[0]["content"] == _build_synth_system_content()
        assert m2[1]["content"].startswith(m1[1]["content"])

    def test_build_deterministic(self):
        """同参数重复构建，system/user 逐字一致（注入内容确定性）。"""
        tool_results = {"get_hydrology": {"station": "吴堡", "flow_m3_s": 3000}}
        m1, _ = _build_synth_messages("查询", tool_results)
        m2, _ = _build_synth_messages("查询", tool_results)
        assert m1[0]["content"] == m2[0]["content"]
        assert m1[1]["content"] == m2[1]["content"]


class TestPlannerContextPersistedInState:
    def test_round1_experiences_history_in_fc_user(self):
        """第 1 轮计算的经验/历史摘要进入首轮 user 消息并写入 state。"""
        with patch("agent.memory.get_relevant_experiences", return_value="EXP_TEXT"), \
             patch("agent.graph.context_compact.extract_history_context", return_value="HIST_TEXT"), \
             patch("agent.skills.match_skill", return_value=None), \
             patch("agent.graph.nodes._plan_via_fc", return_value=_mock_plan([])):
            result = planner_node({
                "user_query": "吴堡站水情",
                "rounds": 0,
                "history": [{"role": "user", "content": "之前聊过龙门站"}],
                "tool_results": {}, "tool_calls": [], "skill_instructions": "",
            })
        assert result["experiences"] == "EXP_TEXT"
        assert result["history_context"] == "HIST_TEXT"
        user = result["fc_messages"][1]["content"]
        assert "EXP_TEXT" in user and "HIST_TEXT" in user

    def test_round2_reuses_state_without_recompute(self):
        """第 2 轮从 state 复用，不再重新计算经验/历史摘要。"""
        def _forbidden():
            raise AssertionError("第 2 轮不应重新计算记忆注入")

        with patch("agent.memory.get_relevant_experiences", side_effect=_forbidden), \
             patch("agent.graph.context_compact.extract_history_context", side_effect=_forbidden), \
             patch("agent.graph.nodes._plan_via_fc", return_value=_mock_plan([])):
            result = planner_node({
                "user_query": "吴堡站水情",
                "rounds": 1,
                "fc_messages": [
                    {"role": "system", "content": "S"},
                    {"role": "user", "content": "吴堡站水情 EXP_TEXT HIST_TEXT"},
                    {"role": "assistant", "content": None, "tool_calls": []},
                ],
                "tool_results": {"get_hydrology": {"station": "吴堡"}},
                "tool_calls": [{"tool_name": "get_hydrology", "arguments": {}}],
                "skill_instructions": "",
                "experiences": "ROUND1_EXP",
                "history_context": "ROUND1_HIST",
            })
        assert result["experiences"] == "ROUND1_EXP"
        assert result["history_context"] == "ROUND1_HIST"
        # 第 1 轮的 user 消息原样保留在序列中（前缀不中途消失）
        assert result["fc_messages"][1]["content"] == "吴堡站水情 EXP_TEXT HIST_TEXT"
