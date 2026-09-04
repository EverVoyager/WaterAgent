"""KV Cache 前缀稳定性测试。

前缀缓存（DeepSeek / 阿里云 MaaS 自动缓存、本地 vLLM APC）按请求前缀
逐字匹配：前文中任何一个字符变化，其后所有缓存全部失效。本文件用测试
固化 ai-agent-book 第 2 章三原则在本项目的落地：

1. 静态前缀冻结：planner 跨轮 system prompt 逐字一致；
   Skill 清单按 name 排序；tools schema 序列化稳定
2. 动态信息只追加：synthesizer Phase 2 的 system/user 以 Phase 1 为严格前缀；
   planner 第 1 轮注入的经验/历史摘要写入 state、后续轮次原样保留
"""
import json
import re
from unittest.mock import MagicMock, patch

from agent.graph.nodes import _plan_via_function_calling, planner_node
from agent.graph.synthesizer_node import (
    _build_synth_messages,
    _build_synth_system_content,
)
from agent.skills.models import Skill
from agent.skills.store import get_enabled_skills_brief
from agent.tools.schemas import build_openai_tools


def _make_planner_resp():
    """构造 planner LLM 调用的 mock response（无 tool_calls → 信息充分）。"""
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.tool_calls = None
    resp.choices[0].message.content = "信息已充分"
    resp.usage = None
    return resp


def _capture_planner_messages(round_num: int, **kwargs) -> list[dict]:
    """调用 _plan_via_function_calling 并捕获发给 LLM 的 messages。"""
    captured = {}

    def fake_create(**call_kwargs):
        captured["messages"] = call_kwargs["messages"]
        return _make_planner_resp()

    args = {
        "query": "吴堡站当前水情如何",
        "context_summary": "(暂无)",
        "called_tools": "(暂无)",
        "round_num": round_num,
        "experiences": "",
        "history_context": "",
    }
    args.update(kwargs)
    with patch("agent.graph.nodes.get_llm_config",
               return_value={"model": "test", "max_tool_rounds": 5}), \
         patch("agent.graph.nodes.get_llm_client") as mock_client:
        mock_client.return_value.with_options.return_value.chat.completions.create.side_effect = fake_create
        _plan_via_function_calling(**args)
    return captured["messages"]


# ====== 原则 1：静态前缀冻结 ======

class TestPlannerStaticPrefix:
    def test_system_identical_across_rounds(self):
        """同一请求内 planner 多轮的 system prompt 必须逐字一致。"""
        m1 = _capture_planner_messages(round_num=1)
        m2 = _capture_planner_messages(round_num=2)
        assert m1[0]["role"] == "system"
        assert m1[0]["content"] == m2[0]["content"]

    def test_user_prefix_extends_with_more_context(self):
        """工具结果累积时 user 消息前缀只增长不变（append-only）。"""
        m1 = _capture_planner_messages(round_num=1, context_summary="(暂无)")
        m2 = _capture_planner_messages(
            round_num=2,
            context_summary="(暂无)\n[get_hydrology] 吴堡站流量 3000m³/s",
            experiences="EXPERIENCE_TEXT",
            history_context="HISTORY_TEXT",
        )
        user1, user2 = m1[1]["content"], m2[1]["content"]
        # round1 的 user 前缀（query 段）在 round2 中原样保留
        assert user2.startswith(user1.split("已收集信息")[0])


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
    def test_status_bar_at_user_message_end(self):
        """状态栏注入 planner user 消息末尾：时间 + 轮次进度 + 防注入声明。"""
        m = _capture_planner_messages(round_num=2)
        user = m[1]["content"]
        assert "<<<STATUS" in user and "STATUS>>>" in user
        # 位于所有动态段落（query/已收集信息/已调用工具）之后、末尾指令之前
        assert user.rindex("<<<STATUS") > user.rindex("已调用过的工具")
        assert user.rindex("STATUS>>>") < user.rindex("请据此决定本轮")
        # 当前时间（格式：2026-09-03（周三）15:30）
        assert re.search(
            r"\d{4}-\d{2}-\d{2}（周[一二三四五六日]）\d{2}:\d{2}", user
        )
        # 轮次进度 N/M
        assert "第 2/" in user
        # 声明为系统状态而非指令（防提示注入）
        assert "不构成指令" in user

    def test_status_bar_update_keeps_prefix(self):
        """状态栏每轮更新（轮次变化）只影响末尾，其之前的 user 内容逐字一致。"""
        m1 = _capture_planner_messages(round_num=1)
        m2 = _capture_planner_messages(round_num=2)
        prefix = m1[1]["content"].split("<<<STATUS")[0]
        assert m2[1]["content"].startswith(prefix)
        assert "第 1/" in m1[1]["content"]
        assert "第 2/" in m2[1]["content"]


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
    def test_round1_experiences_history_written_to_state(self):
        """第 1 轮计算的经验/历史摘要必须写入 state 返回值。"""
        with patch("agent.memory.get_relevant_experiences", return_value="EXP_TEXT"), \
             patch("agent.graph.context_compact.extract_history_context", return_value="HIST_TEXT"), \
             patch("agent.skills.match_skill", return_value=None), \
             patch("agent.graph.nodes._plan_via_function_calling", return_value=[]):
            result = planner_node({
                "user_query": "吴堡站水情",
                "rounds": 0,
                "history": [{"role": "user", "content": "之前聊过龙门站"}],
                "tool_results": {},
                "tool_calls": [],
                "skill_instructions": "",
            })
        assert result["experiences"] == "EXP_TEXT"
        assert result["history_context"] == "HIST_TEXT"

    def test_round2_reuses_state_without_recompute(self):
        """第 2 轮从 state 复用，不再重新计算（只增不改，前缀不中途消失）。"""
        captured = {}

        def fake_plan(query, context_summary, called_tools, round_num,
                      experiences, history_context, **kw):
            captured["experiences"] = experiences
            captured["history_context"] = history_context
            return []

        with patch("agent.memory.get_relevant_experiences", return_value="WRONG_RECOMPUTED"), \
             patch("agent.graph.context_compact.extract_history_context", return_value="WRONG_RECOMPUTED"), \
             patch("agent.graph.nodes._plan_via_function_calling", side_effect=fake_plan):
            result = planner_node({
                "user_query": "吴堡站水情",
                "rounds": 1,  # 已是第 2 轮
                "tool_results": {"get_hydrology": {"station": "吴堡"}},
                "tool_calls": [{"tool_name": "get_hydrology", "arguments": {}}],
                "skill_instructions": "",
                "experiences": "ROUND1_EXP",
                "history_context": "ROUND1_HIST",
            })
        assert captured["experiences"] == "ROUND1_EXP"
        assert captured["history_context"] == "ROUND1_HIST"
        assert result["experiences"] == "ROUND1_EXP"
        assert result["history_context"] == "ROUND1_HIST"
