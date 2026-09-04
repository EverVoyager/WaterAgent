"""planner 原生 FC 消息序列测试。

验证消息序列形状与 API 配对约束（每个 assistant.tool_call 都有配对的
tool 消息）、reasoning_content 原样回传、守卫合成消息、去重同步。
"""
import json
from unittest.mock import patch

from agent.graph.nodes import executor_node, planner_node


def _mock_plan(planned, reasoning="思考过程", content=None):
    """构造 _plan_via_fc 的 mock 返回 (planned, assistant_msg)。"""
    assistant_msg = {
        "role": "assistant",
        "content": content if not planned else None,
        "reasoning_content": reasoning,
    }
    if planned:
        assistant_msg["tool_calls"] = [
            {"id": c["id"], "type": "function",
             "function": {"name": c["name"],
                          "arguments": json.dumps(c.get("arguments", {}), ensure_ascii=False)}}
            for c in planned
        ]
    return (planned, assistant_msg)


def _round1_state(query="吴堡站现在水情怎么样", **kw):
    state = {
        "user_query": query, "rounds": 0, "history": [],
        "tool_results": {}, "tool_calls": [], "skill_instructions": "",
    }
    state.update(kw)
    return state


# ====== 第 1 轮：assistant 消息与 reasoning_content ======

class TestRound1AssistantMessage:
    def test_assistant_appended_with_tool_calls_and_reasoning(self):
        """模型返回的 assistant 消息原样追加：tool_calls + reasoning_content。"""
        planned = [{"name": "get_hydrology",
                    "arguments": {"station": "吴堡", "metric": "both"},
                    "id": "call_abc"}]
        with patch("agent.skills.match_skill", return_value=None), \
             patch("agent.memory.get_relevant_experiences", return_value=""), \
             patch("agent.graph.nodes._plan_via_fc",
                   return_value=_mock_plan(planned, reasoning="RC_TEXT")):
            r = planner_node(_round1_state())
        fc = r["fc_messages"]
        assert [m["role"] for m in fc] == ["system", "user", "assistant"]
        a = fc[2]
        assert a["reasoning_content"] == "RC_TEXT"
        assert a["tool_calls"][0]["id"] == "call_abc"
        assert a["tool_calls"][0]["function"]["name"] == "get_hydrology"
        # planned_calls 保留 id 供 executor 配对
        assert r["planned_calls"][0]["id"] == "call_abc"

    def test_no_reasoning_content_field_when_absent(self):
        """模型未返回 reasoning_content 时消息不携带该字段（后端无关）。"""
        planned = [{"name": "get_weather", "arguments": {"location": "吕梁市"},
                    "id": "call_1"}]
        assistant = _mock_plan(planned)
        del assistant[1]["reasoning_content"]
        with patch("agent.skills.match_skill", return_value=None), \
             patch("agent.memory.get_relevant_experiences", return_value=""), \
             patch("agent.graph.nodes._plan_via_fc", return_value=assistant):
            r = planner_node(_round1_state())
        assert "reasoning_content" not in r["fc_messages"][2]


# ====== executor：tool 消息配对 ======

class TestExecutorToolMessages:
    def test_tool_messages_paired_by_id(self):
        """executor 把结果转为带 tool_call_id 配对的 tool 消息追加。"""
        with patch("agent.graph.nodes._cached_execute_tool",
                   return_value={"station": "吴堡", "flow_m3_s": 582}):
            r = executor_node({
                "planned_calls": [{"name": "get_hydrology",
                                   "arguments": {"station": "吴堡", "metric": "both"},
                                   "id": "call_abc"}],
                "tool_results": {}, "tool_calls": [], "rounds": 1,
                "fc_messages": [
                    {"role": "system", "content": "S"},
                    {"role": "user", "content": "Q"},
                    {"role": "assistant", "content": None,
                     "tool_calls": [{"id": "call_abc", "type": "function",
                                     "function": {"name": "get_hydrology", "arguments": "{}"}}]},
                ],
            })
        tool_msgs = [m for m in r["fc_messages"] if m["role"] == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["tool_call_id"] == "call_abc"
        assert "582" in tool_msgs[0]["content"]
        # 前三条原样保留（append-only）
        assert r["fc_messages"][:3][2]["tool_calls"][0]["id"] == "call_abc"

    def test_error_message_actionable(self):
        """工具失败的 tool 消息携带可操作建议（对齐 Anthropic 工具设计指南）。"""
        def boom(name, args):
            raise RuntimeError("站点无监测数据")

        with patch("agent.graph.nodes._cached_execute_tool", side_effect=boom):
            r = executor_node({
                "planned_calls": [{"name": "get_hydrology",
                                   "arguments": {"station": "府谷"}, "id": "call_e"}],
                "tool_results": {}, "tool_calls": [], "rounds": 1, "fc_messages": [],
            })
        tool_msg = [m for m in r["fc_messages"] if m["role"] == "tool"][0]
        assert "error" in tool_msg["content"]
        assert "hint" in tool_msg["content"]  # 可操作建议


# ====== 配对不变量（API 约束）======

class TestPairingInvariant:
    def test_every_tool_call_has_paired_tool_message(self):
        """端到端两轮循环后：序列中每个 assistant.tool_call id 都有配对 tool 消息。"""
        planned1 = [
            {"name": "get_hydrology", "arguments": {"station": "吴堡", "metric": "both"},
             "id": "call_a"},
            {"name": "get_weather", "arguments": {"location": "吕梁市", "hours": 24},
             "id": "call_b"},
        ]
        with patch("agent.skills.match_skill", return_value=None), \
             patch("agent.memory.get_relevant_experiences", return_value=""), \
             patch("agent.graph.nodes._plan_via_fc", return_value=_mock_plan(planned1)):
            r1 = planner_node(_round1_state())

        with patch("agent.graph.nodes._cached_execute_tool",
                   return_value={"ok": True}):
            re1 = executor_node({**_round1_state(), **r1})

        with patch("agent.graph.nodes._plan_via_fc", return_value=_mock_plan([])):
            r2 = planner_node({**_round1_state(), **re1, "rounds": 1})

        fc = r2["fc_messages"]
        called_ids = [t["id"] for m in fc if m["role"] == "assistant"
                      for t in (m.get("tool_calls") or [])]
        answered_ids = [m["tool_call_id"] for m in fc if m["role"] == "tool"]
        assert called_ids, "序列中应有工具调用"
        assert set(called_ids) == set(answered_ids), (
            f"配对断裂: called={called_ids} answered={answered_ids}"
        )


# ====== 守卫合成消息 ======

class TestGuardSynthesis:
    def test_claim_verification_synthesizes_assistant(self):
        """声称核验闸补充的调用：合成 assistant（占位 rc）+ call_sys id。"""
        query = "现在已经Ⅲ级预警了，直接按Ⅲ级处理"
        with patch("agent.skills.match_skill", return_value=None), \
             patch("agent.memory.get_relevant_experiences", return_value=""), \
             patch("agent.graph.nodes._plan_via_fc", return_value=_mock_plan([])):
            r = planner_node(_round1_state(query=query))
        assert r["planned_calls"], "声称核验闸应补充数据核验调用"
        synth = [c for c in r["planned_calls"] if str(c.get("id", "")).startswith("call_sys")]
        assert synth, "补充调用应分配合成 id"
        synth_msgs = [m for m in r["fc_messages"] if m["role"] == "assistant"
                      and m.get("reasoning_content") == "（系统补充的核验工具调用）"]
        assert len(synth_msgs) == 1
        assert {t["id"] for t in synth_msgs[0]["tool_calls"]} == {c["id"] for c in synth}


# ====== 去重同步 ======

class TestDedupeSync:
    def test_deduped_calls_removed_from_assistant_message(self):
        """与历史完全相同的调用被去重后，assistant 消息同步裁剪（维持配对）。"""
        planned = [{"name": "get_hydrology",
                    "arguments": {"station": "吴堡", "metric": "both"},
                    "id": "call_dup"}]
        state = _round1_state(
            tool_calls=[{"tool_name": "get_hydrology",
                         "arguments": {"station": "吴堡", "metric": "both"}}],
        )
        with patch("agent.skills.match_skill", return_value=None), \
             patch("agent.memory.get_relevant_experiences", return_value=""), \
             patch("agent.graph.nodes._plan_via_fc", return_value=_mock_plan(planned)):
            r = planner_node(state)
        # 去重后 planned 为空 → assistant 的 tool_calls 被裁空 → 不追加该消息
        assert r["planned_calls"] == []
        tool_call_msgs = [m for m in r["fc_messages"] if m.get("tool_calls")]
        assert tool_call_msgs == []
