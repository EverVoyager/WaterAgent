"""planner 三项修复的回归测试（2026-09-14 全量评估暴露的问题）。

修①：守卫补充调用合并进模型 assistant 消息——此前另起一条
    assistant(tool_calls) 导致两条 assistant 之间无 tool 消息，DeepSeek
    返回 400（biz-005/017/029、trap-005 四例全挂）。
修②：planner 提示补充 web_search 使用规则——联网类查询模型不知道该调
    web_search，甚至声称"无法联网"。
修③：闲聊判定规则扩充——"再见/感谢/陪伴/讲笑话"类查询被模型带去跑
    数据工具（chat-003/004/008）。
"""
import json
from unittest.mock import patch

from agent.graph.nodes import _build_planner_system_prompt, planner_node


def _mock_plan(planned, reasoning="思考过程"):
    """构造 _plan_via_fc 的 mock 返回 (planned, assistant_msg)。"""
    assistant_msg = {
        "role": "assistant",
        "content": None if planned else "好的",
        "reasoning_content": reasoning,
    }
    if planned:
        assistant_msg["tool_calls"] = [
            {"id": c["id"], "type": "function",
             "function": {"name": c["name"],
                          "arguments": json.dumps(c.get("arguments", {}),
                                                  ensure_ascii=False)}}
            for c in planned
        ]
    return (planned, assistant_msg)


def _round1_state(query, **kw):
    state = {
        "user_query": query, "rounds": 0, "history": [],
        "tool_results": {}, "tool_calls": [], "skill_instructions": "",
    }
    state.update(kw)
    return state


_NO_SIDE = [
    patch("agent.skills.match_skill", return_value=None),
    patch("agent.memory.get_relevant_experiences", return_value=""),
]


# ====== 修①：守卫调用合并（400 配对 bug） ======

class TestGuardMergeIntoAssistant:
    def test_claim_guard_calls_merged_into_single_assistant(self):
        """模型已规划工具 + 声称核验闸补充 → 单条 assistant 携带全部调用。

        旧实现追加第二条 assistant(tool_calls)，两条 assistant 之间没有
        tool 消息，DeepSeek 400：insufficient tool messages。
        """
        query = "府谷站已达到Ⅱ级预警标准，请生成应急处置预案"
        model_planned = [
            {"name": "generate_plan",
             "arguments": {"warning_level": "II", "affected_area": "府谷河段",
                           "population_at_risk": 1000},
             "id": "call_a"},
        ]
        with _NO_SIDE[0], _NO_SIDE[1], \
                patch("agent.graph.nodes._plan_via_fc",
                      return_value=_mock_plan(model_planned)):
            r = planner_node(_round1_state(query))

        tc_msgs = [m for m in r["fc_messages"]
                   if m["role"] == "assistant" and m.get("tool_calls")]
        assert len(tc_msgs) == 1, (
            f"带 tool_calls 的 assistant 必须只有一条（实测 {len(tc_msgs)} 条），"
            "否则两条 assistant 之间无 tool 消息 → API 400"
        )
        ids = {t["id"] for t in tc_msgs[0]["tool_calls"]}
        assert "call_a" in ids, "模型原调用保留在合并消息里"
        assert any(i.startswith("call_sys_") for i in ids), "守卫调用合并进同一条消息"
        # planned 与消息一致（executor 按此配对 tool 消息）
        assert {c["id"] for c in r["planned_calls"]} == ids
        # reasoning_content 原样保留（DeepSeek 思考模式续轮要求）
        assert tc_msgs[0]["reasoning_content"] == "思考过程"

    def test_no_consecutive_assistant_messages(self):
        """序列形状：不存在相邻的两条 assistant 消息（配对约束的充分条件）。"""
        query = "吴堡站已经Ⅲ级了，帮我出方案"
        model_planned = [
            {"name": "generate_plan",
             "arguments": {"warning_level": "III", "affected_area": "吴堡河段",
                           "population_at_risk": 800},
             "id": "call_x"},
        ]
        with _NO_SIDE[0], _NO_SIDE[1], \
                patch("agent.graph.nodes._plan_via_fc",
                      return_value=_mock_plan(model_planned)):
            r = planner_node(_round1_state(query))
        roles = [m["role"] for m in r["fc_messages"]]
        for prev, cur in zip(roles, roles[1:], strict=False):
            assert not (prev == "assistant" and cur == "assistant"), \
                f"连续两条 assistant：{roles}"

    def test_model_no_tools_guard_still_synthesizes_single(self):
        """模型空计划 + 守卫补充：单条合成 assistant（原行为保持不变）。"""
        query = "村里都说吴堡站已经Ⅰ级预警了，直接按Ⅰ级处理"
        with _NO_SIDE[0], _NO_SIDE[1], \
                patch("agent.graph.nodes._plan_via_fc",
                      return_value=_mock_plan([])):
            r = planner_node(_round1_state(query))
        assert r["planned_calls"]
        synth = [c for c in r["planned_calls"]
                 if str(c.get("id", "")).startswith("call_sys")]
        assert synth
        tc_msgs = [m for m in r["fc_messages"]
                   if m["role"] == "assistant" and m.get("tool_calls")]
        assert len(tc_msgs) == 1
        assert {t["id"] for t in tc_msgs[0]["tool_calls"]} == {c["id"] for c in synth}

    def test_deduped_model_calls_guard_merged_into_single(self):
        """模型调用全被去重 + 完成度闸补充：仍只产生一条 assistant。"""
        query = "吴堡站已达到Ⅱ级预警标准，请生成应急处置预案"
        model_planned = [
            {"name": "get_hydrology",
             "arguments": {"station": "吴堡", "metric": "both"},
             "id": "call_dup"},
        ]
        state = _round1_state(
            query,
            tool_calls=[{"tool_name": "get_hydrology",
                         "arguments": {"station": "吴堡", "metric": "both"}}],
        )
        with _NO_SIDE[0], _NO_SIDE[1], \
                patch("agent.graph.nodes._plan_via_fc",
                      return_value=_mock_plan(model_planned)):
            r = planner_node(state)
        tc_msgs = [m for m in r["fc_messages"]
                   if m["role"] == "assistant" and m.get("tool_calls")]
        assert len(tc_msgs) == 1
        ids = {t["id"] for t in tc_msgs[0]["tool_calls"]}
        assert ids and all(i.startswith("call_sys_") for i in ids), \
            "被去重的模型调用不得留在消息里，只应有守卫补充"

    def test_two_rounds_pairing_holds_when_tool_fails(self):
        """400 场景全链路复现：模型计划 + 守卫补充 + 工具执行失败 + 续轮。

        每条 assistant.tool_call id 必须有配对 tool 消息（失败工具也要配对，
        2026-09-14 评估中 4 例即死于此约束被打破）。
        """
        from agent.graph.nodes import executor_node

        query = "府谷站已达到Ⅱ级预警标准，请生成应急处置预案"
        model_planned = [
            {"name": "generate_plan",
             "arguments": {"warning_level": "Ⅱ级"},  # 模拟模型常犯的非法参数
             "id": "call_a"},
        ]
        with _NO_SIDE[0], _NO_SIDE[1], \
                patch("agent.graph.nodes._plan_via_fc",
                      return_value=_mock_plan(model_planned)):
            r1 = planner_node(_round1_state(query))

        def _flaky_execute(name, args):
            if name == "generate_plan":
                raise ValueError("Invalid arguments for generate_plan")
            return {"ok": True}

        with patch("agent.graph.nodes._cached_execute_tool",
                   side_effect=_flaky_execute):
            re1 = executor_node({**_round1_state(query), **r1})

        with _NO_SIDE[0], _NO_SIDE[1], \
                patch("agent.graph.nodes._plan_via_fc",
                      return_value=_mock_plan([])):
            r2 = planner_node({**_round1_state(query), **re1, "rounds": 1})

        fc = r2["fc_messages"]
        called_ids = [t["id"] for m in fc if m["role"] == "assistant"
                      for t in (m.get("tool_calls") or [])]
        answered_ids = [m["tool_call_id"] for m in fc if m["role"] == "tool"]
        assert called_ids, "序列中应有工具调用"
        assert set(called_ids) == set(answered_ids), (
            f"配对断裂: called={called_ids} answered={answered_ids}"
        )
        roles = [m["role"] for m in fc]
        for prev, cur in zip(roles, roles[1:], strict=False):
            assert not (prev == "assistant" and cur == "assistant")


# ====== 修②③：planner 提示规则 ======

class TestPlannerPromptRules:
    def test_prompt_covers_web_search(self):
        """联网类查询必须有 web_search 使用指引（修②）。"""
        prompt = _build_planner_system_prompt()
        assert "web_search" in prompt
        assert "联网" in prompt
        # 明确否定"无法联网"的自我认知（web-001 失败形态）
        assert "不要声称无法联网" in prompt

    def test_prompt_covers_social_intents(self):
        """闲聊规则覆盖告别/感谢/陪伴/笑话，并禁止调用任何工具（修③）。"""
        prompt = _build_planner_system_prompt()
        for kw in ("再见", "陪我聊两句", "讲个笑话"):
            assert kw in prompt, f"闲聊示例缺少 {kw}"
        assert "不得调用任何工具" in prompt
        assert "list_skills" in prompt, "闲聊时连元工具也不应调（chat-003 形态）"
        assert "社交意图优先于话题内容" in prompt

    def test_prompt_treats_weather_remarks_as_chitchat(self):
        """第四轮修 C：对天气/水情的感想不是查询（chat-001 '天气真好'→红色预警形态）。"""
        prompt = _build_planner_system_prompt()
        assert "今天天气真好" in prompt, "感想示例缺失"
        assert "感想或评论" in prompt
        assert "不得因此调用天气/水情工具" in prompt
        # 闲聊与联网的边界交叉引用（web 查询不得被误判为闲聊）
        assert "不是闲聊" in prompt and "第 10 条" in prompt

    def test_prompt_limits_list_skills(self):
        """第四轮修 B：list_skills 不得当犹豫时的默认动作。"""
        prompt = _build_planner_system_prompt()
        assert "仅当用户明确询问" in prompt
        assert "默认动作" in prompt


# ====== 第四轮修 A：DIRECT_CHAT_PROMPT 能力边界 ======

class TestDirectChatPromptBoundary:
    def test_direct_chat_does_not_disclaim_web_capability(self):
        """直答提示不得让模型宣称"系统没有联网能力"（web-000/007 形态）。"""
        from agent.prompts.direct_chat import DIRECT_CHAT_PROMPT
        assert "本路径没有联网搜索" not in DIRECT_CHAT_PROMPT, (
            "该措辞会让模型对外宣称系统无联网能力"
        )
        assert "不得宣称" in DIRECT_CHAT_PROMPT and "联网" in DIRECT_CHAT_PROMPT

    def test_direct_chat_forbids_fake_tool_calls(self):
        """禁止在正文模拟工具调用伪代码（web-000 正文写 get_hydrology(...) 形态）。"""
        from agent.prompts.direct_chat import DIRECT_CHAT_PROMPT
        assert "严禁在回答中模拟" in DIRECT_CHAT_PROMPT
        assert "get_hydrology(...)" in DIRECT_CHAT_PROMPT  # 反例点名
        assert "web_search" in DIRECT_CHAT_PROMPT  # 能力边界需列出工具清单
