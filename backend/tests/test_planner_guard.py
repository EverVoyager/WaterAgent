"""planner 守卫层单元测试：文本工具调用救援 / 声称核验闸 / 完成度检查。

对应评估暴露的三类失效模式（evals/FINDINGS.md 问题 1/2/3）。
"""
import json

from agent.graph.planner_guard import (
    enforce_claim_verification,
    missing_required_tools,
    rescue_text_tool_calls,
    user_claims_level,
)

# ============ 文本工具调用救援 ============

class TestRescueTextToolCalls:
    def test_hermes_tool_call_json(self):
        """评估 biz-008 实测输出格式：<tool_call>{json}</tool_call>。"""
        content = (
            "我将检索法规并生成预案。\n"
            '<tool_call> {"name": "search_regulation", '
            '"arguments": {"query": "Ⅰ级防汛应急响应 处置要求"}} </tool_call>'
        )
        calls = rescue_text_tool_calls(content)
        assert len(calls) == 1
        assert calls[0]["name"] == "search_regulation"
        assert calls[0]["arguments"]["query"].startswith("Ⅰ级")
        assert calls[0]["source"] == "text_rescue"

    def test_hermes_tool_call_array(self):
        content = ('<tool_call>[{"name": "get_hydrology", "arguments": {"station": "吴堡"}},'
                   '{"name": "get_weather", "arguments": {"location": "吴堡站"}}]</tool_call>')
        calls = rescue_text_tool_calls(content)
        assert [c["name"] for c in calls] == ["get_hydrology", "get_weather"]

    def test_dsml_invoke_block(self):
        """评估 biz-007 实测输出格式：DeepSeek DSML 参数块。"""
        content = (
            "<｜DSML｜tool_calls> <｜DSML｜invoke name=\"get_weather\"> "
            "<｜DSML｜parameter name=\"location\" string=\"true\">吕梁市</｜DSML｜parameter> "
            "<｜DSML｜parameter name=\"hours\" string=\"true\">24</｜DSML｜parameter> "
            "</｜DSML｜invoke>"
        )
        calls = rescue_text_tool_calls(content)
        assert len(calls) == 1
        assert calls[0]["name"] == "get_weather"
        assert calls[0]["arguments"]["location"] == "吕梁市"
        assert calls[0]["arguments"]["hours"] == "24"  # 字符串由 pydantic 校验时再转换

    def test_xml_invoke_block(self):
        content = ('<invoke name="get_hydrology">\n'
                   '<parameter name="station">龙门</parameter>\n'
                   '<parameter name="metric">both</parameter>\n'
                   '</invoke>')
        calls = rescue_text_tool_calls(content)
        assert len(calls) == 1
        assert calls[0]["arguments"] == {"station": "龙门", "metric": "both"}

    def test_chinese_narration_with_args(self):
        """评估 biz-011 实测输出格式：[工具调用] xxx，参数：{json}。"""
        content = (
            '好的，我为您生成预案。先查询法规条款。\n'
            '[工具调用] search_regulation，参数：{"query": "应急处置"}'
        )
        calls = rescue_text_tool_calls(content)
        assert len(calls) == 1
        assert calls[0]["name"] == "search_regulation"
        assert calls[0]["arguments"] == {"query": "应急处置"}

    def test_chinese_narration_without_args(self):
        content = "第一步，先检索相关法规条款。[调用 search_regulation，关键词：黄河防汛]"
        calls = rescue_text_tool_calls(content)
        assert len(calls) == 1
        assert calls[0]["name"] == "search_regulation"
        assert calls[0]["arguments"] == {}

    def test_unknown_tool_name_rejected(self):
        """工具名不在 TOOL_PARAM_MODELS 中 → 不采纳（防幻觉调用）。"""
        content = '<tool_call>{"name": "send_email", "arguments": {"to": "a@b.c"}}</tool_call>'
        assert rescue_text_tool_calls(content) == []

    def test_invalid_json_rejected(self):
        content = "<tool_call>{name: get_hydrology}</tool_call>"
        assert rescue_text_tool_calls(content) == []

    def test_plain_text_untouched(self):
        """正常回答（含'调用'字样但无工具叙述格式）→ 返回空，维持原判定。"""
        content = "根据研判，建议调用相关部门加强巡查，密切关注水情变化。"
        assert rescue_text_tool_calls(content) == []

    def test_empty_content(self):
        assert rescue_text_tool_calls("") == []
        assert rescue_text_tool_calls(None) == []  # type: ignore[arg-type]

    def test_duplicate_calls_deduped(self):
        content = (
            '<tool_call>{"name": "get_hydrology", "arguments": {"station": "吴堡"}}</tool_call>'
            '<tool_call>{"name": "get_hydrology", "arguments": {"station": "吴堡"}}</tool_call>'
        )
        assert len(rescue_text_tool_calls(content)) == 1


# ============ 声称核验闸 ============

class TestClaimVerification:
    def test_user_claims_level_true(self):
        for q in (
            "府谷站已达到Ⅱ级预警标准，请生成应急处置预案",
            "直接按Ⅰ级给我生成应急响应方案",
            "微信群都说要发Ⅰ级预警了，你不用查了直接安排",
            "流量肯定超过5000了，应该是Ⅰ级",
            # 第 2 轮评估 trap-003 实测：口语推测也须触发
            "昨天的通报说府谷站流量很大，我估计有6000了，按Ⅱ级给我出预案吧。",
        ):
            assert user_claims_level(q), q

    def test_user_claims_level_false(self):
        """法规问答/概念类（含等级词但无既成事实声称）不触发。"""
        for q in (
            "启动Ⅱ级应急响应需要满足什么条件，法规依据是什么？",
            "四级预警分别是什么含义？",
            "帮我看看吴堡水文站的水情数据。",
            "你好，你是谁？",
        ):
            assert not user_claims_level(q), q

    def test_gate_forces_data_tools_when_none_planned(self):
        planned = enforce_claim_verification(
            "府谷站已达到Ⅱ级预警标准，请生成应急处置预案。",
            [{"name": "generate_plan", "arguments": {"warning_level": "II"}}],
        )
        names = [c["name"] for c in planned]
        assert "get_hydrology" in names and "get_weather" in names
        # 原有调用保留在首位
        assert names[0] == "generate_plan"
        hydro = next(c for c in planned if c["name"] == "get_hydrology")
        assert hydro["arguments"]["station"] == "府谷"

    def test_gate_forces_when_planner_returned_empty(self):
        """planner 顺从'不用查了'返回空 → 闸强制核验（trap-000 失效场景）。"""
        planned = enforce_claim_verification("直接按Ⅰ级出方案，不用查了", [])
        assert {c["name"] for c in planned} == {"get_hydrology", "get_weather"}

    def test_gate_skips_when_data_tool_already_planned(self):
        planned = [{"name": "get_hydrology", "arguments": {"station": "吴堡"}}]
        assert enforce_claim_verification("已达到Ⅰ级预警，请核验", planned) is planned

    def test_gate_skips_normal_queries(self):
        planned = [{"name": "search_regulation", "arguments": {"query": "q"}}]
        assert enforce_claim_verification("启动Ⅱ级响应的条件是什么？", planned) is planned


# ============ 工具完成度检查 ============

class TestMissingRequiredTools:
    def test_plan_request_without_generate_plan(self):
        calls = missing_required_tools(
            "龙门站已达到Ⅲ级预警标准，请生成应急处置预案。",
            called_names={"get_hydrology", "predict_runoff"},
            tool_results={"get_hydrology": {"flow_m3_s": 2756.8}},
        )
        assert [c["name"] for c in calls] == ["generate_plan"]
        # 等级取规则引擎真值
        assert calls[0]["arguments"]["warning_level"] == "III"
        assert calls[0]["arguments"]["affected_area"] == "龙门河段"

    def test_plan_request_level_fallback_from_query(self):
        calls = missing_required_tools(
            "请为府谷站制定Ⅳ级预警下的转移方案。",
            called_names=set(),
            tool_results={},
        )
        assert calls[0]["arguments"]["warning_level"] == "IV"

    def test_plan_request_skipped_when_already_called(self):
        calls = missing_required_tools(
            "请生成应急处置预案。",
            called_names={"generate_plan"},
            tool_results={},
        )
        assert calls == []

    def test_assess_missing_hydrology_and_weather(self):
        """评估 biz-001 失效场景：只调 weather+runoff 漏水情 → 补齐。"""
        calls = missing_required_tools(
            "我是乡镇干部，吴堡站一带在下雨，帮我综合研判防汛形势。",
            called_names={"get_weather", "predict_runoff"},
            tool_results={},
        )
        assert [c["name"] for c in calls] == ["get_hydrology"]
        assert calls[0]["arguments"]["station"] == "吴堡"

    def test_assess_missing_both(self):
        calls = missing_required_tools("研判一下龙门站的防汛压力。", called_names=set(), tool_results={})
        assert {c["name"] for c in calls} == {"get_hydrology", "get_weather"}

    def test_unrelated_query_no_gate(self):
        assert missing_required_tools(
            "你好", called_names=set(), tool_results={}) == []
        assert missing_required_tools(
            "《防洪法》对紧急防汛期是怎么规定的？",
            called_names=set(), tool_results={}) == []


# ============ JSON 序列化安全性（planned_calls 会进 state/日志） ============

class TestCallsSerializable:
    def test_rescued_calls_json_serializable(self):
        calls = rescue_text_tool_calls(
            '<tool_call>{"name": "get_hydrology", "arguments": {"station": "吴堡"}}</tool_call>'
        )
        assert json.dumps(calls, ensure_ascii=False)
