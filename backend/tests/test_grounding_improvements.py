"""Grounding 改进测试：编号对齐（Perplexica 模式）+ 等级一致性门 + 法规可引用。

对应三处改动：
1. _format_tool_results_for_llm：非引用源用【语义标签】，web/法规保留 [编号] 进 registry
2. _check_level_consistency / _apply_rule_level_override：规则引擎在线校验 LLM 等级
3. 法规检索条款纳入 source_registry（source_type=regulation，可校验）
"""
from agent.graph.synthesizer import compute_warning_level
from agent.graph.synthesizer_node import (
    _apply_rule_level_override,
    _check_level_consistency,
    _format_tool_results_for_llm,
)


def _hydrology(flow=3200.0, level=638.5):
    return {
        "station": "吴堡", "flow_m3_s": flow, "water_level_m": level,
        "warning_level_m": 640.0, "guaranteed_level_m": 642.0,
    }


def _regulation_hits():
    return {
        "hits": [
            {"title": "中华人民共和国防洪法", "article": "第四十条",
             "content": "有防汛抗洪任务的县级以上地方人民政府应当组织做好防汛抗洪工作。"},
        ]
    }


class TestFormatToolResultsAlignment:
    """模型看到的编号 = 用户看到的引用（Perplexica 对齐原则）。"""

    def test_hydrology_uses_label_not_number(self):
        text, registry = _format_tool_results_for_llm({"get_hydrology": _hydrology()})
        assert "【实时水情】" in text
        assert "[1]" not in text.split("【实时水情】")[1].split("\n")[0]
        assert registry == {}

    def test_weather_runoff_gis_threshold_all_labeled(self):
        results = {
            "get_weather": {"location": "吕梁", "total_rainfall_mm": 30},
            "predict_runoff": {"station": "吴堡", "peak_flow_m3_s": 3500, "series": []},
            "query_gis_terrain": {"slope": {"mean_degree": 12.0}},
        }
        text, registry = _format_tool_results_for_llm(results)
        assert "【天气预报】" in text and "【径流预测】" in text and "【GIS 地形分析】" in text
        assert "【预警等级阈值标准】" in text
        assert registry == {}
        # 全文不存在任何 [数字] 编号（无可引用来源时）
        import re
        assert not re.search(r"\[\d+\]", text)

    def test_web_search_numbered_and_registered(self):
        results = {"web_search": {"results": [
            {"title": "汛情通报", "snippet": "吴堡站流量上涨", "url": "https://a.com"},
        ]}}
        text, registry = _format_tool_results_for_llm(results)
        assert "[1] 汛情通报" in text
        assert registry[1]["source_type"] == "web_search"
        assert registry[1]["url"] == "https://a.com"

    def test_regulation_numbered_and_registered(self):
        """法规条款可引用：进 registry，text 与展示文本一致（子串校验可过）。"""
        text, registry = _format_tool_results_for_llm({"search_regulation": _regulation_hits()})
        assert "[1] 中华人民共和国防洪法 第四十条" in text
        assert registry[1]["source_type"] == "regulation"
        assert registry[1]["title"] == "中华人民共和国防洪法"
        # 展示给 LLM 的条款文本与校验原文一致
        assert "组织做好防汛抗洪工作" in registry[1]["text"]

    def test_web_and_regulation_numbering_interleaved(self):
        """两类可引用来源共用一套连续编号。"""
        results = {
            "search_regulation": _regulation_hits(),
            "web_search": {"results": [
                {"title": "汛情", "snippet": "x", "url": "https://a.com"},
            ]},
        }
        text, registry = _format_tool_results_for_llm(results)
        assert set(registry.keys()) == {1, 2}
        types = {registry[1]["source_type"], registry[2]["source_type"]}
        assert types == {"regulation", "web_search"}


class TestLevelConsistencyGate:
    """等级在线一致性门：规则引擎重算 vs LLM 输出。"""

    def test_consistent_level_passes(self):
        rule_level, _ = compute_warning_level({"get_hydrology": _hydrology()})
        result = {"warning_level": rule_level}
        ok, feedback, _ = _check_level_consistency(result, {"get_hydrology": _hydrology()})
        assert ok is True and feedback == ""

    def test_inconsistent_level_fails_with_feedback(self):
        # 流量 3200 → 规则判 II 级；LLM 若说 III 级则拦截
        result = {"warning_level": "III"}
        ok, feedback, rule_level = _check_level_consistency(result, {"get_hydrology": _hydrology()})
        assert ok is False
        assert rule_level == "II"
        assert "II" in feedback and "III" in feedback and "规则引擎" in feedback

    def test_no_data_skips_gate(self):
        """无流量/降雨/水位数据时不强压默认 IV（如纯法规咨询场景）。"""
        ok, feedback, rule_level = _check_level_consistency(
            {"warning_level": "II"}, {"search_regulation": _regulation_hits()},
        )
        assert ok is True and feedback == ""

    def test_empty_level_with_data_fails(self):
        ok, _, rule_level = _check_level_consistency(
            {"warning_level": ""}, {"get_hydrology": _hydrology()},
        )
        assert ok is False and rule_level == "II"

    def test_rainfall_data_triggers_gate(self):
        """24h 降雨 > 100mm → 规则判 I 级。"""
        weather = {"get_weather": {"total_rainfall_mm": 120.0}}
        ok, _, rule_level = _check_level_consistency({"warning_level": "II"}, weather)
        assert ok is False and rule_level == "I"


class TestRuleLevelOverride:
    """重试用尽：规则引擎等级覆盖 + reasoning 留痕。"""

    def test_override_replaces_level_and_annotates(self):
        result = {"warning_level": "IV", "reasoning": "水情平稳", "actions": []}
        _apply_rule_level_override(result, "II", "最大流量 3200m³/s")
        assert result["warning_level"] == "II"
        assert "[等级校正]" in result["reasoning"]
        assert "IV" in result["reasoning"] and "II" in result["reasoning"]
        assert result["reasoning"].endswith("水情平稳")

    def test_override_empty_rule_level_noop(self):
        result = {"warning_level": "IV", "reasoning": "x"}
        _apply_rule_level_override(result, "")
        assert result["warning_level"] == "IV"


# ====== 引用过滤不重生成 + 等级门重生成 + 进度事件 ======

def _mock_resp(content):
    from unittest.mock import MagicMock
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


class TestCitationFilterNoRegen:
    """引用校验失败只过滤，不触发整轮重生成（性能关键回归）。"""

    def test_bad_citation_does_not_regen(self):
        import json
        from unittest.mock import patch

        from agent.graph.synthesizer_node import _synth_via_llm

        # 1 条引用 quote 不在原文中 → 不应重生成（_call_synth_with_fallback 只调 1 次）
        meta = json.dumps({
            "warning_level": "II", "reasoning": "r", "actions": ["a"],
            "citations": [{"ref_id": 1, "quote": "编造的内容", "source_type": "web_search"}],
        })
        with patch("agent.graph.synthesizer_node._call_synth_with_fallback",
                   return_value=_mock_resp(meta)) as mock_call, \
             patch("agent.graph.synthesizer_node.get_llm_config", return_value={"model": "t"}), \
             patch("agent.graph.synthesizer_node.get_llm_client"):
            result, citations = _synth_via_llm("查询", {"get_hydrology": _hydrology()})
            assert mock_call.call_count == 1
            assert citations == []  # 无效引用被过滤
            assert result["warning_level"] == "II"  # 等级一致不受影响

    def test_level_mismatch_still_regens(self):
        import json
        from unittest.mock import patch

        from agent.graph.synthesizer_node import _synth_via_llm

        # 第 1 次等级错（III，规则判 II），第 2 次改正 → 应恰好调用 2 次
        bad = json.dumps({"warning_level": "III", "reasoning": "r", "actions": [], "citations": []})
        good = json.dumps({"warning_level": "II", "reasoning": "r", "actions": [], "citations": []})
        with patch("agent.graph.synthesizer_node._call_synth_with_fallback",
                   side_effect=[_mock_resp(bad), _mock_resp(good)]) as mock_call, \
             patch("agent.graph.synthesizer_node.get_llm_config", return_value={"model": "t"}), \
             patch("agent.graph.synthesizer_node.get_llm_client"):
            result, _ = _synth_via_llm("查询", {"get_hydrology": _hydrology()})
            assert mock_call.call_count == 2
            assert result["warning_level"] == "II"


class TestPhase1ProgressEvents:
    """Phase 1 生成/校验期间推送进度事件（消除 60-90s 静默）。"""

    def test_stream_yields_progress_before_meta(self):
        import json
        from unittest.mock import MagicMock, patch

        from agent.graph.synthesizer_node import _synth_via_llm_stream

        meta = json.dumps({"warning_level": "", "reasoning": "", "actions": [], "citations": []})
        chunks = []
        for tok in ("回答",):
            c = MagicMock()
            c.choices = [MagicMock()]
            c.choices[0].delta.content = tok
            chunks.append(c)
        end = MagicMock()
        end.choices = []
        chunks.append(end)

        with patch("agent.graph.synthesizer_node._call_synth_with_fallback",
                   return_value=_mock_resp(meta)), \
             patch("agent.graph.synthesizer_node.get_llm_config", return_value={"model": "t"}), \
             patch("agent.graph.synthesizer_node.get_llm_client") as mock_client:
            mock_client.return_value.with_options.return_value.chat.completions.create.return_value = iter(chunks)
            events = list(_synth_via_llm_stream("查询", {}))

        types = [e["type"] for e in events]
        # 进度事件先于 synth_meta 到达
        assert "synth_meta" in types
        meta_idx = types.index("synth_meta")
        progress = [e for e in events[:meta_idx] if e["type"] == "reasoning_step"]
        assert any("生成研判结论" in e["message"] for e in progress)
        assert any("校验引用" in e["message"] for e in progress)
        assert all(e["step"] == "synthesizer" for e in progress)
