"""agent/utils.py 单元测试：parse_json_from_llm 多策略容错解析。"""

from agent.utils import (
    LEVEL_DESCRIPTION,
    WARNING_THRESHOLDS,
    now_iso,
    parse_json_from_llm,
)


class TestNowIso:
    def test_returns_iso_format_string(self):
        result = now_iso()
        assert isinstance(result, str)
        # ISO 8601 基本格式校验：包含日期 T 时间
        assert "T" in result
        assert len(result) >= 20

    def test_contains_timezone_utc(self):
        result = now_iso()
        # datetime.isoformat() with timezone.utc 会输出 +00:00
        assert result.endswith("+00:00")

    def test_consecutive_calls_monotonic(self):
        t1 = now_iso()
        t2 = now_iso()
        assert t2 >= t1


class TestLevelDescription:
    def test_covers_four_levels(self):
        assert set(LEVEL_DESCRIPTION.keys()) == {"I", "II", "III", "IV"}

    def test_contains_chinese_level_marker(self):
        assert "Ⅰ级" in LEVEL_DESCRIPTION["I"]
        assert "Ⅱ级" in LEVEL_DESCRIPTION["II"]
        assert "Ⅲ级" in LEVEL_DESCRIPTION["III"]
        assert "Ⅳ级" in LEVEL_DESCRIPTION["IV"]

    def test_contains_color(self):
        assert "红色" in LEVEL_DESCRIPTION["I"]
        assert "橙色" in LEVEL_DESCRIPTION["II"]
        assert "黄色" in LEVEL_DESCRIPTION["III"]
        assert "蓝色" in LEVEL_DESCRIPTION["IV"]


class TestWarningThresholds:
    def test_flow_thresholds_descending(self):
        assert WARNING_THRESHOLDS["flow_level1"] > WARNING_THRESHOLDS["flow_level2"]
        assert WARNING_THRESHOLDS["flow_level2"] > WARNING_THRESHOLDS["flow_level3"]

    def test_rain_thresholds_descending(self):
        assert WARNING_THRESHOLDS["rain_level1"] > WARNING_THRESHOLDS["rain_level2"]

    def test_threshold_values(self):
        assert WARNING_THRESHOLDS["flow_level1"] == 5000
        assert WARNING_THRESHOLDS["flow_level2"] == 3000
        assert WARNING_THRESHOLDS["flow_level3"] == 2000
        assert WARNING_THRESHOLDS["rain_level1"] == 100
        assert WARNING_THRESHOLDS["rain_level2"] == 50


class TestParseJsonFromLlm:
    """测试 4 级 JSON 解析策略。"""

    # ====== 策略 1：直接 json.loads ======

    def test_valid_json_direct_parse(self):
        content = '{"warning_level": "III", "flow": 2500}'
        result = parse_json_from_llm(content)
        assert result == {"warning_level": "III", "flow": 2500}

    def test_empty_string_returns_none(self):
        assert parse_json_from_llm("") is None

    def test_none_returns_none(self):
        assert parse_json_from_llm(None) is None

    def test_valid_json_with_nested_object(self):
        content = '{"data": {"station": "吴堡", "flow": 1234.5}, "ok": true}'
        result = parse_json_from_llm(content)
        assert result["data"]["station"] == "吴堡"
        assert result["ok"] is True

    def test_valid_json_array_at_top_level(self):
        # 顶层是数组，策略1直接解析即可
        content = '[1, 2, 3]'
        result = parse_json_from_llm(content)
        # 注意：策略3只找 {，若顶层是数组策略1已成功；若失败，策略3会返回 None
        assert result == [1, 2, 3]

    # ====== 策略 2：去 ```json ``` 包裹 ======

    def test_json_with_codeblock_prefix(self):
        content = '```json\n{"warning_level": "II"}\n```'
        result = parse_json_from_llm(content)
        assert result == {"warning_level": "II"}

    def test_json_with_plain_codeblock(self):
        content = '```\n{"warning_level": "I"}\n```'
        result = parse_json_from_llm(content)
        assert result == {"warning_level": "I"}

    def test_json_with_codeblock_no_closing(self):
        content = '```json\n{"warning_level": "IV"}'
        result = parse_json_from_llm(content)
        assert result == {"warning_level": "IV"}

    # ====== 策略 3：大括号配对提取 ======

    def test_json_with_leading_text(self):
        content = '根据分析，结果如下：{"warning_level": "III", "flow": 2200}'
        result = parse_json_from_llm(content)
        assert result == {"warning_level": "III", "flow": 2200}

    def test_json_with_trailing_text(self):
        content = '{"warning_level": "II"} 以上是研判结论。'
        result = parse_json_from_llm(content)
        assert result == {"warning_level": "II"}

    def test_json_with_leading_and_trailing_text(self):
        content = '研判结果：{"level": "I"} 请执行响应。'
        result = parse_json_from_llm(content)
        assert result == {"level": "I"}

    def test_json_first_object_extracted_when_multiple(self):
        # 多个 JSON 块时，提取第一个完整 {...}
        content = '{"first": 1} 一些文字 {"second": 2}'
        result = parse_json_from_llm(content)
        assert result == {"first": 1}

    def test_nested_braces_correctly_paired(self):
        content = '结果：{"outer": {"inner": "value"}, "ok": true}'
        result = parse_json_from_llm(content)
        assert result["outer"]["inner"] == "value"
        assert result["ok"] is True

    def test_no_brace_returns_none(self):
        assert parse_json_from_llm("纯文本无 JSON") is None

    # ====== 策略 4：修复单引号 + 尾随逗号 ======

    def test_single_quotes_converted_to_double(self):
        content = "{'warning_level': 'III', 'flow': 2500}"
        result = parse_json_from_llm(content)
        assert result == {"warning_level": "III", "flow": 2500}

    def test_trailing_comma_in_object(self):
        content = '{"warning_level": "III", "flow": 2500,}'
        result = parse_json_from_llm(content)
        assert result == {"warning_level": "III", "flow": 2500}

    def test_trailing_comma_in_nested_array(self):
        content = '{"actions": ["转移", "巡堤",], "level": "I"}'
        result = parse_json_from_llm(content)
        assert result["actions"] == ["转移", "巡堤"]
        assert result["level"] == "I"

    def test_single_quotes_and_trailing_comma_combined(self):
        content = "{'level': 'II', 'actions': ['a', 'b',],}"
        result = parse_json_from_llm(content)
        assert result["level"] == "II"
        assert result["actions"] == ["a", "b"]

    # ====== 边界情况 ======

    def test_empty_object(self):
        assert parse_json_from_llm("{}") == {}

    def test_empty_object_in_text(self):
        assert parse_json_from_llm("结果 {}") == {}

    def test_malformed_json_returns_none(self):
        assert parse_json_from_llm("{ broken json }") is None

    def test_only_opening_brace_returns_none(self):
        assert parse_json_from_llm("{ incomplete") is None

    def test_unicode_content(self):
        content = '{"station": "吴堡", "level": "Ⅰ级"}'
        result = parse_json_from_llm(content)
        assert result["station"] == "吴堡"
        assert result["level"] == "Ⅰ级"

    def test_large_json(self):
        # 构造较大的 JSON（模拟 LLM 返回的复杂预案）
        actions = [f"措施{i}" for i in range(20)]
        content = f'{{"warning_level": "I", "actions": {actions!r}}}'
        # 使用 json 模块构造有效 JSON
        import json
        content = json.dumps({"warning_level": "I", "actions": actions})
        result = parse_json_from_llm(content)
        assert len(result["actions"]) == 20
