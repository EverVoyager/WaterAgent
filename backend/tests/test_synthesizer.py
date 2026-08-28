"""综合研判（synthesizer）单元测试。

覆盖：
- compute_warning_level：基于 flow/rain/water_level 的等级判定
- get_actions_for_level：各级应急措施
- _extract_flow / _extract_rain / _extract_water_level_status：数据提取
"""

from agent.graph.synthesizer import (
    LEVEL_DESCRIPTION,
    _extract_flow,
    _extract_rain,
    _extract_water_level_status,
    compute_warning_level,
    get_actions_for_level,
)

# ============ _extract_flow ============

class TestExtractFlow:
    """从工具结果中提取最大流量。"""

    def test_empty_results(self):
        assert _extract_flow({}) == 0.0

    def test_flow_m3_s_field(self):
        results = {"hydrology": {"flow_m3_s": 537}}
        assert _extract_flow(results) == 537.0

    def test_peak_flow_m3_s_field(self):
        results = {"runoff": {"peak_flow_m3_s": 3000}}
        assert _extract_flow(results) == 3000.0

    def test_takes_max_across_multiple_tools(self):
        results = {
            "hydrology": {"flow_m3_s": 537},
            "runoff": {"peak_flow_m3_s": 4500},
        }
        assert _extract_flow(results) == 4500.0

    def test_series_max_flow(self):
        results = {
            "runoff": {
                "series": [
                    {"predicted_flow_m3_s": 1000},
                    {"predicted_flow_m3_s": 5000},
                    {"predicted_flow_m3_s": 3000},
                ]
            }
        }
        assert _extract_flow(results) == 5000.0

    def test_ignores_non_dict_values(self):
        results = {"meta": "string", "count": 42}
        assert _extract_flow(results) == 0.0


# ============ _extract_rain ============

class TestExtractRain:
    """从工具结果中提取最大降雨量。"""

    def test_empty_results(self):
        assert _extract_rain({}) == 0.0

    def test_total_rainfall_mm(self):
        results = {"weather": {"total_rainfall_mm": 75.5}}
        assert _extract_rain(results) == 75.5

    def test_max_hourly_rainfall_mm_estimated_24h(self):
        """max_hourly_rainfall_mm 应粗估为 24h 累计。"""
        results = {"weather": {"max_hourly_rainfall_mm": 5.0}}
        # 5 * 24 = 120
        assert _extract_rain(results) == 120.0

    def test_takes_max_across_fields(self):
        results = {
            "weather": {"total_rainfall_mm": 80, "max_hourly_rainfall_mm": 3}
        }
        # max(80, 3*24=72) = 80
        assert _extract_rain(results) == 80.0


# ============ _extract_water_level_status ============

class TestExtractWaterLevelStatus:
    """水位状态判断。"""

    def test_empty_results(self):
        assert _extract_water_level_status({}) == "unknown"

    def test_no_water_level_field(self):
        results = {"weather": {"temperature": 25}}
        assert _extract_water_level_status(results) == "unknown"

    def test_normal_level(self):
        results = {
            "hydrology": {
                "water_level_m": 630,
                "warning_level_m": 635,
                "guaranteed_level_m": 640,
            }
        }
        assert _extract_water_level_status(results) == "normal"

    def test_warning_level(self):
        results = {
            "hydrology": {
                "water_level_m": 636,
                "warning_level_m": 635,
                "guaranteed_level_m": 640,
            }
        }
        assert _extract_water_level_status(results) == "warning"

    def test_guaranteed_level(self):
        results = {
            "hydrology": {
                "water_level_m": 641,
                "warning_level_m": 635,
                "guaranteed_level_m": 640,
            }
        }
        assert _extract_water_level_status(results) == "guaranteed"

    def test_boundary_at_warning(self):
        """水位等于警戒水位应判为 warning。"""
        results = {
            "hydrology": {
                "water_level_m": 635,
                "warning_level_m": 635,
                "guaranteed_level_m": 640,
            }
        }
        assert _extract_water_level_status(results) == "warning"


# ============ compute_warning_level ============

class TestComputeWarningLevel:
    """预警等级综合计算。"""

    def test_empty_results_returns_iv(self):
        level, reasoning = compute_warning_level({})
        assert level == "IV"
        assert "Ⅳ级" in reasoning or "暂无足够数据" in reasoning

    def test_level_i_by_high_flow(self):
        """流量 >= 5000 触发 I 级。"""
        results = {"runoff": {"peak_flow_m3_s": 5500}}
        level, reasoning = compute_warning_level(results)
        assert level == "I"
        assert "Ⅰ级" in reasoning
        assert "5500" in reasoning

    def test_level_i_by_guaranteed_water_level(self):
        """水位超保证触发 I 级。"""
        results = {
            "hydrology": {
                "water_level_m": 641,
                "warning_level_m": 635,
                "guaranteed_level_m": 640,
            }
        }
        level, _ = compute_warning_level(results)
        assert level == "I"

    def test_level_i_by_heavy_rain(self):
        """24h 降雨 > 100 触发 I 级。"""
        results = {"weather": {"total_rainfall_mm": 150}}
        level, _ = compute_warning_level(results)
        assert level == "I"

    def test_level_ii_by_medium_flow(self):
        """流量 3000-5000 触发 II 级。"""
        results = {"runoff": {"peak_flow_m3_s": 4000}}
        level, reasoning = compute_warning_level(results)
        assert level == "II"
        assert "Ⅱ级" in reasoning

    def test_level_ii_by_warning_water_level(self):
        """水位超警戒触发 II 级。"""
        results = {
            "hydrology": {
                "water_level_m": 636,
                "warning_level_m": 635,
                "guaranteed_level_m": 640,
            }
        }
        level, _ = compute_warning_level(results)
        assert level == "II"

    def test_level_ii_by_medium_rain(self):
        """24h 降雨 50-100 触发 II 级。"""
        results = {"weather": {"total_rainfall_mm": 75}}
        level, _ = compute_warning_level(results)
        assert level == "II"

    def test_level_iii_by_low_flow(self):
        """流量 2000-3000 触发 III 级。"""
        results = {"runoff": {"peak_flow_m3_s": 2500}}
        level, _ = compute_warning_level(results)
        assert level == "III"

    def test_level_iv_for_normal_conditions(self):
        """正常水情触发 IV 级。"""
        results = {
            "hydrology": {"flow_m3_s": 537, "water_level_m": 630,
                          "warning_level_m": 635, "guaranteed_level_m": 640}
        }
        level, _ = compute_warning_level(results)
        assert level == "IV"

    def test_reasoning_includes_all_factors(self):
        """reasoning 应包含所有数据因素。"""
        results = {
            "hydrology": {"flow_m3_s": 537, "water_level_m": 630,
                          "warning_level_m": 635},
            "weather": {"total_rainfall_mm": 30},
        }
        _, reasoning = compute_warning_level(results)
        assert "537" in reasoning
        assert "30" in reasoning
        assert "normal" in reasoning


# ============ get_actions_for_level ============

class TestGetActionsForLevel:
    """各级应急措施。"""

    def test_level_i_has_5_actions(self):
        actions = get_actions_for_level("I")
        assert len(actions) == 5
        assert any("Ⅰ级" in a for a in actions)

    def test_level_ii_has_5_actions(self):
        actions = get_actions_for_level("II")
        assert len(actions) == 5
        assert any("Ⅱ级" in a for a in actions)

    def test_level_iii_has_4_actions(self):
        actions = get_actions_for_level("III")
        assert len(actions) == 4
        assert any("Ⅲ级" in a for a in actions)

    def test_level_iv_has_3_actions(self):
        actions = get_actions_for_level("IV")
        assert len(actions) == 3
        assert any("Ⅳ级" in a for a in actions)

    def test_unknown_level_falls_back_to_iv(self):
        actions = get_actions_for_level("unknown")
        assert len(actions) == 3  # 同 IV

    def test_custom_area_name(self):
        actions = get_actions_for_level("I", area="太原市")
        assert any("太原市" in a for a in actions)


# ============ LEVEL_DESCRIPTION ============

class TestLevelDescription:
    """等级描述常量。"""

    def test_all_levels_present(self):
        for level in ("I", "II", "III", "IV"):
            assert level in LEVEL_DESCRIPTION

    def test_chinese_color_in_description(self):
        assert "红色" in LEVEL_DESCRIPTION["I"]
        assert "橙色" in LEVEL_DESCRIPTION["II"]
        assert "黄色" in LEVEL_DESCRIPTION["III"]
        assert "蓝色" in LEVEL_DESCRIPTION["IV"]
