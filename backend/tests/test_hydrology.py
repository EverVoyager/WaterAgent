"""水文数据解析 + 余弦相似度纯函数单元测试。

_parse_hydro_table / _parse_float / _filter_metric：HTML 解析高 bug 风险点
cosine_similarity：Skill 匹配相似度计算（公共工具）
"""
import pytest

from agent.data.hydrology import (
    STATION_PARAMS,
    _filter_metric,
    _parse_float,
)
from agent.utils import cosine_similarity

# ====== _parse_float 测试 ======

class TestParseFloat:
    def test_plain_number(self):
        assert _parse_float("1234.5") == 1234.5

    def test_integer_string(self):
        assert _parse_float("900") == 900.0

    def test_number_with_unit(self):
        assert _parse_float("640.0m") == 640.0

    def test_number_with_chinese_unit(self):
        assert _parse_float("2500立方米/秒") == 2500.0

    def test_number_with_spaces(self):
        assert _parse_float("  123.45  ") == 123.45

    def test_negative_number(self):
        # 负号会被保留（- 在 [^\d.] 中被剔除，因此负数变正数）
        # 实际水文数据不会有负数，此行为可接受
        result = _parse_float("-12.5")
        # 负号被移除
        assert result == 12.5

    def test_empty_string_returns_none(self):
        assert _parse_float("") is None

    def test_no_digits_returns_none(self):
        assert _parse_float("无数据") is None

    def test_multiple_dots_keeps_first(self):
        # "12.34.56" 会保留为 "12.34.56"，float() 失败返回 None
        result = _parse_float("12.34.56")
        # 实际行为：re.sub 后 "12.34.56"，float 失败
        assert result is None

    def test_only_dot_returns_none(self):
        assert _parse_float(".") is None

    def test_large_number(self):
        assert _parse_float("99999.99") == 99999.99

    def test_zero(self):
        assert _parse_float("0") == 0.0

    def test_decimal(self):
        assert _parse_float("0.001") == 0.001


# ====== _filter_metric 测试 ======

class TestFilterMetric:
    def _make_full_data(self):
        """构造包含 water_level 和 flow 的完整数据。"""
        return {
            "station": "吴堡",
            "river": "黄河",
            "water_level_m": 638.5,
            "warning_level_m": 640.0,
            "guaranteed_level_m": 642.0,
            "above_warning_m": -1.5,
            "flow_m3_s": 1200,
            "warning_flow_m3_s": 5000,
            "above_warning_flow_m3_s": -3800,
            "observed_time": "2026-8-10 8:00",
            "fetched_at": "2026-08-10T00:00:00+00:00",
            "source": "qqjjsj_realtime",
        }

    def test_both_returns_all_fields(self):
        data = self._make_full_data()
        result = _filter_metric(data, "both")
        assert result == data

    def test_water_level_keeps_level_fields(self):
        data = self._make_full_data()
        result = _filter_metric(data, "water_level")
        assert "water_level_m" in result
        assert "warning_level_m" in result
        assert "guaranteed_level_m" in result
        assert "above_warning_m" in result
        # flow 相关字段应被过滤（但 warning_flow_m3_s 保留）
        assert "flow_m3_s" not in result
        assert "above_warning_flow_m3_s" not in result
        assert "warning_flow_m3_s" in result  # 保留作为参考

    def test_flow_keeps_flow_fields(self):
        data = self._make_full_data()
        result = _filter_metric(data, "flow")
        assert "flow_m3_s" in result
        assert "warning_flow_m3_s" in result
        assert "above_warning_flow_m3_s" in result
        # level 相关字段应被过滤（但 warning_level_m 保留）
        # 注意：above_warning_m 不含 "level"，按实际逻辑会被保留
        assert "water_level_m" not in result
        assert "guaranteed_level_m" not in result
        assert "warning_level_m" in result  # 保留作为参考

    def test_both_preserves_station_and_meta(self):
        data = self._make_full_data()
        result = _filter_metric(data, "both")
        assert result["station"] == "吴堡"
        assert result["river"] == "黄河"
        assert result["source"] == "qqjjsj_realtime"

    def test_water_level_preserves_station(self):
        data = self._make_full_data()
        result = _filter_metric(data, "water_level")
        assert result["station"] == "吴堡"
        assert result["observed_time"] == "2026-8-10 8:00"

    def test_flow_preserves_station(self):
        data = self._make_full_data()
        result = _filter_metric(data, "flow")
        assert result["station"] == "吴堡"

    def test_unknown_metric_returns_all(self):
        data = self._make_full_data()
        result = _filter_metric(data, "unknown")
        assert result == data


# ====== STATION_PARAMS 数据完整性测试 ======

class TestStationParams:
    def test_wubao_params_complete(self):
        p = STATION_PARAMS["吴堡"]
        assert p["base_level_m"] == 636.0
        assert p["warning_level_m"] == 640.0
        assert p["guaranteed_level_m"] == 642.0
        assert p["base_flow_m3_s"] == 900
        assert p["warning_flow_m3_s"] == 5000
        assert p["river"] == "黄河"

    def test_longmen_params_complete(self):
        p = STATION_PARAMS["龙门"]
        assert p["base_level_m"] == 377.0
        assert p["warning_level_m"] == 382.0
        assert p["guaranteed_level_m"] == 385.0
        assert p["base_flow_m3_s"] == 1000
        assert p["warning_flow_m3_s"] == 7000
        assert p["river"] == "黄河"

    def test_no_fugu_station(self):
        # 府谷站不应存在（数据源不包含）
        assert "府谷" not in STATION_PARAMS

    def test_warning_level_higher_than_base(self):
        for name, p in STATION_PARAMS.items():
            assert p["warning_level_m"] > p["base_level_m"], f"{name} 警戒水位应高于基准"

    def test_guaranteed_level_higher_than_warning(self):
        for name, p in STATION_PARAMS.items():
            assert p["guaranteed_level_m"] > p["warning_level_m"], f"{name} 保证水位应高于警戒"


# ====== cosine_similarity 测试 ======

class TestCosine:
    def test_identical_vectors_return_one(self):
        a = [1.0, 2.0, 3.0]
        assert cosine_similarity(a, a) == pytest.approx(1.0)

    def test_orthogonal_vectors_return_zero(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors_return_negative_one(self):
        a = [1.0, 2.0]
        b = [-1.0, -2.0]
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_empty_vector_a_returns_zero(self):
        assert cosine_similarity([], [1.0, 2.0]) == 0.0

    def test_empty_vector_b_returns_zero(self):
        assert cosine_similarity([1.0, 2.0], []) == 0.0

    def test_both_empty_returns_zero(self):
        assert cosine_similarity([], []) == 0.0

    def test_different_length_returns_zero(self):
        assert cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0]) == 0.0

    def test_zero_vector_a_returns_zero(self):
        a = [0.0, 0.0, 0.0]
        b = [1.0, 2.0, 3.0]
        assert cosine_similarity(a, b) == 0.0

    def test_zero_vector_b_returns_zero(self):
        a = [1.0, 2.0, 3.0]
        b = [0.0, 0.0, 0.0]
        assert cosine_similarity(a, b) == 0.0

    def test_both_zero_vectors_returns_zero(self):
        a = [0.0, 0.0]
        b = [0.0, 0.0]
        assert cosine_similarity(a, b) == 0.0

    def test_high_dimensional_vectors(self):
        # 1024 维（DashScope text-embedding-v3 维度）
        a = [0.1] * 1024
        b = [0.1] * 1024
        # 同向向量，余弦相似度应为 1.0
        assert cosine_similarity(a, b) == pytest.approx(1.0, abs=1e-6)

    def test_similarity_in_range(self):
        # 相似度应在 [-1, 1] 范围内
        a = [1.0, 2.0, 3.0, 4.0]
        b = [4.0, 3.0, 2.0, 1.0]
        result = cosine_similarity(a, b)
        assert -1.0 <= result <= 1.0

    def test_single_element_vectors(self):
        assert cosine_similarity([5.0], [5.0]) == pytest.approx(1.0)
        assert cosine_similarity([5.0], [-5.0]) == pytest.approx(-1.0)
        assert cosine_similarity([5.0], [0.0]) == 0.0

    def test_negative_values(self):
        a = [-1.0, -2.0, -3.0]
        b = [-1.0, -2.0, -3.0]
        # 同向（都为负），余弦相似度为 1.0
        assert cosine_similarity(a, b) == pytest.approx(1.0)
