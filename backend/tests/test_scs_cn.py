"""SCS-CN 降雨-径流模型单元测试。

覆盖：
- compute_runoff_depth：边界值与典型场景
- compute_peak_flow：合理化公式
- predict_runoff_scs：站点参数与过程线生成
"""
import pytest

from agent.hydrology.scs_cn import (
    STATION_BASIN,
    compute_peak_flow,
    compute_runoff_depth,
    predict_runoff_scs,
)


# ============ compute_runoff_depth ============

class TestComputeRunoffDepth:
    """径流深计算。"""

    def test_zero_rainfall_returns_zero(self):
        """零降雨应返回 0。"""
        assert compute_runoff_depth(0, 75) == 0.0

    def test_negative_rainfall_returns_zero(self):
        """负降雨应返回 0（防御）。"""
        assert compute_runoff_depth(-10, 75) == 0.0

    def test_below_initial_abstraction_returns_zero(self):
        """降雨量 <= 初损 Ia=0.2*S 时应返回 0。

        CN=75: S = 25400/75 - 254 = 84.67mm, Ia = 0.2*S = 16.93mm
        """
        # 16mm < 16.93mm，应不产流
        assert compute_runoff_depth(16, 75) == 0.0

    def test_just_above_initial_abstraction_produces_flow(self):
        """略大于初损应产流。"""
        # 20mm > 16.93mm，应有正值
        q = compute_runoff_depth(20, 75)
        assert q > 0
        # Q = (P-Ia)^2 / (P-Ia+S) = (20-16.93)^2 / (20-16.93+84.67) ≈ 9.4 / 87.7 ≈ 0.107
        assert 0 < q < 1.0

    def test_typical_storm_event(self):
        """典型暴雨 100mm + CN=75 应有合理径流深。"""
        # S=84.67, Ia=16.93, Q=(100-16.93)^2 / (100-16.93+84.67) = 6907.6 / 167.74 ≈ 41.18
        q = compute_runoff_depth(100, 75)
        assert 40 <= q <= 42

    def test_extreme_rainfall(self):
        """极端暴雨 300mm 应产生较大径流。"""
        q = compute_runoff_depth(300, 75)
        # Q = (300-16.93)^2 / (300-16.93+84.67) = 80121 / 367.74 ≈ 217.8
        assert q > 200

    def test_cn_clamping(self):
        """CN 超出 [1, 100] 应被夹紧。"""
        # CN=0 会被夹紧为 1，S = 25400-254=25146，Ia 巨大，小降雨无流
        q = compute_runoff_depth(10, 0)
        assert q == 0.0
        # CN=200 会被夹为 100，S=0，所有降雨都变径流（但分母 0 会导致异常）
        # 实现中 cn = min(100, max(1, cn))，CN=100 时 S=25400/100-254 = 0
        # Ia=0, Q = (P-0)^2 / (P-0+0) = P，应返回 P
        q = compute_runoff_depth(50, 200)
        assert q == 50.0

    def test_higher_cn_more_runoff(self):
        """CN 越高（土壤越不透水）径流越多。"""
        q_low_cn = compute_runoff_depth(80, 60)
        q_high_cn = compute_runoff_depth(80, 90)
        assert q_high_cn > q_low_cn

    def test_custom_lambda_coef(self):
        """自定义初损系数 lambda 应影响结果。"""
        # 默认 lambda=0.2
        q_default = compute_runoff_depth(30, 75)
        # lambda=0.05（更小初损，更早产流且更多）
        q_small_lambda = compute_runoff_depth(30, 75, lambda_coef=0.05)
        assert q_small_lambda >= q_default


# ============ compute_peak_flow ============

class TestComputePeakFlow:
    """洪峰流量计算。"""

    def test_zero_runoff_returns_base_flow(self):
        """零径流应返回基流。"""
        assert compute_peak_flow(0, 10000, 12, base_flow_m3_s=900) == 900.0

    def test_negative_runoff_returns_base_flow(self):
        """负径流应返回基流（防御）。"""
        assert compute_peak_flow(-5, 10000, 12, base_flow_m3_s=900) == 900.0

    def test_zero_area_returns_base_flow(self):
        """零面积应返回基流（防御）。"""
        assert compute_peak_flow(20, 0, 12, base_flow_m3_s=900) == 900.0

    def test_zero_tc_returns_base_flow(self):
        """零汇流时间应返回基流（防御）。"""
        assert compute_peak_flow(20, 10000, 0, base_flow_m3_s=900) == 900.0

    def test_typical_wubao_scenario(self):
        """吴堡站典型场景：A=10000, Tc=12, base=900, runoff=41mm。

        Q_peak = 0.208 * 10000 * 41 / 12 + 900 = 7106.7 + 900 ≈ 8007
        """
        q = compute_peak_flow(41, 10000, 12, base_flow_m3_s=900)
        assert 7900 <= q <= 8100

    def test_no_base_flow(self):
        """无基流时洪峰=纯径流贡献。"""
        # Q = 0.208 * 100 * 10 / 5 = 41.6
        q = compute_peak_flow(10, 100, 5, base_flow_m3_s=0)
        assert 41 <= q <= 42

    def test_larger_area_larger_peak(self):
        """面积越大洪峰越大。"""
        q_small = compute_peak_flow(20, 5000, 12)
        q_large = compute_peak_flow(20, 15000, 12)
        assert q_large > q_small

    def test_longer_tc_smaller_peak(self):
        """汇流时间越长洪峰越小（汇流平缓）。"""
        q_short = compute_peak_flow(20, 10000, 6)
        q_long = compute_peak_flow(20, 10000, 24)
        assert q_short > q_long


# ============ predict_runoff_scs ============

class TestPredictRunoffScs:
    """径流预测整体流程。"""

    def test_unsupported_station_raises(self):
        """不支持的站点应抛 RuntimeError。"""
        with pytest.raises(RuntimeError, match="不支持的站点"):
            predict_runoff_scs("北京", 50)

    def test_wubao_basic_output_structure(self):
        """吴堡站基本输出结构。"""
        result = predict_runoff_scs("吴堡", 100, lead_time_hours=24)
        assert result["station"] == "吴堡"
        assert result["model"] == "scs-cn-v0.1"
        assert result["source"] == "scs_cn_model"
        assert result["basin_area_km2"] == 433576
        assert result["effective_area_km2"] == 10000
        assert result["curve_number"] == 75
        assert result["tc_hours"] == 12
        assert result["base_flow_m3_s"] == 900
        assert result["total_rainfall_mm"] == 100.0
        assert result["runoff_depth_mm"] > 0
        assert result["peak_flow_m3_s"] > 900  # 大于基流
        assert isinstance(result["series"], list)
        assert len(result["series"]) > 0
        assert result["peak_time"] is not None
        assert "predicted_at" in result

    def test_longmen_uses_correct_params(self):
        """龙门站应使用龙门参数。"""
        result = predict_runoff_scs("龙门", 100)
        assert result["basin_area_km2"] == 497552
        assert result["effective_area_km2"] == 12000
        assert result["tc_hours"] == 14
        assert result["base_flow_m3_s"] == 1000

    def test_rainfall_series_aggregates_total(self):
        """rainfall_series 应聚合为总降雨量。"""
        series = [
            {"time": "2026-07-21T00:00:00Z", "rainfall_mm": 20},
            {"time": "2026-07-21T01:00:00Z", "rainfall_mm": 30},
            {"time": "2026-07-21T02:00:00Z", "rainfall_mm": 10},
        ]
        result = predict_runoff_scs("吴堡", 0, rainfall_series=series)
        assert result["total_rainfall_mm"] == 60.0

    def test_zero_rainfall_returns_base_flow_peak(self):
        """零降雨时洪峰等于基流。"""
        result = predict_runoff_scs("吴堡", 0)
        assert result["peak_flow_m3_s"] == 900
        assert result["runoff_depth_mm"] == 0.0

    def test_series_has_required_fields(self):
        """过程线应包含必要字段。"""
        result = predict_runoff_scs("吴堡", 80, lead_time_hours=12)
        for item in result["series"]:
            assert "time" in item
            assert "predicted_flow_m3_s" in item
            assert "ratio_of_peak" in item
            assert item["predicted_flow_m3_s"] >= result["base_flow_m3_s"]

    def test_stations_in_basin_dict(self):
        """STATION_BASIN 应包含吴堡和龙门两站。"""
        assert "吴堡" in STATION_BASIN
        assert "龙门" in STATION_BASIN
        for name, params in STATION_BASIN.items():
            assert "area_km2" in params
            assert "effective_area_km2" in params
            assert "cn" in params
            assert "tc_hours" in params
            assert "base_flow_m3_s" in params
            # 有效面积应远小于完整集水面积（单场暴雨覆盖子流域）
            assert params["effective_area_km2"] < params["area_km2"]
