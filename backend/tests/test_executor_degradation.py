"""mock_executor 环境分级降级测试（P0-b）。

验证：
- development：真实实现不可用 → 降级到 mock
- production：关键工具真实实现不可用 → 硬失败（raise）
- production：透传工具（list_skills）不受影响
- 训练侧回放（overrides/seed）始终走 mock，不受环境影响
"""
from unittest.mock import patch

import pytest

from agent.tools.mock_executor import (
    _CRITICAL_TOOLS,
    _PASSTHROUGH_TOOLS,
    _should_hard_fail_on_real_failure,
    execute_tool,
)


class TestShouldHardFail:
    """_should_hard_fail_on_real_failure 判定逻辑。"""

    def test_dev_environment_never_hard_fails(self):
        """development 环境即使是关键工具也不硬失败。"""
        with patch("app.core.config.get_settings") as mock_s:
            mock_s.return_value.is_production = False
            for tool in _CRITICAL_TOOLS:
                assert _should_hard_fail_on_real_failure(tool) is False

    def test_production_critical_tool_hard_fails(self):
        """production 环境 + 关键工具 → 硬失败。"""
        with patch("app.core.config.get_settings") as mock_s:
            mock_s.return_value.is_production = True
            # 抽样验证几个关键工具
            for tool in ("get_hydrology", "search_regulation", "generate_plan"):
                assert _should_hard_fail_on_real_failure(tool) is True

    def test_production_passthrough_tool_not_hard_fails(self):
        """production 环境 + 透传工具（list_skills）→ 不硬失败（无 mock 可降级）。"""
        with patch("app.core.config.get_settings") as mock_s:
            mock_s.return_value.is_production = True
            for tool in _PASSTHROUGH_TOOLS:
                assert _should_hard_fail_on_real_failure(tool) is False

    def test_unknown_tool_not_hard_fails(self):
        """未知工具不在 _CRITICAL_TOOLS 中 → 不硬失败（由上层 ValueError 处理）。"""
        with patch("app.core.config.get_settings") as mock_s:
            mock_s.return_value.is_production = True
            assert _should_hard_fail_on_real_failure("nonexistent_tool") is False


class TestDevelopmentDegradation:
    """development 环境：真实实现失败 → 降级到 mock。"""

    def test_runtime_error_falls_back_to_mock_in_dev(self):
        """development 环境 RuntimeError → 降级 mock，source=mock。"""
        with patch("app.core.config.get_settings") as mock_s, \
             patch("agent.tools.real_executor.real_execute_tool") as mock_real:
            mock_s.return_value.is_production = False
            mock_real.side_effect = RuntimeError("Qdrant 索引未构建")
            result = execute_tool("search_regulation", {"query": "防洪法", "top_k": 3})
            assert result["source"] == "mock"
            assert result["query"] == "防洪法"

    def test_generic_exception_falls_back_to_mock_in_dev(self):
        """development 环境 Exception → 降级 mock。"""
        with patch("app.core.config.get_settings") as mock_s, \
             patch("agent.tools.real_executor.real_execute_tool") as mock_real:
            mock_s.return_value.is_production = False
            mock_real.side_effect = ValueError("unexpected")
            result = execute_tool("get_weather", {"location": "吕梁", "hours": 6})
            assert result["source"] == "mock"
            assert result["location"] == "吕梁"


class TestProductionHardFail:
    """production 环境：关键工具真实实现不可用 → 硬失败。"""

    def test_runtime_error_raises_in_production(self):
        """production 环境 RuntimeError → 直接 raise，不降级 mock。"""
        with patch("app.core.config.get_settings") as mock_s, \
             patch("agent.tools.real_executor.real_execute_tool") as mock_real:
            mock_s.return_value.is_production = True
            mock_real.side_effect = RuntimeError("Qdrant 索引未构建")
            with pytest.raises(RuntimeError, match="Qdrant"):
                execute_tool("search_regulation", {"query": "防洪法", "top_k": 3})

    def test_generic_exception_raises_in_production(self):
        """production 环境 Exception → 直接 raise。"""
        with patch("app.core.config.get_settings") as mock_s, \
             patch("agent.tools.real_executor.real_execute_tool") as mock_real:
            mock_s.return_value.is_production = True
            mock_real.side_effect = ConnectionError("AMAP API down")
            with pytest.raises(ConnectionError):
                execute_tool("get_weather", {"location": "吕梁", "hours": 6})

    def test_all_critical_tools_hard_fail_in_production(self):
        """所有关键工具在 production 环境都应硬失败（不漏网）。"""
        valid_args = {
            "get_weather": {"location": "吕梁", "hours": 6},
            "get_hydrology": {"station": "吴堡", "metric": "both"},
            "predict_runoff": {"station": "吴堡", "lead_time_hours": 24},
            "query_gis_terrain": {"analysis_type": "all"},
            "search_regulation": {"query": "防洪法", "top_k": 3},
            "web_search": {"query": "黄河汛情", "max_results": 3},
            "generate_plan": {
                "warning_level": "III",
                "affected_area": "吕梁",
                "population_at_risk": 10000,
            },
        }
        with patch("app.core.config.get_settings") as mock_s, \
             patch("agent.tools.real_executor.real_execute_tool") as mock_real:
            mock_s.return_value.is_production = True
            mock_real.side_effect = RuntimeError("dependency unavailable")
            for tool, args in valid_args.items():
                with pytest.raises(RuntimeError, match="dependency unavailable"):
                    execute_tool(tool, args)


class TestTrainingReplayUnaffected:
    """训练侧回放（overrides/seed）始终走 mock，不受环境影响。"""

    def test_overrides_uses_mock_in_production(self):
        """production 环境 + overrides → 走 mock（训练回放），不触发 real。"""
        with patch("app.core.config.get_settings") as mock_s, \
             patch("agent.tools.real_executor.real_execute_tool") as mock_real:
            mock_s.return_value.is_production = True
            # 即使 real 会失败，overrides 模式也不调用 real
            mock_real.side_effect = RuntimeError("should not be called")
            result = execute_tool(
                "get_weather",
                {"location": "吕梁", "hours": 6},
                overrides={"total_rainfall_mm": 999.9},
            )
            assert result["source"] == "mock"
            assert result["total_rainfall_mm"] == 999.9
            mock_real.assert_not_called()

    def test_seed_uses_mock_in_production(self):
        """production 环境 + seed → 走 mock（训练回放），不触发 real。"""
        with patch("app.core.config.get_settings") as mock_s, \
             patch("agent.tools.real_executor.real_execute_tool") as mock_real:
            mock_s.return_value.is_production = True
            mock_real.side_effect = RuntimeError("should not be called")
            result = execute_tool(
                "get_hydrology",
                {"station": "吴堡", "metric": "both"},
                seed=42,
            )
            assert result["source"] == "mock"
            assert result["station"] == "吴堡"
            mock_real.assert_not_called()


class TestRealSuccessUnaffected:
    """真实实现成功时不受环境影响。"""

    def test_real_success_in_production(self):
        """production 环境 real 成功 → 返回 real 结果。"""
        fake_result = {"data": "real", "source": "rag_qdrant"}
        with patch("app.core.config.get_settings") as mock_s, \
             patch("agent.tools.real_executor.real_execute_tool") as mock_real:
            mock_s.return_value.is_production = True
            mock_real.return_value = fake_result
            result = execute_tool("search_regulation", {"query": "防洪法", "top_k": 3})
            assert result == fake_result
