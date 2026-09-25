"""工具返回值运行时校验（executor 出口闸）测试。

覆盖三层：
1. 全部 7 个 mock 工具的输出必须通过自身返回值模型——mock 与 schema 的
   契约一致性在此锁定，实现漂移先于 62 条评估在本测试变红；
2. validate_tool_result 对缺失字段 / 类型错误 / 非 dict 的检出；
3. nodes._execute_one_tool 接线：校验失败 → error 置位、result 清空，
   坏数据不进 tool_results（fail-fast，交给 planner 重规划）。
"""
import pytest

from agent.tools.mock_executor import clear_replay_context, execute_tool
from agent.tools.schemas import validate_tool_result

# 每个工具的最小合法入参（list_skills 走 real_executor 依赖技能库，
# 不做 mock 回放，单列合成 dict 测试）
_TOOL_ARGS = {
    "get_weather": {"location": "吴堡水文站", "hours": 24},
    "get_hydrology": {"station": "吴堡", "metric": "both"},
    "predict_runoff": {"station": "吴堡", "lead_time_hours": 24},
    "query_gis_terrain": {"analysis_type": "all"},
    "search_regulation": {"query": "黄河防汛条例", "top_k": 3},
    "web_search": {"query": "吴堡水情", "max_results": 3},
    "generate_plan": {"warning_level": "II", "affected_area": "吕梁市",
                      "population_at_risk": 1000},
}


@pytest.fixture(autouse=True)
def _cleanup_replay():
    yield
    clear_replay_context()


class TestMockOutputsConform:
    @pytest.mark.parametrize("tool_name", sorted(_TOOL_ARGS))
    def test_mock_result_passes_schema(self, tool_name):
        # 显式 overrides/seed 强制 mock 分支：无外部依赖、确定性
        result = execute_tool(tool_name, _TOOL_ARGS[tool_name], overrides={}, seed=7)
        assert validate_tool_result(tool_name, result) == ""

    def test_eval_replay_overrides_still_conform(self):
        """评估 overrides 注入后的结果也必须过闸（口径与 62 条评估一致）。"""
        result = execute_tool(
            "get_hydrology", {"station": "吴堡", "metric": "both"},
            overrides={"water_level_m": 807.5, "flow_m3_s": 6200.0}, seed=11,
        )
        assert validate_tool_result("get_hydrology", result) == ""

    def test_list_skills_result_model(self):
        assert validate_tool_result(
            "list_skills",
            {"skills": [{"name": "s"}], "total": 1, "queried_at": "2026-01-01"},
        ) == ""


class TestValidateToolResult:
    def test_missing_required_field(self):
        v = validate_tool_result(
            "get_weather", {"location": "吴堡", "hours": 6})  # 缺降雨总量与序列
        assert "total_rainfall_mm" in v

    def test_wrong_type(self):
        v = validate_tool_result(
            "get_weather",
            {"location": "吴堡", "hours": 6, "total_rainfall_mm": "很大", "series": []},
        )
        assert v

    def test_non_dict_result(self):
        assert validate_tool_result("get_weather", ["not", "a", "dict"])

    def test_unknown_tool_passes(self):
        assert validate_tool_result("no_such_tool", {"anything": 1}) == ""

    def test_optional_variant_fields_tolerated(self):
        """metric=water_level 时无 flow 字段、analysis_type=slope 时无淹没段，均合法。"""
        assert validate_tool_result(
            "get_hydrology",
            {"station": "吴堡", "water_level_m": 805.0, "warning_level_m": 806.0},
        ) == ""
        assert validate_tool_result(
            "query_gis_terrain",
            {"bbox": "110.7,37.4,111.2,37.8", "analysis_type": "slope",
             "slope": {"mean_degree": 8.5}},
        ) == ""

    def test_extra_fields_ignored(self):
        """评估 overrides 注入的附加键不得误伤（Pydantic 默认忽略多余字段）。"""
        assert validate_tool_result(
            "predict_runoff",
            {"station": "吴堡", "lead_time_hours": 24, "peak_flow_m3_s": 7500.0,
             "series": [], "custom_trap_field": "x"},
        ) == ""


class TestExecutorExitGate:
    def _run_one(self, monkeypatch, tool_result):
        import agent.graph.nodes as nodes_mod

        monkeypatch.setattr(nodes_mod, "_cached_execute_tool", lambda name, args: tool_result)
        return nodes_mod._execute_one_tool(
            {"name": "get_weather", "arguments": {"location": "吴堡", "hours": 6}},
            idx=0, weather_rainfall_mm=0.0, weather_series=None,
            round_num=1, existing_keys=set(), duplicate_names=set(),
        )

    def test_schema_violation_becomes_tool_error(self, monkeypatch):
        rec = self._run_one(monkeypatch, {"location": "吴堡"})  # 缺 total_rainfall_mm/series
        assert rec["error"].startswith("result_schema:")
        assert rec["result"] == {}

    def test_valid_result_passes_through(self, monkeypatch):
        good = {"location": "吴堡", "hours": 6, "total_rainfall_mm": 12.5,
                "series": [{"time": "t", "rainfall_mm": 1.0}]}
        rec = self._run_one(monkeypatch, good)
        assert rec["error"] == ""
        assert rec["result"] is good
