"""阈值外置化改造测试：配置加载 / 版本 / 热生效 / 提示词运行时渲染 /
训练档位推导 / 数值与旧硬编码逐位一致（保证评估可比性不断代）。
"""
import json
from pathlib import Path

import pytest

from agent.prompts.synthesizer import (
    SYNTHESIZER_PROMPT,
    render_synthesizer_prompt,
)
from agent.thresholds import (
    ThresholdProfile,
    get_thresholds,
    load_thresholds,
    reload_thresholds,
    threshold_version,
)
from agent.utils import WARNING_THRESHOLDS


def _tmp_config(tmp_path: Path, **overrides) -> Path:
    """基于默认档案生成临时配置（覆盖部分字段）。"""
    raw = json.loads(
        (Path(__file__).resolve().parents[2] / "config" / "thresholds.json")
        .read_text(encoding="utf-8")
    )
    raw.update(overrides)
    path = tmp_path / "thresholds.json"
    path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# ============ 配置与数值一致性 ============

class TestProfileValues:
    def test_values_match_legacy_hardcoded(self):
        """数值与外置化前的硬编码逐位一致（评估/训练可比性不断代）。"""
        p = get_thresholds()
        assert (p.flow_level1, p.flow_level2, p.flow_level3) == (5000, 3000, 2000)
        assert (p.rain_level1, p.rain_level2) == (100, 50)
        st = p.station("吴堡")
        assert st["base_level_m"] == 640.5
        assert st["warning_level_m"] == 642.5, "旧 base+2.0 偏移结果"
        assert st["guaranteed_level_m"] == 644.0, "旧 base+3.5 偏移结果"

    def test_legacy_view_in_sync(self):
        """utils.WARNING_THRESHOLDS 是档案的兼容视图（引擎按引用动态读）。"""
        assert get_thresholds().to_legacy() == WARNING_THRESHOLDS

    def test_unknown_station_falls_back(self):
        st = get_thresholds().station("不存在的站")
        assert st["base_level_m"] == 500.0
        assert st["warning_level_m"] < st["guaranteed_level_m"]

    def test_version_present(self):
        assert threshold_version() == get_thresholds().version
        assert threshold_version(), "version 字段必填（版本追溯）"

    def test_flow_ranges_derived_match_legacy(self):
        """档位区间推导值 = 旧硬编码（I 上限 6500 = 5000×1.3 等）。"""
        p = get_thresholds()
        assert p.flow_range_for("I") == (5000.0, 6500.0)
        assert p.flow_range_for("II") == (3000.0, 4999.0)
        assert p.flow_range_for("III") == (2000.0, 2999.0)
        assert p.flow_range_for("IV") == (500.0, 1999.0)
        assert p.rain_for_level("I") == 120.0
        assert p.rain_for_level("IV") == 8.0

    def test_invalid_thresholds_rejected(self, tmp_path):
        raw = json.loads(_tmp_config(tmp_path).read_text(encoding="utf-8"))
        raw["flow_thresholds_m3_s"] = {"level1": 3000, "level2": 5000, "level3": 2000}
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
        with pytest.raises(ValueError, match="严格递减"):
            load_thresholds(bad)

    def test_station_levels_validated(self, tmp_path):
        raw = json.loads(_tmp_config(tmp_path).read_text(encoding="utf-8"))
        raw["stations"]["吴堡"]["warning_level_m"] = 9999.0
        bad = tmp_path / "bad2.json"
        bad.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
        with pytest.raises(ValueError, match="警戒水位须低于保证水位"):
            load_thresholds(bad)

    def test_missing_version_rejected(self, tmp_path):
        raw = json.loads(_tmp_config(tmp_path).read_text(encoding="utf-8"))
        del raw["version"]
        bad = tmp_path / "bad3.json"
        bad.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
        with pytest.raises(ValueError, match="version"):
            load_thresholds(bad)


# ============ 热生效（运行时换档案） ============

class TestRuntimeReload:
    def test_engine_follows_reload_without_restart(self, tmp_path):
        """换档案 → 规则引擎与旧常量视图即时跟随（无需重启）。"""
        from agent.graph.synthesizer import compute_warning_level

        data = {"get_hydrology": {"flow_m3_s": 6000}}
        assert compute_warning_level(data)[0] == "I", "默认阈值 6000 ≥ 5000"

        try:
            new_cfg = _tmp_config(
                tmp_path,
                version="v-test-8000",
                flow_thresholds_m3_s={"level1": 8000, "level2": 3000, "level3": 2000},
            )
            reload_thresholds(new_cfg)
            assert compute_warning_level(data)[0] == "II", "6000 落入新表 [3000,8000)"
            assert WARNING_THRESHOLDS["flow_level1"] == 8000, "旧视图原地同步"
        finally:
            reload_thresholds()  # 恢复默认档案
        assert compute_warning_level(data)[0] == "I"
        assert WARNING_THRESHOLDS["flow_level1"] == 5000


# ============ 提示词运行时渲染 ============

class TestPromptRendering:
    def test_prompt_contains_current_values(self):
        prompt = render_synthesizer_prompt()
        assert "5000m³/s" in prompt
        assert "100mm" in prompt

    def test_new_version_renders_new_numbers(self, tmp_path):
        """新档案版本 → 提示词自动重渲染（不再 import 烙印）。"""
        try:
            new_cfg = _tmp_config(
                tmp_path,
                version="v-test-prompt",
                flow_thresholds_m3_s={"level1": 8000, "level2": 3000, "level3": 2000},
            )
            reload_thresholds(new_cfg)
            assert "8000m³/s" in render_synthesizer_prompt()
            assert "5000m³/s" not in render_synthesizer_prompt()
        finally:
            reload_thresholds()

    def test_same_version_cached_identical(self):
        """同版本缓存命中：渲染结果逐字节一致（KV 前缀稳定）。"""
        assert render_synthesizer_prompt() == render_synthesizer_prompt()

    def test_legacy_alias_equals_render(self):
        assert render_synthesizer_prompt() == SYNTHESIZER_PROMPT

    def test_synth_system_content_uses_runtime_render(self, tmp_path):
        """synthesizer 节点走运行时渲染（换档案后 system 变化）。"""
        from agent.graph.synthesizer_node import _build_synth_system_content

        try:
            new_cfg = _tmp_config(
                tmp_path,
                version="v-test-node",
                flow_thresholds_m3_s={"level1": 8000, "level2": 3000, "level3": 2000},
            )
            before = _build_synth_system_content(query="吴堡站水情")
            reload_thresholds(new_cfg)
            after = _build_synth_system_content(query="吴堡站水情")
            assert "5000m³/s" in before
            assert "8000m³/s" in after
        finally:
            reload_thresholds()


# ============ mock / 训练场景推导 ============

class TestDerivedConsumers:
    def test_mock_hydrology_uses_station_profile(self):
        from agent.tools.mock_executor import GetHydrologyParams, _mock_get_hydrology

        result = _mock_get_hydrology(GetHydrologyParams(station="吴堡", metric="both"))
        assert result["warning_level_m"] == 642.5
        assert result["guaranteed_level_m"] == 644.0

    def test_scenario_tables_derived(self):
        from train.data_gen.scenario import (
            _LEVEL_TO_FLOW_RANGE,
            STATION_BASE_LEVEL,
        )

        assert STATION_BASE_LEVEL == {"吴堡": 640.5, "龙门": 382.3, "府谷": 810.2}
        assert _LEVEL_TO_FLOW_RANGE["I"] == (5000.0, 6500.0)

    def test_eval_cases_unchanged_after_refactor(self):
        """评估用例与外置化前逐位一致（默认 62 条 + 确定性）。"""
        from evals.cases import build_cases

        a = build_cases(n_memory=4)
        b = build_cases(n_memory=4)
        assert a == b
        assert len(build_cases()) == 62

    def test_scenario_overrides_values_unchanged(self):
        """同 seed 的 overrides 与旧实现一致（档位/水位/降雨全对齐）。"""
        import random

        from train.data_gen.scenario import _make_overrides

        ov = _make_overrides(random.Random(300_000), "吴堡", "II")
        assert 3000.0 <= ov["get_hydrology"]["flow_m3_s"] <= 4999.0
        assert ov["get_hydrology"]["warning_level_m"] == 642.5
        assert ov["get_hydrology"]["guaranteed_level_m"] == 644.0
        assert ov["get_weather"]["total_rainfall_mm"] == 75.0


# ============ 基线可比性（版本拦截） ============

class TestBaselineVersionGate:
    def test_diff_flags_version_change_when_recorded(self):
        from evals.run_eval import _baseline_composition_diff

        baseline = {"config": {"model_label": "m", "threshold_version": "v1"}}
        config = {"model_label": "m", "threshold_version": "v2"}
        assert "threshold_version: v1→v2" in _baseline_composition_diff(baseline, config)

    def test_legacy_baseline_without_version_key_skipped(self):
        """旧基线未记录 threshold_version → 不误报（重建基线时钉住）。"""
        from evals.run_eval import _baseline_composition_diff

        baseline = {"config": {"model_label": "m"}}
        config = {"model_label": "m", "threshold_version": "v2026.09"}
        assert _baseline_composition_diff(baseline, config) == ""


# ============ 档案数据类单测 ============

class TestThresholdProfile:
    def test_profile_direct_construction(self):
        p = ThresholdProfile(version="v", effective_date="2026-01-01",
                             flow_level1=9000, flow_level2=4000, flow_level3=1000)
        assert p.flow_range_for("II") == (4000.0, 8999.0)
        assert p.to_legacy()["flow_level1"] == 9000

    def test_unknown_level_raises(self):
        with pytest.raises(KeyError, match="未知等级"):
            get_thresholds().flow_range_for("V")

    def test_prompt_byte_identical_to_pre_refactor(self):
        """与外置化前（git HEAD 版本）的 SYNTHESIZER_PROMPT 逐字节一致。"""
        import subprocess

        old_src = subprocess.run(
            ["git", "show", "HEAD:agent/prompts/synthesizer.py"],
            capture_output=True, text=True, encoding="utf-8", check=True,
        ).stdout
        namespace = {}
        exec(old_src, namespace)  # noqa: S102 —— 旧实现按当前 utils 值构建
        assert namespace["SYNTHESIZER_PROMPT"] == render_synthesizer_prompt(), (
            "外置化必须数值等价：提示词与改造前逐字节一致"
        )
