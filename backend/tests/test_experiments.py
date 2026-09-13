"""量化实验体系单元测试（离线，不依赖 LLM/MySQL/Qdrant）。

覆盖：消融框架开关（env/patch 生效与恢复）、新用例类型构建（记忆/压缩/
工具边界的确定性与设计约束）、runner 针检查与 history 透传、覆盖矩阵
守门、五个实验模块的编排逻辑（注入假 run_fn/run_case 验证对照计算）、
报告量化声明表渲染。
"""
from unittest.mock import patch

import pytest

from agent.tools.mock_executor import clear_replay_context
from app.core.config import get_settings
from evals import cases as cases_mod
from evals import coverage as coverage_mod
from evals import runner as runner_mod
from evals.case_sets import get_experiment_cases
from evals.cases import EVAL_SEED_BASE, EvalCase, build_cases
from evals.experiments.base import (
    Toggle,
    run_toggle_ablation,
    summarize_contrast,
    toggles_applied,
)


@pytest.fixture(autouse=True)
def _ensure_replay_cleared():
    yield
    clear_replay_context()


def _record(case_id: str, passed: bool, **extra) -> dict:
    record = {
        "case_id": case_id, "case_type": "memory", "capabilities": [],
        "passed": passed, "error": "", "checks": {}, "latency_s": 0.1,
    }
    record.update(extra)
    return record


def _memory_case(cid: str, needle: str = "638.26") -> EvalCase:
    return EvalCase(
        case_id=cid, case_type="memory", query="q", seed=EVAL_SEED_BASE,
        memory_payload={"longterm": f"警戒水位 {needle}"},
        needle_substrings=(needle,),
        expected_intent=None, capabilities=("memory_recall",),
    )


# ============ 消融框架（base.py） ============

class TestToggle:
    def test_invalid_kind_rejected(self):
        with pytest.raises(ValueError, match="未知开关类型"):
            Toggle(kind="magic", target="X")

    def test_env_toggle_takes_effect_and_restores(self):
        original = get_settings().SELF_EVOLUTION_ENABLED
        toggle = Toggle(kind="env", target="SELF_EVOLUTION_ENABLED", off_value=False)
        with toggles_applied([toggle]):
            assert get_settings().SELF_EVOLUTION_ENABLED is False
        assert original == get_settings().SELF_EVOLUTION_ENABLED

    def test_env_toggle_unknown_target_raises(self):
        toggle = Toggle(kind="env", target="NOT_A_REAL_SETTING", off_value=False)
        with pytest.raises(AttributeError, match="NOT_A_REAL_SETTING"), \
                toggles_applied([toggle]):
            pass

    def test_patch_toggle_return_value(self):
        toggle = Toggle(kind="patch",
                        target="agent.memory.build_longterm_section", off_value="")
        import agent.memory as memory_pkg
        with toggles_applied([toggle]):
            assert memory_pkg.build_longterm_section() == ""
        assert callable(memory_pkg.build_longterm_section)
        assert memory_pkg.build_longterm_section is not None

    def test_patch_toggle_off_fn(self):
        toggle = Toggle(kind="patch",
                        target="agent.graph.runner._compact_history_entry",
                        off_fn=lambda history, *a, **kw: history)
        history = [{"role": "user", "content": "x"}]
        import agent.graph.runner as runner_pkg
        with toggles_applied([toggle]):
            assert runner_pkg._compact_history_entry(list(history)) == history

    def test_empty_toggles_noop(self):
        with toggles_applied([]):
            pass  # 不抛错即可


class TestSummarizeContrast:
    def test_delta_relative_and_significance(self):
        # n=60：25pp 分差超 2×组合SE（n=20 时 0.25 < 2×0.148=0.296，不显著）
        treated = [_record(f"c{i}", i < 45) for i in range(60)]   # 75%
        baseline = [_record(f"c{i}", i < 30) for i in range(60)]  # 50%
        result = summarize_contrast(treated, baseline)
        assert result["delta"] == pytest.approx(0.25)
        assert result["relative_lift"] == pytest.approx(0.5)
        assert result["significant"] is True
        assert result["treated_rate"]["n"] == 60
        assert result["baseline_rate"]["p"] == 0.5

    def test_within_noise_band_not_significant(self):
        treated = [_record(f"c{i}", i < 9) for i in range(10)]
        baseline = [_record(f"c{i}", i < 8) for i in range(10)]
        result = summarize_contrast(treated, baseline)
        assert result["significant"] is False

    def test_flips_pairwise(self):
        treated = [_record("a", True), _record("b", False), _record("c", True)]
        baseline = [_record("a", False), _record("b", True), _record("c", True)]
        result = summarize_contrast(treated, baseline)
        assert result["flipped_to_pass"] == ["a"]
        assert result["flipped_to_fail"] == ["b"]

    def test_check_key_variant(self):
        treated = [
            _record("a", True, checks={"needle_found": True}),
            _record("b", True, checks={"needle_found": False}),
        ]
        baseline = [
            _record("a", True, checks={"needle_found": False}),
            _record("b", True, checks={"needle_found": None}),
        ]
        result = summarize_contrast(treated, baseline, check_key="needle_found")
        # baseline 只有 1 条适用（None 不计入）
        assert result["treated_rate"]["n"] == 2
        assert result["baseline_rate"]["n"] == 1
        assert result["check_key"] == "needle_found"

    def test_zero_baseline_relative_lift_none(self):
        treated = [_record("a", True)]
        baseline = [_record("a", False)]
        result = summarize_contrast(treated, baseline)
        assert result["relative_lift"] is None
        assert result["delta"] == 1.0

    def test_empty_records(self):
        result = summarize_contrast([], [])
        assert result["treated_rate"] is None
        assert result["delta"] is None
        assert result["significant"] is None


class TestRunToggleAblation:
    def test_orchestration_with_fake_run_fn(self):
        """run_fn 两次调用：第二次（开关应用中）应看到被 patch 的函数。"""
        calls = []

        def run_fn(cases, label):
            in_ctx = get_settings().SELF_EVOLUTION_ENABLED
            calls.append(in_ctx)
            return [_record(c.case_id, not in_ctx) for c in cases]  # 关=通过

        toggle = Toggle(kind="env", target="SELF_EVOLUTION_ENABLED", off_value=False)
        result = run_toggle_ablation(
            [_memory_case("m1"), _memory_case("m2")], run_fn, [toggle],
        )
        assert calls == [True, False]  # 先机制开，后机制关
        assert result["treated_rate"]["p"] == 0.0
        assert result["baseline_rate"]["p"] == 1.0


# ============ 新用例类型（cases.py 扩容） ============

class TestNewCaseTypes:
    def test_default_composition_unchanged(self):
        """默认 62 条不变——基线可比性（回归门禁组合一致性）。"""
        cases = build_cases()
        assert len(cases) == 62
        assert not any(c.case_type in ("memory", "compression", "tool_edge")
                       for c in cases)

    def test_memory_cases_design(self):
        cases = build_cases(n_memory=24)
        mem = [c for c in cases if c.case_type == "memory"]
        assert len(mem) == 24
        assert all(c.memory_payload and c.needle_substrings for c in mem)
        assert all(c.expected_intent is None for c in mem)
        # 知识更新子类：payload 旧值 + history 更正 + forbidden 旧值
        update_cases = [c for c in mem if c.forbidden_substrings]
        assert update_cases, "知识更新子类应带 forbidden_substrings"
        for c in update_cases:
            assert c.history, "知识更新子类应带更正会话历史"
            assert any(f in c.memory_payload["semantic"]
                       for f in c.forbidden_substrings)
            assert all(n in c.history[0]["content"] for n in c.needle_substrings)
        # 种子隔离
        assert all(300_000 <= c.seed < 400_000 for c in mem)

    def test_compression_cases_exceed_token_budget(self):
        from agent.graph.context_compact import estimate_tokens
        budget = get_settings().HISTORY_MAX_TOKENS
        cases = build_cases(n_compression=12)
        comp = [c for c in cases if c.case_type == "compression"]
        assert len(comp) == 12
        for c in comp:
            tokens = sum(estimate_tokens(m.get("content", "")) for m in c.history)
            assert tokens > budget, f"{c.case_id} 历史 {tokens} 未超预算 {budget}"
            # 针必须在历史里（早轮埋下）
            joined = "".join(m.get("content", "") for m in c.history)
            assert all(n in joined for n in c.needle_substrings)

    def test_compression_needle_not_in_recent_rounds(self):
        keep = get_settings().HISTORY_KEEP_RECENT_ROUNDS
        cases = build_cases(n_compression=12)
        for c in cases:
            recent = "".join(
                m.get("content", "") for m in c.history[-(keep * 2):]
            )
            assert not any(n in recent for n in c.needle_substrings)

    def test_tool_edge_composition(self):
        cases = build_cases(n_tool_edge=20)
        edge = [c for c in cases if c.case_type == "tool_edge"]
        assert len(edge) == 20
        ids = [c.case_id for c in edge]
        assert sum(i.startswith("edge-oop") for i in ids) == 4
        assert sum(i.startswith("edge-adj") for i in ids) == 4
        assert sum(i.startswith("edge-meta") for i in ids) == 4
        assert sum(i.startswith("edge-multi") for i in ids) == 4
        assert sum(i.startswith("edge-seq") for i in ids) == 4
        # 超范围用例：意图不检查 + 严格工具白名单
        oop = [c for c in edge if c.case_id.startswith("edge-oop")]
        assert all(c.expected_intent is None and c.allowed_tools == frozenset()
                   for c in oop)

    def test_new_types_deterministic(self):
        kwargs = {"n_memory": 8, "n_compression": 6, "n_tool_edge": 12}
        a = build_cases(**kwargs)
        b = build_cases(**kwargs)
        assert a == b

    def test_seed_isolation_still_holds(self):
        build_cases(n_memory=24, n_compression=12, n_tool_edge=20)
        cases_mod.assert_seed_isolation()


class TestCaseSets:
    def test_experiment_compositions_fixed(self):
        for name in ("memory", "compression", "self_evolution", "kv_cache", "core"):
            cases = get_experiment_cases(name)
            assert cases, f"{name} 案例集为空"
        mem = get_experiment_cases("memory")
        assert len(mem) == 24 and all(c.case_type == "memory" for c in mem)
        comp = get_experiment_cases("compression")
        assert len(comp) == 12

    def test_unknown_experiment_raises(self):
        with pytest.raises(KeyError, match="未知实验"):
            get_experiment_cases("nope")


# ============ runner 针检查与 history 透传 ============

class TestRunnerNeedle:
    def _run_with_fake_agent(self, case, fake_result):
        with patch("evals.runner.run_graph_agent", return_value=fake_result):
            return runner_mod.run_case(case)

    def test_history_forwarded_to_agent(self):
        case = _memory_case("m1")
        case.history = [{"role": "user", "content": "earlier"}]
        with patch("evals.runner.run_graph_agent",
                   return_value={"final_answer": "警戒水位 638.26",
                                 "warning_level": "", "intent": "chitchat",
                                 "tool_calls": [], "citations": [], "rounds": 1}
                   ) as mock_agent:
            runner_mod.run_case(case)
        _, kwargs = mock_agent.call_args
        assert kwargs.get("history") == [{"role": "user", "content": "earlier"}]

    def test_needle_found_pass(self):
        case = _memory_case("m1")
        record = self._run_with_fake_agent(case, {
            "final_answer": "吴堡站警戒水位为 638.26 米。", "warning_level": "",
            "intent": "chitchat", "tool_calls": [], "citations": [], "rounds": 1,
        })
        assert record["checks"]["needle_found"] is True
        assert record["passed"] is True

    def test_needle_missing_fails(self):
        case = _memory_case("m1")
        record = self._run_with_fake_agent(case, {
            "final_answer": "这个我不太确定。", "warning_level": "",
            "intent": "chitchat", "tool_calls": [], "citations": [], "rounds": 1,
        })
        assert record["checks"]["needle_found"] is False
        assert record["passed"] is False

    def test_forbidden_substring_fails(self):
        case = _memory_case("m1", needle="6500")
        case.forbidden_substrings = ("6200",)
        record = self._run_with_fake_agent(case, {
            "final_answer": "吴堡站警戒流量还是 6200。", "warning_level": "",
            "intent": "chitchat", "tool_calls": [], "citations": [], "rounds": 1,
        })
        assert record["checks"]["needle_found"] is False

    def test_intent_none_skips_check(self):
        case = _memory_case("m1")
        record = self._run_with_fake_agent(case, {
            "final_answer": "警戒水位 638.26 米", "warning_level": "",
            "intent": "chitchat", "tool_calls": [], "citations": [], "rounds": 1,
        })
        assert record["checks"]["intent_ok"] is None

    def test_error_path_needle_counts_as_fail(self):
        case = _memory_case("m1")
        with patch("evals.runner.run_graph_agent",
                   side_effect=RuntimeError("boom")):
            record = runner_mod.run_case(case)
        assert record["checks"]["needle_found"] is False
        assert record["passed"] is False

    def test_case_pass_new_types(self):
        assert runner_mod._case_pass("memory", {"needle_found": True}) is True
        assert runner_mod._case_pass("memory", {"needle_found": False}) is False
        assert runner_mod._case_pass(
            "compression", {"needle_found": None}) is True  # 无针不适用
        assert runner_mod._case_pass(
            "tool_edge", {"tool_precision": False}) is False


# ============ 覆盖矩阵（coverage.py） ============

class TestCoverage:
    def test_full_set_no_gaps(self):
        cases = build_cases(n_memory=24, n_compression=12, n_tool_edge=20)
        gaps = coverage_mod.coverage_gaps(cases)
        assert gaps == [], f"覆盖缺口: {gaps}"

    def test_gap_detected_for_underbudget_history(self):
        cases = build_cases(n_compression=12)
        comp = next(c for c in cases if c.case_type == "compression")
        comp.history = comp.history[:4]  # 砍到预算以下
        gaps = coverage_mod.coverage_gaps(cases, expected_types=("compression",))
        assert any("未超预算" in g for g in gaps)

    def test_gap_detected_missing_level(self):
        cases = build_cases()
        for c in cases:
            if c.case_type == "business" and c.expected_level == "I":
                c.expected_level = "II"
        gaps = coverage_mod.coverage_gaps(cases, expected_types=("business",))
        assert any("等级档位 I 缺失" in g for g in gaps)

    def test_assert_raises_on_gaps(self):
        cases = build_cases(n_compression=12)
        comp = next(c for c in cases if c.case_type == "compression")
        comp.history = []
        with pytest.raises(AssertionError, match="覆盖缺口"):
            coverage_mod.assert_full_coverage(cases, expected_types=("compression",))

    def test_subset_expected_types_no_false_gaps(self):
        """子集口径：只查自己声明的类型，不误报其他类型为空。"""
        cases = get_experiment_cases("memory")
        assert coverage_mod.coverage_gaps(
            cases, expected_types=("memory",)) == []


# ============ 记忆实验（experiments/memory.py） ============

class TestMemoryExperiment:
    def test_scripted_memory_yields_payload(self):
        import agent.memory as memory_pkg
        from evals.experiments.memory import (
            memory_payload_set,
            scripted_memory,
        )
        with scripted_memory():
            assert memory_pkg.build_longterm_section() == ""  # 无 payload = 空
            with memory_payload_set({"longterm": "站点档案：警戒水位 638.26"}):
                assert memory_pkg.build_longterm_section() == "站点档案：警戒水位 638.26"
                assert memory_pkg.get_semantic_knowledge("q") == ""
            assert memory_pkg.build_longterm_section() == ""  # 退出恢复

    def test_run_memory_experiment_contrast(self):
        """假 run_case：记忆在场才答对——验证对照结构与子类拆分。"""
        from evals.experiments import memory as memory_exp

        def fake_run_case(case, model_label=""):
            injected = memory_exp._CURRENT_PAYLOAD.get() or {}
            has_memory = bool(injected)
            found = has_memory and case.case_id != "mem-001"  # 一条记忆也救不了
            return _record(
                case.case_id, found,
                checks={"needle_found": found if case.needle_substrings else None},
            )

        cases = [_memory_case(f"mem-{i:03d}") for i in range(8)]
        with patch.object(memory_exp, "run_case", side_effect=fake_run_case):
            result = memory_exp.run_memory_experiment(cases)

        assert result["contrast"]["treated_rate"]["p"] == pytest.approx(7 / 8)
        assert result["contrast"]["baseline_rate"]["p"] == 0.0
        assert result["contrast"]["significant"] is True
        assert "fact" in result["by_subtype"]
        assert "temporal" in result["by_subtype"]


# ============ 压缩实验（experiments/compression.py） ============

class TestCompressionExperiment:
    def test_measure_token_savings_with_fake_compact(self):
        from evals.experiments.compression import measure_token_savings
        history = [
            {"role": "user", "content": "很长的中文内容" * 100},
            {"role": "assistant", "content": "同样很长的回答" * 100},
        ]
        savings = measure_token_savings(
            history, compact_fn=lambda h: h[:1],  # 假压缩：只留第一条
        )
        assert savings["tokens_before"] > savings["tokens_after"]
        assert 0 < savings["saved_pct"] < 100

    def test_compression_disabled_identity(self):
        from evals.experiments.compression import compression_disabled
        history = [{"role": "user", "content": "x"}]
        import agent.graph.runner as runner_pkg
        original = runner_pkg._compact_history_entry
        with compression_disabled():
            assert runner_pkg._compact_history_entry(list(history)) == history
        assert runner_pkg._compact_history_entry is original

    def test_run_compression_experiment_structure(self):
        from evals.experiments import compression as comp_exp

        cases = get_experiment_cases("compression")
        # 前半调用=基线遍（全量历史，针必在）；后半=机制遍（压缩丢 comp-000 的针）
        state = {"calls": 0}

        def fake_run_case(case, model_label=""):
            state["calls"] += 1
            is_treated_pass = state["calls"] > len(cases)
            found = not (is_treated_pass and case.case_id == "comp-000")
            return _record(case.case_id, found,
                           checks={"needle_found": found})

        with patch.object(comp_exp, "run_case", side_effect=fake_run_case), \
             patch.object(comp_exp, "measure_token_savings",
                          return_value={"tokens_before": 5000,
                                        "tokens_after": 2000, "saved_pct": 60.0}):
            result = comp_exp.run_compression_experiment(cases)

        assert result["retention_contrast"]["baseline_rate"]["p"] == 1.0
        assert result["retention_contrast"]["treated_rate"]["p"] < 1.0
        assert result["token_savings"]["mean_saved_pct"] == 60.0


# ============ 自进化实验（experiments/self_evolution.py） ============

class TestSelfEvolutionExperiment:
    def test_curves_and_toggle_direction(self):
        from evals.experiments import self_evolution as evo_exp

        seen_settings = []

        def fake_run_cases(cases, model_label="", log_every=10):
            seen_settings.append(get_settings().SELF_EVOLUTION_ENABLED)
            enabled = seen_settings[-1]
            it = seen_settings.count(enabled)  # 该方向的第几轮
            # 实验组随轮次进步（0.3 → 0.6 → 0.9），对照组恒 0.3
            rate = min(0.3 + 0.3 * (it - 1), 0.9) if enabled else 0.3
            n = len(cases)
            ok = int(rate * n)
            return [_record(c.case_id, i < ok) for i, c in enumerate(cases)]

        cases = [_memory_case(f"biz-{i:03d}") for i in range(10)]
        with patch.object(evo_exp, "run_cases", side_effect=fake_run_cases):
            result = evo_exp.run_self_evolution_experiment(cases, iterations=3)

        assert seen_settings == [False, False, False, True, True, True]
        assert result["learning_curve_control"] == [0.3, 0.3, 0.3]
        assert result["learning_curve_experimental"] == [0.3, 0.6, 0.9]
        assert result["final_contrast"]["delta"] == pytest.approx(0.6)


# ============ KV Cache 实验（experiments/kv_cache.py） ============

class TestKvCacheExperiment:
    def test_stats_snapshot_aggregation(self):
        from app.core.llm_stats import record_llm_usage, reset_cache_stats
        from evals.experiments.kv_cache import _stats_snapshot
        reset_cache_stats()
        record_llm_usage("planner", {"prompt_tokens": 100,
                                     "prompt_tokens_details": {"cached_tokens": 80}})
        record_llm_usage("planner", {"prompt_tokens": 120,
                                     "prompt_tokens_details": {"cached_tokens": 60}})
        snapshot = _stats_snapshot()
        assert snapshot["nodes"]["planner"]["calls"] == 2
        assert snapshot["nodes"]["planner"]["hit_rate"] == pytest.approx(
            140 / 220, abs=1e-4)
        assert snapshot["total"]["prompt_tokens"] == 220
        reset_cache_stats()

    def test_run_experiment_reports_planner_delta(self):
        from app.core.llm_stats import reset_cache_stats
        from evals.experiments import kv_cache as kv_exp

        def fake_run_script(station, model_label=""):
            from app.core.llm_stats import record_llm_usage
            reset_cache_stats()
            # 冻结版高命中；破坏版零命中（由 settings 探测区分不可行——
            # 直接按调用次序：第 1 遍冻结、第 2 遍破坏）
            calls = fake_run_script.calls
            fake_run_script.calls += 1
            hit = 90 if calls == 0 else 0
            record_llm_usage("planner", {
                "prompt_tokens": 100,
                "prompt_tokens_details": {"cached_tokens": hit},
            })
        fake_run_script.calls = 0

        with patch.object(kv_exp, "_run_session_script",
                          side_effect=fake_run_script):
            result = kv_exp.run_kv_cache_experiment()
        assert result["planner_hit_frozen"] == pytest.approx(0.9)
        assert result["planner_hit_broken"] == 0.0
        assert result["planner_hit_delta"] == pytest.approx(0.9)
        reset_cache_stats()


# ============ 训练阶梯实验（experiments/model_ladder.py） ============

class TestModelLadder:
    def test_ladder_rows_and_step_contrasts(self):
        from evals.experiments import model_ladder as ladder_exp
        cleared = []

        def fake_run_cases(cases, model_label="", log_every=10):
            rate = {"base": 0.3, "sft": 0.6, "dpo": 0.8}.get(model_label, 0.5)
            ok = int(rate * len(cases))
            return [_record(c.case_id, i < ok,
                            checks={"level_exact": i < ok}) for i, c in enumerate(cases)]

        cases = [_memory_case(f"biz-{i:03d}") for i in range(10)]
        with patch.object(ladder_exp, "run_cases", side_effect=fake_run_cases), \
             patch("app.core.llm.get_llm_client") as fake_client:
            fake_client.cache_clear = lambda: cleared.append(True)
            result = ladder_exp.run_model_ladder(["base", "sft", "dpo"], cases)

        assert [r["model"] for r in result["rungs"]] == ["base", "sft", "dpo"]
        assert result["rungs"][0]["case_pass_rate"]["p"] == pytest.approx(0.3)
        assert len(result["step_contrasts"]) == 2
        assert result["step_contrasts"][0]["contrast"]["delta"] == pytest.approx(0.3)
        assert cleared  # 每级都清了客户端缓存

    def test_requires_two_models(self):
        from evals.experiments.model_ladder import run_model_ladder
        with pytest.raises(ValueError, match="至少需要 2 个"):
            run_model_ladder(["only-one"], [])


# ============ 报告量化声明表（report.py） ============

class TestClaimsSection:
    def _memory_result(self):
        treated = [_record(f"m{i}", i < 8) for i in range(10)]
        baseline = [_record(f"m{i}", i < 3) for i in range(10)]
        return {
            "contrast": summarize_contrast(treated, baseline),
            "needle_contrast": summarize_contrast(
                treated, baseline, check_key="needle_found"),
            "by_subtype": {"fact": summarize_contrast(treated[:5], baseline[:5])},
        }

    def test_claims_table_renders(self):
        from evals.report import render_claims_section
        md = "\n".join(render_claims_section({"memory": self._memory_result()}))
        assert "量化声明表" in md
        assert "记忆注入（脚本化）" in md
        assert "+50.0 pp" in md
        assert "显著" in md

    def test_claims_for_all_experiment_types(self):
        from evals.report import experiment_claims
        # compression / self_evolution / kv_cache / model_ladder 各出 ≥1 行
        comp = {
            "retention_contrast": summarize_contrast(
                [_record("c0", True, checks={"needle_found": True})],
                [_record("c0", True, checks={"needle_found": True})],
                check_key="needle_found"),
            "token_savings": {"mean_saved_pct": 55.5},
        }
        evo = {
            "final_contrast": summarize_contrast(
                [_record("e0", True)], [_record("e0", False)]),
            "learning_curve_experimental": [0.3, 0.6],
            "learning_curve_control": [0.3, 0.3],
        }
        kv = {
            "planner_hit_frozen": 0.9, "planner_hit_broken": 0.1,
            "planner_hit_delta": 0.8,
            "frozen": {"nodes": {"planner": {
                "calls": 3, "prompt_tokens": 300, "cached_tokens": 270,
                "hit_rate": 0.9}}},
        }
        ladder = {
            "step_contrasts": [{
                "from": "base", "to": "sft",
                "contrast": summarize_contrast(
                    [_record("l0", True)], [_record("l0", False)]),
                "level_exact_contrast": summarize_contrast(
                    [_record("l0", True, checks={"level_exact": True})],
                    [_record("l0", True, checks={"level_exact": False})],
                    check_key="level_exact"),
            }],
        }
        assert experiment_claims("compression", comp)
        assert experiment_claims("self_evolution", evo)
        assert experiment_claims("kv_cache", kv)
        assert experiment_claims("model_ladder", ladder)
        assert experiment_claims("unknown", {}) == []
