"""系统级评估体系单元测试（离线，不依赖 LLM/MySQL/Qdrant）。

覆盖：回放上下文、用例构建与种子隔离、确定性检查逻辑、指标与置信区间、
回归门禁阈值、Judge 解析容错、记忆消融 patch。
"""
import json

import pytest

from agent.tools.mock_executor import (
    clear_replay_context,
    execute_tool,
    is_replay_active,
    replay_context,
    set_replay_context,
)
from evals import cases as cases_mod
from evals import judge as judge_mod
from evals import metrics as metrics_mod
from evals import regression as regression_mod
from evals import runner as runner_mod
from evals.cases import EVAL_SEED_BASE, EvalCase, assert_seed_isolation, build_cases


@pytest.fixture(autouse=True)
def _ensure_replay_cleared():
    """每条测试后强制清理回放上下文，防止泄漏影响其他测试。"""
    yield
    clear_replay_context()


# ============ 回放上下文 ============

class TestReplayContext:
    def test_set_and_clear(self):
        set_replay_context({"get_hydrology": {"flow_m3_s": 6000}}, seed=123)
        assert is_replay_active() is True
        clear_replay_context()
        assert is_replay_active() is False

    def test_execute_tool_uses_replay_overrides(self):
        """回放激活：overrides 生效且强制走 mock（忽略真实实现）。"""
        set_replay_context(
            {"get_hydrology": {"flow_m3_s": 6000.0, "station": "吴堡"}}, seed=7,
        )
        result = execute_tool("get_hydrology", {"station": "吴堡", "metric": "flow"})
        assert result["flow_m3_s"] == 6000.0
        assert result["source"] == "mock"

    def test_execute_tool_seed_deterministic(self):
        """同 seed 同参数 → 数据完全一致（可复现；时间戳字段除外）。"""
        set_replay_context({}, seed=42)
        r1 = execute_tool("get_weather", {"location": "吕梁", "hours": 6})
        r2 = execute_tool("get_weather", {"location": "吕梁", "hours": 6})
        assert r1["total_rainfall_mm"] == r2["total_rainfall_mm"]
        def strip_time(series):
            return [{k: v for k, v in s.items() if k != "time"} for s in series]

        assert strip_time(r1["series"]) == strip_time(r2["series"])

    def test_replay_forces_mock_even_without_overrides(self):
        """回放激活但该工具无 overrides 且 seed=None：仍强制 mock 分支。"""
        set_replay_context({"get_weather": {"total_rainfall_mm": 1.0}}, seed=None)
        result = execute_tool("get_hydrology", {"station": "吴堡", "metric": "flow"})
        assert result["source"] == "mock"

    def test_explicit_overrides_take_precedence(self):
        """显式传 overrides（训练侧用法）优先于回放上下文。"""
        set_replay_context({"get_hydrology": {"flow_m3_s": 6000.0}}, seed=1)
        result = execute_tool(
            "get_hydrology", {"station": "吴堡", "metric": "flow"},
            overrides={"flow_m3_s": 2500.0},
        )
        assert result["flow_m3_s"] == 2500.0

    def test_context_manager_restores(self):
        with replay_context({"get_weather": {"total_rainfall_mm": 99.0}}, seed=3):
            assert is_replay_active() is True
            r = execute_tool("get_weather", {"location": "吕梁", "hours": 3})
            assert r["total_rainfall_mm"] == 99.0
        assert is_replay_active() is False

    def test_cache_bypassed_during_replay(self, monkeypatch):
        """回放期间绕过 TTL 缓存：不同 case 的 overrides 不会互相污染。"""
        from agent.graph import cache
        cache._TOOL_RESULT_CACHE.clear()
        try:
            set_replay_context({}, seed=11)
            r1 = execute_tool("get_hydrology", {"station": "吴堡", "metric": "flow"})
            # 模拟缓存里已有旧结果：回放中 _cached_execute_tool 不应命中
            cache._TOOL_RESULT_CACHE[cache._cache_key(
                "get_hydrology", {"station": "吴堡", "metric": "flow"},
            )] = (0.0, {"flow_m3_s": 9999.0})
            from agent.graph.cache import _cached_execute_tool
            r2 = _cached_execute_tool("get_hydrology", {"station": "吴堡", "metric": "flow"})
            assert r2.get("flow_m3_s") != 9999.0
            assert r2["source"] == "mock"
            assert r2["flow_m3_s"] == r1["flow_m3_s"]
        finally:
            cache._TOOL_RESULT_CACHE.clear()

    def test_cache_used_outside_replay(self):
        """回放关闭时缓存路径行为不变（生产路径零影响）。"""
        from agent.graph import cache
        cache._TOOL_RESULT_CACHE.clear()
        key = cache._cache_key("get_weather", {"location": "吕梁", "hours": 3})
        cache._TOOL_RESULT_CACHE[key] = (0.0 + __import__("time").time(),
                                         {"total_rainfall_mm": 5.0})
        result = cache._cached_execute_tool("get_weather", {"location": "吕梁", "hours": 3})
        assert result.get("_from_cache") is True


# ============ 用例构建与种子隔离 ============

class TestCases:
    def test_seed_isolation_assertion_passes(self):
        assert_seed_isolation()

    def test_seed_isolation_detects_overlap(self, monkeypatch):
        monkeypatch.setattr(cases_mod, "EVAL_SEED_BASE", 100_000)
        with pytest.raises(AssertionError):
            assert_seed_isolation()

    def test_build_cases_composition_and_types(self):
        cases = build_cases()
        types = {c.case_type for c in cases}
        assert types == {"business", "chitchat", "regulation", "web_search", "trap"}
        assert len([c for c in cases if c.case_type == "business"]) == 30
        assert len(cases) == 62

    def test_build_cases_deterministic(self):
        c1 = build_cases(seed=EVAL_SEED_BASE)
        c2 = build_cases(seed=EVAL_SEED_BASE)
        assert [(c.case_id, c.query, c.seed) for c in c1] == \
               [(c.case_id, c.query, c.seed) for c in c2]

    def test_case_fields_complete(self):
        for case in build_cases():
            assert case.case_id and case.query
            assert case.case_type in cases_mod.CASE_TYPES
            assert cases_mod.EVAL_SEED_BASE <= case.seed < cases_mod.EVAL_SEED_LIMIT
            assert case.capabilities, f"{case.case_id} 缺能力标签"

    def test_business_expected_levels_balanced(self):
        cases = [c for c in build_cases() if c.case_type == "business"]
        levels = [c.expected_level for c in cases]
        assert set(levels) == {"I", "II", "III", "IV"}

    def test_trap_case_design(self):
        """陷阱用例：口头声称Ⅰ/Ⅱ级，数据为Ⅳ级，抗误导能力标签在列。"""
        traps = [c for c in build_cases() if c.case_type == "trap"]
        assert traps
        for c in traps:
            assert c.claimed_level in ("I", "II")
            assert c.expected_level == "IV"
            assert cases_mod.CAP_RESIST in c.capabilities
            assert c.required_any  # 必须核验数据工具

    def test_chitchat_forbids_all_tools(self):
        for c in build_cases():
            if c.case_type == "chitchat":
                assert c.allowed_tools == frozenset()
                assert c.expected_intent == "chitchat"


# ============ 确定性检查逻辑（runner 纯函数） ============

def _mk_case(**kwargs):
    defaults = {
        "case_id": "t-001", "case_type": "business", "query": "q",
        "seed": 300_000, "expected_level": "II",
    }
    defaults.update(kwargs)
    return EvalCase(**defaults)


class TestCheckSequence:
    def test_weather_before_runoff_ok(self):
        assert runner_mod._check_sequence(
            ["get_weather", "predict_runoff", "generate_plan"]) is True

    def test_runoff_before_weather_fails(self):
        assert runner_mod._check_sequence(
            ["predict_runoff", "get_weather"]) is False

    def test_plan_not_last_with_data_after_fails(self):
        assert runner_mod._check_sequence(
            ["generate_plan", "get_hydrology"]) is False

    def test_plan_last_ok(self):
        assert runner_mod._check_sequence(
            ["get_hydrology", "generate_plan"]) is True

    def test_no_constraints_returns_none(self):
        assert runner_mod._check_sequence(["get_hydrology"]) is None
        assert runner_mod._check_sequence([]) is None


class TestCheckCitations:
    def _tool_calls(self):
        return [{
            "tool_name": "web_search",
            "error": "",
            "result": {"results": [{
                "url": "https://a.com", "title": "通报",
                "snippet": "吴堡水文站流量持续监测中",
            }]},
        }]

    def test_valid_quote_passes(self):
        citations = [{"ref_id": 1, "quote": "流量持续监测中",
                      "url": "https://a.com", "title": "通报"}]
        assert runner_mod._check_citations(citations, self._tool_calls()) is True

    def test_fabricated_quote_fails(self):
        citations = [{"ref_id": 1, "quote": "流量已达8000",
                      "url": "https://a.com", "title": "通报"}]
        assert runner_mod._check_citations(citations, self._tool_calls()) is False

    def test_empty_citations_fails_for_web_search(self):
        assert runner_mod._check_citations([], self._tool_calls()) is False


class TestEvaluateCase:
    def _result(self, **overrides):
        base = {
            "final_answer": "回答",
            "warning_level": "II",
            "intent": "agent_task",
            "tool_calls": [
                {"tool_name": "get_weather", "result": {"total_rainfall_mm": 75.0},
                 "error": "", "round": 1},
                {"tool_name": "get_hydrology", "result": {"flow_m3_s": 3500.0},
                 "error": "", "round": 1},
            ],
            "citations": [],
            "rounds": 2,
        }
        base.update(overrides)
        return base

    def test_happy_business_case(self):
        case = _mk_case(required_tools=frozenset({"get_hydrology", "get_weather"}),
                        allowed_tools=frozenset({"get_weather", "get_hydrology"}))
        checks = runner_mod._evaluate_case(case, self._result())["checks"]
        assert checks["level_exact"] is True
        assert checks["tool_recall"] is True
        assert checks["tool_precision"] is True
        assert checks["intent_ok"] is True

    def test_level_mismatch_detected(self):
        case = _mk_case(expected_level="IV")
        record = runner_mod._evaluate_case(case, self._result())
        assert record["checks"]["level_exact"] is False
        assert record["passed"] is False

    def test_env_mismatch_surfaced(self):
        """数据工具结果与用例设计等级冲突时显式暴露环境缺陷。"""
        case = _mk_case(expected_level="II")
        result = self._result()
        result["tool_calls"][1]["result"]["flow_m3_s"] = 6000.0  # 引擎会算出Ⅰ级
        record = runner_mod._evaluate_case(case, result)
        assert "环境缺陷" in record["env_mismatch"]

    def test_trap_obeying_claim_fails(self):
        """陷阱：Agent 盲信用户声称的Ⅰ级 → trap_resisted 失败。"""
        case = _mk_case(case_type="trap", expected_level="IV", claimed_level="I",
                        required_any=frozenset({"get_hydrology"}))
        result = self._result(warning_level="I")
        record = runner_mod._evaluate_case(case, result)
        assert record["checks"]["trap_resisted"] is False
        assert record["passed"] is False

    def test_trap_verified_data_passes(self):
        case = _mk_case(case_type="trap", expected_level="IV", claimed_level="I",
                        required_any=frozenset({"get_hydrology"}))
        record = runner_mod._evaluate_case(case, self._result(warning_level="IV"))
        assert record["checks"]["trap_resisted"] is True

    def test_chitchat_no_tools_passes(self):
        case = _mk_case(case_type="chitchat", expected_level=None,
                        expected_intent="chitchat",
                        allowed_tools=frozenset())
        result = self._result(warning_level="", intent="chitchat", tool_calls=[])
        record = runner_mod._evaluate_case(case, result)
        assert record["passed"] is True

    def test_duplicate_tool_results_all_counted(self):
        """同名数据工具多次调用：结果不互相覆盖，规则引擎取全部 max。"""
        case = _mk_case(expected_level="II")
        result = self._result()
        result["tool_calls"].append(
            {"tool_name": "get_hydrology", "result": {"flow_m3_s": 6000.0},
             "error": "", "round": 2})
        record = runner_mod._evaluate_case(case, result)
        # 6000 ≥ 5000 → 引擎Ⅰ级 ≠ 设计Ⅱ级 → 环境缺陷显式暴露
        assert "环境缺陷" in record["env_mismatch"]


# ============ 指标与置信区间 ============

class TestMetrics:
    def test_binomial_ci_known_values(self):
        ci = metrics_mod.binomial_ci(70, 100)
        assert ci["p"] == 0.7
        # SE = sqrt(0.7*0.3/100) ≈ 0.0458 → 95% CI ≈ [0.61, 0.79]
        assert 0.60 <= ci["ci95"][0] <= 0.62
        assert 0.78 <= ci["ci95"][1] <= 0.80

    def test_binomial_ci_extremes_clamped(self):
        ci = metrics_mod.binomial_ci(100, 100)
        assert ci["ci95"] == [1.0, 1.0] or ci["ci95"][1] <= 1.0
        assert metrics_mod.binomial_ci(0, 0)["n"] == 0

    def _records(self):
        return [
            {"case_id": "a", "case_type": "business", "capabilities": ["level_decision"],
             "passed": True, "latency_s": 1.0, "rounds": 2,
             "checks": {"level_exact": True, "intent_ok": True, "tool_recall": True,
                        "tool_precision": True, "sequence_valid": True}},
            {"case_id": "b", "case_type": "business", "capabilities": ["level_decision"],
             "passed": False, "latency_s": 3.0, "rounds": 3,
             "checks": {"level_exact": False, "intent_ok": True, "tool_recall": True,
                        "tool_precision": True, "sequence_valid": True}},
            {"case_id": "c", "case_type": "chitchat", "capabilities": ["intent"],
             "passed": True, "latency_s": 0.5, "rounds": 1,
             "checks": {"level_exact": None, "intent_ok": True, "tool_recall": None,
                        "tool_precision": True, "sequence_valid": None}},
        ]

    def test_compute_metrics_aggregation(self):
        m = metrics_mod.compute_metrics(self._records())
        assert m["n_cases"] == 3
        assert m["case_pass_rate"]["p"] == pytest.approx(2 / 3, abs=1e-3)
        # level_exact 仅 2 条适用（chitchat 为 None 不计）
        assert m["level_exact"]["n"] == 2
        assert m["level_exact"]["p"] == 0.5
        assert m["by_type"]["business"]["n"] == 2
        assert m["capability_matrix"]["level_decision"]["p"] == 0.5
        assert m["latency"]["p50"] == 1.0

    def test_capability_matrix_not_inflated(self):
        """能力矩阵用例级 passed：等级错 → level_decision 能力分不虚高。"""
        records = self._records()
        m = metrics_mod.compute_metrics(records)
        # case b 等级错 → level_decision 0.5 而非 1.0
        assert m["capability_matrix"]["level_decision"]["p"] == 0.5

    def test_pass_power_k(self):
        r1 = [{"case_id": "x", "passed": True}, {"case_id": "y", "passed": False}]
        r2 = [{"case_id": "x", "passed": True}, {"case_id": "y", "passed": True}]
        r3 = [{"case_id": "x", "passed": True}, {"case_id": "y", "passed": False}]
        result = metrics_mod.compute_pass_power_k([r1, r2, r3], k=3)
        assert result["n"] == 2
        assert result["p"] == 0.5  # x 三次全对，y 有失败

    def test_pass_power_k_missing_repeat_fails_case(self):
        r1 = [{"case_id": "x", "passed": True}]
        r2 = [{"case_id": "x", "passed": True}]
        r3 = []  # 第三次缺整个 case → 不能算 pass^3
        result = metrics_mod.compute_pass_power_k([r1, r2, r3], k=3)
        assert result["p"] == 0.0


# ============ 回归门禁 ============

class TestRegression:
    def _baseline(self, level_p=0.9):
        return {
            "config": {"model_label": "m", "n_business": 1},
            "metrics": {
                "case_pass_rate": {"p": level_p, "se": 0.03},
                "level_exact": {"p": level_p, "se": 0.03},
                "trap_resisted": {"p": 0.9, "se": 0.05},
            },
            "per_case": {"a": {"case_type": "business", "passed": True},
                         "b": {"case_type": "trap", "passed": False}},
        }

    def _metrics(self, level_p, trap_p=0.9):
        return {
            "case_pass_rate": {"p": level_p, "se": 0.03},
            "level_exact": {"p": level_p, "se": 0.03},
            "trap_resisted": {"p": trap_p, "se": 0.05},
        }

    def test_small_regression_within_noise_not_flagged(self):
        """降幅 < max(下限, 2×SE) → 不算回归（分差在噪声带宽内不下结论）。"""
        report = regression_mod.compare_with_baseline(
            self._metrics(0.88), self._baseline())  # 降 2 个点 < 下限 5 个点
        assert report.regressed is False

    def test_large_regression_flagged(self):
        report = regression_mod.compare_with_baseline(
            self._metrics(0.7), self._baseline())  # 降 20 个点
        assert report.regressed is True
        assert any(it["metric"] == "level_exact" and it["regressed"] for it in report.items)

    def test_trap_floor_stricter(self):
        """安全指标固定下限更严（0.10）：降 15 个点即触发（8 个点未超带宽不触发）。"""
        within = regression_mod.compare_with_baseline(
            self._metrics(0.9, trap_p=0.82), self._baseline())
        assert next(it for it in within.items
                    if it["metric"] == "trap_resisted")["regressed"] is False
        beyond = regression_mod.compare_with_baseline(
            self._metrics(0.9, trap_p=0.75), self._baseline())
        assert next(it for it in beyond.items
                    if it["metric"] == "trap_resisted")["regressed"] is True

    def test_flipped_cases_pairwise(self):
        records = [
            {"case_id": "a", "passed": False},   # pass→fail
            {"case_id": "b", "passed": True},    # fail→pass
            {"case_id": "z", "passed": True},    # 基线无此 case，忽略
        ]
        report = regression_mod.compare_with_baseline(
            self._metrics(0.9), self._baseline(), current_records=records)
        flips = {f["case_id"]: f["flip"] for f in report.flipped_cases}
        assert flips == {"a": "pass→fail", "b": "fail→pass"}

    def test_baseline_payload_roundtrip(self):
        records = [{"case_id": "a", "case_type": "business", "passed": True,
                    "predicted_level": "II", "error": "",
                    "checks": {}, "final_answer": "长回答不含入基线"}]
        metrics = {"case_pass_rate": {"p": 1.0, "se": 0.0, "ci95": [1.0, 1.0], "n": 1}}
        payload = regression_mod.build_baseline_payload(metrics, records, {"model_label": "m"})
        text = json.dumps(payload, ensure_ascii=False)
        loaded = json.loads(text)
        assert loaded["per_case"]["a"]["passed"] is True
        assert "final_answer" not in loaded["per_case"]["a"]


# ============ Judge（假客户端，不触网） ============

class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, content):
        self._content = content

    def create(self, **kwargs):
        return _FakeResponse(self._content)


class _FakeClient:
    def __init__(self, content):
        self.chat = type("Chat", (), {})()
        self.chat.completions = _FakeCompletions(content)


_VALID_JUDGE_JSON = json.dumps({
    "faithfulness": {"total_claims": 4, "supported_claims": 3,
                     "unsupported_claims": ["某数据"]},
    "answer_quality": {"necessary_score": 1.0, "important_score": 0.8,
                       "optional_score": 0.5, "trap_deductions": 0.0,
                       "veto_triggered": False, "veto_reason": "", "rationale": "ok"},
}, ensure_ascii=False)


class TestJudge:
    def _record(self, answer="回答"):
        return {"case_id": "a", "query": "q", "final_answer": answer,
                "_tool_calls": [{"tool_name": "get_hydrology",
                                 "result": {"flow_m3_s": 3000.0}, "error": ""}]}

    def test_judge_record_parses(self):
        result = judge_mod.judge_record(
            self._record(), client=_FakeClient(_VALID_JUDGE_JSON), model="judge-x")
        assert result is not None
        assert result["faithfulness"]["rate"] == 0.75
        # 加权：0.5*1.0 + 0.3*0.8 + 0.2*0.5 = 0.84
        assert result["answer_quality"]["score"] == pytest.approx(0.84)
        assert result["answer_quality"]["veto_triggered"] is False

    def test_veto_scores_zero(self):
        vetoed = json.dumps({
            "faithfulness": {"total_claims": 2, "supported_claims": 0,
                             "unsupported_claims": []},
            "answer_quality": {"necessary_score": 1.0, "important_score": 1.0,
                               "optional_score": 1.0, "trap_deductions": 0.0,
                               "veto_triggered": True, "veto_reason": "编造数值",
                               "rationale": ""},
        }, ensure_ascii=False)
        result = judge_mod.judge_record(
            self._record(), client=_FakeClient(vetoed), model="j")
        assert result["answer_quality"]["score"] == 0.0
        assert result["answer_quality"]["veto_triggered"] is True

    def test_non_json_output_returns_none(self):
        assert judge_mod.judge_record(
            self._record(), client=_FakeClient("抱歉我无法评分"), model="j") is None

    def test_empty_answer_skipped(self):
        assert judge_mod.judge_record(
            self._record(answer=""), client=_FakeClient(_VALID_JUDGE_JSON), model="j") is None

    def test_aggregate(self):
        j = judge_mod.judge_record(
            self._record(), client=_FakeClient(_VALID_JUDGE_JSON), model="j")
        agg = judge_mod.aggregate_judge([j, None])
        assert agg["n_judged"] == 1
        assert agg["n_unavailable"] == 1
        assert agg["quality_score_mean"] == pytest.approx(0.84)

    def test_fmt_tool_results_strips_series(self):
        """series/时间戳剔除，控制评判上下文长度。"""
        text = judge_mod._fmt_tool_results([
            {"tool_name": "get_weather", "error": "",
             "result": {"total_rainfall_mm": 75.0, "series": [{"t": 1}] * 30,
                        "fetched_at": "2025-01-01"}},
        ])
        assert "series" not in text
        assert "75.0" in text


# ============ 记忆消融 patch ============

class TestMemoryDisabled:
    def test_patch_takes_effect_and_restores(self):
        import agent.memory as memory_pkg
        from evals.ablation import memory_disabled
        original = memory_pkg.build_longterm_section
        with memory_disabled():
            assert memory_pkg.build_longterm_section() == ""
            assert memory_pkg.get_relevant_experiences("q") == ""
            assert memory_pkg.get_semantic_knowledge("q") == ""
        assert memory_pkg.build_longterm_section is original
