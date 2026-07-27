"""GRPO prompts 与 SFT 集零重叠；rollout 回放确定性。"""
from train.data_gen.scenario import generate_scenarios
from train.grpo.prompts import build_grpo_prompts
from train.grpo.rollouts import replay_tool_call


def test_prompts_disjoint_from_sft():
    sft = generate_scenarios(n=100, seed=1000)
    grpo = build_grpo_prompts(n=100)
    assert {s.scenario_id for s in sft}.isdisjoint({p["scenario"].scenario_id for p in grpo})


def test_prompt_carries_system_and_query():
    p = build_grpo_prompts(n=1)[0]
    assert p["prompt"][0]["role"] == "system"
    assert p["prompt"][-1]["role"] == "user"
    assert p["scenario"].expected_level in ("I", "II", "III", "IV", "")


def test_replay_uses_scenario_overrides():
    scn = next(s for s in generate_scenarios(n=10, seed=100500) if s.query_type != "chatty")
    out = replay_tool_call(scn, "get_hydrology", {"station": scn.station, "metric": "both"})
    assert out["flow_m3_s"] == scn.tool_overrides["get_hydrology"]["flow_m3_s"]
    assert out["source"] == "mock"
