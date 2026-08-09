"""奖励合成：reward = gate × (r1 + r2 + r3)；分量日志供曲线绘制。"""
import json
import time
from pathlib import Path

from agent.graph.synthesizer import compute_warning_level
from train.data_gen.scenario import Scenario
from train.rewards.format_gate import gate_pass
from train.rewards.r1_level import r1_score
from train.rewards.r2_tool_call import r2_score
from train.rewards.r3_plan import r3_score

_LOG_PATH = Path("train/grpo/outputs/reward_log.jsonl")


def _truth_level(scn: Scenario) -> str:
    """等级真值：用场景覆盖值重构工具结果，经规则引擎重算（单一权威来源）。

    与 scenario 生成时的 expected_level 一致（生成器按阈值构造），此处重算
    是为了让奖励逻辑直接依赖规则引擎而非元数据副本，防双份规则漂移。
    """
    tool_results = {}
    for tool_name, key in (("get_hydrology", "hydro"), ("get_weather", "wx"),
                           ("predict_runoff", "runoff")):
        if scn.tool_overrides.get(tool_name):
            tool_results[key] = scn.tool_overrides[tool_name]
    if not tool_results:
        return scn.expected_level  # plan_only 等无水文数据场景，直接用场景真值
    level, _ = compute_warning_level(tool_results)
    return level


def compute_reward(completion: str, scn: Scenario, rag_hits: list) -> tuple:
    """返回 (reward, parts)。格式门控失败 → (0.0, {})。"""
    if not gate_pass(completion):
        return 0.0, {}
    parts = {
        "r1": r1_score(completion, _truth_level(scn)),
        "r2": r2_score(completion),
        "r3": r3_score(completion, rag_hits),
    }
    return sum(parts.values()), parts


def log_reward_parts(step: int, parts_list: list, path: Path = _LOG_PATH) -> None:
    """每 step 记录三分量均值（NFR-7）。"""
    if not parts_list:
        return
    valid = [p for p in parts_list if p]
    means = {k: round(sum(p.get(k, 0.0) for p in valid) / max(len(valid), 1), 4)
             for k in ("r1", "r2", "r3")}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": time.time(), "step": step, **means}) + "\n")
