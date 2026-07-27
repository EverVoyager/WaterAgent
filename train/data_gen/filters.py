"""三道规则过滤：F1 参数合法 / F2 序列合法 / F3 等级一致。

过滤权威：等级真值由 agent.graph.synthesizer.compute_warning_level 对回放
工具结果重算得出，与线上规则引擎单一同源。
"""
from enum import Enum

from agent.graph.synthesizer import compute_warning_level
from agent.tools.schemas import TOOL_PARAM_MODELS
from train.data_gen.hermes_format import parse_final_answer, parse_trace
from train.data_gen.scenario import Scenario


class FilterResult(Enum):
    ACCEPT = "accept"
    REJECT_F1 = "reject_f1_params"
    REJECT_F2 = "reject_f2_sequence"
    REJECT_F3 = "reject_f3_level"


def _f1_params_valid(tool_calls: list) -> bool:
    """参数合法性：仅校验已知工具的参数；未知工具交由 _f2 处理。"""
    for call in tool_calls:
        model = TOOL_PARAM_MODELS.get(call["name"])
        if model is None:
            continue
        try:
            model(**call["arguments"])
        except Exception:
            return False
    return True


def _f2_sequence_valid(tool_calls: list) -> bool:
    """序列合法性：工具已知 / 无重复 / predict_runoff 前有 weather / generate_plan 在末尾。"""
    names = [c["name"] for c in tool_calls]
    if any(n not in TOOL_PARAM_MODELS for n in names):
        return False
    if len(names) != len(set(names)):  # 重复调用
        return False
    if "predict_runoff" in names and "get_weather" not in names[: names.index("predict_runoff")]:
        return False
    if "generate_plan" in names and names.index("generate_plan") != len(names) - 1:
        return False
    return True


def filter_trace(messages: list, scenario: Scenario) -> FilterResult:
    """对单条轨迹执行三道过滤。"""
    if scenario.query_type == "chatty":
        trace = parse_trace(messages)
        if trace is None or trace["tool_calls"]:
            return FilterResult.REJECT_F2
        # 闲聊样本要求最终段不含任何等级字样
        return FilterResult.ACCEPT if parse_final_answer(messages) is None else FilterResult.REJECT_F3

    trace = parse_trace(messages)
    if trace is None:
        return FilterResult.REJECT_F1
    if not _f1_params_valid(trace["tool_calls"]):
        return FilterResult.REJECT_F1
    if not _f2_sequence_valid(trace["tool_calls"]):
        return FilterResult.REJECT_F2
    # F3：用回放结果重算规则等级，与轨迹最终等级比对
    tool_results = {f"tool_{i}": r for i, r in enumerate(trace["tool_results"])}
    truth, _ = compute_warning_level(tool_results)
    model_level = parse_final_answer(messages)
    if model_level is None or model_level != truth or truth != scenario.expected_level:
        return FilterResult.REJECT_F3
    return FilterResult.ACCEPT
