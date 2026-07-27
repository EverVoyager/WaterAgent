"""r2 工具调用正确性（0.3）：参数校验 0.1 + 工具命中 0.1 + 顺序合法 0.1。"""
from agent.tools.schemas import TOOL_PARAM_MODELS
from train.data_gen.hermes_format import extract_tool_calls


def r2_score(completion: str, reference_tools: list) -> float:
    calls = extract_tool_calls(completion)
    if not calls:
        return 0.0
    score = 0.0
    # 0.1 参数全部合法
    ok = True
    for c in calls:
        model = TOOL_PARAM_MODELS.get(c["name"])
        if model is None:
            ok = False
            break
        try:
            model(**c["arguments"])
        except Exception:
            ok = False
            break
    if ok:
        score += 0.1
    # 0.1 调用集合 ⊆ 参考集且至少 1 个必需工具
    names = {c["name"] for c in calls}
    if names and names.issubset(set(reference_tools)) and names:
        score += 0.1
    # 0.1 顺序合法
    seq = [c["name"] for c in calls]
    order_ok = True
    if "predict_runoff" in seq and "get_weather" not in seq[: seq.index("predict_runoff")]:
        order_ok = False
    if "generate_plan" in seq and seq.index("generate_plan") != len(seq) - 1:
        order_ok = False
    if order_ok:
        score += 0.1
    # 避免 0.1+0.1+0.1 = 0.30000000000000004 浮点误差
    return round(score, 6)
