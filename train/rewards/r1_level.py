"""r1 等级正确性（0.4）：与规则引擎真值一致满分，相邻一级半分。

chatty 场景（truth_level=""）：模型不输出等级得满分，输出等级得 0 分。
"""
from train.data_gen.hermes_format import _LEVEL_MAP

_ADJACENT = {"I": {"II"}, "II": {"I", "III"}, "III": {"II", "IV"}, "IV": {"III"}}


def _extract(text: str) -> str | None:
    for pattern, level in _LEVEL_MAP:
        if pattern.search(text):
            return level
    return None


def r1_score(completion: str, truth_level: str) -> float:
    final = completion.split("</tool_call>")[-1]
    level = _extract(final)
    if truth_level == "":
        return 0.4 if level is None else 0.0
    if level is None:
        return 0.0
    if level == truth_level:
        return 0.4
    if level in _ADJACENT.get(truth_level, set()):
        return 0.2
    return 0.0
