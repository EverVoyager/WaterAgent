"""格式门控：工具调用块全部可解析且最终段非空才放行。"""
from train.data_gen.hermes_format import extract_tool_calls


def gate_pass(completion: str) -> bool:
    has_tag = "<tool_call>" in completion
    calls = extract_tool_calls(completion)
    if has_tag and not calls:
        return False  # 有标签但 JSON 全坏
    final = completion.split("</tool_call>")[-1].strip()
    return bool(final)
