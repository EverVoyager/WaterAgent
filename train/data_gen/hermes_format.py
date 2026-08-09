"""Hermes 格式序列化/解析（训练集、过滤器、奖励函数、评估共用唯一实现）。

标签约定（Qwen 原生）：
- assistant 内容中的工具调用：<tool_call>\n{json}\n</tool_call>
- tool 消息内容：<tool_response>\n{json}\n</tool_response>（裸 JSON 也接受）
"""
import json
import re
from typing import Any, Optional

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
# 等级归一化：先匹配长词（Ⅳ>Ⅲ>Ⅱ>Ⅰ），避免子串误配
_LEVEL_MAP = [
    (re.compile(r"Ⅳ|IV|4\s*级|四\s*级"), "IV"),
    (re.compile(r"Ⅲ|III|3\s*级|三\s*级"), "III"),
    (re.compile(r"Ⅱ|II|2\s*级|二\s*级"), "II"),
    (re.compile(r"Ⅰ|(?<!I)I(?!I)|1\s*级|一\s*级"), "I"),
]


def make_tool_call_text(name: str, arguments: dict) -> str:
    """序列化单个工具调用块。"""
    return f"<tool_call>\n{json.dumps({'name': name, 'arguments': arguments}, ensure_ascii=False)}\n</tool_call>"


def make_tool_response_text(result: dict) -> str:
    """序列化 tool 消息内容。"""
    return f"<tool_response>\n{json.dumps(result, ensure_ascii=False)}\n</tool_response>"


def extract_tool_calls(assistant_content: str) -> list[dict]:
    """从 assistant 文本提取全部工具调用；任一 JSON 非法则该次返回 []。"""
    calls = []
    for m in _TOOL_CALL_RE.finditer(assistant_content):
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            return []
        if not isinstance(obj, dict) or "name" not in obj or "arguments" not in obj:
            return []
        if not isinstance(obj["arguments"], dict):
            return []
        calls.append({"name": obj["name"], "arguments": obj["arguments"]})
    return calls


def parse_trace(messages: list[dict]) -> Optional[dict]:
    """解析完整轨迹。返回 {"tool_calls": [...], "tool_results": [...], "final": str}；
    任一结构非法（坏 JSON、tool 消息无前置调用）返回 None。"""
    tool_calls: list[dict] = []
    tool_results: list[dict] = []
    final = ""
    pending_calls = 0
    for msg in messages:
        role, content = msg.get("role"), msg.get("content", "")
        if role == "assistant":
            calls = extract_tool_calls(content)
            if "<tool_call>" in content and not calls:
                return None  # 有标签但全部解析失败
            if calls:
                tool_calls.extend(calls)
                pending_calls += len(calls)
            else:
                final = content  # 无调用的 assistant 段 = 最终回答
        elif role == "tool":
            if pending_calls <= 0:
                return None
            raw = content
            m = re.match(r"\s*<tool_response>\s*(\{.*\})\s*</tool_response>\s*$", raw, re.DOTALL)
            payload = m.group(1) if m else raw
            try:
                tool_results.append(json.loads(payload))
            except json.JSONDecodeError:
                return None
            pending_calls -= 1
    return {"tool_calls": tool_calls, "tool_results": tool_results, "final": final}


def parse_final_answer(messages: list[dict]) -> Optional[str]:
    """从轨迹最后一个 assistant 段提取归一化等级（I/II/III/IV），无则 None。"""
    final = next((m["content"] for m in reversed(messages) if m.get("role") == "assistant"), "")
    for pattern, level in _LEVEL_MAP:
        if pattern.search(final):
            return level
    return None


def round_trip_ok(messages: list[dict]) -> bool:
    """parse_trace 成功即通过。"""
    return parse_trace(messages) is not None
