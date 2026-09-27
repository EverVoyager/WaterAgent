"""prompt_capture 序列化契约测试：tools 在前 + 只追加保持前缀。

背景（2026-09-15）：静态前缀复用占比曾因序列化按字母序（messages 在前）
被系统性低估——消息追加会整体挪动 tools 块，LCP 提前断裂。序列化改为
tools 在前（对齐服务端渲染顺序：工具 schema 属于系统区）后，messages 只
追加时串的前缀性质与真实服务端视图一致。本文件锁死该契约。
"""
import json

from evals.experiments.prefix_reuse_ratio import _normalize
from evals.experiments.prompt_capture import serialize_request


def _req(tools, messages):
    return serialize_request({"tools": tools, "messages": messages})


def test_tools_come_before_messages():
    s = _req([{"function": {"name": "get_hydrology"}}],
             [{"role": "user", "content": "查水情"}])
    assert s.startswith('{"tools":')
    assert s.index('"tools"') < s.index('"messages"')


def test_append_only_messages_keep_tools_region_and_prefix():
    """消息追加时：tools 区逐字节不动，分叉点只允许出现在前一轮结尾。"""
    tools = [{"function": {"name": "t"}}]
    m1 = [{"role": "system", "content": "sys"}, {"role": "user", "content": "q1"}]
    m2 = m1 + [{"role": "assistant", "content": "a1"}]
    p1, p2 = _req(tools, m1), _req(tools, m2)
    cut = p1.index('"messages"')
    assert p1[:cut] == p2[:cut]
    k = 0
    while k < min(len(p1), len(p2)) and p1[k] == p2[k]:
        k += 1
    # 分叉点只允许出现在 p1 结尾的收括号处（messages 的 "]" 与根对象的 "}"）
    assert k >= len(p1) - 2


def test_deterministic_across_dict_key_order():
    """内部字典键序不影响结果（确定性序列化契约）。"""
    a = _req(None, [{"role": "user", "content": "q", "extra": 1}])
    b = _req(None, [{"extra": 1, "content": "q", "role": "user"}])
    assert a == b


def test_normalize_old_alphabetical_dump():
    """旧落盘（字母序，messages 在前）归一化为 tools 在前。"""
    old = json.dumps(
        {"messages": [{"role": "user", "content": "q"}],
         "tools": [{"function": {"name": "t"}}]},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    fixed = _normalize(old)
    assert fixed.startswith('{"tools":')
    assert _normalize(fixed) == fixed  # 幂等


def test_normalize_passthrough_non_dict():
    assert _normalize("not-json") == "not-json"
