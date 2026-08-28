"""Agent 公共工具函数。

消除跨模块重复代码：时间戳、预警等级描述、LLM JSON 解析、余弦相似度。
"""
import json
import math
import re
from datetime import datetime, timezone
from typing import Any


def now_iso() -> str:
    """当前 UTC 时间的 ISO 8601 字符串。"""
    return datetime.now(timezone.utc).isoformat()


# 预警等级描述（全项目统一，避免多处硬编码不一致）
LEVEL_DESCRIPTION: dict[str, str] = {
    "I": "Ⅰ级（红色）特别重大",
    "II": "Ⅱ级（橙色）重大",
    "III": "Ⅲ级（黄色）较大",
    "IV": "Ⅳ级（蓝色）一般",
}

# 预警等级阈值（全项目统一，synthesizer 逻辑判断 + prompt 描述 + llm guardrail 三处引用）
WARNING_THRESHOLDS: dict[str, int] = {
    "flow_level1": 5000,  # Ⅰ级流量阈值 m³/s
    "flow_level2": 3000,  # Ⅱ级流量阈值
    "flow_level3": 2000,  # Ⅲ级流量阈值
    "rain_level1": 100,   # Ⅰ级 24h 降雨阈值 mm
    "rain_level2": 50,    # Ⅱ级 24h 降雨阈值
}


def parse_json_from_llm(content: str) -> dict[str, Any] | None:
    """解析 LLM 返回的 JSON，多策略容错。

    策略 1：直接 json.loads
    策略 2：去除 ```json ``` 代码块包裹
    策略 3：大括号配对提取第一个完整 {...} 块
    策略 4：修复单引号→双引号、去尾随逗号后重试

    返回 None 表示无法解析。
    """
    if not content:
        return None
    # 策略 1：直接解析
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    # 策略 2：去除 ```json ``` 包裹
    text = content
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 策略 3：提取第一个完整 {...} 块（大括号配对）
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                block = text[start:i + 1]
                try:
                    return json.loads(block)
                except json.JSONDecodeError:
                    # 策略 4：修复常见问题（单引号→双引号、去尾随逗号）
                    fixed = block.replace("'", '"')
                    fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
                    try:
                        return json.loads(fixed)
                    except json.JSONDecodeError:
                        return None
    return None


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """余弦相似度。

    用于 Skill 匹配（description embedding 相似度计算）等场景。
    从已弃用的 agent/router/semantic_router.py 提取为公共工具。

    Args:
        a: 向量 A
        b: 向量 B

    Returns:
        相似度 [-1, 1]。长度不等或空向量返回 0.0。
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ====== 引用标记过滤（Citation Grounding 展示层兜底）======
# 规则：answer 中的 [N] 标记只有对应真实联网搜索引用（source_registry 校验通过）
# 才允许保留；闲聊/纯工具路径一律剥离，防止模型凭空编造"参考文献"标注。

_CITE_MARKER_RE = re.compile(r"\[(\d{1,3})\]")
_CITE_PARTIAL_RE = re.compile(r"\[(?:\d{1,3})?$")


def strip_citation_markers(text: str, valid_ref_ids: set[int] | None = None) -> str:
    """剥离文本中的 [N] 引用标记。

    Args:
        valid_ref_ids: None 表示无任何有效引用（全部剥离，如闲聊路径）；
            传入集合时仅保留编号在集合内的标记（联网搜索真实引用）。
    """
    if not text:
        return text
    if valid_ref_ids is None:
        return _CITE_MARKER_RE.sub("", text)
    return _CITE_MARKER_RE.sub(
        lambda m: m.group(0) if int(m.group(1)) in valid_ref_ids else "", text
    )


class CitationMarkerFilter:
    """流式安全版引用过滤：处理 [N] 跨 token 分片到达的情况。

    用法：
        f = CitationMarkerFilter(valid_ref_ids)
        for delta in stream:
            emit = f.feed(delta)          # 末尾疑似半个标记的片段自动暂存
            ...
        tail = f.flush()                  # 流结束时结算残留
    """

    def __init__(self, valid_ref_ids: set[int] | None = None):
        self._valid = valid_ref_ids
        self._buf = ""

    def feed(self, text: str) -> str:
        """喂入增量，返回可安全下发的文本（已过滤引用标记）。"""
        self._buf += text
        hold = 0
        m = _CITE_PARTIAL_RE.search(self._buf)
        if m:
            hold = len(m.group(0))
        emit = self._buf[: len(self._buf) - hold]
        self._buf = self._buf[len(self._buf) - hold:] if hold else ""
        return strip_citation_markers(emit, self._valid)

    def flush(self) -> str:
        """流结束：结算残留 buffer（未闭合的半个标记按原文下发）。"""
        out = self._buf
        self._buf = ""
        return strip_citation_markers(out, self._valid)
