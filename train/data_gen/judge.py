"""LLM-as-Judge 评判器（双模型蒸馏的评判角色）。

用 qwen-max 对教师生成的轨迹做 4 维语义打分：
1. 工具调用合理性（该调的调了没、不该调的没调）
2. 参数正确性（工具参数是否合理）
3. 等级判定准确性（预警等级是否正确）
4. 回答专业性与完整性（是否专业、是否完整回答用户问题）

总分 1-5，≥4 进 SFT 候选，<4 进 DPO 负样本候选。
"""
import json
import logging
from dataclasses import dataclass

from train.data_gen.hermes_format import parse_trace

logger = logging.getLogger(__name__)

_JUDGE_PROMPT = """你是防汛预警领域的专家评审员。请对以下 AI 助手的对话轨迹进行打分。

用户问题：{query}

AI 助手的完整轨迹（含工具调用和最终回答）：
{trace}

场景真值信息：
- 期望预警等级：{expected_level}
- 期望站点：{station}

请从以下 4 个维度分别打分（1-5 分），并给出总分：

1. tool_reasonableness（工具调用合理性）：是否调用了合适的工具？有没有多余或遗漏？
2. param_correctness（参数正确性）：工具参数是否合理？
3. level_accuracy（等级准确性）：最终判定的预警等级是否正确？
4. answer_quality（回答质量）：回答是否专业、完整、准确？

请返回 JSON：
{{
  "tool_reasonableness": 1-5,
  "param_correctness": 1-5,
  "level_accuracy": 1-5,
  "answer_quality": 1-5,
  "total": 1-5,
  "rationale": "简要说明扣分原因（50字以内）"
}}

评分标准：
- 5分：完美，无任何问题
- 4分：良好，有小瑕疵但不影响使用
- 3分：及格，有明显问题但仍可用
- 2分：不及格，存在严重错误
- 1分：完全不可用

仅返回 JSON 对象，不要其他内容。"""


@dataclass
class JudgeResult:
    tool_reasonableness: int
    param_correctness: int
    level_accuracy: int
    answer_quality: int
    total: int
    rationale: str

    @property
    def is_sft_quality(self) -> bool:
        """总分 >= 4 进 SFT 候选。"""
        return self.total >= 4

    @property
    def is_dpo_negative(self) -> bool:
        """总分 < 4 但 > 1 进 DPO 负样本候选（1分完全不可用，丢弃）。"""
        return 1 < self.total < 4


def _format_trace_for_judge(messages: list, max_chars: int = 2000) -> str:
    """把轨迹格式化为评判 LLM 可读的文本（截断防超长）。"""
    parts = []
    total = 0
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        role_cn = {"system": "系统", "user": "用户", "assistant": "助手", "tool": "工具结果"}.get(role, role)
        # 工具结果截断到 300 字（评判只需了解概要）
        if role == "tool" and len(content) > 300:
            content = content[:300] + "...(truncated)"
        line = f"[{role_cn}] {content}"
        if total + len(line) > max_chars:
            line = line[:max_chars - total] + "...(truncated)"
            parts.append(line)
            break
        parts.append(line)
        total += len(line) + 1
    return "\n".join(parts)


def _parse_judge_response(content: str) -> JudgeResult | None:
    """解析评判 LLM 返回的 JSON。"""
    import re
    text = content.strip()
    # 去掉 ```json ``` 包裹
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
    # 提取第一个 {...} 块
    start = text.find("{")
    if start == -1:
        return None
    end = text.rfind("}")
    if end == -1:
        return None
    block = text[start:end + 1]
    try:
        data = json.loads(block)
    except json.JSONDecodeError:
        # 修复常见问题
        fixed = block.replace("'", '"')
        fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
        try:
            data = json.loads(fixed)
        except json.JSONDecodeError:
            return None
    return JudgeResult(
        tool_reasonableness=int(data.get("tool_reasonableness", 0)),
        param_correctness=int(data.get("param_correctness", 0)),
        level_accuracy=int(data.get("level_accuracy", 0)),
        answer_quality=int(data.get("answer_quality", 0)),
        total=int(data.get("total", 0)),
        rationale=data.get("rationale", ""),
    )


def judge_trace(client, model: str, messages: list, scenario,
                max_retries: int = 3) -> JudgeResult | None:
    """对单条轨迹做 LLM-as-Judge 评判。

    Args:
        client: OpenAI 兼容客户端
        model: 评判模型名（如 qwen-max）
        messages: Hermes 格式轨迹
        scenario: Scenario 对象（提供 query/expected_level/station）
        max_retries: 限流/超时重试次数

    Returns:
        JudgeResult 或 None（评判失败时）
    """
    import time as _time

    trace_text = _format_trace_for_judge(messages, max_chars=1500)
    prompt = _JUDGE_PROMPT.format(
        query=scenario.query,
        trace=trace_text,
        expected_level=scenario.expected_level,
        station=scenario.station,
    )
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=300,
                timeout=90,
            )
            content = resp.choices[0].message.content or ""
            result = _parse_judge_response(content)
            if result is None:
                logger.warning("[judge] 评判返回非 JSON: %s", content[:200])
            return result
        except Exception as e:
            err_str = str(e)[:200]
            # 限流/超时 → 等待重试
            if any(k in err_str.lower() for k in ("rate_limit", "429", "timeout", "timed out")):
                wait = 5 * (attempt + 1)
                logger.warning("[judge] 评判限流/超时，%ds 后重试 (%d/%d): %s",
                               wait, attempt + 1, max_retries, err_str)
                _time.sleep(wait)
                continue
            logger.warning("[judge] 评判异常: %s", err_str)
            return None
    logger.warning("[judge] 评判重试 %d 次仍失败", max_retries)
    return None
