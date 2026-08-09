"""DPO 正负对构建模块。

从同场景的高分轨迹(chosen)和低分轨迹(rejected)构建偏好对，
用于 DPO（Direct Preference Optimization）训练。

流程：
1. 第一轮合成 + 评判，高分进 SFT，低分场景收集
2. 对低分场景进行第二轮合成 + 评判
3. 第二轮高分 → 与第一轮低分配成 DPO 对
4. 第二轮仍低分 → 丢弃

输出格式（兼容 trl DPOTrainer）：
{
  "prompt": [{"role": "system", ...}, {"role": "user", ...}],
  "chosen": [{"role": "assistant", "content": "高质量最终回答"}],
  "rejected": [{"role": "assistant", "content": "低质量最终回答"}]
}
"""
import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class DPOPair:
    """单个 DPO 偏好对。"""
    prompt: list  # [{"role": "system/user", "content": "..."}]
    chosen: str   # 高质量最终回答
    rejected: str  # 低质量最终回答
    scenario_id: str = ""


def _extract_final_answer(messages: list) -> str:
    """从轨迹中提取最终回答（最后一个 assistant 消息，不含 tool_call 标签）。

    Hermes 格式的 assistant 消息可能含 <tool_call>...</tool_call> 块，
    最终回答是纯文本的 assistant 消息。
    """
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            # 最终回答不含 <tool_call> 标签
            if "<tool_call>" not in content:
                return content
    return ""


def _extract_prompt(messages: list) -> list:
    """提取 prompt 部分（system + user 消息）。"""
    prompt = []
    for msg in messages:
        role = msg.get("role", "")
        if role in ("system", "user"):
            prompt.append({"role": role, "content": msg.get("content", "")})
    return prompt


def build_dpo_pair(high_record: dict, low_record: dict) -> DPOPair | None:
    """从同场景的高分和低分轨迹构建一个 DPO 对。

    Args:
        high_record: 高分轨迹 {"scenario_id", "level", "messages"}
        low_record: 低分轨迹 {"scenario_id", "level", "messages"}

    Returns:
        DPOPair 或 None（无法提取最终回答时）
    """
    high_answer = _extract_final_answer(high_record["messages"])
    low_answer = _extract_final_answer(low_record["messages"])
    if not high_answer or not low_answer:
        return None
    # 回答完全相同则无偏好价值
    if high_answer == low_answer:
        return None
    prompt = _extract_prompt(high_record["messages"])
    return DPOPair(
        prompt=prompt,
        chosen=high_answer,
        rejected=low_answer,
        scenario_id=high_record.get("scenario_id", ""),
    )


def write_dpo_jsonl(pairs: list[DPOPair], out_path: Path) -> int:
    """将 DPO 对写入 JSONL 文件。

    Returns:
        写入条数。
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps({
                "prompt": pair.prompt,
                "chosen": [{"role": "assistant", "content": pair.chosen}],
                "rejected": [{"role": "assistant", "content": pair.rejected}],
                "scenario_id": pair.scenario_id,
            }, ensure_ascii=False) + "\n")
            written += 1
    return written
