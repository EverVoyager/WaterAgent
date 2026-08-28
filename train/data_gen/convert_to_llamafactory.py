"""将 Hermes 格式数据集转换为 LlamaFactory sharegpt 格式。

LlamaFactory 工具调用格式（sharegpt）：
- conversations: [{"from": "human/gpt/function_call/observation", "value": "..."}]
- tools: JSON 字符串，工具描述列表
- system: 系统提示词

转换规则：
- system 消息 → system 字段
- user 消息 → {"from": "human", "value": content}
- assistant 无工具调用 → {"from": "gpt", "value": content}
- assistant 含 <tool_call> → 拆成多个 function_call 消息（每个工具调用一条）
- tool 消息（<tool_response>）→ {"from": "observation", "value": json_str}

用法：
  python -m train.data_gen.convert_to_llamafactory
"""
import json
import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
_BACKEND_ROOT = str(Path(__file__).resolve().parents[2] / "backend")
for _p in (_PROJECT_ROOT, _BACKEND_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dotenv import load_dotenv

load_dotenv(Path(_BACKEND_ROOT) / ".env")

from agent.tools.schemas import TOOL_DESCRIPTIONS, TOOL_PARAM_MODELS
from app.core.llm import get_default_system_prompt
from train.data_gen.hermes_format import extract_tool_calls


def build_tools_json() -> str:
    """构建 LlamaFactory tools 字段（JSON 字符串）。"""
    tools = []
    for name, model in TOOL_PARAM_MODELS.items():
        schema = model.model_json_schema()
        params_schema = {
            "type": schema.get("type", "object"),
            "properties": schema.get("properties", {}),
        }
        if "required" in schema:
            params_schema["required"] = schema["required"]
        tools.append({
            "name": name,
            "description": TOOL_DESCRIPTIONS[name],
            "parameters": params_schema,
        })
    return json.dumps(tools, ensure_ascii=False)


def convert_record(messages: list, tools_json: str) -> dict | None:
    """单条 Hermes 轨迹 → LlamaFactory sharegpt 格式。

    返回 {"conversations": [...], "system": ..., "tools": ...} 或 None（转换失败）。

    LlamaFactory 要求奇偶位置交替：human/observation 在偶数位，gpt/function_call 在奇数位。
    当一轮 assistant 含多个工具调用时，合并为一条 function_call（JSON 数组）；
    对应的多个 tool 结果也合并为一条 observation（JSON 数组）。
    """
    import re
    system_prompt = get_default_system_prompt()
    # 先按 Hermes 消息序列收集原始角色块
    raw_blocks = []  # [(role, value), ...]

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "system":
            system_prompt = content
            continue

        if role == "user":
            raw_blocks.append(("human", content))
            continue

        if role == "assistant":
            tool_calls = extract_tool_calls(content)
            if tool_calls:
                # 合并本轮所有工具调用为一条 function_call
                calls = [{"name": tc["name"], "arguments": tc["arguments"]} for tc in tool_calls]
                raw_blocks.append(("function_call", json.dumps(calls, ensure_ascii=False)))
            else:
                clean_content = content.replace("<tool_call>", "").replace("</tool_call>", "").strip()
                if clean_content:
                    raw_blocks.append(("gpt", clean_content))
            continue

        if role == "tool":
            m = re.match(r"\s*<tool_response>\s*(\{.*\})\s*</tool_response>\s*$", content, re.DOTALL)
            payload = m.group(1) if m else content
            try:
                json.loads(payload)
            except json.JSONDecodeError:
                return None
            raw_blocks.append(("observation", payload))
            continue

    # 合并连续相同角色的块
    # （多个 function_call 连续出现时合并，多个 observation 连续出现时也合并）
    conversations = []
    for role, value in raw_blocks:
        if conversations and conversations[-1]["from"] == role:
            # 同角色连续：合并 JSON
            prev = conversations[-1]["value"]
            try:
                prev_list = json.loads(prev)
                curr = json.loads(value)
                if not isinstance(prev_list, list):
                    prev_list = [prev_list]
                if not isinstance(curr, list):
                    curr = [curr]
                merged = json.dumps(prev_list + curr, ensure_ascii=False)
                conversations[-1]["value"] = merged
            except json.JSONDecodeError:
                # 合并失败：直接拼接（用换行分隔）
                conversations[-1]["value"] = prev + "\n" + value
        else:
            conversations.append({"from": role, "value": value})

    if not conversations:
        return None

    # 校验奇偶位置
    for i, conv in enumerate(conversations):
        from_tag = conv["from"]
        if i % 2 == 0 and from_tag not in ("human", "observation") or i % 2 == 1 and from_tag not in ("gpt", "function_call"):
            return None

    return {
        "conversations": conversations,
        "system": system_prompt,
        "tools": tools_json,
    }


def main() -> None:
    # 输入：WaterAgents 的训练/验证集
    src_dir = Path("train/lora/data")
    train_path = src_dir / "hermes_fc_v1.jsonl"
    val_path = src_dir / "hermes_fc_v1.val.jsonl"

    # 输出：WaterAgents 项目内（训练时通过 dataset_dir 指定）
    out_dir = Path("train/lora/llamafactory_data")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_train = out_dir / "wateragents_sft.jsonl"
    out_val = out_dir / "wateragents_sft_val.jsonl"

    tools_json = build_tools_json()

    total = 0
    converted = 0
    failed = 0

    for src_file, out_file, label in [(train_path, out_train, "train"), (val_path, out_val, "val")]:
        if not src_file.exists():
            print(f"[convert] 跳过 {label}：源文件 {src_file} 不存在")
            continue

        records = []
        for line in src_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            total += 1
            rec = json.loads(line)
            result = convert_record(rec["messages"], tools_json)
            if result is None:
                failed += 1
                continue
            records.append(result)
            converted += 1

        with out_file.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        print(f"[convert] {label}: {len(records)} 条 → {out_file}")

    print(f"\n[convert] 总计: {total} 条，成功 {converted}，失败 {failed}")
    print(f"[convert] tools 字段长度: {len(tools_json)} 字符")

    # 打印一条样本供检查
    if out_train.exists():
        sample = json.loads(out_train.read_text(encoding="utf-8").splitlines()[0])
        print("\n[convert] 样本 conversations:")
        for c in sample["conversations"][:6]:
            val_preview = c["value"][:100] + "..." if len(c["value"]) > 100 else c["value"]
            print(f"  {c['from']:15s} | {val_preview}")
        if len(sample["conversations"]) > 6:
            print(f"  ... (共 {len(sample['conversations'])} 条)")
        print(f"  system: {sample['system'][:80]}...")
        print(f"  tools: {sample['tools'][:80]}...")


if __name__ == "__main__":
    main()
