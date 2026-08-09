"""GRPO 对齐入口（trl GRPOTrainer + vLLM colocate + 规则奖励）。

用法：
  smoke: python -m train.grpo.train_grpo --smoke
  全量:  python -m train.grpo.train_grpo
奖励接线：trl 的 reward_funcs 接收 completions 与 dataset 列；
scenario 以 JSON 字符串存列（避免 HF datasets 对嵌套 dict 的 struct 类型强转），
奖励函数内还原并调 compute_reward。
补全格式：单轮——模型一次输出多个 <tool_call> 块 + 最终研判段。
"""
import argparse
import json
import sys
from pathlib import Path

# 路径引导：项目根（agent/train 可导入）+ backend 目录（app.* 可导入）
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
_BACKEND_ROOT = str(Path(__file__).resolve().parents[2] / "backend")
for _p in (_PROJECT_ROOT, _BACKEND_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import yaml  # noqa: E402
from datasets import Dataset  # noqa: E402
from trl import GRPOConfig, GRPOTrainer  # noqa: E402

from train.data_gen.scenario import Scenario  # noqa: E402
from train.grpo.prompts import build_grpo_prompts  # noqa: E402
from train.rewards.composite import compute_reward, log_reward_parts  # noqa: E402

_STEP = {"n": 0}


def _serialize(scn: Scenario) -> str:
    return json.dumps({
        "scenario_id": scn.scenario_id,
        "expected_level": scn.expected_level,
        "tool_overrides": scn.tool_overrides,
        "station": scn.station,
        "query": scn.query,
    }, ensure_ascii=False)


def _deserialize(s: str) -> Scenario:
    d = json.loads(s)
    return Scenario(
        scenario_id=d["scenario_id"], station=d["station"], query=d["query"],
        expected_level=d["expected_level"], tool_overrides=d["tool_overrides"],
    )


def rule_reward(completions, scenario, **kwargs) -> list:
    """trl reward_func：每条补全算 reward = gate × (r1+r2+r3)。

    rag_hits：训练期用确定性 mock 检索结果（mock_executor._mock_search_regulation
    的固定文档集），与场景无关，保证可复现。
    """
    from agent.tools.mock_executor import _mock_search_regulation
    from agent.tools.schemas import SearchRegulationParams

    rewards = []
    parts_list = []
    for completion, scn_json in zip(completions, scenario, strict=True):
        scn = _deserialize(scn_json)
        rag = _mock_search_regulation(SearchRegulationParams(query=scn.query, top_k=3))["hits"]
        text = completion if isinstance(completion, str) else completion[-1]["content"]
        r, parts = compute_reward(text, scn, rag_hits=rag)
        rewards.append(r)
        parts_list.append(parts)
    _STEP["n"] += 1
    log_reward_parts(_STEP["n"], parts_list)
    return rewards


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="train/grpo/configs/grpo.yaml")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    n = cfg["smoke"]["n_prompts"] if args.smoke else cfg["n_prompts"]
    prompts = build_grpo_prompts(n=n)
    dataset = Dataset.from_list([
        {"prompt": p["prompt"], "scenario": _serialize(p["scenario"])} for p in prompts
    ])

    gcfg = GRPOConfig(
        output_dir=cfg["output_dir"],
        learning_rate=cfg["grpo"]["lr"],
        beta=cfg["grpo"]["beta"],
        num_generations=cfg["grpo"]["group_size"],
        temperature=cfg["grpo"]["temperature"],
        max_completion_length=cfg["grpo"]["max_completion_length"],
        per_device_train_batch_size=cfg["grpo"]["per_device_batch"],
        gradient_accumulation_steps=cfg["grpo"]["grad_accum"],
        num_train_epochs=1 if args.smoke else cfg["grpo"]["epochs"],
        max_steps=cfg["smoke"]["max_steps"] if args.smoke else -1,
        gradient_checkpointing=cfg["grpo"]["gradient_checkpointing"],
        save_steps=cfg["grpo"]["save_steps"],
        logging_steps=1,
        bf16=True,
        seed=cfg["grpo"]["seed"],
        report_to=[],
        use_vllm=True,
        vllm_mode=cfg["vllm"]["mode"],
        vllm_gpu_memory_utilization=cfg["vllm"]["gpu_memory_utilization"],
    )
    trainer = GRPOTrainer(
        model=cfg["sft_model"],
        args=gcfg,
        train_dataset=dataset,
        reward_funcs=rule_reward,
    )
    trainer.train()
    trainer.save_model(f"{cfg['output_dir']}/adapter")
    print(f"[grpo] adapter → {cfg['output_dir']}/adapter；合并复用 train/lora/merge.py")


if __name__ == "__main__":
    main()
