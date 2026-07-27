"""QLoRA SFT 入口（trl SFTTrainer）。

用法：
  smoke: python -m train.lora.train_sft --smoke
  全量:  python -m train.lora.train_sft
"""
import argparse
import sys
from pathlib import Path

# 路径引导：项目根 + backend 目录（app.* 可导入）
_PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
_BACKEND_ROOT = str(Path(__file__).resolve().parents[3] / "backend")
for _p in (_PROJECT_ROOT, _BACKEND_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402
import yaml  # noqa: E402
from peft import LoraConfig  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402

from train.lora.dataset import build_sft_dataset, load_jsonl  # noqa: E402


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="train/lora/configs/sft_qlora.yaml")
    parser.add_argument("--smoke", action="store_true", help="10 样本 5 步验证全流程")
    args = parser.parse_args()
    cfg = load_config(args.config)

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    bnb = BitsAndBytesConfig(
        load_in_4bit=cfg["qlora"]["load_in_4bit"],
        bnb_4bit_quant_type=cfg["qlora"]["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=cfg["qlora"]["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], quantization_config=bnb, device_map="auto",
    )

    records = load_jsonl(cfg["data"])
    if args.smoke:
        records = records[: cfg["smoke"]["n_samples"]]
    train_ds = build_sft_dataset(records, tokenizer, cfg["max_len"])

    sft_cfg = SFTConfig(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=cfg["train"]["per_device_batch"],
        gradient_accumulation_steps=cfg["train"]["grad_accum"],
        learning_rate=cfg["train"]["lr"],
        num_train_epochs=1 if args.smoke else cfg["train"]["epochs"],
        max_steps=cfg["smoke"]["max_steps"] if args.smoke else -1,
        warmup_ratio=cfg["train"]["warmup_ratio"],
        lr_scheduler_type=cfg["train"]["lr_scheduler"],
        gradient_checkpointing=cfg["train"]["gradient_checkpointing"],
        logging_steps=cfg["train"]["logging_steps"],
        save_steps=cfg["train"]["save_steps"],
        save_total_limit=3,
        bf16=True,
        seed=cfg["train"]["seed"],
        report_to=[],
        dataset_num_proc=1,
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=train_ds,
        peft_config=LoraConfig(
            r=cfg["lora"]["r"],
            lora_alpha=cfg["lora"]["alpha"],
            lora_dropout=cfg["lora"]["dropout"],
            target_modules=cfg["lora"]["target_modules"],
            task_type="CAUSAL_LM",
        ),
    )
    trainer.train()
    trainer.save_model(f"{cfg['output_dir']}/adapter")
    print(f"[sft] adapter → {cfg['output_dir']}/adapter")


if __name__ == "__main__":
    main()
