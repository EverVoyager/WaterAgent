"""LoRA adapter 合并为全量权重（BF16，供 vLLM 加载）。

用法：python -m train.lora.merge [--adapter train/lora/outputs/sft/adapter]
"""
import argparse
import sys
from pathlib import Path

# 路径引导：项目根 + backend 目录
_PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
_BACKEND_ROOT = str(Path(__file__).resolve().parents[3] / "backend")
for _p in (_PROJECT_ROOT, _BACKEND_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:  # noqa: E402
    import torch  # noqa: E402
    from peft import PeftModel  # noqa: E402
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
except ImportError:  # 训练重型依赖未装时允许模块导入（单测用 mock 替换）
    torch = None  # type: ignore[assignment]
    PeftModel = None  # type: ignore[assignment]
    AutoModelForCausalLM = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]


def merge_adapter(base_model: str, adapter_path: str, out_dir: str) -> str:
    base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, adapter_path)
    merged = model.merge_and_unload()
    merged.save_pretrained(out_dir)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.save_pretrained(out_dir)
    print(f"[merge] → {out_dir}")
    return out_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter", default="train/lora/outputs/sft/adapter")
    parser.add_argument("--out", default="train/lora/outputs/sft/merged")
    args = parser.parse_args()
    merge_adapter(args.base, args.adapter, args.out)
