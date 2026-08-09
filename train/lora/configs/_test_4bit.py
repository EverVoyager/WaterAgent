"""测试 4bit 量化是否实际生效。"""
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

model_path = "D:/hf_cache/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"

print("加载 4bit 量化模型...")
q = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
m = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=q,
    trust_remote_code=True,
    device_map="auto",
)
mem_gb = torch.cuda.memory_allocated() / 1024**3
print(f"4bit 模型显存占用: {mem_gb:.2f} GB")
if mem_gb > 4.0:
    print("⚠️ 显存过高，4bit 量化可能未生效！")
else:
    print("✅ 4bit 量化正常工作")
