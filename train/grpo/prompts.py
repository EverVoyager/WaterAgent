"""GRPO prompt 集：独立种子区间 [100_000, 200_000)，与 SFT/评估零重叠。"""
from app.core.llm import get_default_system_prompt
from train.data_gen.scenario import generate_scenarios

GRPO_SEED_BASE = 100_000


def build_grpo_prompts(n: int, seed: int = GRPO_SEED_BASE) -> list:
    scenarios = generate_scenarios(n=n, seed=seed)
    return [
        {
            "prompt": [
                {"role": "system", "content": get_default_system_prompt()},
                {"role": "user", "content": scn.query},
            ],
            "scenario": scn,
        }
        for scn in scenarios
    ]
