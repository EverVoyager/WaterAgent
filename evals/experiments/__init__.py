"""量化实验包：机制消融框架 + 五个"从 X% 到 Y%"数字生产线。

每个实验对应一个可对外引用的量化声明，统一方法论（调研 Mem0/AWM/
Reflexion/LLMLingua/vLLM 的报告口径后固化）：
- 同一案例集 × 机制开关对 → 基线/机制两版通过率
- 绝对值 + 相对提升 + 95% CI + 显著性 + n + 副指标（token/延迟）
- 报告嵌复现命令，history/ 落盘留档（docs/eval-experiments.md 台账）

模块：
- base.py           消融框架（Toggle 开关 + 对照汇总，memory 实验等的公共原语）
- memory.py         记忆增益（脚本化记忆注入 vs 无记忆，LongMemEval 式）
- compression.py    压缩等价性（针保留率 + token 节省，LLMLingua 式）
- self_evolution.py 反思学习曲线（Reflexion 式多轮迭代）
- kv_cache.py       前缀冻结命中率对照（vLLM 式工程指标）
- model_ladder.py   训练阶梯（base→SFT→DPO→GRPO 同集对比）
"""
