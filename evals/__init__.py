"""WaterAgents 系统级评估包。

按《AI Agents in Depth》第 6 章「Agent 的评估」方法论构建，
评估对象是"模型 + Harness 组合体"——驱动真实 Agent 链路
（planner → executor → synthesizer），不走裸模型合成循环。

模块：
- cases.py      评估数据集（8 类用例 + 能力标签 + 种子隔离）
- case_sets.py  实验 ↔ 固定案例组合映射（量化实验的子集口径）
- coverage.py   覆盖矩阵守门（类型/档位/针埋设位置断言）
- replay.py     评估环境（可重置：mock overrides + seed 回放）
- runner.py     执行协议（驱动 run_graph_agent，采集轨迹与结果）
- metrics.py    指标体系（过程/结果/安全三层 + 95% 置信区间 + 能力矩阵）
- judge.py      LLM-as-Judge（Rubric 式，可 --no-judge 跳过）
- ablation.py   记忆消融（有/无记忆注入对比，定位 Harness 组件贡献）
- regression.py 基线回归门禁（阈值 = max(固定下限, 2×SE)）
- report.py     Markdown 报告生成（含量化声明表）
- run_eval.py   CLI 入口（--experiment 量化实验 / --models 训练阶梯）
- experiments/  量化实验包（docs/eval-experiments.md）：
                base 消融框架 / memory 记忆增益 / compression 压缩等价性 /
                self_evolution 反思学习曲线 / kv_cache 前缀冻结 / model_ladder 训练阶梯
"""
