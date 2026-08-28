# 黄河吕梁段防汛预警智能体 — 需求分析

> 版本：v1.0 | 日期：2026-07-26 | 状态：已评审
> 关联文档：[设计文档](design.md) | [实施计划](implementation-plan.md)
>
> 本文档覆盖整个项目（推理链路 + 训练流水线 + 部署），其中推理链路已建成，
> 训练流水线与全栈 Docker 化为本期核心交付内容。

---

## 1. 项目概述

### 1.1 背景

黄河吕梁段（重点断面：吴堡、龙门、府谷水文站）汛期洪水威胁大，基层防汛决策
需要融合天气、实时水情、径流预测、GIS 地形河床、政策法规等多源信息，人工研判
耗时长、口径不一。本项目构建防汛预警智能体，通过大模型 Function Calling 自主
编排工具调用，输出**标准化预警等级（Ⅰ/Ⅱ/Ⅲ/Ⅳ）与具体应急预案**。

### 1.2 目标

1. 建成覆盖"感知 → 研判 → 预案"全链路的防汛预警智能体（已完成）。
2. 构建 Hermes 范式 Function Calling 训练集，通过 LoRA 半参微调将通用基座模型
   （Qwen2.5-7B-Instruct）领域化为防汛专用模型（本期）。
3. 开发基于规则奖励的强化学习对齐机制，通过 GRPO 算法更新模型权重，使模型输出
   的预警等级与规则引擎保持一致、工具调用合规、预案有法规依据（本期）。
4. 全栈 Docker 化交付，微调模型经 vLLM 以 OpenAI 兼容接口接入现有系统（本期）。

### 1.3 范围

| 范围内 | 范围外 |
|---|---|
| Hermes 训练集合成与校验 | 径流预测物理模型的改进（SCS-CN 维持现状） |
| QLoRA 微调与 GRPO 对齐训练 | 水文/气象数据源的接入扩展（维持现有源） |
| 微调模型 vLLM 部署与链路切换 | 多区域推广（本期仅黄河吕梁段） |
| 全栈 docker-compose 编排 | 预警信息发布渠道（短信/广播对接） |
| 训练侧评估与测试 | 前端界面改版 |

### 1.4 术语

| 术语 | 含义 |
|---|---|
| Hermes 范式 | NousResearch Hermes 系列的 Function Calling 数据格式：`<tool_call>`/`<tool_response>` 标签包裹的多轮工具调用对话 |
| LoRA / QLoRA | 低秩适配半参微调；QLoRA 为 4bit 量化基座上的 LoRA |
| GRPO | Group Relative Policy Optimization，组内相对优势策略优化 RL 算法 |
| 规则奖励 | 不依赖奖励模型，用确定性规则代码计算 reward 的 RL 对齐方式 |
| 综合研判 | 融合多工具结果计算预警等级并生成预案的过程（`agent/graph/synthesizer.py`） |

---

## 2. 现状分析

### 2.1 已建成能力（基线）

| 模块 | 位置 | 能力 |
|---|---|---|
| 执行链路 | `agent/graph/` | LangGraph 4 节点状态机：planner → executor → synthesizer（+ direct_chat 闲聊路径），planner 原生 Function Calling 统一路由 |
| 工具层 | `agent/tools/` | 8 个工具：get_weather / get_hydrology / predict_runoff / query_gis_terrain / search_regulation / generate_plan / web_search / list_skills（元工具），Pydantic schema + OpenAI tools 导出 |
| 规则引擎 | `agent/graph/synthesizer.py` | 基于流量/降雨/水位状态计算 I-IV 级预警，输出标准应急措施（GRPO 奖励的规则来源） |
| 径流模型 | `agent/hydrology/scs_cn.py` | SCS-CN 降雨-径流预测 |
| GIS 分析 | `agent/gis/` | DEM 加载、坡度/河床断面/淹没分析 |
| 法规 RAG | `agent/rag/` + Qdrant | 5 部真实法规向量检索（防洪法/防汛条例/黄河水量调度/山西预案/黄河预案） |
| 后端 | `backend/app/` | FastAPI + SSE 流式 + 限流 + 结构化日志 |
| 前端 | `frontend/` | Vue 3 + TypeScript（无 UI 库，自写组件），SSE 对话界面 |
| 记忆 | `agent/memory/` | 自进化三层记忆（长期/技能/反思），经验检索注入 + Curator 定期治理 |
| 测试/CI | `backend/tests/` + `.github/workflows/ci.yml` | 400+ 单测、ruff、前端构建 |

### 2.2 缺口（本期交付）

| 缺口 | 现状 | 影响 |
|---|---|---|
| Hermes 训练集 | `train/` 为空 | 无法微调 |
| LoRA 微调 | `train/lora/` 为空 | 无法领域化基座模型 |
| GRPO 对齐 | `train/grpo/` 为空 | 模型输出无法与规则引擎对齐 |
| Docker 部署 | `docker/` 为空 | 无法全栈交付、微调模型无法接入 |

### 2.3 用户故事

- US-1（防汛值班员）：输入"吴堡站现在水情怎么样，需要预警吗"，智能体自动调用
  水情/天气/径流工具，输出预警等级、研判依据和应急措施。
- US-2（模型工程师）：运行一条命令合成指定规模的 Hermes 训练集，且每条样本的
  最终等级与规则引擎一致（不合格样本被自动过滤）。
- US-3（模型工程师）：在单卡 24GB GPU 上完成 QLoRA 微调和 GRPO 对齐，产出可
  部署的合并权重。
- US-4（运维工程师）：`docker compose up` 一键拉起 vLLM + 后端 + 前端 + Qdrant，
  通过环境变量切换 DashScope / 本地微调模型。
- US-5（项目评审）：查看 base / SFT / SFT+GRPO 三版模型在 held-out 评估集上的
  对比报告，证明 GRPO 对齐有效。

---

## 3. 功能需求

优先级：P0 = 本期必须交付，P1 = 应交付，P2 = 可选增强。

### 3.1 训练数据生成（`train/data_gen/`）

| 编号 | 需求 | 优先级 |
|---|---|---|
| FR-D1 | 场景生成器：组合水文站 × 流量档位 × 降雨档位 × 水位状态 × 查询类型，生成多样化防汛查询场景；I/II/III/IV 四级场景配额可配置且默认均衡 | P0 |
| FR-D2 | 教师合成：调用 DashScope qwen-plus + 现有 OpenAI tools schema，生成 Hermes 格式多轮工具调用轨迹；工具结果由 mock_executor 确定性回放 | P0 |
| FR-D3 | 规则过滤三道硬门槛：① JSON 合法 + Pydantic 参数校验；② 调用序列合法（如 predict_runoff 之前须有 get_weather）；③ 最终等级与 `compute_warning_level()` 一致 | P0 |
| FR-D4 | 输出 `hermes_fc_v1.jsonl`（3,000-5,000 条），train/val = 95:5 切分，含样本元数据（场景标签、等级、工具序列） | P0 |
| FR-D5 | 闲聊/越界查询负样本占比 5%-10%，训练意图边界 | P1 |
| FR-D6 | 数据集统计报告：等级分布、工具分布、平均轮次、过滤率 | P1 |

### 3.2 LoRA 微调（`train/lora/`）

| 编号 | 需求 | 优先级 |
|---|---|---|
| FR-L1 | QLoRA 4bit NF4 + BF16 计算，基座 Qwen2.5-7B-Instruct，单卡 24GB 可运行（gradient checkpointing 开启） | P0 |
| FR-L2 | LoRA r=16, α=32, dropout=0.05，作用于 q/k/v/o/gate/up/down 投影 | P0 |
| FR-L3 | assistant-only loss（completion mask），序列长 4096，有效 batch 32，lr=1e-4 cosine，3 epochs | P0 |
| FR-L4 | 训练产出：LoRA adapter + 合并后全量权重（供 vLLM 加载） | P0 |
| FR-L5 | 训练日志（loss 曲线）与 checkpoint 管理（保留 best + last） | P1 |

### 3.3 GRPO 对齐（`train/grpo/` + `train/rewards/`）

| 编号 | 需求 | 优先级 |
|---|---|---|
| FR-G1 | 从 SFT checkpoint 出发，trl GRPOTrainer + vLLM colocate 采样，每组 G=8，temperature=1.0 | P0 |
| FR-G2 | 规则奖励三维度（总分 1.0）：r1 预警等级正确性 0.4（与规则引擎一致）；r2 工具调用正确性 0.3（参数校验 + 工具命中 + 顺序合法）；r3 预案质量与法规依据 0.3（四要素完备 + 引用条款与 RAG 检索一致） | P0 |
| FR-G3 | 格式门控：JSON 解析失败或标签非法 → 总奖励 0 | P0 |
| FR-G4 | GRPO prompts 与 SFT 数据集不重叠（场景生成器新种子） | P0 |
| FR-G5 | KL β=0.04，lr=1e-6，2-3 轮迭代；奖励函数直接复用 `agent.graph.synthesizer` | P0 |
| FR-G6 | 训练中工具执行走 mock_executor（确定性、无外部依赖） | P1 |

### 3.4 评估（`train/eval/`）

| 编号 | 需求 | 优先级 |
|---|---|---|
| FR-E1 | 300 条 held-out 评估场景（与训练/GRPO prompts 均不重叠） | P0 |
| FR-E2 | 指标：端到端等级准确率、三维奖励均值、工具调用成功率 | P0 |
| FR-E3 | 对比报告：base vs SFT vs SFT+GRPO 三版模型 | P0 |

### 3.5 部署（`docker/`）

| 编号 | 需求 | 优先级 |
|---|---|---|
| FR-P1 | docker-compose 编排四服务：vllm（GPU，合并权重，OpenAI /v1 接口）、backend、frontend（nginx 托管 Vue build）、qdrant | P0 |
| FR-P2 | backend 通过环境变量 `LLM_PROVIDER=local|dashscope` + `LLM_BASE_URL` 切换模型源，零代码改动 | P0 |
| FR-P3 | 训练容器可选（`--profile train`），封装 PyTorch + CUDA + 训练依赖，保证训练可复现 | P1 |
| FR-P4 | 模型权重以 volume 挂载，不进镜像 | P0 |

### 3.6 已建成功能的验收基线（回归要求）

| 编号 | 需求 | 优先级 |
|---|---|---|
| FR-R1 | 现有 90 个后端单测全部保持绿色 | P0 |
| FR-R2 | 切换到本地微调模型后，SSE 流式对话链路端到端可用 | P0 |
| FR-R3 | 预警等级规则引擎行为不变（GRPO 对齐模型输出须向其收敛，而非修改规则） | P0 |

---

## 4. 非功能需求

| 编号 | 类别 | 需求 |
|---|---|---|
| NFR-1 | 硬件 | 训练：单卡 24GB（RTX 4090 级）；推理：同卡可同时托管 vLLM 7B 服务 |
| NFR-2 | 性能 | 数据合成：5k 条 ≤ 24h（受教师 API 限流约束）；SFT ≤ 12h；GRPO ≤ 24h |
| NFR-3 | 可复现 | 场景生成器、数据切分、训练配置全部 seed 固定 + 配置文件化 |
| NFR-4 | 质量 | 新增代码遵循现有规范：ruff lint 零告警、pytest 覆盖核心逻辑（数据过滤、奖励函数、场景生成器） |
| NFR-5 | 安全 | API Key 仅经环境变量注入；训练数据不含真实个人信息；镜像不含硬编码密钥 |
| NFR-6 | 兼容 | 训练侧不修改 `agent/` 推理代码的公共接口；仅允许复用 import |
| NFR-7 | 可观测 | 训练过程日志结构化；GRPO 奖励分量按 step 记录可绘图 |

---

## 5. 数据需求

| 数据 | 来源 | 说明 |
|---|---|---|
| 教师模型 | DashScope qwen-plus | 复用现有 `LLM_API_KEY`，合成训练轨迹 |
| 工具执行环境 | `agent/tools/mock_executor.py` | 确定性回放，保证数据可复现、训练无外部依赖 |
| 场景参数 | 规则引擎阈值（2000/3000/5000 m³/s 等） | 反推流量/降雨/水位组合，保证四级覆盖 |
| 法规语料 | `data/raw/regulations/`（5 部） | r3 奖励的一致性校验来源 |
| 基座模型 | Qwen2.5-7B-Instruct（ModelScope/HF） | 约 15GB，训练前下载 |

## 6. 接口需求

| 接口 | 方向 | 约定 |
|---|---|---|
| 微调模型 → backend | OpenAI 兼容 | vLLM `/v1/chat/completions`，支持 tools 参数；backend 仅改 `LLM_BASE_URL`/`LLM_MODEL` |
| 奖励函数 → 规则引擎 | Python import | `train/rewards/` 直接 `from agent.graph.synthesizer import compute_warning_level` |
| 数据生成 → 工具 schema | Python import | 复用 `agent.tools.schemas.build_openai_tools()`，单一事实来源 |
| GRPO → 工具执行 | Python import | `mock_executor` 确定性执行 |

## 7. 约束条件

1. **技术栈**：PyTorch、trl、peft、vLLM、LangChain/LangGraph（现状）、GDAL（GIS 现状）、Docker、FastAPI；训练框架限定 trl + peft（方案 A 已评审）。
2. **硬件**：单卡 24GB 显存硬约束 → 必须 QLoRA；GRPO 采样组 G ≤ 8。
3. **基座模型**：Qwen2.5-7B-Instruct（与线上 qwen 同族，原生 Function Calling）。
4. **数据规模**：3k-5k 条（教师 API 成本与质量平衡）。
5. **规则权威**：预警等级阈值以 `synthesizer.py` 为唯一权威，训练向其对齐，禁止在训练侧复制阈值逻辑。

## 8. 验收标准

| 编号 | 标准 | 度量 |
|---|---|---|
| AC-1 | Hermes 数据集交付 | 3k-5k 条，三道过滤后等级一致率 100%（抽样 200 条人工核验 ≥ 98%） |
| AC-2 | SFT 模型可用 | held-out 集端到端等级准确率显著高于 base 模型（目标 ≥ 80%，base 预计 ≤ 40%） |
| AC-3 | GRPO 对齐有效 | SFT+GRPO 相比 SFT：等级准确率提升 ≥ 5pp，三维奖励均值提升且训练曲线收敛 |
| AC-4 | 全栈一键部署 | `docker compose up` 后前端可对话，本地模型链路端到端输出预警等级与预案 |
| AC-5 | 回归无损 | 现有 90 单测绿色；ruff 零告警；DashScope 链路可切回 |

## 9. 风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| 7B 模型 Function Calling 能力弱，SFT 后仍格式漂移 | 中 | 高 | 数据三道硬过滤保证训练信号纯净；GRPO 格式门控奖励兜底；保留 DashScope 作为降级 |
| 24GB 显存 GRPO OOM | 中 | 高 | vLLM colocate 模式 + G=8 起步，OOM 则降 G=4 / 缩短 max_completion_length / 关 vLLM 显存余量 |
| 教师合成数据多样性不足（场景坍缩） | 中 | 中 | 场景生成器强制配额均衡；query 相似度去重；FR-D6 统计报告监控分布 |
| 教师 API 限流/成本超支 | 低 | 中 | 合成限速 + 断点续合成；预算上限告警 |
| GRPO 奖励 hacking（如堆关键词骗 r3） | 中 | 中 | r3 要求引用条款与 RAG 检索结果一致（非纯关键词）；评估集人工抽审 |
| Docker GPU 环境差异（Windows 开发机 vs Linux 部署机） | 中 | 低 | 训练/部署容器均基于 NVIDIA 官方 CUDA 镜像；compose 文件注明 nvidia-container-toolkit 前置要求 |

## 10. 依赖

- 外部服务：DashScope API（教师合成 + 线上降级）、高德天气 API（现状）、Qdrant（现状）
- 开源模型与库：Qwen2.5-7B-Instruct、trl ≥ 0.12、peft、vLLM、bitsandbytes、transformers、datasets
- 硬件：24GB GPU（训练 + 本地推理）
