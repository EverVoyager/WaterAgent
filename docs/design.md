# 黄河吕梁段防汛预警智能体 — 设计文档

> 版本：v1.0 | 日期：2026-07-26 | 状态：已评审
> 关联文档：[需求分析](requirements-analysis.md) | [实施计划](implementation-plan.md)
>
> 本文档覆盖整个系统：已建成的推理链路（第 2 章简述接口契约）、
> 本期新建的训练流水线（第 3-6 章详设）、部署（第 7 章）、测试（第 8 章）。

---

## 1. 设计总览

### 1.1 核心设计决策

| 决策 | 选择 | 理由 |
|---|---|---|
| 训练/推理解耦方式 | OpenAI 兼容协议 | backend `llm.py` 已是 OpenAI 客户端，微调模型经 vLLM 托管后仅改环境变量即可切换，零代码侵入 |
| 训练框架 | trl + peft 同框架（方案 A） | SFT 与 GRPO 聊天模板/tokenizer 无缝衔接；规则奖励可直接 import 现有规则引擎 |
| 半参微调 | QLoRA 4bit NF4 | 单卡 24GB 硬约束下唯一可行的 7B 微调方案 |
| 对齐算法 | GRPO + 规则奖励（非奖励模型） | 防汛等级有确定性规则（`synthesizer.py`），规则奖励比训练奖励模型更准、更省 |
| 规则权威 | 单一来源 `agent/graph/synthesizer.py` | 训练侧只做 import 复用，禁止复制阈值逻辑，防止双份规则漂移 |
| 工具执行（训练时） | `mock_executor` | 确定性、无网络依赖、可复现；与线上 real_executor 同一 schema |

### 1.2 系统上下文

```
教师模型(qwen-plus)                ┌────────── 现有推理链路（不改动）──────────┐
      │ 合成轨迹                   │ 前端 Vue ──SSE──▶ FastAPI ──▶ LangGraph  │
      ▼                           │                          │  planner FC  │
┌───────────┐   ┌───────────┐    │                          ▼              │
│ 场景生成器 │──▶│ Hermes 数据集│   │                     6 工具执行层          │
└───────────┘   └─────┬─────┘    └──────────────▲───────────────────────────┘
                      ▼                         │ OpenAI 兼容 /v1
              ┌───────────────┐        ┌────────┴─────────┐
              │ QLoRA SFT     │        │ DashScope (现状)  │
              └───────┬───────┘        │ vLLM 本地模型(新)  │◀── 合并权重
                      ▼                └──────────────────┘
              ┌───────────────┐                ▲
              │ GRPO + 规则奖励 │── 合并权重 ────┘
              └───────────────┘
```

---

## 2. 推理链路接口契约（现状，训练侧依赖）

训练侧对推理代码仅有**只读 import 依赖**，以下为被依赖的公共接口（冻结面）：

| 接口 | 位置 | 训练侧用途 |
|---|---|---|
| `build_openai_tools() -> list[dict]` | `agent/tools/schemas.py` | 教师合成与 SFT 数据的 tools 字段唯一来源 |
| `TOOL_PARAM_MODELS` | `agent/tools/schemas.py` | 过滤器/r2 奖励的 Pydantic 参数校验 |
| `compute_warning_level(tool_results) -> (level, reasoning)` | `agent/graph/synthesizer.py` | 数据过滤第③道 + r1 奖励的等级真值 |
| `get_actions_for_level(level, area)` | `agent/graph/synthesizer.py` | r3 奖励的预案要素参照 |
| `mock_executor` | `agent/tools/mock_executor.py` | 合成回放 + GRPO 训练时工具执行 |
| RAG 检索 | `agent/rag/vector_store.py` | r3 奖励的法规一致性校验 |

**约束**：训练侧不得修改上述接口签名；如需扩展（如 mock 场景注入参数），
以新增可选参数方式演进，保持向后兼容。

---

## 3. 训练数据生成子系统（`train/data_gen/`）

### 3.1 模块结构

```
train/data_gen/
├── scenario.py        # 场景生成器
├── teacher.py         # 教师合成客户端（DashScope + 限速 + 断点续传）
├── filters.py         # 三道规则过滤
├── hermes_format.py   # Hermes 文本序列化/解析（与 Qwen 聊天模板对齐）
├── build_dataset.py   # 编排入口：场景 → 合成 → 过滤 → JSONL → 切分
└── stats.py           # FR-D6 分布统计报告
```

### 3.2 场景生成器（scenario.py）

确定性伪随机（seed 固定）组合以下维度，每条场景带唯一 `scenario_id`：

| 维度 | 取值 | 说明 |
|---|---|---|
| station | 吴堡 / 龙门 / 府谷 | 现状三个重点站 |
| flow_tier | <2000 / [2000,3000) / [3000,5000) / ≥5000 m³/s | 对齐规则引擎阈值，决定等级真值 |
| rain_tier | <50 / [50,100] / >100 mm/24h | 同上 |
| level_status | normal / warning / guaranteed | 同上 |
| query_type | single_tool / multi_tool / plan_only / chatty | 单工具查询 / 综合研判 / 预案生成 / 闲聊负样本 |
| persona | 值班员 / 乡镇干部 / 企业负责人 | 丰富 query 表述 |

- 等级配额：I/II/III/IV 默认各 25%（可配置）；`chatty` 占 5%-10%（FR-D5）。
- 由 flow/rain/level 三元组**预计算等级真值** `expected_level`（按规则引擎阈值构造），
  写入场景元数据，供过滤与评估复用；`predict_runoff` 的 peak 覆盖值取 flow 的
  1.0-1.1 倍且**不越过本档上限**（防止跨档导致规则引擎重算等级与场景真值不一致）。
- query 文本由模板 × persona × 参数渲染 + 教师模型润色，随后 embedding 相似度
  去重（阈值 cosine ≥ 0.92 视为重复）。

### 3.3 教师合成（teacher.py）

- 输入：场景 + `build_openai_tools()` + 系统提示（复用 `agent/prompts/` 防汛提示词）
- 循环：教师输出 tool_call → `mock_executor`（按场景参数注入水情/天气数值）回放
  tool_response → 追加消息，直至教师输出最终研判（含等级 + 预案）或达到 8 轮上限
- 限速：令牌桶（默认 30 RPM，可配置）；失败指数退避重试 3 次
- 断点续传：每完成一条追加写入 `raw_traces.jsonl`，重跑跳过已有 `scenario_id`

### 3.4 Hermes 格式（hermes_format.py）

采用 Qwen 聊天模板承载 Hermes 标签（与 Qwen2.5-Instruct 原生格式一致）：

```
<|im_start|>system
{system_prompt}
# Tools
{tools_json}
<|im_end|>
<|im_start|>user
{query}
<|im_end|>
<|im_start|>assistant
<tool_call>
{"name": "get_hydrology", "arguments": {"station": "吴堡", "metric": "both"}}
</tool_call>
<|im_end|>
<|im_start|>tool
<tool_response>
{"station": "吴堡", "water_level_m": 818.2, "flow_m3_s": 3250.0, ...}
</tool_response>
<|im_end|>
<|im_start|>assistant
……最终研判（预警等级 + 依据 + 应急预案）……
<|im_end|>
```

- 序列化与解析互逆，`parse_trace()` 供过滤器、奖励函数、评估三处复用（单一解析实现）。
- JSONL 每行：`{scenario_id, level, query_type, messages: [...], tools_used: [...], rounds: n}`。

### 3.5 三道规则过滤（filters.py）

| # | 过滤 | 判定 |
|---|---|---|
| F1 | 格式与参数 | `parse_trace` 成功；每个 tool_call 的 arguments 通过 `TOOL_PARAM_MODELS[name]` 校验 |
| F2 | 序列合法 | 工具名 ∈ 6 工具；`predict_runoff` 之前出现过 `get_weather`；无重复相同调用 |
| F3 | 等级一致 | 对回放工具结果重算 `compute_warning_level`，与轨迹最终等级相同（chatty 样本豁免：要求无 tool_call 且无等级输出） |

任一不过 → 丢弃并记录原因（供 FR-D6 报告过滤率）。F3 是保证"训练信号与
规则引擎一致"的核心门槛，**通过率目标 ≥ 70%**，低于该值应调整教师提示词而非放宽过滤。

### 3.6 切分与产出

- 产出 `train/lora/data/hermes_fc_v1.jsonl`；按 `scenario_id` 分层（保持等级比例）
  95:5 切分 train/val，seed 固定。
- GRPO 与评估场景由场景生成器以**独立种子区间**生成（见 5.2、6.1），与 SFT 集零重叠。

---

## 4. LoRA 微调子系统（`train/lora/`）

### 4.1 结构与配置

```
train/lora/
├── configs/sft_qlora.yaml   # 全部超参（唯一事实来源）
├── train_sft.py             # trl SFTTrainer 入口
├── merge.py                 # adapter 合并为全量权重（供 vLLM）
└── data/                    # hermes_fc_v1.jsonl（数据生成子系统产出）
```

关键配置（`sft_qlora.yaml`）：

| 项 | 值 |
|---|---|
| base_model | Qwen/Qwen2.5-7B-Instruct |
| 量化 | bnb 4bit NF4 + double_quant，compute bf16 |
| LoRA | r=16, α=32, dropout=0.05, target=[q,k,v,o,gate,up,down]_proj |
| 序列 | max_len=4096，packing=off，assistant-only loss（completion mask 仅计 assistant 段） |
| batch | per_device=2 × grad_accum=16（有效 32） |
| 优化 | AdamW，lr=1e-4，cosine，warmup 3%，3 epochs，grad ckpt on |
| 显存估算 | 权重 4bit ≈ 4.5GB + LoRA 优化器 ≈ 2GB + 激活/碎片 ≈ 12-15GB → 合计 ~18-22GB |
| 输出 | `outputs/sft/adapter/` + `merge.py` 产出 `outputs/sft/merged/` |

### 4.2 数据流

```
hermes_fc_v1.jsonl
  → datasets.load_dataset(json)
  → 按 messages 渲染 Qwen 聊天模板 + completion mask
  → SFTTrainer → adapter → merge → merged/（HF 格式，vLLM 可直接加载）
```

### 4.3 容错

- OOM 降级路径（文档化）：per_device 2→1、seq 4096→3072、LoRA target 裁剪 mlp 投影
- checkpoint：每 200 step 保存，保留 best(val loss) + last；中断可从 last 续训

---

## 5. GRPO 对齐子系统（`train/grpo/` + `train/rewards/`）

### 5.1 结构

```
train/rewards/
├── format_gate.py      # 门控：parse_trace 失败 → 0
├── r1_level.py         # 等级正确性 0.4
├── r2_tool_call.py     # 工具调用正确性 0.3
├── r3_plan.py          # 预案质量与法规依据 0.3
└── composite.py        # 加权合成 + 分量日志
train/grpo/
├── configs/grpo.yaml
├── prompts.py          # 独立种子场景 → prompt 集（与 SFT 零重叠）
├── rollouts.py         # mock_executor 工具回放环境
└── train_grpo.py       # trl GRPOTrainer 入口
```

### 5.2 训练流程

```
SFT merged 模型 ──▶ GRPOTrainer（policy）
prompts.py（独立种子区间 [100_000, 200_000)）生成查询
每个 prompt 经 vLLM colocate 采样 G=8 条补全（temperature=1.0）
每条补全：format_gate →（过）→ r1+r2+r3 → 合成 reward
              └（不过）→ reward = 0
组内相对优势 → 策略更新（KL β=0.04 锚定 SFT 模型）
```

**补全格式约定**：trl GRPOTrainer 标准流程为**单轮补全**——模型在一次生成中
输出完整调用计划（多个 `<tool_call>` 块拼接）+ 最终研判段。奖励函数对该单段
文本提取全部调用（r2）与最终段（r1/r3），无需交互式多轮 rollout；
确定性工具回放（`replay_tool_call`）供多轮扩展与评估复用。

| 超参 | 值 |
|---|---|
| G（组大小） | 8（OOM 降 4） |
| temperature / max_completion | 1.0 / 1024 |
| lr / β(KL) / 迭代 | 1e-6 / 0.04 / 2-3 轮（按奖励曲线早停） |
| prompts 规模 | 每轮 512 条 |

### 5.3 规则奖励设计

**门控（format_gate.py）**：`parse_trace` 解析失败、标签非法、最终段缺失 → reward=0，
不再计算分量。格式分不单列，避免模型学会"格式正确但内容空"。

**r1 等级正确性（0.4，r1_level.py）**：
- 从补全最终段提取等级（正则 `Ⅰ|Ⅱ|Ⅲ|Ⅳ|I|II|III|IV` 归一化）；
- 真值：对该 prompt 场景回放结果重算 `compute_warning_level`；
- 一致 → 0.4；相邻一级（如真 II 出 III）→ 0.2（部分分，缓解稀疏奖励）；其余 → 0。

**r2 工具调用正确性（0.3，r2_tool_call.py）**，按子项累加：
- 0.1：所有 tool_call 参数通过 Pydantic 校验；
- 0.1：调用集合 ⊆ 场景参考工具集（场景元数据携带）且至少含 1 个必需工具；
- 0.1：顺序合法（predict_runoff 前有 get_weather；generate_plan 在最终段之前）。

**r3 预案质量与法规依据（0.3，r3_plan.py）**，按子项累加：
- 0.15：预案四要素齐备——转移、物资、责任人/责任单位、时限（模板要素匹配，
  参照 `get_actions_for_level(level)` 的要素类别）；
- 0.15：引用法规条款命中 RAG 检索结果（对 query 执行 `search_regulation` top_k=3，
  预案中引用的条款名/条文号须 ∩ 检索结果非空），防关键词堆砌式 reward hacking。

**composite.py**：`reward = gate × (r1 + r2 + r3)`；每 step 记录三分量均值，
写 JSONL 日志供曲线绘制（NFR-7）。

### 5.4 24GB 显存预算（GRPO）

policy 4bit 推理(vLLM colocate, gpu_memory_utilization=0.35) + 训练态 LoRA 反传
+ ref 模型（冻结 SFT merged，4bit 加载）→ 峰值 ~20-22GB；OOM 降级顺序：
G 8→4 → max_completion 1024→768 → vLLM 显存占比 0.35→0.25。

---

## 6. 评估子系统（`train/eval/`）

- `eval_scenarios.py`：第三段独立种子生成 300 条场景（等级分层均衡）；
- `run_eval.py`：对 base / SFT / SFT+GRPO 三版模型，经 vLLM 批量推理 + mock 回放，
  复用 `train/rewards/` 与等级真值计算；
- 指标：端到端等级准确率（主指标）、三维奖励均值、工具调用成功率、平均轮次；
- `report.py`：输出 Markdown 对比表（验收 AC-2/AC-3 的证据）。

## 7. 部署设计（`docker/`）

### 7.1 拓扑

```
docker/
├── docker-compose.yml
├── backend.Dockerfile        # python:3.11-slim + uvicorn
├── frontend.Dockerfile       # node:20 build → nginx:alpine 托管
├── train.Dockerfile          # nvidia/cuda:12.4 + torch + trl/peft/vllm（profile=train）
├── nginx/default.conf        # /api 反代 backend，/ 托管前端静态
└── .env.example              # LLM_PROVIDER / LLM_BASE_URL / LLM_API_KEY / 模型路径
```

### 7.2 compose 服务

| 服务 | 镜像 | 关键点 |
|---|---|---|
| vllm | vllm/vllm-openai | `--model /models/grpo-merged --served-model-name water-agent-fc --enable-auto-tool-choice --tool-call-parser hermes`（**必需**，否则服务端不解析 `<tool_call>`，FC 链路失效）；GPU 预留；volume 挂权重 |
| backend | backend.Dockerfile | `LLM_PROVIDER=local` 时 `LLM_BASE_URL=http://vllm:8000/v1`；depends_on vllm+qdrant（healthcheck） |
| qdrant | qdrant/qdrant | volume 挂现有 `tools/qdrant/storage` 数据 |
| frontend | frontend.Dockerfile | nginx 反代 `/api → backend:8000` |
| train（可选） | train.Dockerfile | `profiles: ["train"]`，挂载仓库与数据集 |

### 7.3 模型切换契约

backend 现状已读 `LLM_BASE_URL`/`LLM_MODEL`/`LLM_API_KEY`（OpenAI 客户端），
仅约定取值：`LLM_PROVIDER=dashscope` → DashScope 默认；`LLM_PROVIDER=local` →
指向 vllm 服务 + `LLM_MODEL=water-agent-fc`。**不新增配置项**，如现状代码缺
`LLM_PROVIDER` 则仅作文档层约定，不改变现有行为。

### 7.4 部署安全

- `.env` 不入库；镜像构建不复制 `.env`；密钥仅 runtime 注入；
- vLLM 服务仅 compose 内网暴露，不映射宿主机端口；
- 权重以只读 volume 挂载。

---

## 8. 测试设计

### 8.1 训练侧单元测试（并入 `backend/tests/` 或新增 `train/tests/`）

| 测试 | 断言 |
|---|---|
| 场景生成器 | seed 固定输出确定；等级配额 ±2%；四维组合无非法值 |
| Hermes 序列化 | serialize/parse 互逆（round-trip）；非法 JSON/缺标签被 parse 拒绝 |
| 三道过滤 | 构造正/反例：参数越界被 F1 拒、乱序被 F2 拒、等级不符被 F3 拒、chatty 豁免生效 |
| 奖励函数 | 全对 → 1.0；格式坏 → 0；等级相邻 → r1 部分分；r3 无 RAG 命中 → 扣除 0.15 |
| GRPO prompts | 与 SFT 集 scenario_id 零交集 |

### 8.2 回归

- 现有 90 单测全绿（FR-R1）；ruff 对 `train/` 零告警（并入现有配置）；
- 端到端冒烟：docker compose 起栈后 `curl /api/health/ready` + 一条 SSE 查询断言等级字段。

### 8.3 TDD 顺序

按实施计划阶段推进，每个模块先写失败测试再实现（红→绿→重构），
训练脚本类（长耗时）以"小样本 smoke run"作为测试替身（10 条数据跑通全流程）。

---

## 9. 里程碑产物

| 里程碑 | 产物 |
|---|---|
| M1 数据 | `hermes_fc_v1.jsonl` + 统计报告 + 过滤单测 |
| M2 SFT | `outputs/sft/merged/` + loss 曲线 + smoke 推理样例 |
| M3 GRPO | `outputs/grpo/merged/` + 奖励分量曲线 |
| M4 评估 | base/SFT/GRPO 对比报告（AC-2/AC-3 证据） |
| M5 部署 | docker-compose 全栈 + 冒烟通过 + README 更新 |
