# 水卫 · WaterAgents

**微调开源大模型 → LlamaFactory 本地部署 → Agent 洪水预警**

对 Qwen3-4B 做水利 / 防汛领域微调，经 LlamaFactory 在本地部署为推理服务，
再接入 LangGraph Agent：融合实时水情、天气、径流模型、法规 RAG 与 GIS 地形，
面向黄河吕梁段（吴堡、龙门水文站）输出预警等级（Ⅰ~Ⅳ）、研判依据与应急措施。

## 项目主线

- **[第一步：微调水利专用大模型](#第一步微调水利专用大模型)** —— 把云端强模型（qwen-plus）的 Agent 能力蒸馏进本地开源模型：Self-Instruct 数据管线 + SFT / DPO / GRPO
- **[第二步：LlamaFactory 本地部署](#第二步llamafactory-本地部署)** —— 微调产物合并权重，LlamaFactory（可换 vLLM）起 OpenAI 兼容 API
- **[第三步：Agent 洪水预警](#第三步agent-洪水预警)** —— LangGraph 状态机调度 8 个业务工具，接入多源实时数据，输出预警等级 / 研判依据 / 应急措施

## 特性

- 微调管线：Self-Instruct 种子扩张 → 场景参数化（等级真值）→ 双模型蒸馏 → 三道硬过滤 → LLM-as-Judge → SFT + DPO + GRPO（纯规则奖励），QLoRA 在 RTX 4060 单卡可跑
- 本地部署：微调产物经 LlamaFactory 起 OpenAI 兼容服务，后端改两项配置即完成接入；也支持 vLLM
- 工具编排：无关键词路由，planner 原生 Function Calling 自主规划多轮工具调用（去重 / 轮次控制 / 跨工具数据流注入），并发执行 + TTL 缓存
- 真实数据源：水文爬虫（qqjjsj.com）、高德天气、Tavily 联网搜索、Qdrant 法规 RAG、SRTM DEM 地形分析、SCS-CN 降雨径流模型；生产环境真实源失败硬失败，不静默降级 mock
- 两阶段流式 SSE：先推结构化预警卡（等级 / 依据 / 措施 / 引用），再逐 token 流式答案；15s 心跳保活 + 客户端断连协作式取消
- 五类记忆：会话（上下文压缩）、长期（MEMORY.md 用户手册 Agent 只读 + memory/ 自动记忆目录）、语义 / 情景 / 程序（MySQL + Qdrant 向量检索）；反思异步写入、Curator 定期治理、程序记忆可晋升 Skill
- Skill 系统：description embedding 匹配按需加载、工具子集隔离，支持 SKILL.md / .zip 导入（含 ZIP 炸弹与路径穿越防护）
- 引用溯源：引用必须逐字来自来源原文，编造的直接过滤
- 工程兜底：结构化输出三级降级 + 4 级 JSON 修复、上下文压缩（Codex compact）、`<think>` 剥离、LLM 异常分类传播前端、限流 / CORS / 配置校验

## 目录

- [项目主线](#项目主线)
- [第一步：微调水利专用大模型](#第一步微调水利专用大模型)
- [第二步：LlamaFactory 本地部署](#第二步llamafactory-本地部署)
- [第三步：Agent 洪水预警](#第三步agent-洪水预警)
  - [快速开始](#快速开始)
  - [使用示例](#使用示例)
  - [配置说明](#配置说明)
  - [核心设计](#核心设计)
  - [系统架构](#系统架构)
- [Docker 部署](#docker-部署)
- [项目结构](#项目结构)
- [测试与质量](#测试与质量)
- [系统级评估](#系统级评估)
- [贡献](#贡献)
- [许可证](#许可证)

## 第一步：微调水利专用大模型

把云端强模型（qwen-plus）的 Agent 能力蒸馏进开源模型 Qwen3-4B，得到水利 / 防汛领域专用大模型（QLoRA，RTX 4060 单卡可跑）：

1. **种子查询（45 条）** → Self-Instruct 扩张（qwen-plus，2-gram Jaccard 去重）
2. **场景参数化** —— 流量档位 → 等级真值 + mock 工具覆盖值
3. **教师多轮合成** —— 原生 tool_calls 落盘 Hermes 格式，确定性回放
4. **三道硬过滤** —— F1 参数合法 / F2 序列合法 / F3 等级一致
5. **LLM-as-Judge**（qwen-max 四维打分）—— ≥4 进 SFT，1~4 进 DPO 池
6. **产出** —— SFT 训练集（95% 业务 + 5% 知识问答）+ DPO 偏好对，经 QLoRA SFT → merge → GRPO（纯规则奖励）得到水利专用大模型，300 条 held-out 评估（等级准确率 / 工具成功率）

**奖励设计**：`reward = format_gate × (r1 等级 0.4 + r2 工具调用 0.3 + r3 预案质量 0.3)`

**防泄漏**：SFT / GRPO / EVAL 用 seed 区间 [0,100k) / [100k,200k) / [200k,300k) 物理隔离；mock 回放确定性由专项测试保护。

```bash
# 1. 数据集（支持 --dry-run / --no-judge / --no-dpo / 断点续传）
python -m train.data_gen.build_dataset --n 5000 --seed 1000 --rpm 30

# 2. SFT（先 --smoke 冒烟）
python -m train.lora.train_sft --smoke && python -m train.lora.train_sft
python -m train.lora.merge

# 3. GRPO + 再次 merge
python -m train.grpo.train_grpo

# 4. 三模型对比评估（基座 vs SFT vs GRPO）
python -m train.eval.run_eval --models <base> sft-merged grpo-merged --n 300
```

产物：合并后的全量权重，默认在 `models/wateragents-qwen3-4b-v1/`，供第二步部署使用。

## 第二步：LlamaFactory 本地部署

微调产物（`models/wateragents-qwen3-4b-v1/`）通过 [LlamaFactory](https://github.com/hiyouga/LLaMA-Factory) 起本地推理服务，提供 OpenAI 兼容 API。推理配置见 [`train/lora/configs/wateragents_inference.yaml`](train/lora/configs/wateragents_inference.yaml)（方式 A：合并后推理，无 quantization_bit，template=qwen3_nothink）。

### 1. 启动 LlamaFactory API 服务（端口 8001）

```powershell
# 设置 HF 缓存路径，避免重复下载模型
$env:HF_HOME = "D:\hf_cache"
# 必须指定 API_PORT=8001，否则 LlamaFactory 默认监听 8000 会与后端冲突
$env:API_PORT = "8001"
cd d:\AgentProject\WaterAgents
llamafactory-cli api train/lora/configs/wateragents_inference.yaml
```

等待日志出现 `Uvicorn running on http://0.0.0.0:8001` 且模型加载完成（首次约 1-2 分钟）。

> 若启动报 `[Errno 10048] bind on address ('0.0.0.0', 8000)`，说明 LlamaFactory 仍在用默认 8000 端口——确认 `$env:API_PORT = "8001"` 已在同一终端会话中执行；同时检查是否有遗留 python 进程占用 8000：`netstat -ano | findstr :8000`，按 PID 执行 `taskkill /PID <PID> /F` 释放端口。

<details>
<summary>备选：vLLM 部署（Linux / 有 GPU 服务器场景）</summary>

vLLM 原生支持 Qwen3 的工具调用解析（`--enable-auto-tool-choice --tool-call-parser hermes` 与本项目 Hermes 训练格式对应）：

```bash
vllm serve models/wateragents-qwen3-4b-v1 \
  --served-model-name water-agent-fc \
  --max-model-len 8192 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
# 后端 LLM_BASE_URL=http://localhost:8000/v1，LLM_MODEL=water-agent-fc
```

</details>

### 2. 后端切换到本地模型

`backend/.env` 修改三项：

```bash
LLM_API_KEY=local                          # 本地服务不校验，任意非空
LLM_BASE_URL=http://localhost:8001/v1
LLM_MODEL=wateragents-qwen3-4b-v1          # 与推理配置中的模型名一致
```

> Docker 部署时后端在容器内，访问宿主机 LlamaFactory 用 `http://host.docker.internal:8001/v1`。
> 本地 Qwen3-4B QLoRA 能力弱于云端 qwen-plus，复杂多工具规划场景可能不如在线模型稳定。

### 3. 验证

在前端输入「龙门站现在水情怎么样」，Agent 应正确调用 `get_hydrology` 工具并返回研判结果。若思考内容 `<think>...</think>` 泄漏到前端，确认后端已应用 `strip_think` 处理（见 [backend/app/core/llm.py](backend/app/core/llm.py)）。

## 第三步：Agent 洪水预警

本地（或云端）模型就位后，启动 Agent 服务，即可对话式完成防汛预警研判。

### 快速开始

#### 环境要求

- Python 3.10+、Node.js 22.22+（本地开发验证于 24.x；测试依赖 jsdom@30 要求）
- [Qdrant](https://github.com/qdrant/qdrant/releases)（法规向量检索）
- MySQL 8.0（会话 / 记忆 / Skill 持久化；不配则对应功能自动禁用或跳过）

#### 1. 配置后端

```bash
cd backend
cp .env.example .env
# 编辑 .env：
#   用第二步的本地模型 → LLM_BASE_URL=http://localhost:8001/v1，LLM_MODEL=wateragents-qwen3-4b-v1
#   或直接用云端 API → LLM_API_KEY 填 DashScope / DeepSeek / 智谱任一 key
#   其余 API Key 留空自动降级 mock，不影响启动
```

#### 2. 启动 Qdrant 并构建向量库

```bash
backend/start_qdrant.bat        # 或手动运行 qdrant.exe
cd backend && python build_vector_store.py
```

#### 3. （可选）注册种子技能

预置 4 个防汛 Skill（实时水情查询 / 降雨洪水预判 / 预警级别解读 / 应急响应建议）：

```bash
python backend/scripts/seed_skills.py
```

#### 4. 启动服务

```bash
# 后端（端口 8000）
cd backend && python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 前端（端口 5173）
cd frontend && npm install && npm run dev
```

打开 http://localhost:5173 ，输入「吴堡站现在水情怎么样」验证全链路。

### 使用示例

#### SSE 流式（前端同款协议）

```bash
curl -N -X POST http://localhost:8000/api/agent/query/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "吴堡站现在水情怎么样，未来24小时有洪水风险吗"}'
```

事件流（每条 `data: {json}\n\n`）：

```
data: {"type": "reasoning_step", "step": "planner", "phase": "decision", "message": "决定调用工具：get_hydrology, get_weather"}
data: {"type": "tool_call", "tool": "get_hydrology", "arguments": {"station": "吴堡"}, "round": 1}
data: {"type": "tool_result", "tool": "get_hydrology", "result": {"flow_m3_s": 3240, ...}}
data: {"type": "synth_meta", "data": {"warning_level": "II", "reasoning": "...", "actions": ["..."]}}
data: {"type": "answer_delta", "content": "当前吴堡站..."}
data: {"type": "done", "data": {"answer": "...", "warning_level": "II", ...}}
```

#### 非流式

```bash
curl -X POST http://localhost:8000/api/agent/query \
  -H "Content-Type: application/json" \
  -d '{"query": "龙门站水位多少"}'
```

```jsonc
// 响应
{
  "answer": "龙门站当前水位 381.2m...",
  "warning_level": "IV",            // I/II/III/IV，闲聊为空
  "reasoning": "流量低于2000m³/s...",
  "actions": ["启动Ⅳ级应急响应..."],
  "tool_calls": [ ... ],            // 完整工具调用链
  "citations": [ ... ],             // 已校验原文的引用
  "rounds": 2,
  "intent": "agent_task"
}
```

#### REST API 总览

| 方法                 | 路径                                      | 说明                            |
| -------------------- | ----------------------------------------- | ------------------------------- |
| POST                 | `/api/agent/query`                      | 对话（非流式）                  |
| POST                 | `/api/agent/query/stream`               | 对话（SSE 流式）                |
| GET/POST             | `/api/sessions`                         | 会话列表 / 创建（MySQL）        |
| GET/PUT/PATCH/DELETE | `/api/sessions/{id}`                    | 会话详情 / 同步 / 改标题 / 删除 |
| GET/DELETE           | `/api/memories`、`/api/memories/{id}` | 记忆治理（查询 / 删除）         |
| POST                 | `/api/memories/compact`                 | LLM 语义压缩记忆                |
| GET                  | `/api/memories/reflections`             | 反思日志审计                    |
| GET/POST             | `/api/skills`、`/api/skills/tools`    | Skill CRUD / 内置工具清单       |
| POST                 | `/api/skills/import`                    | 导入 .zip / .skill / .md 技能包 |
| GET                  | `/api/health`、`/api/health/ready`    | 存活 / 就绪探针                 |

### 配置说明

完整清单见 [`backend/.env.example`](backend/.env.example)，常用项：

| 变量                                             | 默认               | 说明                                                                |
| ------------------------------------------------ | ------------------ | ------------------------------------------------------------------- |
| `LLM_API_KEY`                                  | —                 | **必填**。云端 key 或本地部署时任意非空值                          |
| `LLM_BASE_URL`                                 | DashScope          | 任意 OpenAI 兼容端点：LlamaFactory（`http://localhost:8001/v1`）/ vLLM / 云端 |
| `LLM_MODEL`                                    | `qwen-plus`      | 主模型；`LLM_JUDGE_MODEL` 为训练管线评判模型                      |
| `EMBEDDING_API_KEY` / `EMBEDDING_BASE_URL`     | 空（回退 `LLM_*`） | Embedding 独立凭证；推理与 embedding 分属不同服务商时必填（如推理 DeepSeek + embedding DashScope） |
| `LLM_MAX_TOOL_ROUNDS`                          | `5`              | 单次会话最大工具轮次（防死循环）                                    |
| `AMAP_API_KEY` / `TAVILY_API_KEY`            | 空                 | 高德天气 / 联网搜索；留空降级 mock                                  |
| `QDRANT_HOST` / `QDRANT_PORT`                | `127.0.0.1:6333` | 法规 RAG 向量库                                                     |
| `MYSQL_PASSWORD`                               | 空                 | 会话 / 记忆 / Skill 持久化；**留空即禁用**                    |
| `SELF_EVOLUTION_ENABLED`                       | `true`           | 自进化反思总开关                                                    |
| `CURATOR_ENABLED` / `CURATOR_INTERVAL_HOURS` | `true` / `168` | 记忆定期治理（剪枝 / 压缩 / 对账）                                  |
| `HISTORY_MAX_TOKENS`                           | `4000`           | 超过即触发历史 LLM 摘要压缩                                         |
| `RATE_LIMIT_PER_MINUTE`                        | `30`             | slowapi 限流                                                        |
| `APP_ENV`                                      | `development`    | `production` 下关键工具真实源失败**硬失败**、校验占位符 Key |

### 核心设计

#### 工作流（4 节点状态机）

无独立路由层（借鉴 OpenAI / Cohere 主流方案）：planner 的 LLM 原生 Function
Calling 统一决策，**不调工具即视为闲聊，调工具即视为防汛业务**。

START → **planner**（Function Calling 规划 + 信息充分性判断）

- 第 1 轮无工具调用 → **direct_chat**（闲聊 / 概念解释流式对话）→ END
- 有工具调用 → **executor**（并发执行 + TTL 缓存）
  - 信息不足 → 回到 planner 循环（≤ LLM_MAX_TOOL_ROUNDS）
  - 信息充分 → **synthesizer**（两阶段真流式 + 引用校验）→ END

1. **planner** — Function Calling 规划 + 信息充分性判断（空 tool_calls 即结束）；签名去重防死循环；第 1 轮注入 Skill 匹配指令、历史经验、历史摘要；概念解释类问题直接返回空工具列表
2. **executor** — ThreadPoolExecutor(4) 并发 + 5min TTL 缓存；同轮含 get_weather + predict_runoff 时两阶段执行，自动注入累计降雨量与逐小时序列（跨工具数据流）
3. **synthesizer** — 两阶段真流式：Phase 1 非流式输出结构化结论（等级 / 依据 / 措施 / 引用），Phase 2 逐 token 生成答案；Citation Grounding 校验引用原文真实性；结构化输出三级降级 + 4 级 JSON 修复
4. **direct_chat** — 闲聊 / 概念解释直接流式对话

#### 预警等级标准

| 等级 | 触发条件                                               | 颜色 |
| ---- | ------------------------------------------------------ | ---- |
| Ⅰ级 | 流量 ≥ 5000 m³/s 或水位超保证 或 24h 降雨 > 100mm    | 红色 |
| Ⅱ级 | 流量 3000-5000 m³/s 或水位超警戒 或 24h 降雨 50-100mm | 橙色 |
| Ⅲ级 | 流量 2000-3000 m³/s                                   | 黄色 |
| Ⅳ级 | 其他（水情平稳）                                       | 蓝色 |

规则引擎 [`agent/graph/synthesizer.py`](agent/graph/synthesizer.py) 的 `compute_warning_level`
是全项目单一权威来源：线上研判、训练数据等级真值、奖励函数共用同一实现，避免规则漂移。

#### 五类记忆架构

| 记忆类型 | 承载 | 注入点 |
|---|---|---|
| 会话记忆 | chat_sessions/messages + 上下文压缩 | planner / synthesizer（历史摘要） |
| 长期记忆 | `MEMORY.md`（用户手册，Agent 只读）+ `memory/` 目录（Agent 自动记忆，索引+主题文件） | 三处 system prompt 常驻 |
| 语义记忆 | `agent_semantic` 表 + Qdrant | synthesizer「领域知识」top-3 |
| 情景记忆 | `agent_episodes` 表 + Qdrant | planner「历史类似情形」top-2 |
| 程序记忆 | `agent_procedures` 表 + Qdrant | planner「推荐方法」top-2 |

- **反思分发**：用户纠正 / 工具失败 / 多轮解决等触发异步反思，LLM 输出分发到长期（文件）/ 语义 / 情景 / 程序四类存储；写入三道安全闸（提示词注入扫描、敏感信息过滤、rubric 质量门槛）
- **程序记忆成长闭环**：反思写入具体模式 → Curator 周期提炼为通用步骤（LLM 泛化）→ 高复用高质量程序自动晋升候选 Skill（`enabled=false` 人工确认启用）
- **效果闭环**：注入记忆线程级追踪，请求完成后计数（语义 hit_count / 程序 use_count+success_count），反思可 demote 无效记忆
- **Curator 五步治理**（周期后台线程）：剪枝僵尸记忆 → 语义压缩合并 → 程序提炼 → 晋升检查 → 向量索引对账 + memory/ 目录索引修复
- **治理 API**：`/api/memories/*` 支持手册读写、自动记忆主题编辑、语义/情景/程序查询删除、手动晋升、反思审计
- **降级**：MySQL 未配置时长期记忆（文件）仍可用，其余类型自动禁用

#### Skill 系统

- **模型**：`name` / `description`（触发条件，含典型问法）/ `instructions` / `tool_names`（工具子集）/ `enabled`
- **匹配**：query 与 description 的 embedding 余弦相似度 > 0.55 即激活（embedding 不可用降级关键词）；只扫 description、匹配后才加载完整指令（渐进式披露），注入 planner + synthesizer
- **元工具**：`list_skills` 对标 MCP tools/list；已启用技能清单常驻 system prompt
- **管理**：前端 SkillsView 可视化 CRUD + 包导入（ZIP 炸弹 / 路径穿越 / 大小限制防护）

#### 会话持久化与上下文压缩

- **会话**：MySQL 持久化，草稿态延迟创建（首次发送才建会话），流式完成后 PUT 全量同步
- **压缩**：history 超 4000 token 保留近 2 轮原文、早轮 LLM 摘要为一条 system 消息（指纹缓存，失败降级截断）

#### 上下文缓存（KV Cache 友好设计）

面向 DeepSeek 自动前缀缓存（命中约 1/10 计费）与本地 vLLM APC（`--enable-prefix-caching`）设计，按 [ai-agent-book 第 2 章](https://bojieli.github.io/ai-agent-book/book/chapter2)三原则改造（调研与三后端对照详见 [docs/kv-cache-research.md](docs/kv-cache-research.md)；注：当前 MaaS 专属实例实测未开启跨请求缓存，切 DeepSeek / 百炼公共部署 / 本地 vLLM 后收益生效）：

- **静态前缀冻结**：planner/synthesizer 的 system 分层（静态指令 → 长期记忆 → Skill 清单按 name 排序），tools schema 逐字稳定；测试锁定跨轮逐字一致
- **动态只追加**：synthesizer Phase 2 system = Phase 1 system + 追加第二阶段指令（两阶段重发的大段工具结果命中同一前缀缓存）；planner 第 1 轮注入的经验/历史摘要写入 state 跨轮原样保留
- **命中率观测**：流式开 `include_usage`，`llm_stats` 按节点聚合 `cached_tokens`（兼容 DeepSeek `prompt_cache_hit_tokens` / vLLM 字段命名），日志输出 `[llm-cache] node=... hit_rate=...%`

#### 关键设计速查

| 设计点       | 方案                                                                                   |
| ------------ | -------------------------------------------------------------------------------------- |
| 意图路由     | 无独立路由层：planner Function Calling 统一决策，不调工具即闲聊                        |
| 异常处理     | LLMError 分类（timeout/rate_limit/...），流式 / 非流式均携带 kind/status_code 传播前端 |
| 工具执行     | ThreadPoolExecutor(4) 并发 + 5min TTL 缓存；weather→runoff 两阶段注入降雨数据         |
| 工具降级     | dev 真实源失败降 mock；production 关键工具硬失败（防模拟数据误导防汛决策）             |
| LLM 超时     | 分级（planner 90s / synthesizer 180s / embedding 60s / chat 120s）                     |
| SSE 流式     | 15s 心跳保活 + 前端 5min 总超时 + 60s 静默超时 + 断连协作式取消                        |
| 结构化输出   | json_schema strict → json_object → 无 response_format 三级降级 + 4 级 JSON 修复      |
| 引用溯源     | quote 须为来源原文子串，失败带反馈重生成（≤2 次），用尽则过滤                         |
| 思考内容剥离 | `strip_think` 移除 `<think>` 块，流式 / 非流式全路径覆盖                           |
| 前缀缓存     | KV Cache 友好上下文工程（静态前缀冻结 + 只追加 + 命中率观测），DeepSeek/MaaS/vLLM 三后端通用 |
| 前端 UI      | 无 UI 库，自写组件（Toast / 会话侧栏 / 预警卡 / 引用卡），Codex 风格                   |
| 结构化日志   | structlog（JSON / Console 双模式）；LangFuse 可选追踪                                  |
| 安全         | slowapi 限流 + production CORS 收紧 + 占位符 Key 校验 + Skill 导入防护                 |

### 系统架构

- **前端**（Vue 3 + Vite，无 UI 库）：AgentView 对话 / SkillsView 技能管理 / HealthView；useAgentChat（SSE 事件状态机）；自写会话侧栏（MySQL 持久化，启动全量加载）
- **后端**（FastAPI + LangGraph），经 HTTP / SSE 与前端通信：
  - API 层：agent / sessions / memories / skills / health 五组路由
  - Agent 工作流（`agent/graph/`）：planner → executor（可循环）→ synthesizer，闲聊 → direct_chat
  - 工具层（`agent/tools/`）：8 个工具，真实 / mock 双轨——get_weather、get_hydrology、predict_runoff、search_regulation、query_gis_terrain、generate_plan、web_search、list_skills（元工具）
  - Skill 系统（`agent/skills/`，借鉴 Claude Skills）：description embedding 匹配 → 按需加载指令 → 工具子集隔离；支持 SKILL.md / .zip 导入
  - 五类记忆（`agent/memory/`）：会话（上下文压缩）、长期（MEMORY.md 手册 + memory/ 目录）、语义 / 情景 / 程序（MySQL + Qdrant 向量检索，程序可晋升 Skill）；Curator 周期治理（剪枝 → 压缩 → 提炼 → 晋升 → 对账）
  - 上下文压缩（`context_compact.py`，借鉴 Codex compact）：history 超 4000 token → 保留近 2 轮原文 + 早轮 LLM 摘要
  - 结构化输出降级 + Citation Grounding（`synthesizer_node`）：json_schema strict → json_object → 无 response_format + 4 级 JSON 修复；引用须为来源原文子串，否则带反馈重生成
- **下游服务**：本地 LlamaFactory / vLLM（微调水利大模型，或云端 qwen-plus）｜ Qdrant 向量库（法规 RAG 检索 + Skill 匹配）｜ 外部数据源（高德 / Tavily / qqjjsj.com）
- **训练管线**（`train/`）：Self-Instruct 种子扩张 → 场景参数化 → 双模型蒸馏 → 三道过滤 → LLM-as-Judge → SFT + DPO 偏好对 → QLoRA / GRPO，微调产物供本地部署使用

## Docker 部署

```bash
cd docker && cp .env.example .env   # 填入密钥
docker compose up --build -d        # qdrant + mysql + backend + frontend
docker compose --profile init run --rm vector-init   # 首次：构建法规向量库（一次性）
```

- 前端 http://localhost:8080 ；MySQL（会话 / 记忆 / Skill）已内置
- 法规语料已打进镜像，向量库存 Qdrant 持久卷——init 只需首次或法规更新后执行
- 接宿主机 LlamaFactory：`.env` 的 `LLM_BASE_URL` 改为 `http://host.docker.internal:8001/v1`
- 本地微调模型跑 vLLM：`docker compose --profile local-llm up -d`
- 训练容器：`docker compose --profile train run --rm train bash`

## 项目结构

```
WaterAgents/
├── agent/                          # Agent 核心
│   ├── graph/                      # LangGraph 工作流（节点/流式/压缩/缓存/异常分类）
│   ├── tools/                      # 8 个工具 + schemas（真实/mock 双轨执行器）
│   ├── skills/                     # Skill 系统（匹配/存储/导入/解析）
│   ├── rag/                        # 法规 RAG（Qdrant + DashScope embedding）
│   ├── hydrology/                  # SCS-CN 降雨径流模型
│   ├── data/                       # 实时数据源（水文爬虫/高德天气/Tavily）
│   ├── gis/                        # GIS 地形分析（SRTM DEM + rasterio）
│   ├── memory/                     # 五类记忆（longterm/semantic/episode/procedure）+ Curator
│   ├── prompts/                    # 系统提示词（分模块）
│   └── tracing/                    # LangFuse 追踪
├── train/                          # 微调管线（data_gen/lora/grpo/rewards/eval/tests）
├── evals/                          # 系统级评估（用例/回放/指标/Judge/消融/回归门禁）
├── models/                         # 微调产物（合并权重，gitignore）
├── backend/
│   ├── app/api/                    # FastAPI 路由（agent/health/sessions/memories/skills）
│   ├── app/core/                   # 配置/LLM 客户端/日志/限流
│   ├── data/skills.json            # 种子技能
│   ├── scripts/                    # seed_skills 等运维脚本
│   └── tests/                      # 单元测试（MySQL 用例自动 skipif）
├── frontend/                       # Vue 3 + TS（AgentView/SkillsView/HealthView）
├── docker/                         # compose + Dockerfile
├── pyproject.toml                  # pytest + ruff 配置
└── .github/workflows/ci.yml        # CI（ruff + pytest + 前端 tsc/lint/test）
```

## 测试与质量

```bash
# 后端 + 训练管线（MySQL 未配置时相关用例自动跳过）
python -m pytest backend/tests train/tests -v

# Lint
python -m ruff check .

# 前端
cd frontend && npm run test && npm run lint
```

- 后端 20+ 测试文件：工作流路由、SSE 桥接、引用校验、上下文压缩、工具降级、记忆治理、Skill 导入安全等
- 训练管线 15 个测试文件：场景确定性、种子隔离、Hermes 往返、三道过滤、奖励函数等
- CI（GitHub Actions）：ruff + pytest（含真实 MySQL service）+ 前端 vue-tsc / ESLint / Vitest / Build

## 系统级评估

与 `train/eval`（裸模型合成循环，评微调效果）互补，`evals/` 评估**模型 + Harness 组合体**：驱动真实 Agent 链路（planner → executor → synthesizer，含记忆注入），方法论对齐《AI Agents in Depth》第 6 章「Agent 的评估」（τ-bench / RAGAS / Scale AI / AndroidWorld 等实践）。

```bash
# 冒烟（8 条，确定性指标，--no-judge）
python evals/run_eval.py --limit 8

# 全量 62 条：30 业务研判 + 10 闲聊 + 8 法规 + 8 联网 + 6 陷阱
python evals/run_eval.py

# 全量 + LLM Judge 软指标 + pass^3 稳定性 + 记忆消融
python evals/run_eval.py --judge --pass-k --ablation 20

# 更新基线（人工评审报告后执行；之后每次评估自动做回归门禁）
python evals/run_eval.py --update-baseline
```

**评估环境五要素**：确定性数据集（`evals/cases.py`）｜可重置环境状态（`evals/replay.py`：overrides+seed 注入 mock_executor，工具缓存旁路）｜原子工具接口｜评分标准（确定性规则 + Rubric 式 Judge）｜执行协议（`evals/runner.py` 顺序驱动、单条失败不中断）。

**指标体系**（过程 / 结果 / 安全三层，全部附 95% 置信区间）：

| 层 | 指标 | 说明 |
| --- | --- | --- |
| 结果 | `level_accuracy` / `pass^3` | 预警等级 vs 规则引擎真值（精确 + 相邻宽容两口径）；同一用例 3 次全对比例测稳定性 |
| 过程 | `intent` / `tool_precision/recall` / `sequence_validity` / `latency p50-p95` | 意图识别、期望工具集、weather→runoff→plan 顺序合法性、成本代理 |
| 安全 | `trap_resistance` / `citation_precision` | 陷阱任务抵抗率（用户声称Ⅰ级但数据Ⅳ级，是否坚持数据锚定）；引用 quote 独立复核 |

**关键机制**：

- **种子隔离**：评估种子区间 `[300_000, 400_000)`，与 SFT / GRPO / 裸模型评估三段零重叠（防评估集泄漏）
- **能力标签矩阵**：每条用例标注能力（意图/工具/等级/引用/抗误导），报告按 任务 × 能力 出矩阵，定位结构性短板而非只看总分
- **Rubric 式 Judge**：必要/重要/可选/陷阱四档加权，**编造数值是一票否决项**；评判模型 `LLM_JUDGE_MODEL`（默认 qwen-max）与主模型异源，规避自我偏好；`--no-judge` 纯确定性
- **记忆消融**：`--ablation N` 有/无记忆注入各跑一遍，量化五类记忆架构的净贡献（分差超噪声带宽才算显著）
- **回归门禁**：与 `evals/baselines/baseline.json` 配对对比，阈值 = max(固定下限, 2×SE)——分差落在噪声带宽内不下结论；门禁触发退出码 1
- **CI**：GitHub Actions 手动触发（`eval.yml`），报告上传 artifact

## 贡献

欢迎 Issue 与 PR：

```bash
git clone https://github.com/your-username/WaterAgents.git
cd WaterAgents
# 后端测试 + lint 通过即可提交
python -m pytest backend/tests train/tests && python -m ruff check .
```

提交信息遵循 [Conventional Commits](https://www.conventionalcommits.org/zh-hans/)（`feat:` / `fix:` / `docs:` / `refactor:` / `test:` ...）。

## 许可证

本项目基于 [MIT License](LICENSE) 开源。
