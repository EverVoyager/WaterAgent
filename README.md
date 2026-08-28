<div align="center">

---

## ✨ 特性亮点

- 🔧 **LLM 原生 Function Calling 工具编排** —— 无关键词路由，planner 自主规划多轮工具调用（去重 / 轮次控制 / 跨工具数据流注入），并发执行 + TTL 缓存
- 🌊 **真实数据源接入** —— 水文爬虫（qqjjsj.com）、高德天气、Tavily 联网搜索、Qdrant 法规 RAG、SRTM DEM 地形分析、SCS-CN 降雨径流模型；生产环境真实源失败**硬失败**而非静默降级 mock
- 📡 **两阶段真流式 SSE** —— 先推结构化预警卡（等级 / 依据 / 措施 / 引用），再逐 token 流式答案；15s 心跳保活 + 客户端断连协作式取消
- 🧠 **自进化三层记忆** —— 长期记忆 / 技能记忆 / 反思日志 MySQL 持久化，异步反思不阻塞响应，planner / synthesizer 按需注入经验（借鉴 Hermes 范式），配套治理 API（查询 / 删除 / 压缩）
- 🎯 **Skill 系统（兼容 Claude Skills）** —— description embedding 语义匹配按需加载指令、工具子集隔离、支持 SKILL.md / .zip 导入（含 ZIP 炸弹 / 路径穿越防护）
- 📚 **Citation Grounding 引用溯源** —— 引用必须逐字来自来源原文，校验失败带反馈重生成，杜绝编造
- 🏋️ **完整训练管线** —— Self-Instruct 种子扩张 → 场景参数化（等级真值）→ 双模型蒸馏 → 三道硬过滤 → LLM-as-Judge → SFT + DPO + GRPO（纯规则奖励），QLoRA 单卡可跑
- 🛡️ **工程化兜底** —— 结构化输出三级降级 + 4 级 JSON 修复、上下文压缩（Codex compact）、`<think>` 剥离、LLM 异常分类传播前端、限流 / CORS / 配置校验

## 📑 目录

- [系统架构](#%EF%B8%8F-系统架构)
- [快速开始](#-快速开始)
- [使用示例](#-使用示例)
- [配置说明](#%EF%B8%8F-配置说明)
- [核心设计](#-核心设计)
- [训练管线](#%EF%B8%8F-训练管线)
- [本地微调模型部署](#-本地微调模型部署可选)
- [Docker 部署](#-docker-部署)
- [项目结构](#-项目结构)
- [测试与质量](#%EF%B8%8F-测试与质量)
- [贡献](#-贡献)
- [许可证](#-许可证)

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    前端（Vue 3 + Vite，无 UI 库）            │
│  AgentView（对话）/ SkillsView（技能管理）/ HealthView       │
│  useAgentChat.ts（SSE 事件状态机）── api/agent.ts（fetch 流式）│
│  自写 Toast / 会话侧栏（MySQL 持久化，启动全量加载）           │
└──────────────────────────────┬──────────────────────────────┘
                               │ HTTP / SSE
┌──────────────────────────────▼──────────────────────────────┐
│                后端（FastAPI + LangGraph）                   │
│  ┌────────────────────┐  ┌───────────────────────────────┐ │
│  │API 层（5 组路由）   │  │   Agent 工作流（agent/graph/） │ │
│  │agent / sessions    │─▶│  planner → executor（可循环）  │ │
│  │memories / skills   │  │    │        ↓                 │ │
│  │health              │  │  闲聊→direct_chat  合成→synth  │ │
│  └────────────────────┘  └───────────────────────────────┘ │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  工具层（agent/tools/）— 8 个工具（真实/mock 双轨）    │   │
│  │  get_weather | get_hydrology | predict_runoff        │   │
│  │  search_regulation | query_gis_terrain               │   │
│  │  generate_plan | web_search | list_skills（元工具）   │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Skill 系统（agent/skills/，借鉴 Claude Skills）       │   │
│  │  description embedding 匹配 → 按需加载 instructions   │   │
│  │  → 工具子集隔离；支持 SKILL.md / .zip 导入            │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  自进化记忆（agent/memory/）— 三层记忆体系             │   │
│  │  · 长期记忆（用户偏好/纠正/领域知识）─┐               │   │
│  │  · 技能记忆（query→工具组合）       ├─ MySQL 持久化    │   │
│  │  · 反思日志（审计记录）             ┘（含会话持久化）  │   │
│  │  反思触发：用户纠正/工具失败/格式错误/多轮解决          │   │
│  │  经验注入：planner 注入技能+教训，synth 注入偏好+知识    │   │
│  │  治理面：/api/memories 查询/删除/压缩 + 反思日志审计   │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  上下文压缩（context_compact.py，借鉴 Codex compact）  │   │
│  │  history 超 4000 token → 保留近 2 轮原文 + 早轮 LLM 摘要│   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  结构化输出降级 + Citation Grounding（synthesizer_node）│   │
│  │  json_schema strict → json_object → 无 response_format │   │
│  │  + 4 级 JSON 修复；引用须为来源原文子串否则带反馈重生成 │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────────────┬──────────────────────────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        ▼                      ▼                      ▼
   DashScope LLM          Qdrant 向量库         外部数据源
   (qwen + embedding)     (法规 RAG 检索       (高德/Tavily/
                          + Skill 匹配)        qqjjsj.com)
                               │
                               ▼
                   训练管线（train/）
                   Self-Instruct 种子扩张 → 场景参数化
                   → 双模型蒸馏 → 三道过滤 → LLM-as-Judge
                   → SFT + DPO 偏好对 → QLoRA/GRPO
```

## 🚀 快速开始

### 环境要求

- Python 3.10+、Node.js 22.22+（本地开发验证于 24.x；测试依赖 jsdom@30 要求）
- [Qdrant](https://github.com/qdrant/qdrant/releases)（法规向量检索）
- MySQL 8.0（会话 / 记忆 / Skill 持久化；不配则对应功能自动禁用或跳过）

### 1️⃣ 配置后端

```bash
cd backend
cp .env.example .env
# 编辑 .env，至少填入 LLM_API_KEY（DashScope / DeepSeek / 智谱任一）
# 其余 API Key 留空自动降级 mock，不影响启动
```

### 2️⃣ 启动 Qdrant 并构建向量库

```bash
backend/start_qdrant.bat        # 或手动运行 qdrant.exe
cd backend && python build_vector_store.py
```

### 3️⃣（可选）注册种子技能

预置 4 个防汛 Skill（实时水情查询 / 降雨洪水预判 / 预警级别解读 / 应急响应建议）：

```bash
python backend/scripts/seed_skills.py
```

### 4️⃣ 启动服务

```bash
# 后端（端口 8000）
cd backend && python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 前端（端口 5173）
cd frontend && npm install && npm run dev
```

打开 http://localhost:5173 ，问一句「吴堡站现在水情怎么样」即可体验完整链路。

## 💬 使用示例

### SSE 流式（前端同款协议）

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

### 非流式

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

### REST API 总览

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

## ⚙️ 配置说明

完整清单见 [`backend/.env.example`](backend/.env.example)，常用项：

| 变量                                             | 默认               | 说明                                                                |
| ------------------------------------------------ | ------------------ | ------------------------------------------------------------------- |
| `LLM_API_KEY`                                  | —                 | **必填**。DashScope / DeepSeek / 智谱（OpenAI 兼容）          |
| `LLM_BASE_URL`                                 | DashScope          | 任意 OpenAI 兼容端点（含本地 vLLM / LlamaFactory）                  |
| `LLM_MODEL`                                    | `qwen-plus`      | 主模型；`LLM_JUDGE_MODEL` 为训练管线评判模型                      |
| `LLM_MAX_TOOL_ROUNDS`                          | `5`              | 单次会话最大工具轮次（防死循环）                                    |
| `AMAP_API_KEY` / `TAVILY_API_KEY`            | 空                 | 高德天气 / 联网搜索；留空降级 mock                                  |
| `QDRANT_HOST` / `QDRANT_PORT`                | `127.0.0.1:6333` | 法规 RAG 向量库                                                     |
| `MYSQL_PASSWORD`                               | 空                 | 会话 / 记忆 / Skill 持久化；**留空即禁用**                    |
| `SELF_EVOLUTION_ENABLED`                       | `true`           | 自进化反思总开关                                                    |
| `CURATOR_ENABLED` / `CURATOR_INTERVAL_HOURS` | `true` / `168` | 记忆定期治理（剪枝 / 压缩 / 对账）                                  |
| `HISTORY_MAX_TOKENS`                           | `4000`           | 超过即触发历史 LLM 摘要压缩                                         |
| `RATE_LIMIT_PER_MINUTE`                        | `30`             | slowapi 限流                                                        |
| `APP_ENV`                                      | `development`    | `production` 下关键工具真实源失败**硬失败**、校验占位符 Key |

## 🧠 核心设计

### 工作流（4 节点状态机）

无独立路由层（借鉴 OpenAI / Cohere 主流方案）：planner 的 LLM 原生 Function
Calling 统一决策，**不调工具即视为闲聊，调工具即视为防汛业务**。

```
START → planner ──(第 1 轮无工具调用)──→ direct_chat → END
           │
           └─(有工具调用)→ executor ─(should_continue?)─┬─ 是 → planner（循环，≤ LLM_MAX_TOOL_ROUNDS）
                                                       └─ 否 → synthesizer → END
```

1. **planner** — Function Calling 规划 + 信息充分性判断（空 tool_calls 即结束）；签名去重防死循环；第 1 轮注入 Skill 匹配指令、历史经验、历史摘要；概念解释类问题直接返回空工具列表
2. **executor** — ThreadPoolExecutor(4) 并发 + 5min TTL 缓存；同轮含 get_weather + predict_runoff 时两阶段执行，自动注入累计降雨量与逐小时序列（跨工具数据流）
3. **synthesizer** — 两阶段真流式：Phase 1 非流式输出结构化结论（等级 / 依据 / 措施 / 引用），Phase 2 逐 token 生成答案；Citation Grounding 校验引用原文真实性；结构化输出三级降级 + 4 级 JSON 修复
4. **direct_chat** — 闲聊 / 概念解释直接流式对话

### 预警等级标准

| 等级 | 触发条件                                               | 颜色 |
| ---- | ------------------------------------------------------ | ---- |
| Ⅰ级 | 流量 ≥ 5000 m³/s 或水位超保证 或 24h 降雨 > 100mm    | 红色 |
| Ⅱ级 | 流量 3000-5000 m³/s 或水位超警戒 或 24h 降雨 50-100mm | 橙色 |
| Ⅲ级 | 流量 2000-3000 m³/s                                   | 黄色 |
| Ⅳ级 | 其他（水情平稳）                                       | 蓝色 |

规则引擎 [`agent/graph/synthesizer.py`](agent/graph/synthesizer.py) 的 `compute_warning_level`
是全项目**单一权威来源**：线上研判、训练数据等级真值、奖励函数共用，防规则漂移。

### 自进化记忆

- **触发**：用户纠正 / 工具失败 / 格式错误 / 多轮解决（异步反思，不阻塞 SSE）
- **写入**：成功工具模式 → 技能记忆；偏好 / 纠正 / 领域知识 → 长期记忆（MySQL 三表）
- **注入**：planner 注入「过往经验」、synthesizer 注入「用户偏好」（各上限 3 条，防 prompt 膨胀）
- **治理**：`/api/memories` 查询 / 删除 / LLM 压缩 + 反思日志审计（借鉴 Letta，防固执记忆）；Curator 定期剪枝 / 合并对账
- **降级**：MySQL 不可用自动回退无记忆模式

### Skill 系统

- **模型**：`name` / `description`（触发条件，含典型问法）/ `instructions` / `tool_names`（工具子集）/ `enabled`
- **匹配**：query 与 description 的 embedding 余弦相似度 > 0.55 即激活（embedding 不可用降级关键词）；只扫 description、匹配后才加载完整指令（渐进式披露），注入 planner + synthesizer
- **元工具**：`list_skills` 对标 MCP tools/list；已启用技能清单常驻 system prompt
- **管理**：前端 SkillsView 可视化 CRUD + 包导入（ZIP 炸弹 / 路径穿越 / 大小限制防护）

### 会话持久化与上下文压缩

- **会话**：MySQL 持久化，草稿态延迟创建（首次发送才建会话），流式完成后 PUT 全量同步
- **压缩**：history 超 4000 token 保留近 2 轮原文、早轮 LLM 摘要为一条 system 消息（指纹缓存，失败降级截断）

### 关键设计速查

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
| 前端 UI      | 无 UI 库，自写组件（Toast / 会话侧栏 / 预警卡 / 引用卡），Codex 风格                   |
| 结构化日志   | structlog（JSON / Console 双模式）；LangFuse 可选追踪                                  |
| 安全         | slowapi 限流 + production CORS 收紧 + 占位符 Key 校验 + Skill 导入防护                 |

## 🏋️ 训练管线

把线上 Agent 蒸馏进本地 Qwen3-4B（QLoRA，RTX 4060 单卡可跑）：

```
种子查询（45 条）
   │ Self-Instruct 扩张（qwen-plus，2-gram Jaccard 去重）
   ▼
场景参数化（流量档位 → 等级真值 + mock 工具覆盖值）
   │ 教师多轮合成（原生 tool_calls 双轨落盘 Hermes 格式，确定性回放）
   ▼
三道硬过滤（F1 参数合法 / F2 序列合法 / F3 等级一致）
   │ LLM-as-Judge（qwen-max 四维打分：≥4 进 SFT，1~4 进 DPO 池）
   ▼
SFT 训练集（95% 业务 + 5% 知识问答）＋ DPO 偏好对
   │ QLoRA SFT → merge → GRPO（trl + vLLM，纯规则奖励）
   ▼
评估（300 条 held-out，等级准确率 / 工具成功率）
```

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

# 4. 三模型对比评估
python -m train.eval.run_eval --models <base> sft-merged grpo-merged --n 300
```

## 🦙 本地微调模型部署（可选）

完成 SFT 并合并 LoRA adapter 后（产物默认在 `models/wateragents-qwen3-4b-v1/`），可用 LlamaFactory 起本地服务替代 DashScope。推理配置见 [`train/lora/configs/wateragents_inference.yaml`](train/lora/configs/wateragents_inference.yaml)。

**终端 1 — LlamaFactory API（端口 8001）**

```powershell
$env:HF_HOME = "D:\hf_cache"
$env:API_PORT = "8001"   # 必须，否则与后端 8000 冲突
cd d:\AgentProject\WaterAgents
llamafactory-cli api train/lora/configs/wateragents_inference.yaml
```

等待日志出现 `Uvicorn running on http://0.0.0.0:8001` 且模型加载完成（首次约 1-2 分钟）。

**终端 2 — 后端（端口 8000）**

`backend/.env` 切换为：

```bash
LLM_API_KEY=local                          # 本地服务不校验，任意非空
LLM_BASE_URL=http://localhost:8001/v1
LLM_MODEL=wateragents-qwen3-4b-v1
```

```bash
cd backend && python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**终端 3 — 前端**：同快速开始。

> 本地 Qwen3-4B QLoRA 能力弱于 qwen-plus，复杂多工具规划场景可能不如在线模型稳定。

## 🐳 Docker 部署

```bash
cd docker && cp .env.example .env   # 填入密钥
docker compose up --build -d        # qdrant + mysql + backend + frontend
docker compose --profile init run --rm vector-init   # 首次：构建法规向量库（一次性）
```

- 前端 http://localhost:8080 ；MySQL（会话 / 记忆 / Skill）已内置
- 法规语料已打进镜像，向量库存 Qdrant 持久卷——init 只需首次或法规更新后执行
- 本地微调模型：`.env` 改指 vllm 后 `docker compose --profile local-llm up -d`
- 训练容器：`docker compose --profile train run --rm train bash`

## 📁 项目结构

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
│   ├── memory/                     # 自进化记忆 + 会话存储 + Curator
│   ├── prompts/                    # 系统提示词（分模块）
│   └── tracing/                    # LangFuse 追踪
├── train/                          # 微调管线（data_gen/lora/grpo/rewards/eval/tests）
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

## ✅️ 测试与质量

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

## 🤝 贡献

欢迎 Issue 与 PR：

```bash
git clone https://github.com/your-username/WaterAgents.git
cd WaterAgents
# 后端测试 + lint 通过即可提交
python -m pytest backend/tests train/tests && python -m ruff check .
```

提交信息遵循 [Conventional Commits](https://www.conventionalcommits.org/zh-hans/)（`feat:` / `fix:` / `docs:` / `refactor:` / `test:` ...）。

## 📄 许可证

待添加（建议推送到 GitHub 前选定 MIT / Apache-2.0 并添加 LICENSE 文件）。

---

<div align="center">
