# 水卫 · 黄河吕梁段防汛预警智能体

基于 LangGraph 状态机 + LLM 原生 Function Calling 的防汛预警智能体。聚焦黄河吕梁段（重点吴堡、龙门水文站），通过实时水情/天气/法规检索/GIS 地形等多源数据，输出预警等级、研判依据和应急措施。

## 系统架构

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

## 核心流程

4 节点状态机驱动。无独立路由层（借鉴 OpenAI / Cohere 主流方案）：由 planner 的
LLM 原生 Function Calling 统一决策，**不调工具即视为闲聊，调工具即视为防汛业务**
（原 Semantic Router embedding 二分类与 planner 决策重复，已移除，
`test_router_removal.py` 回归保护）。

```
START → planner ──(第 1 轮无工具调用)──→ direct_chat → END
           │
           └─(有工具调用)→ executor ─(should_continue?)─┬─ 是 → planner（循环，≤ LLM_MAX_TOOL_ROUNDS）
                                                       └─ 否 → synthesizer → END
```

1. **planner** — LLM 原生 Function Calling 规划工具调用，同时承担"信息是否充分"判断（返回空 tool_calls 即信息充分）；name+参数签名去重防死循环、轮次控制；第 1 轮注入：Skill 匹配指令 + 历史经验（技能记忆/失败教训）+ 历史对话摘要；概念解释类问题直接返回空工具列表
2. **executor** — ThreadPoolExecutor(4) 并发执行工具，进程级 TTL 缓存（5 分钟）避免重复调用；同轮含 get_weather + predict_runoff 时两阶段执行，自动将累计降雨量与逐小时序列注入 predict_runoff（跨工具数据流）
3. **synthesizer** — 两阶段真流式：Phase 1 非流式输出结构化结论（预警等级 Ⅰ/Ⅱ/Ⅲ/Ⅳ、研判依据、措施、引用），Phase 2 `stream=True` 逐 token 生成答案；Citation Grounding 校验引用原文真实性（quote 须为来源子串，失败带反馈重生成）；结构化输出分级降级（json_schema → json_object → 无格式）+ 4 级 JSON 修复；注入用户偏好 + 领域知识
4. **direct_chat** — 闲聊/概念解释路径直接 LLM 流式对话（注入 Skill 指令但禁止调工具）

### 自进化记忆

会话结束后异步触发反思（不阻塞 SSE 响应），将经验写入三层记忆：

- **触发条件**：用户纠正、工具失败、格式错误、多轮问题解决
- **写入**：安全闸门拦截提示词注入 + Rubric 质量评分过滤低价值记忆；成功工具调用模式 → 技能记忆，用户偏好/纠正/领域知识 → 长期记忆
- **注入**：Qdrant 向量索引按与当前 query 的语义相关性检索记忆（索引不可用时降级时间序）；planner 节点注入"过往经验"few-shot，synthesizer 节点注入"用户偏好"（上限 3 技能 + 3 偏好，避免 prompt 膨胀），注入内容隔离包裹防记忆被当作指令执行
- **效果闭环**：反思时追踪被注入记忆的实际效用，无效记忆自动降权（demote）
- **治理**：Curator 后台线程定期剪枝僵尸记忆/LLM 压缩合并/向量索引对账（高命中核心记忆受保护门控）；`/api/memories` 支持按类型查询/删除/LLM 语义压缩记忆，`/api/memories/reflections` 提供反思日志审计（借鉴 Letta，防"固执记忆"污染行为）
- **降级**：MySQL 不可用时记忆功能返回空值，Agent 回退到无记忆模式（`MYSQL_PASSWORD` 为空即禁用）

### Skill 系统（借鉴 Claude Skills）

- **模型**：`name` / `description`（触发条件，含典型问法）/ `instructions`（行为指令）/ `tool_names`（工具子集，空 = 不限制）/ `enabled`
- **匹配**：query embedding 与所有 enabled Skill 的 description 做余弦相似度，最高分 > 0.55 即激活（embedding 不可用时降级关键词匹配）；只扫描 description（轻量），匹配后才加载完整 instructions（渐进式披露），注入 planner + synthesizer
- **元工具**：`list_skills` 供 LLM 主动获取技能完整指令（对标 MCP tools/list）；已启用技能的 name+description 始终注入 system prompt，LLM 可自然回答"你有哪些技能"
- **管理**：`/api/skills` 提供 CRUD + `.zip`/`.skill`/`.md` 包导入（含 ZIP 炸弹/路径穿越/大小限制防护），前端 SkillsView 可视化管理；预置 4 个防汛种子技能（`backend/scripts/seed_skills.py` 幂等注册）

### 会话持久化与上下文压缩

- **会话**：多会话 MySQL 持久化（`/api/sessions`，未配置 MySQL 时接口硬失败，不降级 localStorage）；前端启动全量加载、草稿态延迟创建（首次发送才建会话）、流式完成后 PUT 全量同步
- **压缩**：history 估算 token 超 `HISTORY_MAX_TOKENS`(4000) 时，保留最近 `HISTORY_KEEP_RECENT_ROUNDS`(2) 轮原文、早轮用 LLM 总结为一条 system 摘要（借鉴 Codex compact.rs；带 history 指纹缓存避免重复调用，LLM 失败降级为截断）

## 项目结构

```
WaterAgents/
├── agent/                          # Agent 核心逻辑
│   ├── graph/                      # LangGraph 工作流
│   │   ├── state.py                # 状态定义（AgentState）
│   │   ├── errors.py               # LLM 异常分类（LLMError）
│   │   ├── cache.py                # 工具结果缓存
│   │   ├── llm_helpers.py          # LLM 调用辅助（JSON 解析/流式/思考剥离）
│   │   ├── context_compact.py      # 上下文 token 压缩（LLM 摘要）
│   │   ├── nodes.py                # 图节点（planner/executor/direct_chat）
│   │   ├── direct_chat_stream.py   # 闲聊真流式（<think> 剥离状态机）
│   │   ├── synthesizer_node.py     # 综合研判（两阶段流式+引用校验+降级）
│   │   ├── synthesizer.py          # 规则引擎（等级计算，训练真值来源）
│   │   ├── runner.py               # 图构建与运行入口（含流式 v2）
│   │   └── workflow.py             # 入口（re-export 公共 API）
│   ├── tools/                      # 8 个业务工具 + schemas（真实/mock 双轨）
│   ├── skills/                     # Skill 系统（匹配/存储/导入，借鉴 Claude Skills）
│   ├── rag/                        # 法规 RAG 检索（Qdrant + DashScope embedding）
│   ├── hydrology/                  # SCS-CN 降雨-径流模型
│   ├── data/                       # 实时数据源（水文爬虫/高德天气/Tavily 搜索）
│   ├── gis/                        # GIS 地形分析（SRTM DEM + rasterio）
│   ├── memory/                     # 自进化记忆（三层体系）+ 会话存储
│   │   ├── memory_store.py         # MySQL 持久化（长期/技能/反思）
│   │   ├── vector_index.py         # Qdrant 记忆语义检索（相关记忆精准注入）
│   │   ├── experience.py           # 经验注入（planner/synth 节点）
│   │   ├── reflection.py           # 异步反思循环（安全闸门+Rubric 评分）
│   │   ├── curator.py               # 定期治理（剪枝/压缩/索引对账）
│   │   └── session_store.py        # 会话/消息持久化
│   ├── prompts/                    # 系统提示词（含 few-shot / Skill / 压缩 / 反思）
├── train/                          # 微调数据管线 + 训练
│   ├── data_gen/                   # 数据集生成（Self-Instruct + 蒸馏 + Judge + DPO）
│   ├── lora/                       # QLoRA SFT + adapter 合并
│   ├── grpo/                       # GRPO 强化学习
│   ├── rewards/                    # 奖励函数（等级/工具调用/计划）
│   ├── eval/                       # 评估
│   └── tests/                      # 训练管线测试
├── data/raw/regulations/           # 5 个真实法规 Markdown（RAG 语料，gitignore）
├── backend/
│   ├── app/
│   │   ├── api/                    # FastAPI 路由（agent/health/sessions/memories/skills）
│   │   ├── core/                   # 配置/LLM 客户端/结构化日志/限流
│   │   └── main.py                 # 应用入口
│   ├── data/skills.json            # 种子技能数据
│   ├── scripts/                    # 运维脚本（seed_skills 等）
│   ├── build_vector_store.py       # 向量库构建脚本
│   └── tests/                      # 单元测试（unit/integration 分级标记）
├── frontend/                       # Vue 3 + TS（无 UI 库，自写组件）
│   └── src/views/                  # AgentView（对话）/ SkillsView（技能）/ HealthView（健康）
├── pyproject.toml                  # pytest + ruff 配置
└── .github/workflows/ci.yml        # CI（后端 pytest + 前端 vue-tsc/lint/test）
```

## 快速开始

### 环境要求

- Python 3.10+
- Node.js 20+
- Qdrant 向量数据库（Windows 原生二进制，无需 Docker）

### 1. 后端配置

```bash
cd backend
cp .env.example .env
# 编辑 .env 填入：
#   LLM_API_KEY     — DashScope / DeepSeek / 智谱 API Key
#   AMAP_API_KEY    — 高德天气 API Key（留空则降级 mock）
#   TAVILY_API_KEY  — Tavily 联网搜索 Key（留空则降级 mock）
#   MYSQL_PASSWORD  — MySQL 密码（留空则禁用记忆/会话持久化，其余功能不受影响）
#   其他项可保持默认
```

### 2. 启动 Qdrant（法规检索）

下载 [qdrant.exe](https://github.com/qdrant/qdrant/releases) 到 `tools/qdrant/`，双击启动或：

```bash
backend/start_qdrant.bat
```

### 3. 构建向量库（首次或法规更新后）

```bash
cd backend
python build_vector_store.py
```

### 3b. 注册种子技能（可选，需 MySQL）

预置 4 个防汛 Skill（实时水情查询/降雨洪水预判/预警级别解读/应急响应建议）：

```bash
python backend/scripts/seed_skills.py
```

### 4. 启动后端

```bash
cd backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. 启动前端

```bash
cd frontend
npm install
npm run dev
# 打开 http://localhost:5173
```

## 验证

```bash
# 健康检查
curl http://localhost:8000/api/health
curl http://localhost:8000/api/health/ready

# 单元测试（与 CI 一致；可用 -m unit / -m integration 按标记筛选）
python -m pytest backend/tests/ train/tests/ -v

# 代码检查
python -m ruff check agent/ backend/app/ backend/scripts/ train/
```

## 训练（微调对齐）

完整数据管线见 [train/data_gen/](train/data_gen/)。流程：Self-Instruct 种子扩张 → 场景参数化（绑定 mock 工具返回值保证等级真值）→ 双模型蒸馏（双轨消息：原生 tool_calls + Hermes 文本）→ 三道硬规则过滤（F1 参数合法/F2 序列合法/F3 等级一致）→ LLM-as-Judge（qwen-max 四维打分）→ SFT + DPO 偏好对构建 → QLoRA 微调。

简版命令：
1. 数据集：`python -m train.data_gen.build_dataset --n 5000 --seed 1000 --rpm 30`（支持 `--dry-run` 干跑验证、`--no-judge`/`--no-dpo` 调试、`--knowledge-ratio` 知识问答占比、`--val-ratio` 验证集比例、断点续传）
2. SFT：`python -m train.lora.train_sft --smoke` 验证后全量 `python -m train.lora.train_sft`
3. 合并：`python -m train.lora.merge`
4. GRPO：`python -m train.grpo.train_grpo`，再次 merge
5. 评估：`python -m train.eval.run_eval --models <base> sft-merged grpo-merged --n 300`

> 数据集按 I/II/III/IV 四等级 1:1:1:1 轮换生成保证均衡；SFT/GRPO/EVAL 通过 seed 区间物理隔离避免数据重叠；输出 JSONL 符合 Hermes Function Calling 训练格式，DPO 输出兼容 trl 的 prompt/chosen/rejected 结构。

## 本地微调模型部署（可选）

完成 SFT 训练并合并 LoRA adapter 后（合并产物默认在 `models/wateragents-qwen3-4b-v1/`），可启动本地 LLM API 服务替代 DashScope。推理配置见 [train/lora/configs/wateragents_inference.yaml](train/lora/configs/wateragents_inference.yaml)（方式 A：合并后推理，无 quantization_bit，template=qwen3_nothink）。

### 启动顺序（3 个终端，严格按序）

**终端 1 — 启动 LlamaFactory API 服务（端口 8001）**

```powershell
# 设置 HF 缓存路径，避免重复下载模型
$env:HF_HOME = "D:\hf_cache"
# 必须指定 API_PORT=8001，否则 LlamaFactory 默认监听 8000 会与后端冲突
$env:API_PORT = "8001"
cd d:\AgentProject\WaterAgents
llamafactory-cli api train/lora/configs/wateragents_inference.yaml
```

等待日志出现 `Uvicorn running on http://0.0.0.0:8001` 且模型加载完成（首次加载约 1-2 分钟），再启动后端。

> 若启动报 `[Errno 10048] bind on address ('0.0.0.0', 8000)`，说明 LlamaFactory 仍在用默认 8000 端口——确认 `$env:API_PORT = "8001"` 已在同一终端会话中执行；同时检查是否有遗留 python 进程占用 8000：`netstat -ano | findstr :8000`，按 PID 执行 `taskkill /PID <PID> /F` 释放端口。

**终端 2 — 启动后端（端口 8000）**

先确认 `backend/.env` 已切换到本地模型：

```bash
LLM_API_KEY=local                          # 本地服务不校验 key，填任意非空值
LLM_BASE_URL=http://localhost:8001/v1      # 指向 LlamaFactory API
LLM_MODEL=wateragents-qwen3-4b-v1          # 与推理配置中的模型名一致
```

再启动后端（必须在 backend 目录下执行，否则会报 `No module named 'app'`）：

```bash
cd d:\AgentProject\WaterAgents\backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
# 或直接运行 backend/run.bat
```

**终端 3 — 启动前端（端口 5173）**

```bash
cd d:\AgentProject\WaterAgents\frontend
npm run dev
# 打开 http://localhost:5173
```

### 验证本地模型部署

在前端输入"龙门站现在水情怎么样"，Agent 应正确调用 `get_hydrology` 工具并返回研判结果。若思考内容 `<think>...</think>` 泄漏到前端，确认后端已应用 `strip_think` 处理（见 [backend/app/core/llm.py](backend/app/core/llm.py)）。

> 注：本地微调模型（Qwen3-4B QLoRA）的能力弱于 DashScope qwen-plus，复杂多工具规划场景可能不如在线模型稳定。

## Docker 部署

见 docker/.env.example 配置密钥后：
- DashScope 链路：`cd docker && docker compose up --build -d`，前端 http://localhost:8080
- 会话/记忆/技能依赖的 MySQL 已包含在 compose 中（mysql:8.0 服务，密码对齐 `MYSQL_PASSWORD`）
- 本地微调模型：.env 改指 vllm 后 `docker compose --profile local-llm up -d`
- 训练容器：`docker compose --profile train run --rm train bash`

## 关键设计

| 设计点 | 方案 |
|---|---|
| 意图路由 | 无独立路由层：planner LLM 原生 Function Calling 统一决策，不调工具即闲聊（移除原 Semantic Router） |
| 异常处理 | LLMError 分类（timeout/rate_limit/...），直接传播前端（无 fallback） |
| 工具执行 | ThreadPoolExecutor(4) 并发 + 5min TTL 缓存；get_weather→predict_runoff 两阶段执行自动注入降雨数据 |
| 工具降级 | development 下真实源失败降级 mock；production 下关键工具硬失败（防模拟数据误导防汛决策） |
| LLM 超时 | 分级（planner 90s / synthesizer 180s / embedding 60s / chat 120s） |
| SSE 流式 | 15s 心跳保活 + 前端 5min 总超时 + 60s 静默超时 |
| 推理过程 | 手动状态机驱动，分阶段推送 reasoning_step 事件 |
| 结构化输出 | json_schema strict → json_object → 无 response_format 分级降级 + 4 级 JSON 修复（直接解析/去代码块/大括号配对提取/单引号+尾随逗号修复） |
| 引用溯源 | Citation Grounding：引用 quote 须为来源原文子串，校验失败带反馈重生成（≤2 次），重试用尽过滤无效引用 |
| 自进化记忆 | 三层（长期/技能/反思）MySQL 持久化，异步反思不阻塞响应，planner/synth 节点经验注入（上限 3+3），治理 API 支持查询/删除/压缩 |
| Skill 系统 | description embedding 匹配（>0.55）按需加载 instructions，工具子集隔离，SKILL.md/.zip 导入（含 ZIP 炸弹/路径穿越防护） |
| 上下文压缩 | history 超 4000 token 保留近 2 轮原文 + 早轮 LLM 摘要（指纹缓存，失败降级截断） |
| 思考内容剥离 | `strip_think` 移除 `</think>...` 块，流式/非流式全路径覆盖 |
| 前端 UI | 无 UI 库，自写组件（Toast/会话侧栏/预警卡/引用卡），Codex 风格 |
| 会话持久化 | MySQL（`/api/sessions`，硬失败不降级 localStorage），草稿态延迟创建，流式完成后全量同步 |
| 结构化日志 | structlog（JSON/Console 双模式，支持 request_id 上下文绑定） |
| 安全 | slowapi 限流 + production 模式 CORS 收紧 + 配置校验 + Skill 导入防护 |

## 预警等级标准

| 等级 | 触发条件 | 颜色 |
|---|---|---|
| Ⅰ级 | 流量 ≥ 5000 m³/s 或水位超保证 或 24h 降雨 > 100mm | 红色 |
| Ⅱ级 | 流量 3000-5000 m³/s 或水位超警戒 或 24h 降雨 50-100mm | 橙色 |
| Ⅲ级 | 流量 2000-3000 m³/s | 黄色 |
| Ⅳ级 | 其他（水情平稳） | 蓝色 |

## 技术栈

- **后端**：FastAPI、LangGraph、Pydantic、Qdrant、structlog、slowapi
- **LLM**：OpenAI 兼容接口（DashScope qwen-plus + text-embedding-v3）
- **前端**：Vue 3、TypeScript、Vite（无 UI 库，Codex 风格自写组件）
- **训练**：LlamaFactory（QLoRA）、trl（DPO）、Self-Instruct 数据管线
- **测试**：pytest、pytest-cov、ruff（后端）；Vitest、ESLint、Playwright（前端）
- **CI**：GitHub Actions（后端 pytest + 前端 vue-tsc/lint/test）
