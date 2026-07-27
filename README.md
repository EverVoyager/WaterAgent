# 水卫 · 黄河吕梁段防汛预警智能体

基于 LangGraph 状态机 + LLM 原生 Function Calling 的防汛预警智能体。聚焦黄河吕梁段（重点吴堡、龙门水文站），通过实时水情/天气/法规检索/GIS 地形等多源数据，输出预警等级、研判依据和应急措施。

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    前端（Vue 3 + Vite）                      │
│  AgentView.vue ──useAgentChat.ts──api/agent.ts（SSE 流式）   │
└──────────────────────────────┬──────────────────────────────┘
                               │ HTTP / SSE
┌──────────────────────────────▼──────────────────────────────┐
│                后端（FastAPI + LangGraph）                   │
│  ┌──────────┐  ┌──────────────────────────────────────────┐ │
│  │API 层    │  │       Agent 工作流（agent/graph/）        │ │
│  │agent.py  │─▶│  router → planner → executor             │ │
│  │health.py │  │              ↑          ↓                │ │
│  └──────────┘  │              └─ should_continue? ─synth── │ │
│                └──────────────────────────────────────────┘ │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  工具层（agent/tools/）— 6 个真实工具                 │   │
│  │  get_weather | get_hydrology | predict_runoff        │   │
│  │  search_regulation | query_gis_terrain | generate_plan│   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────────────┬──────────────────────────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        ▼                      ▼                      ▼
   DashScope LLM          Qdrant 向量库         外部数据源
   (qwen + embedding)     (法规 RAG 检索)       (高德/qqjjsj.com)
```

## 核心流程

5 节点状态机驱动：

1. **router** — Semantic Router（embedding 余弦相似度）识别意图：闲聊 / 防汛业务
2. **planner** — LLM 原生 Function Calling，根据查询规划工具调用（含去重、轮次控制）
3. **executor** — ThreadPoolExecutor 并发执行工具，进程级 TTL 缓存避免重复调用
4. **synthesizer** — 基于工具结果计算预警等级（Ⅰ/Ⅱ/Ⅲ/Ⅳ），LLM 综合研判输出预案
5. **direct_chat** — 闲聊路径直接 LLM 流式对话

## 项目结构

```
WaterAgents/
├── agent/                          # Agent 核心逻辑
│   ├── graph/                      # LangGraph 工作流
│   │   ├── state.py                # 状态定义（AgentState）
│   │   ├── errors.py               # LLM 异常分类（LLMError）
│   │   ├── cache.py                # 工具结果缓存
│   │   ├── llm_helpers.py          # LLM 调用辅助（含 LangFuse 追踪）
│   │   ├── nodes.py                # 图节点（router/planner/executor/...）
│   │   ├── synthesizer_node.py     # 综合研判节点
│   │   ├── runner.py               # 图构建与运行入口
│   │   └── workflow.py             # 入口（re-export 公共 API）
│   ├── tools/                      # 6 个业务工具 + schemas
│   ├── router/                     # Semantic Router 意图识别
│   ├── rag/                        # 法规 RAG 检索（Qdrant）
│   ├── hydrology/                  # SCS-CN 降雨-径流模型
│   ├── data/                       # 实时数据源（水文/天气爬虫）
│   ├── gis/                        # GIS 地形分析
│   ├── prompts/                    # 系统提示词（含 few-shot）
│   └── tracing/                    # LangFuse LLM 追踪
├── backend/
│   ├── app/
│   │   ├── api/                    # FastAPI 路由（agent + health）
│   │   ├── core/                   # 配置/LLM 客户端/结构化日志
│   │   └── main.py                 # 应用入口
│   ├── data/raw/regulations/       # 5 个真实法规 Markdown
│   ├── build_vector_store.py       # 向量库构建脚本
│   └── tests/                      # 单元测试（90 个）
├── frontend/                       # Vue 3 + Element Plus + TS
├── pyproject.toml                  # pytest + ruff 配置
└── .github/workflows/ci.yml        # CI（后端测试 + 前端构建）
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

# 单元测试
python -m pytest backend/tests/ -v

# 代码检查
python -m ruff check agent/ backend/app/
```

## 训练（微调对齐）

见 docs/implementation-plan.md 完整计划。简版：
1. 数据集：`python -m train.data_gen.build_dataset --n 5000 --seed 1000 --rpm 30`
2. SFT：`python -m train.lora.train_sft --smoke` 验证后全量 `python -m train.lora.train_sft`
3. 合并：`python -m train.lora.merge`
4. GRPO：`python -m train.grpo.train_grpo`，再次 merge
5. 评估：`python -m train.eval.run_eval --models <base> sft-merged grpo-merged --n 300`

## Docker 部署

见 docker/.env.example 配置密钥后：
- DashScope 链路：`cd docker && docker compose up --build -d`，前端 http://localhost:8080
- 本地微调模型：.env 改指 vllm 后 `docker compose --profile local-llm up -d`
- 训练容器：`docker compose --profile train run --rm train bash`

## 关键设计

| 设计点 | 方案 |
|---|---|
| 工具选择 | LLM 原生 Function Calling（非关键词路由） |
| 异常处理 | LLMError 分类（timeout/rate_limit/...），直接传播前端（无 fallback） |
| 工具执行 | ThreadPoolExecutor 并发 + 5min TTL 缓存 |
| LLM 超时 | 分级（planner 30s / synthesizer 90s / embedding 20s） |
| SSE 流式 | 15s 心跳保活 + 前端 5min 总超时 + 60s 静默超时 |
| 推理过程 | 手动状态机驱动，分阶段推送 reasoning_step 事件 |
| 结构化日志 | structlog（JSON/Console 双模式） |
| LLM 追踪 | LangFuse（可选，环境变量控制） |
| 安全 | slowapi 限流 + production 模式 CORS 收紧 + 配置校验 |

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
- **前端**：Vue 3、TypeScript、Element Plus、Vite
- **测试**：pytest、pytest-cov、ruff
- **CI**：GitHub Actions（后端 pytest + 前端 vue-tsc/build）
