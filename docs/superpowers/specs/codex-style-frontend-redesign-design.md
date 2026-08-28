# Codex 风格前端重设计 · 设计规格

- 日期：2026-08-09
- 范围：`frontend/`（Vue 3 + TS + Vite）
- 状态：已获用户批准（对话内逐节确认）

## 1. 背景与目标

当前前端为「深色渐变顶栏 + el-menu 导航 + 底栏」的传统后台外壳，AgentView 以聊天气泡 + 大渐变预警横幅为主，视觉偏重、Element Plus 痕迹明显。

目标：重设计为 **Codex Web 风格** —— 简单大气、操作顺滑：

- 左侧窄边栏 + 右侧对话主线程的双栏布局
- 浅色为主、深色可切换
- 多会话历史（localStorage 持久化）
- 核心 UI 纯手写，Element Plus 仅保留 `ElMessage`
- 仅桌面端，不做移动端适配
- 后端 API / SSE 协议零改动

## 2. 关键决策（用户逐项确认）

| 决策点 | 结论 |
|---|---|
| 布局方向 | Codex Web 版：左边栏 + 右侧主线程 |
| 主题 | 浅色为主 + 深色可切换（CSS 变量驱动） |
| 边栏内容 | 多会话历史（新建/切换/删除/自动命名，localStorage） |
| 组件库 | 核心界面纯手写，仅保留 ElMessage |
| 响应式 | 仅桌面 |
| 排版语言 | 方案 A：无气泡流式排版（AI 纯文本、用户右对齐胶囊） |

## 3. 整体布局（App Shell）

废弃顶栏/底栏，改为：

```
┌──────────────┬────────────────────────────────────┐
│ 边栏 260px    │  主区（flex-1，内容居中 max-w-768px）│
│ ┌──────────┐ │                                    │
│ │＋ 新会话  │ │      对话线程（无气泡流式）          │
│ └──────────┘ │                                    │
│ 今天 / 昨天  │                                    │
│ 更早（分组） │   ┌──────────────────────────┐     │
│  · 会话项    │   │ 大圆角输入框（悬浮感）      │     │
│              │   └──────────────────────────┘     │
│ ──────────  │     Enter 发送 · Shift+Enter 换行    │
│ ◉ 智能研判   │                                    │
│ ○ 服务健康   │                                    │
│ ──────────  │                                    │
│ ☾ 深色模式   │                                    │
└──────────────┴────────────────────────────────────┘
```

- 边栏宽 260px，可折叠为 60px 图标栏（`width 0.2s ease` 过渡）；折叠态保留：新会话、两个导航项、主题切换（纯图标）
- 折叠状态仅保留在内存（不持久化）
- 主区无任何顶栏/底栏；路由仅两个：`/agent`（默认）、`/health`

## 4. 主题系统

新增 `frontend/src/styles/theme.css`，全部设计变量集中定义；所有组件只引用变量，不写死颜色。

**变量分组**：

- 背景层级：`--bg-base`（主区）、`--bg-subtle`（边栏）、`--bg-elevated`（卡片/悬浮）
- 文字：`--text-primary / --text-secondary / --text-tertiary`
- 边框：`--border-default / --border-strong`
- 强调色：`--accent / --accent-hover / --accent-soft`（淡底）
- 状态色：`--success / --warning / --error`
- 预警四色：`--level-1 / --level-2 / --level-3 / --level-4` 及对应 `--level-*-soft`（淡底）

**浅色**（`:root`）：底 `#ffffff / #f7f7f5`，文字 `#1f2328 / #6e7781 / #9aa0a6`，边框 `#e6e4e0`，accent `#2563eb`

**深色**（`[data-theme="dark"]`）：底 `#171716 / #101010`，文字 `#ececec` 系，边框 `#2c2c28`，预警色整体调亮

**切换逻辑**（`useTheme` composable）：

- `html[data-theme]` 驱动；初始值读 localStorage，缺失时跟随 `prefers-color-scheme`
- 切换写回 localStorage；背景/文字色加 `transition 0.15s`（仅 color/background，避免全局闪烁）

## 5. 会话管理（新增）

**`useChatSessions` composable**：

- 数据模型：`Session { id: string, title: string, createdAt: number, updatedAt: number, messages: Message[] }`
- 持久化：`localStorage["water-agents:sessions"]`（全量 sessions）+ `localStorage["water-agents:active-session"]`（当前 id）
- 操作：新建、切换、删除、自动命名（首条用户消息前 20 字）
- 分组展示：今天 / 昨天 / 更早（按 `updatedAt`）
- 刷新后恢复当前会话及全部历史
- **改造 `useAgentChat`**：`messages` 绑定当前会话；流式回合结束（`done`/`error`/`stop`）时落盘；存储异常（quota/损坏）静默降级为内存态，不阻断对话
- 边栏会话项：hover 显示删除 ×；当前项高亮
- 移除现有「清空」按钮（新会话替代其功能）

## 6. 对话主区（无气泡流式排版）

- **欢迎屏**：居中大标题「水卫 · 黄河吕梁段防汛预警智能体」+ 一句话副标题 + 2×2 建议卡片（沿用现有 4 条 SUGGESTIONS，hover 微浮起 `translateY(-2px)` + 阴影）
- **用户消息**：右对齐胶囊，`--bg-subtle` 底、圆角 16px、max-width 70%
- **AI 消息**：左对齐纯文本，无气泡、无头像，整宽（768px 列内）
- **推理过程块**：单行状态条——进行中 `spinner + 当前步骤名`，完成后 `✓ 已完成 N 步推理`；点击展开时间线。时间线节点统一中性灰圆点 + 状态图标（spinner/✓/◆），不再按 step 类型着色（废除现有紫/蓝/绿/橙/红彩色编码）
- **工具调用块**：等宽字体状态行 `▸ get_hydrology {"station":"wubu"}` + 右侧状态（spinner/✓/✗）；默认折叠为一行「调用 N 个工具」，点击展开
- **预警等级卡片**：左侧 4px 色条（等级色）+ 淡底色卡片，含等级大字、描述（沿用 LEVEL_DESC）、轮次；替代现有大渐变横幅
- **应急措施**：编号列表，小标题 uppercase 灰色小字
- **流式**：细竖线光标 `▏` 闪烁；新消息 `opacity + translateY(4px) 0.2s ease-out`；滚动沿用现有 rAF + userScrolledUp 方案

## 7. 输入区

- 底部居中，大圆角（16px）输入框，1px 边框；focus 时 `--accent` 边框 + 淡 ring（`box-shadow 0 0 0 3px accent-soft`）
- textarea 自动增高，上限约 200px
- **Enter 发送、Shift+Enter 换行**（保留 Ctrl+Enter 兼容）
- 右下角圆形图标发送按钮（accent 色，箭头图标）；loading 时变为停止方块按钮
- 输入框下方一行小字提示（`--text-tertiary`）

## 8. 服务健康页

同风格手写重写：简洁卡片 + 绿/红脉冲状态点 + 描述列表（服务名/版本/环境/时间戳）+ 刷新按钮；从边栏导航进入。功能与数据源不变。

## 9. 动效规范

| 场景 | 参数 |
|---|---|
| 边栏折叠 | width 0.2s ease |
| 会话项/按钮 hover | background 0.1s |
| 消息进入 | opacity + translateY(4px) 0.2s ease-out |
| 折叠块展开 | grid-rows/max-height 0.25s |
| 按钮按压 | transform scale(0.98) |
| 主题切换 | color/background 0.15s |
| 对话滚动 | scroll-behavior: auto（流式场景） |

尊重 `prefers-reduced-motion`：关闭非必要动画。

## 10. 文件结构

```
frontend/src/
├── styles/theme.css            # 新增：设计变量 + 全局基础样式
├── components/                 # 新增目录
│   ├── AppSidebar.vue          # 边栏（会话列表 + 导航 + 主题切换）
│   ├── ChatInput.vue           # 输入区
│   ├── ChatMessage.vue         # 单条消息渲染（组合下列块组件）
│   ├── ReasoningBlock.vue      # 推理过程状态条/时间线
│   ├── ToolChainBlock.vue      # 工具调用链
│   └── WarningCard.vue         # 预警等级卡片 + 应急措施
├── composables/
│   ├── useChatSessions.ts      # 新增：会话 CRUD + localStorage 持久化
│   ├── useTheme.ts             # 新增：主题切换
│   └── useAgentChat.ts         # 改造：消息绑定会话、回合结束落盘
├── views/
│   ├── AgentView.vue           # 重写为组合层（大幅瘦身）
│   └── HealthView.vue          # 手写风格重写
├── App.vue                     # 重写为边栏布局
└── main.ts                     # 引入 theme.css
```

不改动：`api/`（SSE 协议）、`router/`（仅两条路由）、后端全部代码。

## 11. 错误处理

- localStorage 写入失败（quota）→ 静默降级内存态，对话不受影响
- localStorage 读取损坏（JSON 解析失败）→ 重置为空会话列表
- SSE 错误/中断 → 沿用现有 `useAgentChat` 的 error/done 处理 + ElMessage 提示
- 删除当前会话 → 自动跳到最新会话，无会话时回到欢迎屏

## 12. 验证

- `cd frontend && npm run build`（vue-tsc 类型检查 + vite build）通过
- 建议：为 `useChatSessions`、`useTheme` 两个纯逻辑 composable 配 vitest 单测（新建/切换/删除/分组/持久化/主题初值与切换）
- 人工走查清单：发送/停止/流式渲染、推理与工具块折叠、预警卡片四色、会话 CRUD 与刷新恢复、主题切换与持久化、边栏折叠、欢迎屏建议卡片

## 13. 明确不做（YAGNI）

- 移动端/窄屏适配
- 会话重命名手动编辑、导出、搜索
- 实时水情摘要边栏 widget
- 完全移除 Element Plus（ElMessage 保留）
- 后端任何改动
