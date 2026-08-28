# Codex 风格前端重设计 · 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 `frontend/` 重设计为 Codex Web 风格（边栏 + 无气泡流式对话 + 双主题 + 多会话持久化），后端零改动。

**Architecture:** 设计变量集中于 `styles/theme.css`；会话状态收敛到模块级单例 composable（`useChatSessions` / `useAgentChat` / `useTheme`）；视图层拆为 6 个纯手写组件 + 2 个重写的 view + 1 个重写的 App Shell。SSE 协议层 `api/agent.ts` 不动。

**Tech Stack:** Vue 3.5 + TS（strict）、Vite 5、Element Plus（仅 ElMessage）、vitest + jsdom（新增）。

**关联规格：** `docs/superpowers/specs/2026-08-09-codex-style-frontend-redesign-design.md`

## Global Constraints

- 仅桌面端；不做移动端适配
- 核心 UI 纯手写；Element Plus 仅保留 `ElMessage`（保留 `element-plus/dist/index.css` 全量样式供其使用）
- 所有颜色必须引用 CSS 变量，禁止写死色值
- 浅色 `:root` / 深色 `[data-theme="dark"]` 双套变量
- 后端 API、SSE 事件协议、`api/`、`router/` 零改动
- localStorage 键：`water-agents:sessions`、`water-agents:active-session`、`water-agents:theme`
- 存储/读取异常一律静默降级（try/catch 空捕获），不阻断对话
- 每个 Task 结束：`cd frontend && npm run build` 与 `npm run test` 全绿后提交
- 动画须尊重 `prefers-reduced-motion`
- 提交信息格式：`<type>: <description>`（feat/fix/refactor/test/chore/docs）

---

### Task 1: 主题系统 + 测试基建

**Files:**
- Create: `frontend/src/styles/theme.css`
- Create: `frontend/src/composables/useTheme.ts`
- Create: `frontend/src/composables/__tests__/useTheme.spec.ts`
- Modify: `frontend/package.json`（加 vitest/jsdom + test 脚本）
- Modify: `frontend/vite.config.ts`（加 vitest 配置）
- Modify: `frontend/src/main.ts`（换主题样式入口）
- Delete: `frontend/src/assets/styles/main.css`（reset 并入 theme.css）

**Interfaces:**
- Produces: `useTheme(): { theme: Ref<'light'|'dark'>, toggleTheme: () => void, setTheme: (t: Theme) => void }`；`Theme = 'light' | 'dark'`；CSS 变量全集（后续所有 Task 消费）

- [ ] **Step 1: 安装测试依赖**

```powershell
cd frontend; npm install -D vitest jsdom
```

- [ ] **Step 2: package.json 加脚本**

`frontend/package.json` 的 `scripts` 中追加：

```json
"test": "vitest run",
"test:watch": "vitest"
```

- [ ] **Step 3: vite.config.ts 加 vitest 配置**

文件首行加三斜线引用，return 的对象中加 `test` 字段：

```ts
/// <reference types="vitest/config" />
import { defineConfig, loadEnv } from 'vite'
// ...其余不变
    test: {
      environment: 'jsdom',
      include: ['src/**/*.spec.ts'],
    },
```

- [ ] **Step 4: 写失败测试 `frontend/src/composables/__tests__/useTheme.spec.ts`**

```ts
import { beforeEach, describe, expect, it, vi } from 'vitest'

const STORAGE_KEY = 'water-agents:theme'

function stubMatchMedia(dark: boolean) {
  Object.defineProperty(window, 'matchMedia', {
    writable: true,
    configurable: true,
    value: vi.fn().mockImplementation((query: string) => ({
      matches: dark,
      media: query,
      onchange: null,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      addListener: vi.fn(),
      removeListener: vi.fn(),
      dispatchEvent: vi.fn(),
    })),
  })
}

describe('useTheme', () => {
  beforeEach(() => {
    localStorage.clear()
    vi.resetModules()
  })

  it('无存储且系统浅色时默认 light', async () => {
    stubMatchMedia(false)
    const { useTheme } = await import('@/composables/useTheme')
    const { theme } = useTheme()
    expect(theme.value).toBe('light')
    expect(document.documentElement.getAttribute('data-theme')).toBe('light')
  })

  it('无存储且系统深色时默认 dark', async () => {
    stubMatchMedia(true)
    const { useTheme } = await import('@/composables/useTheme')
    expect(useTheme().theme.value).toBe('dark')
    expect(document.documentElement.getAttribute('data-theme')).toBe('dark')
  })

  it('localStorage 存储值优先于系统偏好', async () => {
    stubMatchMedia(true)
    localStorage.setItem(STORAGE_KEY, 'light')
    const { useTheme } = await import('@/composables/useTheme')
    expect(useTheme().theme.value).toBe('light')
  })

  it('toggleTheme 切换并写入 html[data-theme] 与 localStorage', async () => {
    stubMatchMedia(false)
    const { useTheme } = await import('@/composables/useTheme')
    const { theme, toggleTheme } = useTheme()
    expect(theme.value).toBe('light')
    toggleTheme()
    await Promise.resolve()
    expect(theme.value).toBe('dark')
    expect(document.documentElement.getAttribute('data-theme')).toBe('dark')
    expect(localStorage.getItem(STORAGE_KEY)).toBe('dark')
  })
})
```

- [ ] **Step 5: 运行测试确认失败**

Run: `cd frontend; npm run test`
Expected: FAIL（`@/composables/useTheme` 不存在）

- [ ] **Step 6: 实现 `frontend/src/composables/useTheme.ts`**

```ts
/**
 * 主题管理 composable：light/dark 切换 + localStorage 持久化 + 系统偏好兜底
 *
 * 模块级单例，html[data-theme] 驱动 theme.css 变量切换。
 */
import { ref, watchEffect, type Ref } from 'vue'

export type Theme = 'light' | 'dark'

export interface UseThemeReturn {
  theme: Ref<Theme>
  toggleTheme: () => void
  setTheme: (t: Theme) => void
}

const STORAGE_KEY = 'water-agents:theme'

function resolveInitialTheme(): Theme {
  try {
    const saved = localStorage.getItem(STORAGE_KEY)
    if (saved === 'light' || saved === 'dark') return saved
  } catch {
    // 存储不可用时静默降级
  }
  if (typeof window !== 'undefined' && window.matchMedia?.('(prefers-color-scheme: dark)').matches) {
    return 'dark'
  }
  return 'light'
}

const theme = ref<Theme>(resolveInitialTheme())

watchEffect(() => {
  document.documentElement.setAttribute('data-theme', theme.value)
  try {
    localStorage.setItem(STORAGE_KEY, theme.value)
  } catch {
    // quota 满等异常静默降级
  }
})

export function useTheme(): UseThemeReturn {
  function toggleTheme() {
    theme.value = theme.value === 'dark' ? 'light' : 'dark'
  }

  function setTheme(t: Theme) {
    theme.value = t
  }

  return { theme, toggleTheme, setTheme }
}
```

- [ ] **Step 7: 运行测试确认通过**

Run: `cd frontend; npm run test`
Expected: 4 passed

- [ ] **Step 8: 写 `frontend/src/styles/theme.css`（设计变量全集）**

```css
/* ===== 设计变量（Codex 风格 · 浅色为主）===== */
:root {
  --bg-base: #ffffff;
  --bg-subtle: #f7f7f5;
  --bg-elevated: #ffffff;
  --text-primary: #1f2328;
  --text-secondary: #6e7781;
  --text-tertiary: #9aa0a6;
  --border-default: #e6e4e0;
  --border-strong: #d4d2cc;
  --accent: #2563eb;
  --accent-hover: #1d4ed8;
  --accent-soft: rgba(37, 99, 235, 0.1);
  --success: #16a34a;
  --warning: #d97706;
  --error: #dc2626;
  --level-1: #dc2626;
  --level-1-soft: rgba(220, 38, 38, 0.08);
  --level-2: #ea580c;
  --level-2-soft: rgba(234, 88, 12, 0.08);
  --level-3: #d97706;
  --level-3-soft: rgba(217, 119, 6, 0.1);
  --level-4: #2563eb;
  --level-4-soft: rgba(37, 99, 235, 0.08);
  --sidebar-width: 260px;
  --sidebar-collapsed-width: 60px;
  --radius-sm: 8px;
  --radius-md: 12px;
  --radius-lg: 16px;
  color-scheme: light;
}

[data-theme='dark'] {
  --bg-base: #171716;
  --bg-subtle: #101010;
  --bg-elevated: #1f1f1d;
  --text-primary: #ececec;
  --text-secondary: #a3a3a3;
  --text-tertiary: #737373;
  --border-default: #2c2c28;
  --border-strong: #3a3a35;
  --accent: #3b82f6;
  --accent-hover: #60a5fa;
  --accent-soft: rgba(59, 130, 246, 0.16);
  --success: #4ade80;
  --warning: #fbbf24;
  --error: #f87171;
  --level-1: #f87171;
  --level-1-soft: rgba(248, 113, 113, 0.12);
  --level-2: #fb923c;
  --level-2-soft: rgba(251, 146, 60, 0.12);
  --level-3: #fbbf24;
  --level-3-soft: rgba(251, 191, 36, 0.14);
  --level-4: #60a5fa;
  --level-4-soft: rgba(96, 165, 250, 0.12);
  color-scheme: dark;
}

/* ===== 基础 reset（原 assets/styles/main.css 并入）===== */
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

html,
body,
#app {
  height: 100%;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC',
    'Hiragino Sans GB', 'Microsoft YaHei', Arial, sans-serif;
  background: var(--bg-base);
  color: var(--text-primary);
  -webkit-font-smoothing: antialiased;
  transition: background-color 0.15s ease, color 0.15s ease;
}

a {
  text-decoration: none;
  color: inherit;
}

button {
  font-family: inherit;
}

::selection {
  background: var(--accent-soft);
}

/* 细滚动条 */
::-webkit-scrollbar {
  width: 8px;
  height: 8px;
}
::-webkit-scrollbar-thumb {
  background: var(--border-strong);
  border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
  background: var(--text-tertiary);
}
::-webkit-scrollbar-track {
  background: transparent;
}

/* 尊重减少动效偏好 */
@media (prefers-reduced-motion: reduce) {
  *,
  *::before,
  *::after {
    animation-duration: 0.01ms !important;
    transition-duration: 0.01ms !important;
  }
}
```

- [ ] **Step 9: main.ts 换样式入口 + 删旧样式**

`frontend/src/main.ts`：将 `import './assets/styles/main.css'` 改为 `import './styles/theme.css'`（其余行不动）。

删除 `frontend/src/assets/styles/main.css`。

- [ ] **Step 10: 构建 + 测试全绿后提交**

```powershell
cd frontend; npm run build; if ($?) { npm run test }
```
Expected: vue-tsc + vite build 成功；4 tests passed

```powershell
git add frontend/package.json frontend/package-lock.json frontend/vite.config.ts frontend/src/main.ts frontend/src/styles/theme.css frontend/src/composables/useTheme.ts frontend/src/composables/__tests__/useTheme.spec.ts frontend/src/assets/styles/main.css
git commit -m "feat(frontend): 主题系统（CSS 变量 + useTheme）与 vitest 基建"
```

---

### Task 2: 会话存储 useChatSessions

**Files:**
- Create: `frontend/src/composables/useChatSessions.ts`
- Test: `frontend/src/composables/__tests__/useChatSessions.spec.ts`

**Interfaces:**
- Consumes: `ReasoningStep`、`AgentQueryResponse`（来自 `@/api/agent`，已有）
- Produces（Task 3/7/8 消费）:
  - 类型 `ToolEvent` / `ReasoningStepEntry` / `Message` / `ChatSession` / `SessionGroup`
  - `useChatSessions(): UseChatSessionsReturn`：
    - `sessions: ComputedRef<ChatSession[]>`（按 updatedAt 降序）
    - `activeSessionId: Ref<string>`（`''` 表示草稿态）
    - `activeSession: ComputedRef<ChatSession | null>`
    - `activeMessages: ComputedRef<Message[]>`
    - `groupedSessions: ComputedRef<SessionGroup[]>`
    - `createSession() / ensureActiveSession() / switchSession(id) / deleteSession(id) / persistActiveSession()`
  - 纯函数 `loadSessions() / loadActiveId() / groupSessions(list, now)`
  - 常量 `SESSIONS_KEY` / `ACTIVE_KEY`

- [ ] **Step 1: 写失败测试 `frontend/src/composables/__tests__/useChatSessions.spec.ts`**

```ts
import { beforeEach, describe, expect, it, vi } from 'vitest'
import {
  ACTIVE_KEY,
  SESSIONS_KEY,
  groupSessions,
  loadSessions,
  type ChatSession,
} from '@/composables/useChatSessions'

function makeSession(partial: Partial<ChatSession>): ChatSession {
  return { id: 'x', title: '新会话', createdAt: 0, updatedAt: 0, messages: [], ...partial }
}

describe('纯函数', () => {
  beforeEach(() => localStorage.clear())

  it('loadSessions: 无数据返回空数组', () => {
    expect(loadSessions()).toEqual([])
  })

  it('loadSessions: 损坏 JSON 返回空数组', () => {
    localStorage.setItem(SESSIONS_KEY, '{bad json')
    expect(loadSessions()).toEqual([])
  })

  it('loadSessions: 过滤非法会话项', () => {
    localStorage.setItem(
      SESSIONS_KEY,
      JSON.stringify([makeSession({ id: 'a' }), { id: 1 }, null, { title: '无 id' }]),
    )
    const list = loadSessions()
    expect(list).toHaveLength(1)
    expect(list[0].id).toBe('a')
  })

  it('groupSessions: 按今天/昨天/更早分组', () => {
    const now = new Date(2026, 7, 9, 15, 0, 0).getTime()
    const startToday = new Date(now).setHours(0, 0, 0, 0)
    const list = [
      makeSession({ id: 'today', updatedAt: now }),
      makeSession({ id: 'yesterday', updatedAt: startToday - 1000 }),
      makeSession({ id: 'earlier', updatedAt: startToday - 2 * 24 * 60 * 60 * 1000 }),
    ]
    const groups = groupSessions(list, now)
    expect(groups.map((g) => g.label)).toEqual(['今天', '昨天', '更早'])
    expect(groups[0].items[0].id).toBe('today')
    expect(groups[1].items[0].id).toBe('yesterday')
    expect(groups[2].items[0].id).toBe('earlier')
  })

  it('groupSessions: 空列表返回空数组', () => {
    expect(groupSessions([], Date.now())).toEqual([])
  })
})

describe('useChatSessions 单例', () => {
  beforeEach(() => {
    localStorage.clear()
    vi.resetModules()
  })

  it('ensureActiveSession 在无会话时创建并激活', async () => {
    const { useChatSessions } = await import('@/composables/useChatSessions')
    const s = useChatSessions()
    expect(s.activeSession.value).toBeNull()
    s.ensureActiveSession()
    expect(s.activeSession.value).not.toBeNull()
    expect(s.activeMessages.value).toEqual([])
  })

  it('persistActiveSession 用首条用户消息自动命名（前 20 字）并写盘', async () => {
    const { useChatSessions } = await import('@/composables/useChatSessions')
    const s = useChatSessions()
    s.ensureActiveSession()
    s.activeMessages.value.push({ role: 'user', content: '吴堡站当前水情如何？请详细说明一下情况' })
    s.persistActiveSession()
    expect(s.activeSession.value!.title).toBe('吴堡站当前水情如何？请详细说明一下情况'.slice(0, 20))
    const raw = localStorage.getItem(SESSIONS_KEY)
    expect(raw).toBeTruthy()
    expect(JSON.parse(raw!)[0].messages).toHaveLength(1)
  })

  it('createSession 进入草稿态', async () => {
    const { useChatSessions } = await import('@/composables/useChatSessions')
    const s = useChatSessions()
    s.ensureActiveSession()
    s.createSession()
    expect(s.activeSession.value).toBeNull()
    expect(s.activeMessages.value).toEqual([])
  })

  it('switchSession 切换并持久化 activeId', async () => {
    const { useChatSessions } = await import('@/composables/useChatSessions')
    const s = useChatSessions()
    s.ensureActiveSession()
    const firstId = s.activeSessionId.value
    s.activeMessages.value.push({ role: 'user', content: '问题一' })
    s.persistActiveSession()
    s.createSession()
    s.ensureActiveSession()
    expect(s.activeSessionId.value).not.toBe(firstId)
    s.switchSession(firstId)
    expect(s.activeSessionId.value).toBe(firstId)
    expect(localStorage.getItem(ACTIVE_KEY)).toBe(firstId)
  })

  it('deleteSession 删除当前会话后回退到最新会话', async () => {
    const { useChatSessions } = await import('@/composables/useChatSessions')
    const s = useChatSessions()
    s.ensureActiveSession()
    const firstId = s.activeSessionId.value
    s.persistActiveSession()
    s.createSession()
    s.ensureActiveSession()
    const secondId = s.activeSessionId.value
    s.deleteSession(secondId)
    expect(s.activeSessionId.value).toBe(firstId)
    s.deleteSession(firstId)
    expect(s.activeSessionId.value).toBe('')
    expect(s.sessions.value).toHaveLength(0)
  })

  it('启动时 activeId 失效回退到最新会话', async () => {
    localStorage.setItem(
      SESSIONS_KEY,
      JSON.stringify([makeSession({ id: 'live', updatedAt: 5 })]),
    )
    localStorage.setItem(ACTIVE_KEY, 'ghost')
    const { useChatSessions } = await import('@/composables/useChatSessions')
    expect(useChatSessions().activeSessionId.value).toBe('live')
  })
})
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd frontend; npm run test`
Expected: FAIL（模块不存在）；Task 1 的 4 个测试仍通过

- [ ] **Step 3: 实现 `frontend/src/composables/useChatSessions.ts`**

```ts
/**
 * 会话管理 composable：多会话 CRUD + localStorage 持久化
 *
 * 模块级单例，跨组件共享。存储异常（quota/损坏）静默降级为内存态。
 * 草稿态约定：activeSessionId === '' 表示欢迎屏草稿，首次发送时真正建会话。
 */
import { computed, ref, type ComputedRef, type Ref } from 'vue'
import type { AgentQueryResponse, ReasoningStep } from '@/api/agent'

// ====== 类型定义（持久化数据结构，useAgentChat 复用）======

export interface ToolEvent {
  tool: string
  arguments: Record<string, any>
  result?: Record<string, any>
  error?: string
  round: number
  status: 'running' | 'done' | 'error'
}

export interface ReasoningStepEntry extends ReasoningStep {
  timestamp: number
}

export interface Message {
  role: 'user' | 'assistant'
  content: string
  toolEvents?: ToolEvent[]
  reasoningSteps?: ReasoningStepEntry[]
  thinking?: boolean
  chainExpanded?: boolean
  reasoningExpanded?: boolean
  response?: AgentQueryResponse
}

export interface ChatSession {
  id: string
  title: string
  createdAt: number
  updatedAt: number
  messages: Message[]
}

export interface SessionGroup {
  label: string
  items: ChatSession[]
}

export interface UseChatSessionsReturn {
  sessions: ComputedRef<ChatSession[]>
  activeSessionId: Ref<string>
  activeSession: ComputedRef<ChatSession | null>
  activeMessages: ComputedRef<Message[]>
  groupedSessions: ComputedRef<SessionGroup[]>
  createSession: () => void
  ensureActiveSession: () => void
  switchSession: (id: string) => void
  deleteSession: (id: string) => void
  persistActiveSession: () => void
}

// ====== 常量 ======

export const SESSIONS_KEY = 'water-agents:sessions'
export const ACTIVE_KEY = 'water-agents:active-session'
const DEFAULT_TITLE = '新会话'
const TITLE_MAX_LEN = 20
const DAY_MS = 24 * 60 * 60 * 1000

// ====== 纯函数（可单测）======

export function loadSessions(): ChatSession[] {
  try {
    const raw = localStorage.getItem(SESSIONS_KEY)
    if (!raw) return []
    const data = JSON.parse(raw)
    if (!Array.isArray(data)) return []
    return (data as ChatSession[]).filter(
      (s) =>
        s && typeof s.id === 'string' && typeof s.title === 'string' && Array.isArray(s.messages),
    )
  } catch {
    return []
  }
}

export function loadActiveId(): string {
  try {
    return localStorage.getItem(ACTIVE_KEY) || ''
  } catch {
    return ''
  }
}

export function groupSessions(list: ChatSession[], now: number): SessionGroup[] {
  const startOfToday = new Date(now).setHours(0, 0, 0, 0)
  const startOfYesterday = startOfToday - DAY_MS
  const today: ChatSession[] = []
  const yesterday: ChatSession[] = []
  const earlier: ChatSession[] = []
  for (const s of list) {
    if (s.updatedAt >= startOfToday) today.push(s)
    else if (s.updatedAt >= startOfYesterday) yesterday.push(s)
    else earlier.push(s)
  }
  const groups: SessionGroup[] = []
  if (today.length) groups.push({ label: '今天', items: today })
  if (yesterday.length) groups.push({ label: '昨天', items: yesterday })
  if (earlier.length) groups.push({ label: '更早', items: earlier })
  return groups
}

function genId(): string {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 8)
}

// ====== 模块级单例状态 ======

const sessions = ref<ChatSession[]>(loadSessions())
const activeSessionId = ref<string>(loadActiveId())

// 启动校正：activeId 失效时回退到最新会话
if (activeSessionId.value && !sessions.value.some((s) => s.id === activeSessionId.value)) {
  activeSessionId.value = sessions.value[0]?.id || ''
}

function persistAll() {
  try {
    localStorage.setItem(SESSIONS_KEY, JSON.stringify(sessions.value))
    localStorage.setItem(ACTIVE_KEY, activeSessionId.value)
  } catch {
    // quota 满等异常：静默降级为内存态
  }
}

// ====== composable ======

export function useChatSessions(): UseChatSessionsReturn {
  const sorted = computed(() => [...sessions.value].sort((a, b) => b.updatedAt - a.updatedAt))
  const activeSession = computed(
    () => sessions.value.find((s) => s.id === activeSessionId.value) || null,
  )
  const activeMessages = computed(() => activeSession.value?.messages ?? [])
  const groupedSessions = computed(() => groupSessions(sorted.value, Date.now()))

  function persistActiveSession() {
    const active = activeSession.value
    if (active) {
      active.updatedAt = Date.now()
      if (active.title === DEFAULT_TITLE) {
        const firstUser = active.messages.find((m) => m.role === 'user')
        if (firstUser) {
          const t = firstUser.content.slice(0, TITLE_MAX_LEN).trim()
          if (t) active.title = t
        }
      }
    }
    persistAll()
  }

  function ensureActiveSession() {
    if (activeSession.value) return
    const now = Date.now()
    const s: ChatSession = {
      id: genId(),
      title: DEFAULT_TITLE,
      createdAt: now,
      updatedAt: now,
      messages: [],
    }
    sessions.value.push(s)
    activeSessionId.value = s.id
  }

  function createSession() {
    // 草稿态：置空 activeId，首次发送时由 ensureActiveSession 真正创建，避免空会话堆积
    activeSessionId.value = ''
    persistAll()
  }

  function switchSession(id: string) {
    if (!sessions.value.some((s) => s.id === id)) return
    activeSessionId.value = id
    persistAll()
  }

  function deleteSession(id: string) {
    const idx = sessions.value.findIndex((s) => s.id === id)
    if (idx === -1) return
    sessions.value.splice(idx, 1)
    if (activeSessionId.value === id) {
      const rest = [...sessions.value].sort((a, b) => b.updatedAt - a.updatedAt)
      activeSessionId.value = rest[0]?.id || ''
    }
    persistAll()
  }

  return {
    sessions: sorted,
    activeSessionId,
    activeSession,
    activeMessages,
    groupedSessions,
    createSession,
    ensureActiveSession,
    switchSession,
    deleteSession,
    persistActiveSession,
  }
}
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd frontend; npm run test`
Expected: 全部通过（Task1 4 个 + Task2 11 个）

- [ ] **Step 5: 提交**

```powershell
git add frontend/src/composables/useChatSessions.ts frontend/src/composables/__tests__/useChatSessions.spec.ts
git commit -m "feat(frontend): 多会话存储 useChatSessions（localStorage 持久化 + 分组）"
```

---

### Task 3: useAgentChat 接入会话存储

**Files:**
- Modify: `frontend/src/composables/useAgentChat.ts`（整体重写）

**Interfaces:**
- Consumes: Task 2 的 `useChatSessions` / `Message`
- Produces（保持对旧 AgentView 的兼容，Task 8 前不破坏构建）:
  - 原有导出全部保留：`messages / loading / inputText / userScrolledUp / sendQuery / stopQuery / clearChat / useSuggestion / handleStreamEvent` + `SUGGESTIONS / LEVEL_DESC / STEP_NAME / levelDesc / stepName`
  - 类型改为 re-export：`export type { Message, ReasoningStepEntry, ToolEvent } from './useChatSessions'`
  - 状态提升为模块级单例（多组件共享）

- [ ] **Step 1: 整体重写 `frontend/src/composables/useAgentChat.ts`**

```ts
/**
 * Agent 对话状态管理 composable
 *
 * 消息数据接入 useChatSessions（localStorage 持久化），
 * 本 composable 负责 SSE 流式交互与消息状态推进。
 * 模块级单例：多组件（AgentView / ChatInput / ...）共享同一状态。
 */
import { ref, type ComputedRef, type Ref } from 'vue'
import { ElMessage } from 'element-plus'
import {
  queryAgentStream,
  type AgentQueryResponse,
  type AgentStreamEvent,
  type ChatMessage,
} from '@/api/agent'
import { useChatSessions, type Message } from '@/composables/useChatSessions'

export type { Message, ReasoningStepEntry, ToolEvent } from '@/composables/useChatSessions'

export interface UseAgentChatReturn {
  // 状态
  messages: ComputedRef<Message[]>
  loading: Ref<boolean>
  inputText: Ref<string>
  userScrolledUp: Ref<boolean>

  // 方法
  sendQuery: () => Promise<void>
  stopQuery: () => void
  clearChat: () => void
  useSuggestion: (s: string) => void
  handleStreamEvent: (event: AgentStreamEvent, aiMsgIdx: number) => void
}

// ====== 常量 ======

export const SUGGESTIONS = [
  '吴堡站当前水情如何？',
  '发生Ⅱ级预警时应该怎么响应？',
  '未来24小时降雨会不会引发洪水？',
  '你好',
]

export const LEVEL_DESC: Record<string, string> = {
  I: 'Ⅰ级（红色）特别重大',
  II: 'Ⅱ级（橙色）重大',
  III: 'Ⅲ级（黄色）较大',
  IV: 'Ⅳ级（蓝色）一般',
}

export const STEP_NAME: Record<string, string> = {
  router: '意图识别',
  planner: '工具规划',
  executor: '工具执行',
  reflector: '信息评估', // 兼容历史事件
  synthesizer: '综合研判',
  direct_chat: '对话生成',
}

export function levelDesc(level: string): string {
  return LEVEL_DESC[level] || level
}

export function stepName(step: string): string {
  return STEP_NAME[step] || step
}

// ====== 模块级单例状态 ======

const { activeMessages, ensureActiveSession, persistActiveSession } = useChatSessions()

const inputText = ref('')
const loading = ref(false)
const userScrolledUp = ref(false)
let abortController: AbortController | null = null

async function sendQuery() {
  const q = inputText.value.trim()
  if (!q || loading.value) return

  ensureActiveSession()

  const history: ChatMessage[] = activeMessages.value.map((m) => ({
    role: m.role,
    content: m.content,
  }))

  activeMessages.value.push({ role: 'user', content: q })
  activeMessages.value.push({
    role: 'assistant',
    content: '',
    toolEvents: [],
    reasoningSteps: [],
    thinking: true,
    chainExpanded: true,
    reasoningExpanded: true,
    response: undefined,
  })
  const aiMsgIdx = activeMessages.value.length - 1
  inputText.value = ''
  loading.value = true
  userScrolledUp.value = false

  abortController = queryAgentStream(
    { query: q, history },
    (event) => handleStreamEvent(event, aiMsgIdx),
    (err) => {
      const msg = activeMessages.value[aiMsgIdx]
      if (msg) {
        msg.thinking = false
        msg.content = `调用失败：${err.message}`
      }
      loading.value = false
      persistActiveSession()
      ElMessage.error(err.message)
    },
  )
}

function handleStreamEvent(event: AgentStreamEvent, aiMsgIdx: number) {
  const aiMsg = activeMessages.value[aiMsgIdx]
  if (!aiMsg) return
  switch (event.type) {
    case 'reasoning_step':
      if (!aiMsg.reasoningSteps) aiMsg.reasoningSteps = []
      aiMsg.reasoningSteps.push({
        step: event.step || 'router',
        phase: event.phase || 'start',
        message: event.message || '',
        details: event.details,
        timestamp: Date.now(),
      })
      break

    case 'intent':
      if (event.intent === 'chitchat') {
        aiMsg.chainExpanded = false
      }
      break

    case 'tool_call':
      if (!aiMsg.toolEvents) aiMsg.toolEvents = []
      aiMsg.toolEvents.push({
        tool: event.tool || 'unknown',
        arguments: event.arguments || {},
        round: event.round || 1,
        status: 'running',
      })
      break

    case 'tool_result': {
      const events = aiMsg.toolEvents || []
      const last = [...events].reverse().find(
        (e) => e.tool === event.tool && e.status === 'running',
      )
      if (last) {
        last.result = event.result
        last.error = event.error
        last.status = event.error ? 'error' : 'done'
      }
      break
    }

    case 'synth_meta':
      if (event.data && typeof event.data === 'object') {
        const meta = event.data as Record<string, any>
        if (!aiMsg.response) {
          aiMsg.response = {
            answer: '',
            warning_level: meta.warning_level || '',
            reasoning: meta.reasoning || '',
            actions: meta.actions || [],
            tool_calls: [],
            rounds: 0,
            intent: 'agent_task',
          } as AgentQueryResponse
        } else {
          aiMsg.response.warning_level = meta.warning_level || ''
          aiMsg.response.reasoning = meta.reasoning || ''
          aiMsg.response.actions = meta.actions || []
        }
      }
      break

    case 'answer_delta':
      aiMsg.content += event.content || ''
      break

    case 'done':
      aiMsg.thinking = false
      if (event.data) {
        const data = event.data as AgentQueryResponse
        aiMsg.response = data
        if (!aiMsg.content && data.answer) {
          aiMsg.content = data.answer
        }
      }
      aiMsg.chainExpanded = false
      aiMsg.reasoningExpanded = false
      loading.value = false
      persistActiveSession()
      break

    case 'error':
      aiMsg.thinking = false
      aiMsg.content = `运行失败：${event.message}`
      loading.value = false
      persistActiveSession()
      ElMessage.error(event.message || 'Agent 运行失败')
      break
  }
}

function stopQuery() {
  if (abortController) {
    abortController.abort()
    abortController = null
  }
  const last = activeMessages.value[activeMessages.value.length - 1]
  if (last && last.role === 'assistant') {
    last.thinking = false
    if (!last.content) last.content = '（已中断）'
  }
  loading.value = false
  persistActiveSession()
}

/** @deprecated Task 8 后随旧视图一起移除（新设计用「新会话」替代） */
function clearChat() {
  activeMessages.value.splice(0, activeMessages.value.length)
  persistActiveSession()
}

function useSuggestion(s: string) {
  inputText.value = s
  sendQuery()
}

// ====== composable 主函数（返回共享单例）======

export function useAgentChat(): UseAgentChatReturn {
  return {
    messages: activeMessages,
    loading,
    inputText,
    userScrolledUp,
    sendQuery,
    stopQuery,
    clearChat,
    useSuggestion,
    handleStreamEvent,
  }
}
```

- [ ] **Step 2: 构建 + 测试**

Run: `cd frontend; npm run build; if ($?) { npm run test }`
Expected: 全绿（旧 AgentView 仍编译通过，行为不变）

- [ ] **Step 3: 提交**

```powershell
git add frontend/src/composables/useAgentChat.ts
git commit -m "refactor(frontend): useAgentChat 接入会话存储并提升为模块级单例"
```

---

### Task 4: 推理块 + 工具链块组件

**Files:**
- Create: `frontend/src/components/ReasoningBlock.vue`
- Create: `frontend/src/components/ToolChainBlock.vue`

**Interfaces:**
- Consumes: `ReasoningStepEntry` / `ToolEvent`（Task 2）；`stepName`（Task 3 re-export）
- Produces（Task 5 ChatMessage 消费）:
  - `<ReasoningBlock :steps="ReasoningStepEntry[]" :thinking="boolean" v-model:expanded="boolean" />`
  - `<ToolChainBlock :events="ToolEvent[]" v-model:expanded="boolean" />`

- [ ] **Step 1: 创建 `frontend/src/components/ReasoningBlock.vue`**

```vue
<template>
  <div class="reasoning-block">
    <button class="block-header" type="button" @click="emit('update:expanded', !expanded)">
      <svg
        class="chevron"
        :class="{ expanded }"
        viewBox="0 0 24 24"
        width="14"
        height="14"
        fill="none"
        stroke="currentColor"
        stroke-width="2"
      >
        <polyline points="9 18 15 12 9 6" />
      </svg>
      <span v-if="thinking" class="spinner" />
      <span v-else class="status-ok">✓</span>
      <span class="header-text">{{ headerText }}</span>
      <span class="step-count">{{ steps.length }} 步</span>
    </button>
    <div v-show="expanded" class="timeline">
      <div v-for="(rs, i) in steps" :key="i" class="timeline-item">
        <div class="dot">
          <span v-if="rs.phase === 'done'" class="dot-icon">✓</span>
          <span v-else-if="rs.phase === 'decision'" class="dot-icon">◆</span>
          <span v-else class="dot-spinner" />
        </div>
        <div class="item-body">
          <div class="item-name">{{ stepName(rs.step) }}</div>
          <div class="item-message">{{ rs.message }}</div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { stepName } from '@/composables/useAgentChat'
import type { ReasoningStepEntry } from '@/composables/useChatSessions'

const props = withDefaults(
  defineProps<{
    steps: ReasoningStepEntry[]
    thinking: boolean
    expanded?: boolean
  }>(),
  { expanded: false },
)

const emit = defineEmits<{ 'update:expanded': [value: boolean] }>()

const headerText = computed(() => {
  if (props.thinking && props.steps.length) {
    return stepName(props.steps[props.steps.length - 1].step)
  }
  return `已完成 ${props.steps.length} 步推理`
})
</script>

<style scoped>
.reasoning-block {
  border: 1px solid var(--border-default);
  border-radius: var(--radius-md);
  background: var(--bg-subtle);
  overflow: hidden;
}

.block-header {
  display: flex;
  align-items: center;
  gap: 8px;
  width: 100%;
  padding: 9px 14px;
  border: none;
  background: transparent;
  cursor: pointer;
  font-size: 12px;
  color: var(--text-secondary);
  user-select: none;
  text-align: left;
  transition: background-color 0.1s;
}

.block-header:hover {
  background: var(--accent-soft);
}

.chevron {
  flex-shrink: 0;
  color: var(--text-tertiary);
  transition: transform 0.2s;
}

.chevron.expanded {
  transform: rotate(90deg);
}

.spinner {
  flex-shrink: 0;
  width: 12px;
  height: 12px;
  border: 2px solid var(--border-strong);
  border-top-color: var(--accent);
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

.status-ok {
  flex-shrink: 0;
  font-size: 11px;
  color: var(--success);
}

.header-text {
  font-weight: 500;
}

.step-count {
  margin-left: auto;
  font-size: 11px;
  color: var(--text-tertiary);
  background: var(--bg-elevated);
  border: 1px solid var(--border-default);
  padding: 1px 8px;
  border-radius: 10px;
}

.timeline {
  padding: 2px 14px 12px;
}

.timeline-item {
  display: flex;
  gap: 10px;
  padding: 7px 0;
  position: relative;
}

.timeline-item:not(:last-child)::before {
  content: '';
  position: absolute;
  left: 6px;
  top: 22px;
  bottom: -4px;
  width: 2px;
  background: var(--border-default);
}

.dot {
  flex-shrink: 0;
  width: 14px;
  height: 14px;
  margin-top: 2px;
  border-radius: 50%;
  background: var(--text-tertiary);
  display: flex;
  align-items: center;
  justify-content: center;
  color: #fff;
  z-index: 1;
}

.dot-icon {
  font-size: 8px;
  line-height: 1;
}

.dot-spinner {
  width: 8px;
  height: 8px;
  border: 2px solid rgba(255, 255, 255, 0.4);
  border-top-color: #fff;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

.item-body {
  flex: 1;
  min-width: 0;
}

.item-name {
  font-size: 12px;
  font-weight: 600;
  color: var(--text-primary);
  margin-bottom: 1px;
}

.item-message {
  font-size: 12px;
  color: var(--text-secondary);
  line-height: 1.5;
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}
</style>
```

- [ ] **Step 2: 创建 `frontend/src/components/ToolChainBlock.vue`**

```vue
<template>
  <div class="toolchain">
    <button class="block-header" type="button" @click="emit('update:expanded', !expanded)">
      <svg
        class="chevron"
        :class="{ expanded }"
        viewBox="0 0 24 24"
        width="14"
        height="14"
        fill="none"
        stroke="currentColor"
        stroke-width="2"
      >
        <polyline points="9 18 15 12 9 6" />
      </svg>
      <span v-if="runningTool" class="spinner" />
      <span v-else class="status-ok">✓</span>
      <span class="header-text">{{ headerText }}</span>
    </button>
    <div v-show="expanded" class="tool-list">
      <div v-for="(ev, i) in events" :key="i" class="tool-item" :class="{ 'has-error': ev.error }">
        <div class="tool-line">
          <span class="tool-status">
            <span v-if="ev.status === 'running'" class="spinner" />
            <span v-else-if="ev.status === 'error'" class="status-err">✗</span>
            <span v-else class="status-ok">✓</span>
          </span>
          <code class="tool-call">{{ ev.tool }}({{ formatArgs(ev.arguments) }})</code>
          <span class="round-tag">R{{ ev.round }}</span>
        </div>
        <div v-if="ev.error" class="tool-err">错误：{{ ev.error }}</div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import type { ToolEvent } from '@/composables/useChatSessions'

const props = withDefaults(
  defineProps<{
    events: ToolEvent[]
    expanded?: boolean
  }>(),
  { expanded: false },
)

const emit = defineEmits<{ 'update:expanded': [value: boolean] }>()

const runningTool = computed(() => props.events.find((e) => e.status === 'running'))
const headerText = computed(() =>
  runningTool.value ? `正在调用 ${runningTool.value.tool}…` : `调用 ${props.events.length} 个工具`,
)

function formatArgs(args: Record<string, any>): string {
  return Object.entries(args)
    .map(([k, v]) => `${k}=${JSON.stringify(v)}`)
    .join(', ')
}
</script>

<style scoped>
.toolchain {
  border: 1px solid var(--border-default);
  border-radius: var(--radius-md);
  background: var(--bg-subtle);
  overflow: hidden;
}

.block-header {
  display: flex;
  align-items: center;
  gap: 8px;
  width: 100%;
  padding: 9px 14px;
  border: none;
  background: transparent;
  cursor: pointer;
  font-size: 12px;
  color: var(--text-secondary);
  user-select: none;
  text-align: left;
  transition: background-color 0.1s;
}

.block-header:hover {
  background: var(--accent-soft);
}

.chevron {
  flex-shrink: 0;
  color: var(--text-tertiary);
  transition: transform 0.2s;
}

.chevron.expanded {
  transform: rotate(90deg);
}

.spinner {
  flex-shrink: 0;
  width: 12px;
  height: 12px;
  border: 2px solid var(--border-strong);
  border-top-color: var(--accent);
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

.status-ok {
  font-size: 11px;
  color: var(--success);
}

.status-err {
  font-size: 11px;
  color: var(--error);
}

.header-text {
  font-weight: 500;
}

.tool-list {
  padding: 2px 14px 10px;
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.tool-item {
  background: var(--bg-elevated);
  border: 1px solid var(--border-default);
  border-radius: var(--radius-sm);
  padding: 7px 10px;
}

.tool-item.has-error {
  border-color: var(--error);
}

.tool-line {
  display: flex;
  align-items: center;
  gap: 8px;
  min-width: 0;
}

.tool-status {
  flex-shrink: 0;
  display: inline-flex;
  align-items: center;
}

.tool-call {
  flex: 1;
  min-width: 0;
  font-family: 'SF Mono', Monaco, Consolas, monospace;
  font-size: 12px;
  color: var(--text-secondary);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.round-tag {
  flex-shrink: 0;
  font-size: 11px;
  color: var(--text-tertiary);
  background: var(--bg-subtle);
  border: 1px solid var(--border-default);
  padding: 0 6px;
  border-radius: 4px;
}

.tool-err {
  margin-top: 4px;
  font-size: 11px;
  color: var(--error);
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}
</style>
```

- [ ] **Step 3: 构建 + 测试**

Run: `cd frontend; npm run build; if ($?) { npm run test }`
Expected: 全绿（组件暂未接线，仅编译校验）

- [ ] **Step 4: 提交**

```powershell
git add frontend/src/components/ReasoningBlock.vue frontend/src/components/ToolChainBlock.vue
git commit -m "feat(frontend): 推理过程块与工具调用链块组件"
```

---

### Task 5: 预警卡片 + 消息组件

**Files:**
- Create: `frontend/src/components/WarningCard.vue`
- Create: `frontend/src/components/ChatMessage.vue`

**Interfaces:**
- Consumes: Task 4 两个块组件；`levelDesc`（Task 3）；`Message`（Task 2）；`AgentQueryResponse`（`@/api/agent`）
- Produces（Task 8 AgentView 消费）: `<ChatMessage :msg="Message" />`

- [ ] **Step 1: 创建 `frontend/src/components/WarningCard.vue`**

```vue
<template>
  <div class="warning-wrap">
    <div class="warning-card" :class="`lv-${response.warning_level}`">
      <span class="level-badge">{{ response.warning_level }} 级</span>
      <span class="level-desc">{{ levelDesc(response.warning_level) }}</span>
      <span class="rounds">轮次 {{ response.rounds }}</span>
    </div>
    <div v-if="response.actions && response.actions.length" class="actions-block">
      <div class="actions-title">应急措施</div>
      <ol class="actions-list">
        <li v-for="(a, i) in response.actions" :key="i">{{ a }}</li>
      </ol>
    </div>
  </div>
</template>

<script setup lang="ts">
import { levelDesc } from '@/composables/useAgentChat'
import type { AgentQueryResponse } from '@/api/agent'

defineProps<{ response: AgentQueryResponse }>()
</script>

<style scoped>
.warning-wrap {
  display: flex;
  flex-direction: column;
  gap: 10px;
  animation: fade-in 0.25s ease;
}

.warning-card {
  display: flex;
  align-items: center;
  gap: 14px;
  padding: 13px 16px;
  border-radius: var(--radius-md);
  border: 1px solid var(--border-default);
  border-left: 4px solid transparent;
}

.lv-I {
  border-left-color: var(--level-1);
  background: var(--level-1-soft);
}
.lv-I .level-badge {
  color: var(--level-1);
}
.lv-II {
  border-left-color: var(--level-2);
  background: var(--level-2-soft);
}
.lv-II .level-badge {
  color: var(--level-2);
}
.lv-III {
  border-left-color: var(--level-3);
  background: var(--level-3-soft);
}
.lv-III .level-badge {
  color: var(--level-3);
}
.lv-IV {
  border-left-color: var(--level-4);
  background: var(--level-4-soft);
}
.lv-IV .level-badge {
  color: var(--level-4);
}

.level-badge {
  font-size: 22px;
  font-weight: 700;
  letter-spacing: -0.5px;
}

.level-desc {
  font-size: 13px;
  font-weight: 500;
  color: var(--text-primary);
}

.rounds {
  margin-left: auto;
  font-size: 11px;
  color: var(--text-tertiary);
  background: var(--bg-elevated);
  border: 1px solid var(--border-default);
  padding: 2px 10px;
  border-radius: 10px;
}

.actions-block {
  border: 1px solid var(--border-default);
  border-radius: var(--radius-md);
  background: var(--bg-subtle);
  padding: 12px 16px;
}

.actions-title {
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.5px;
  text-transform: uppercase;
  color: var(--text-tertiary);
  margin-bottom: 6px;
}

.actions-list {
  padding-left: 18px;
  font-size: 13px;
  line-height: 1.8;
  color: var(--text-primary);
}

@keyframes fade-in {
  from {
    opacity: 0;
    transform: translateY(4px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}
</style>
```

- [ ] **Step 2: 创建 `frontend/src/components/ChatMessage.vue`**

```vue
<template>
  <!-- 用户消息：右对齐胶囊 -->
  <div v-if="msg.role === 'user'" class="user-row">
    <div class="user-capsule">{{ msg.content }}</div>
  </div>

  <!-- AI 消息：无气泡流式排版 -->
  <div v-else class="ai-row">
    <ReasoningBlock
      v-if="msg.reasoningSteps && msg.reasoningSteps.length"
      :steps="msg.reasoningSteps"
      :thinking="!!msg.thinking"
      v-model:expanded="msg.reasoningExpanded"
    />
    <ToolChainBlock
      v-if="msg.toolEvents && msg.toolEvents.length"
      :events="msg.toolEvents"
      v-model:expanded="msg.chainExpanded"
    />
    <div class="answer">
      <span v-if="!msg.content && msg.thinking" class="typing"><i /><i /><i /></span>
      <template v-else>
        <span class="answer-text">{{ msg.content }}</span>
        <span v-if="msg.thinking" class="cursor">▏</span>
      </template>
    </div>
    <WarningCard v-if="showWarning && msg.response" :response="msg.response" />
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import ReasoningBlock from '@/components/ReasoningBlock.vue'
import ToolChainBlock from '@/components/ToolChainBlock.vue'
import WarningCard from '@/components/WarningCard.vue'
import type { Message } from '@/composables/useChatSessions'

const props = defineProps<{ msg: Message }>()

const showWarning = computed(
  () => props.msg.response?.intent === 'agent_task' && !!props.msg.response?.warning_level,
)
</script>

<style scoped>
.user-row {
  display: flex;
  justify-content: flex-end;
  margin-bottom: 20px;
  animation: fade-in 0.2s ease-out;
}

.user-capsule {
  max-width: 70%;
  padding: 9px 15px;
  background: var(--bg-subtle);
  border: 1px solid var(--border-default);
  border-radius: var(--radius-lg);
  font-size: 14px;
  line-height: 1.6;
  color: var(--text-primary);
  white-space: pre-wrap;
  word-break: break-word;
}

.ai-row {
  display: flex;
  flex-direction: column;
  gap: 10px;
  margin-bottom: 24px;
  animation: fade-in 0.2s ease-out;
}

.answer {
  font-size: 14px;
  line-height: 1.75;
  color: var(--text-primary);
}

.answer-text {
  white-space: pre-wrap;
  word-break: break-word;
}

.typing {
  display: inline-flex;
  gap: 4px;
  align-items: center;
  padding: 4px 0;
}

.typing i {
  width: 6px;
  height: 6px;
  background: var(--text-tertiary);
  border-radius: 50%;
  animation: bounce 1.4s infinite ease-in-out;
}

.typing i:nth-child(2) {
  animation-delay: 0.2s;
}
.typing i:nth-child(3) {
  animation-delay: 0.4s;
}

.cursor {
  display: inline-block;
  margin-left: 1px;
  color: var(--accent);
  animation: blink 1s infinite;
}

@keyframes bounce {
  0%,
  80%,
  100% {
    transform: scale(0.6);
    opacity: 0.5;
  }
  40% {
    transform: scale(1);
    opacity: 1;
  }
}

@keyframes blink {
  0%,
  50% {
    opacity: 1;
  }
  51%,
  100% {
    opacity: 0;
  }
}

@keyframes fade-in {
  from {
    opacity: 0;
    transform: translateY(4px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}
</style>
```

- [ ] **Step 3: 构建 + 测试**

Run: `cd frontend; npm run build; if ($?) { npm run test }`
Expected: 全绿

- [ ] **Step 4: 提交**

```powershell
git add frontend/src/components/WarningCard.vue frontend/src/components/ChatMessage.vue
git commit -m "feat(frontend): 预警等级卡片与无气泡消息组件"
```

---

### Task 6: 输入区组件 ChatInput

**Files:**
- Create: `frontend/src/components/ChatInput.vue`

**Interfaces:**
- Produces（Task 8 AgentView 消费）: `<ChatInput v-model:text="inputText" :loading="loading" @send="sendQuery" @stop="stopQuery" />`
- 交互约定：Enter 发送；Shift+Enter 换行；Ctrl+Enter 发送；`e.isComposing`（中文输入法组词中）不发送；textarea 自动增高上限 200px

- [ ] **Step 1: 创建 `frontend/src/components/ChatInput.vue`**

```vue
<template>
  <footer class="input-area">
    <div class="input-box">
      <textarea
        ref="textareaRef"
        class="input-el"
        :value="text"
        :disabled="loading"
        placeholder="输入防汛相关问题，Enter 发送…"
        rows="1"
        @input="onInput"
        @keydown.enter="onEnterKey"
      />
      <div class="input-bar">
        <span class="input-hint">Enter 发送 · Shift+Enter 换行</span>
        <button v-if="loading" class="send-btn stop" type="button" title="停止" @click="emit('stop')">
          <svg viewBox="0 0 24 24" width="13" height="13" fill="currentColor">
            <rect x="6" y="6" width="12" height="12" rx="2" />
          </svg>
        </button>
        <button
          v-else
          class="send-btn"
          type="button"
          :disabled="!text.trim()"
          title="发送"
          @click="emit('send')"
        >
          <svg
            viewBox="0 0 24 24"
            width="15"
            height="15"
            fill="none"
            stroke="currentColor"
            stroke-width="2"
            stroke-linecap="round"
            stroke-linejoin="round"
          >
            <line x1="12" y1="19" x2="12" y2="5" />
            <polyline points="5 12 12 5 19 12" />
          </svg>
        </button>
      </div>
    </div>
  </footer>
</template>

<script setup lang="ts">
import { nextTick, onMounted, ref, watch } from 'vue'

const props = defineProps<{ text: string; loading: boolean }>()
const emit = defineEmits<{ 'update:text': [value: string]; send: []; stop: [] }>()

const textareaRef = ref<HTMLTextAreaElement | null>(null)

function onInput(e: Event) {
  emit('update:text', (e.target as HTMLTextAreaElement).value)
}

function onEnterKey(e: KeyboardEvent) {
  if (e.isComposing) return // 中文输入法组词中不发送
  if (e.shiftKey) return // Shift+Enter 换行
  e.preventDefault()
  if (props.text.trim() && !props.loading) emit('send')
}

function resize() {
  const el = textareaRef.value
  if (!el) return
  el.style.height = 'auto'
  el.style.height = `${Math.min(el.scrollHeight, 200)}px`
}

watch(
  () => props.text,
  () => nextTick(resize),
)
onMounted(resize)
</script>

<style scoped>
.input-area {
  flex-shrink: 0;
  padding: 10px 24px 16px;
  background: var(--bg-base);
}

.input-box {
  max-width: 768px;
  margin: 0 auto;
  background: var(--bg-elevated);
  border: 1px solid var(--border-default);
  border-radius: var(--radius-lg);
  padding: 12px 14px 8px;
  transition:
    border-color 0.15s,
    box-shadow 0.15s;
}

.input-box:focus-within {
  border-color: var(--accent);
  box-shadow: 0 0 0 3px var(--accent-soft);
}

.input-el {
  display: block;
  width: 100%;
  border: none;
  outline: none;
  resize: none;
  font-family: inherit;
  font-size: 14px;
  line-height: 1.55;
  color: var(--text-primary);
  background: transparent;
  max-height: 200px;
}

.input-el::placeholder {
  color: var(--text-tertiary);
}

.input-el:disabled {
  opacity: 0.6;
}

.input-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-top: 6px;
}

.input-hint {
  font-size: 11px;
  color: var(--text-tertiary);
}

.send-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 30px;
  height: 30px;
  border: none;
  border-radius: 50%;
  background: var(--accent);
  color: #fff;
  cursor: pointer;
  transition:
    background-color 0.15s,
    transform 0.1s;
}

.send-btn:hover:not(:disabled) {
  background: var(--accent-hover);
}

.send-btn:active:not(:disabled) {
  transform: scale(0.94);
}

.send-btn:disabled {
  background: var(--border-strong);
  cursor: not-allowed;
}

.send-btn.stop {
  background: var(--error);
}
</style>
```

- [ ] **Step 2: 构建 + 测试**

Run: `cd frontend; npm run build; if ($?) { npm run test }`
Expected: 全绿

- [ ] **Step 3: 提交**

```powershell
git add frontend/src/components/ChatInput.vue
git commit -m "feat(frontend): 悬浮输入区组件（自动增高 + IME 安全发送）"
```

---

### Task 7: 边栏组件 AppSidebar

**Files:**
- Create: `frontend/src/components/AppSidebar.vue`

**Interfaces:**
- Consumes: `useChatSessions`（Task 2）、`useTheme`（Task 1）、`vue-router` 的 `useRoute/useRouter`
- Produces（Task 10 App.vue 消费）: `<AppSidebar />`（无 props/emits；内部管理折叠状态）

- [ ] **Step 1: 创建 `frontend/src/components/AppSidebar.vue`**

```vue
<template>
  <aside class="sidebar" :class="{ collapsed }">
    <div class="side-head">
      <div v-if="!collapsed" class="brand">
        <svg
          class="brand-icon"
          viewBox="0 0 24 24"
          width="18"
          height="18"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
          stroke-linejoin="round"
        >
          <path d="M12 2.69l5.66 5.66a8 8 0 1 1-11.31 0z" />
        </svg>
        <span class="brand-name">水卫</span>
      </div>
      <button
        class="icon-btn"
        type="button"
        :title="collapsed ? '展开边栏' : '收起边栏'"
        @click="collapsed = !collapsed"
      >
        <svg
          v-if="collapsed"
          viewBox="0 0 24 24"
          width="16"
          height="16"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
          stroke-linejoin="round"
        >
          <polyline points="9 18 15 12 9 6" />
        </svg>
        <svg
          v-else
          viewBox="0 0 24 24"
          width="16"
          height="16"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
          stroke-linejoin="round"
        >
          <polyline points="15 18 9 12 15 6" />
        </svg>
      </button>
    </div>

    <button class="new-chat" type="button" title="新会话" @click="onNewChat">
      <svg
        viewBox="0 0 24 24"
        width="15"
        height="15"
        fill="none"
        stroke="currentColor"
        stroke-width="2"
        stroke-linecap="round"
      >
        <line x1="12" y1="5" x2="12" y2="19" />
        <line x1="5" y1="12" x2="19" y2="12" />
      </svg>
      <span v-if="!collapsed">新会话</span>
    </button>

    <nav v-if="!collapsed" class="session-list">
      <div v-for="g in groupedSessions" :key="g.label" class="session-group">
        <div class="group-label">{{ g.label }}</div>
        <button
          v-for="s in g.items"
          :key="s.id"
          class="session-item"
          :class="{ active: s.id === activeSessionId }"
          type="button"
          :title="s.title"
          @click="onSwitch(s.id)"
        >
          <span class="session-title">{{ s.title }}</span>
          <span class="session-del" title="删除会话" @click.stop="onDelete(s.id)">×</span>
        </button>
      </div>
      <div v-if="!groupedSessions.length" class="empty-hint">暂无历史会话</div>
    </nav>
    <div v-else class="side-spacer" />

    <div class="side-bottom">
      <router-link
        to="/agent"
        class="nav-item"
        :class="{ active: route.path === '/agent' }"
        title="智能研判"
      >
        <svg
          viewBox="0 0 24 24"
          width="16"
          height="16"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
          stroke-linejoin="round"
        >
          <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
        </svg>
        <span v-if="!collapsed">智能研判</span>
      </router-link>
      <router-link
        to="/health"
        class="nav-item"
        :class="{ active: route.path === '/health' }"
        title="服务健康"
      >
        <svg
          viewBox="0 0 24 24"
          width="16"
          height="16"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
          stroke-linejoin="round"
        >
          <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
        </svg>
        <span v-if="!collapsed">服务健康</span>
      </router-link>
      <button
        class="nav-item"
        type="button"
        :title="theme === 'dark' ? '切换到浅色模式' : '切换到深色模式'"
        @click="toggleTheme"
      >
        <svg
          v-if="theme === 'dark'"
          viewBox="0 0 24 24"
          width="16"
          height="16"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
          stroke-linejoin="round"
        >
          <circle cx="12" cy="12" r="5" />
          <line x1="12" y1="1" x2="12" y2="3" />
          <line x1="12" y1="21" x2="12" y2="23" />
          <line x1="4.22" y1="4.22" x2="5.64" y2="5.64" />
          <line x1="18.36" y1="18.36" x2="19.78" y2="19.78" />
          <line x1="1" y1="12" x2="3" y2="12" />
          <line x1="21" y1="12" x2="23" y2="12" />
          <line x1="4.22" y1="19.78" x2="5.64" y2="18.36" />
          <line x1="18.36" y1="5.64" x2="19.78" y2="4.22" />
        </svg>
        <svg
          v-else
          viewBox="0 0 24 24"
          width="16"
          height="16"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
          stroke-linejoin="round"
        >
          <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
        </svg>
        <span v-if="!collapsed">{{ theme === 'dark' ? '浅色模式' : '深色模式' }}</span>
      </button>
      <div v-if="!collapsed" class="version">v0.1.0</div>
    </div>
  </aside>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useChatSessions } from '@/composables/useChatSessions'
import { useTheme } from '@/composables/useTheme'

const route = useRoute()
const router = useRouter()
const { activeSessionId, groupedSessions, createSession, switchSession, deleteSession } =
  useChatSessions()
const { theme, toggleTheme } = useTheme()

const collapsed = ref(false)

function onNewChat() {
  createSession()
  if (route.path !== '/agent') router.push('/agent')
}

function onSwitch(id: string) {
  switchSession(id)
  if (route.path !== '/agent') router.push('/agent')
}

function onDelete(id: string) {
  deleteSession(id)
}
</script>

<style scoped>
.sidebar {
  width: var(--sidebar-width);
  flex-shrink: 0;
  height: 100vh;
  background: var(--bg-subtle);
  border-right: 1px solid var(--border-default);
  display: flex;
  flex-direction: column;
  padding: 12px 10px;
  gap: 8px;
  transition: width 0.2s ease;
  overflow: hidden;
}

.sidebar.collapsed {
  width: var(--sidebar-collapsed-width);
}

.side-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  min-height: 32px;
}

.collapsed .side-head {
  justify-content: center;
}

.brand {
  display: flex;
  align-items: center;
  gap: 8px;
  color: var(--accent);
  padding-left: 4px;
}

.brand-name {
  font-size: 15px;
  font-weight: 700;
  color: var(--text-primary);
  letter-spacing: 0.5px;
}

.icon-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 28px;
  border: none;
  border-radius: var(--radius-sm);
  background: transparent;
  color: var(--text-secondary);
  cursor: pointer;
  transition: background-color 0.1s;
}

.icon-btn:hover {
  background: var(--accent-soft);
  color: var(--text-primary);
}

.new-chat {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  width: 100%;
  padding: 8px 12px;
  border: 1px solid var(--border-default);
  border-radius: var(--radius-md);
  background: var(--bg-elevated);
  color: var(--text-primary);
  font-size: 13px;
  font-weight: 500;
  cursor: pointer;
  transition:
    border-color 0.15s,
    box-shadow 0.15s;
}

.new-chat:hover {
  border-color: var(--accent);
  box-shadow: 0 0 0 3px var(--accent-soft);
}

.session-list {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding: 2px;
}

.side-spacer {
  flex: 1;
}

.group-label {
  font-size: 11px;
  color: var(--text-tertiary);
  padding: 0 8px 4px;
}

.session-group {
  display: flex;
  flex-direction: column;
  gap: 1px;
}

.session-item {
  display: flex;
  align-items: center;
  gap: 6px;
  width: 100%;
  padding: 7px 8px;
  border: none;
  border-radius: var(--radius-sm);
  background: transparent;
  color: var(--text-secondary);
  font-size: 13px;
  text-align: left;
  cursor: pointer;
  transition: background-color 0.1s;
}

.session-item:hover {
  background: var(--accent-soft);
}

.session-item.active {
  background: var(--accent-soft);
  color: var(--text-primary);
  font-weight: 500;
}

.session-title {
  flex: 1;
  min-width: 0;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.session-del {
  flex-shrink: 0;
  width: 18px;
  height: 18px;
  display: none;
  align-items: center;
  justify-content: center;
  border-radius: 4px;
  color: var(--text-tertiary);
  font-size: 14px;
  line-height: 1;
}

.session-item:hover .session-del {
  display: inline-flex;
}

.session-del:hover {
  background: var(--bg-elevated);
  color: var(--error);
}

.empty-hint {
  padding: 12px 8px;
  font-size: 12px;
  color: var(--text-tertiary);
  text-align: center;
}

.side-bottom {
  display: flex;
  flex-direction: column;
  gap: 2px;
  border-top: 1px solid var(--border-default);
  padding-top: 8px;
}

.nav-item {
  display: flex;
  align-items: center;
  gap: 8px;
  width: 100%;
  padding: 7px 8px;
  border: none;
  border-radius: var(--radius-sm);
  background: transparent;
  color: var(--text-secondary);
  font-size: 13px;
  cursor: pointer;
  transition: background-color 0.1s;
}

.collapsed .nav-item {
  justify-content: center;
}

.nav-item:hover {
  background: var(--accent-soft);
  color: var(--text-primary);
}

.nav-item.active {
  color: var(--accent);
  font-weight: 500;
}

.version {
  padding: 6px 8px 0;
  font-size: 11px;
  color: var(--text-tertiary);
}
</style>
```

- [ ] **Step 2: 构建 + 测试**

Run: `cd frontend; npm run build; if ($?) { npm run test }`
Expected: 全绿

- [ ] **Step 3: 提交**

```powershell
git add frontend/src/components/AppSidebar.vue
git commit -m "feat(frontend): Codex 风格边栏（会话分组 + 导航 + 主题切换）"
```

---

### Task 8: AgentView 重写（欢迎屏 + 组合层）

**Files:**
- Modify: `frontend/src/views/AgentView.vue`（整体重写）

**Interfaces:**
- Consumes: `ChatMessage`（Task 5）、`ChatInput`（Task 6）、`useAgentChat`（Task 3）、`useChatSessions`（Task 2）
- Produces: `/agent` 页面最终形态

- [ ] **Step 1: 整体重写 `frontend/src/views/AgentView.vue`**

```vue
<template>
  <div class="agent-view">
    <main ref="streamRef" class="chat-stream" @scroll="onStreamScroll">
      <!-- 欢迎屏 -->
      <div v-if="!messages.length && !loading" class="welcome">
        <div class="welcome-icon">
          <svg
            viewBox="0 0 24 24"
            width="40"
            height="40"
            fill="none"
            stroke="currentColor"
            stroke-width="1.5"
            stroke-linecap="round"
            stroke-linejoin="round"
          >
            <path d="M12 2.69l5.66 5.66a8 8 0 1 1-11.31 0z" />
          </svg>
        </div>
        <h1 class="welcome-title">水卫 · 黄河吕梁段防汛预警智能体</h1>
        <p class="welcome-sub">实时水情 / 降雨 / 径流预测 / 预警等级 / 法规预案</p>
        <div class="suggest-grid">
          <button
            v-for="s in suggestions"
            :key="s"
            class="suggest-card"
            type="button"
            @click="useSuggestion(s)"
          >
            {{ s }}
          </button>
        </div>
      </div>

      <!-- 消息列表 -->
      <ChatMessage v-for="(msg, i) in messages" :key="i" :msg="msg" />
    </main>

    <ChatInput v-model:text="inputText" :loading="loading" @send="sendQuery" @stop="stopQuery" />
  </div>
</template>

<script setup lang="ts">
import { ref, watch } from 'vue'
import ChatMessage from '@/components/ChatMessage.vue'
import ChatInput from '@/components/ChatInput.vue'
import { useAgentChat, SUGGESTIONS } from '@/composables/useAgentChat'
import { useChatSessions } from '@/composables/useChatSessions'

const { messages, loading, inputText, userScrolledUp, sendQuery, stopQuery, useSuggestion } =
  useAgentChat()
const { activeSessionId } = useChatSessions()

const streamRef = ref<HTMLElement | null>(null)
let scrollRafId: number | null = null

function onStreamScroll() {
  if (!streamRef.value) return
  const { scrollTop, scrollHeight, clientHeight } = streamRef.value
  userScrolledUp.value = scrollHeight - scrollTop - clientHeight >= 60
}

function scrollToBottom(force = false) {
  if (scrollRafId !== null) return
  scrollRafId = requestAnimationFrame(() => {
    scrollRafId = null
    if (!streamRef.value) return
    if (!force && userScrolledUp.value) return
    streamRef.value.scrollTop = streamRef.value.scrollHeight
  })
}

// 新消息到达：强制滚到底
watch(
  () => messages.value.length,
  () => scrollToBottom(true),
)

// 流式更新：跟随滚动（用户上翻时除外）
watch(
  () => messages.value[messages.value.length - 1]?.content,
  () => scrollToBottom(),
)

// 切换会话：滚到底
watch(activeSessionId, () => scrollToBottom(true))

const suggestions = SUGGESTIONS
</script>

<style scoped>
.agent-view {
  display: flex;
  flex-direction: column;
  height: 100%;
  overflow: hidden;
  background: var(--bg-base);
}

.chat-stream {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  padding: 24px 24px 8px;
  scroll-behavior: auto;
}

.chat-stream > * {
  max-width: 768px;
  margin-left: auto;
  margin-right: auto;
}

/* 消息组件自带 margin-bottom，此处仅约束列宽 */
.chat-stream :deep(.user-row),
.chat-stream :deep(.ai-row) {
  max-width: 768px;
  margin-left: auto;
  margin-right: auto;
}

/* ===== 欢迎屏 ===== */
.welcome {
  text-align: center;
  padding: 72px 20px 40px;
}

.welcome-icon {
  width: 72px;
  height: 72px;
  margin: 0 auto 18px;
  border-radius: 18px;
  background: var(--accent-soft);
  color: var(--accent);
  display: flex;
  align-items: center;
  justify-content: center;
}

.welcome-title {
  margin: 0 0 8px;
  font-size: 22px;
  font-weight: 650;
  color: var(--text-primary);
  letter-spacing: -0.3px;
}

.welcome-sub {
  margin: 0 0 28px;
  font-size: 13px;
  color: var(--text-secondary);
}

.suggest-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
  max-width: 560px;
  margin: 0 auto;
}

.suggest-card {
  padding: 13px 16px;
  background: var(--bg-elevated);
  border: 1px solid var(--border-default);
  border-radius: var(--radius-md);
  font-size: 13px;
  color: var(--text-secondary);
  text-align: left;
  cursor: pointer;
  transition:
    transform 0.15s ease,
    border-color 0.15s,
    box-shadow 0.15s;
}

.suggest-card:hover {
  transform: translateY(-2px);
  border-color: var(--accent);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);
  color: var(--text-primary);
}
</style>
```

- [ ] **Step 2: 构建 + 测试**

Run: `cd frontend; npm run build; if ($?) { npm run test }`
Expected: 全绿。注意旧 AgentView 的 `clearChat`/`levelDesc`/`stepName` 模板引用已随重写移除

- [ ] **Step 3: 提交**

```powershell
git add frontend/src/views/AgentView.vue
git commit -m "feat(frontend): AgentView 重写为无气泡流式组合层"
```

---

### Task 9: HealthView 手写风格重写

**Files:**
- Modify: `frontend/src/views/HealthView.vue`（整体重写）

**Interfaces:**
- Consumes: `getHealth` / `HealthResponse`（`@/api/health`，不变）
- Produces: `/health` 页面最终形态（不再使用任何 el-* 组件）

- [ ] **Step 1: 整体重写 `frontend/src/views/HealthView.vue`**

```vue
<template>
  <div class="health-view">
    <div class="health-card">
      <div class="card-head">
        <span class="status-dot" :class="dotClass" />
        <h2 class="card-title">后端服务健康检查</h2>
        <button class="refresh-btn" type="button" :disabled="loading" @click="fetchHealth">
          <svg
            viewBox="0 0 24 24"
            width="14"
            height="14"
            fill="none"
            stroke="currentColor"
            stroke-width="2"
            stroke-linecap="round"
            stroke-linejoin="round"
          >
            <polyline points="23 4 23 10 17 10" />
            <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10" />
          </svg>
          刷新
        </button>
      </div>

      <div v-if="loading && !healthData" class="skeleton-list">
        <div v-for="i in 4" :key="i" class="skeleton-row" />
      </div>

      <div v-else-if="healthData" class="info-rows">
        <div class="info-row">
          <span class="info-key">服务状态</span>
          <span class="info-value">
            <span class="status-tag" :class="healthData.status === 'ok' ? 'ok' : 'bad'">
              {{ healthData.status === 'ok' ? '运行中' : '异常' }}
            </span>
          </span>
        </div>
        <div class="info-row">
          <span class="info-key">服务名称</span>
          <span class="info-value">{{ healthData.service }}</span>
        </div>
        <div class="info-row">
          <span class="info-key">版本号</span>
          <span class="info-value mono">{{ healthData.version }}</span>
        </div>
        <div class="info-row">
          <span class="info-key">环境</span>
          <span class="info-value mono">{{ healthData.env }}</span>
        </div>
        <div class="info-row">
          <span class="info-key">时间戳</span>
          <span class="info-value">{{ formatTime(healthData.timestamp) }}</span>
        </div>
      </div>

      <div v-if="errorMsg" class="error-box">{{ errorMsg }}</div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { getHealth, type HealthResponse } from '@/api/health'

const healthData = ref<HealthResponse | null>(null)
const loading = ref(false)
const errorMsg = ref('')

const dotClass = computed(() => {
  if (errorMsg.value) return 'bad'
  if (healthData.value?.status === 'ok') return 'ok'
  if (healthData.value) return 'bad'
  return 'unknown'
})

async function fetchHealth() {
  loading.value = true
  errorMsg.value = ''
  try {
    healthData.value = await getHealth()
  } catch (e: any) {
    errorMsg.value = `后端连接失败：${e?.message || '未知错误'}`
    healthData.value = null
  } finally {
    loading.value = false
  }
}

function formatTime(ts?: string): string {
  if (!ts) return '-'
  try {
    return new Date(ts).toLocaleString('zh-CN', { timeZone: 'Asia/Shanghai' })
  } catch {
    return ts
  }
}

onMounted(fetchHealth)
</script>

<style scoped>
.health-view {
  height: 100%;
  overflow-y: auto;
  padding: 32px 24px;
  background: var(--bg-base);
}

.health-card {
  max-width: 640px;
  margin: 0 auto;
  background: var(--bg-elevated);
  border: 1px solid var(--border-default);
  border-radius: var(--radius-lg);
  padding: 20px 24px;
}

.card-head {
  display: flex;
  align-items: center;
  gap: 10px;
  padding-bottom: 16px;
  border-bottom: 1px solid var(--border-default);
}

.status-dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  flex-shrink: 0;
}

.status-dot.ok {
  background: var(--success);
  box-shadow: 0 0 0 3px var(--accent-soft);
  animation: pulse 2s infinite;
}

.status-dot.bad {
  background: var(--error);
}

.status-dot.unknown {
  background: var(--text-tertiary);
}

.card-title {
  flex: 1;
  font-size: 15px;
  font-weight: 600;
  color: var(--text-primary);
}

.refresh-btn {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 14px;
  border: 1px solid var(--border-default);
  border-radius: var(--radius-sm);
  background: var(--bg-elevated);
  color: var(--text-secondary);
  font-size: 12px;
  cursor: pointer;
  transition:
    border-color 0.15s,
    color 0.15s;
}

.refresh-btn:hover:not(:disabled) {
  border-color: var(--accent);
  color: var(--accent);
}

.refresh-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.skeleton-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
  padding: 16px 0 4px;
}

.skeleton-row {
  height: 18px;
  border-radius: 6px;
  background: linear-gradient(90deg, var(--bg-subtle) 25%, var(--border-default) 50%, var(--bg-subtle) 75%);
  background-size: 200% 100%;
  animation: shimmer 1.4s infinite;
}

.info-rows {
  display: flex;
  flex-direction: column;
  padding: 8px 0 4px;
}

.info-row {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 9px 0;
  border-bottom: 1px solid var(--border-default);
}

.info-row:last-child {
  border-bottom: none;
}

.info-key {
  width: 88px;
  flex-shrink: 0;
  font-size: 13px;
  color: var(--text-tertiary);
}

.info-value {
  font-size: 13px;
  color: var(--text-primary);
}

.mono {
  font-family: 'SF Mono', Monaco, Consolas, monospace;
}

.status-tag {
  display: inline-block;
  padding: 2px 10px;
  border-radius: 10px;
  font-size: 12px;
  font-weight: 500;
}

.status-tag.ok {
  color: var(--success);
  background: var(--level-4-soft);
}

.status-tag.bad {
  color: var(--error);
  background: var(--level-1-soft);
}

.error-box {
  margin-top: 14px;
  padding: 10px 14px;
  border: 1px solid var(--error);
  border-radius: var(--radius-sm);
  background: var(--level-1-soft);
  color: var(--error);
  font-size: 13px;
}

@keyframes pulse {
  0% {
    box-shadow: 0 0 0 0 rgba(22, 163, 74, 0.3);
  }
  70% {
    box-shadow: 0 0 0 6px rgba(22, 163, 74, 0);
  }
  100% {
    box-shadow: 0 0 0 0 rgba(22, 163, 74, 0);
  }
}

@keyframes shimmer {
  0% {
    background-position: 200% 0;
  }
  100% {
    background-position: -200% 0;
  }
}
</style>
```

- [ ] **Step 2: 构建 + 测试**

Run: `cd frontend; npm run build; if ($?) { npm run test }`
Expected: 全绿

- [ ] **Step 3: 提交**

```powershell
git add frontend/src/views/HealthView.vue
git commit -m "feat(frontend): HealthView 手写风格重写（脉冲状态点 + 骨架屏）"
```

---

### Task 10: App Shell + main.ts 精简 + 清理 + 终验

**Files:**
- Modify: `frontend/src/App.vue`（整体重写）
- Modify: `frontend/src/main.ts`（去 Element Plus 全量注册与图标循环）
- Modify: `frontend/src/composables/useAgentChat.ts`（移除 `clearChat` shim 及其接口声明）
- Modify: `README.md`（技术栈一行更新）

**Interfaces:**
- Consumes: `AppSidebar`（Task 7）、`useTheme`（Task 1）
- Produces: 最终 App Shell；`useAgentChat` 不再导出 `clearChat`（`UseAgentChatReturn` 同步删除该字段）

- [ ] **Step 1: 整体重写 `frontend/src/App.vue`**

```vue
<template>
  <div class="app-shell">
    <AppSidebar />
    <div class="main-area">
      <router-view />
    </div>
  </div>
</template>

<script setup lang="ts">
import AppSidebar from '@/components/AppSidebar.vue'
import { useTheme } from '@/composables/useTheme'

// 确保主题在应用启动时初始化
useTheme()
</script>

<style scoped>
.app-shell {
  display: flex;
  height: 100vh;
  overflow: hidden;
  background: var(--bg-base);
  color: var(--text-primary);
}

.main-area {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
</style>
```

- [ ] **Step 2: 精简 `frontend/src/main.ts`**

完整内容替换为：

```ts
import { createApp } from 'vue'
import 'element-plus/dist/index.css'

import App from './App.vue'
import router from './router'
import './styles/theme.css'

const app = createApp(App)

app.use(router)
app.mount('#app')
```

（保留 Element Plus 全量 CSS 供 ElMessage 使用；`ElMessage` 为函数式调用，无需 `app.use(ElementPlus)`；图标全局注册与 `zhCn` locale 随旧壳一并移除。）

- [ ] **Step 3: 移除 useAgentChat 中的 clearChat**

`frontend/src/composables/useAgentChat.ts`：
1. 删除 `clearChat` 函数及其上方 `@deprecated` 注释
2. `UseAgentChatReturn` 接口中删除 `clearChat: () => void` 一行
3. `useAgentChat()` return 对象中删除 `clearChat,` 一行

- [ ] **Step 4: README 技术栈行更新**

`README.md` 第 186 行 `- **前端**：Vue 3、TypeScript、Element Plus、Vite` 改为：

```markdown
- **前端**：Vue 3、TypeScript、Vite（Codex 风格手写 UI，Element Plus 仅用于全局提示）
```

- [ ] **Step 5: 全量验证**

```powershell
cd frontend; npm run build; if ($?) { npm run test }
```
Expected: vue-tsc + vite build 成功；15 个测试全部通过

- [ ] **Step 6: 人工走查（启动 dev server 逐项核对规格 §12）**

```powershell
cd frontend; npm run dev
```

走查清单：
- [ ] 欢迎屏：居中大标题 + 2×2 建议卡片，点击卡片发起对话
- [ ] 发送/停止/流式渲染正常；Enter 发送、Shift+Enter 换行、中文输入法组词中 Enter 不发送
- [ ] 推理块/工具块折叠展开、完成后自动折叠
- [ ] 预警卡片四色（Ⅰ红/Ⅱ橙/Ⅲ黄/Ⅳ蓝）+ 应急措施列表
- [ ] 会话：新建/切换/hover 删除/自动命名/今天·昨天·更早分组；刷新后恢复
- [ ] 主题：边栏底部切换，刷新后保持；系统偏好兜底
- [ ] 边栏折叠为图标栏，过渡顺滑
- [ ] 服务健康页：状态点/刷新/错误提示

- [ ] **Step 7: 提交**

```powershell
git add frontend/src/App.vue frontend/src/main.ts frontend/src/composables/useAgentChat.ts README.md
git commit -m "feat(frontend): Codex 风格 App Shell（边栏布局）与 Element Plus 精简"
```

---

## Self-Review 记录

- **Spec coverage**：规格 §3→Task10；§4→Task1；§5→Task2/3；§6→Task4/5/8；§7→Task6；§8→Task9；§9→各组件样式（fade-in/spin/transition + theme.css reduced-motion）；§11→Task2 纯函数容错 + Task3 错误路径；§12→Task10 终验。
- **类型一致性**：`Message/ToolEvent/ReasoningStepEntry` 唯一定义于 Task 2，Task 3 re-export，组件统一从 `@/composables/useChatSessions` 导入类型、`@/composables/useAgentChat` 导入 `stepName/levelDesc/SUGGESTIONS`。`v-model:expanded` 对应 `expanded?: boolean` + `update:expanded` emit；`v-model:text` 对应 `text: string` + `update:text` emit。
- **构建连续性**：Task 3 保留 `clearChat` shim 保证旧 AgentView 编译；Task 8 重写 AgentView 后 Task 10 才移除 shim。HealthView（Task 9）先于 main.ts 精简（Task 10），el-* 组件在移除注册前已全部退场。
