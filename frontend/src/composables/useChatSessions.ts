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
