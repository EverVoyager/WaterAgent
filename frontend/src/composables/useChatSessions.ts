/**
 * 会话管理 composable：多会话 CRUD + 后端 MySQL 持久化（P1-b）
 *
 * 模块级单例，跨组件共享。硬失败策略：MySQL 不可用时抛错，不降级到 localStorage。
 * 草稿态约定：activeSessionId === '' 表示欢迎屏草稿，首次发送时真正建会话。
 */
import { computed, ref, type ComputedRef, type Ref } from 'vue'
import type { AgentQueryResponse, ReasoningStep } from '@/api/agent'
import {
  createSession as createSessionApi,
  deleteSession as deleteSessionApi,
  listSessions,
  syncSession,
} from '@/api/sessions'

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
  // 状态
  sessions: ComputedRef<ChatSession[]>
  activeSessionId: Ref<string>
  activeSession: ComputedRef<ChatSession | null>
  activeMessages: ComputedRef<Message[]>
  groupedSessions: ComputedRef<SessionGroup[]>
  sessionsLoaded: Ref<boolean>
  sessionsError: Ref<string>

  // 方法
  initSessions: () => Promise<void>
  createSession: () => void
  ensureActiveSession: () => Promise<void>
  switchSession: (id: string) => void
  deleteSession: (id: string) => Promise<void>
  persistActiveSession: () => Promise<void>
  persistSessionById: (id: string) => Promise<void>
}

// ====== 常量 ======

const DEFAULT_TITLE = '新会话'
const TITLE_MAX_LEN = 20
const DAY_MS = 24 * 60 * 60 * 1000

// ====== 纯函数（可单测）======

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

const sessions = ref<ChatSession[]>([])
const activeSessionId = ref<string>('')
const sessionsLoaded = ref(false)
const sessionsError = ref('')
let initPromise: Promise<void> | null = null

// ====== 异步初始化 ======

async function initSessions(): Promise<void> {
  if (initPromise) return initPromise
  if (sessionsLoaded.value) return
  initPromise = (async () => {
    try {
      const list = await listSessions()
      sessions.value = list
      sessionsError.value = ''
      sessionsLoaded.value = true
    } catch (e: any) {
      sessionsError.value = e?.response?.data?.detail || e?.message || '加载会话失败'
      console.error('[useChatSessions] 加载会话失败', e)
      // 清除缓存的 rejected promise，允许下次调用重试（网络恢复后自愈）
      initPromise = null
      throw e
    }
  })()
  return initPromise
}

// ====== composable ======

export function useChatSessions(): UseChatSessionsReturn {
  const sorted = computed(() => [...sessions.value].sort((a, b) => b.updatedAt - a.updatedAt))
  const activeSession = computed(
    () => sessions.value.find((s) => s.id === activeSessionId.value) || null,
  )
  const activeMessages = computed(() => activeSession.value?.messages ?? [])
  const groupedSessions = computed(() => groupSessions(sorted.value, Date.now()))

  async function ensureActiveSession() {
    await initSessions()
    if (activeSession.value) return
    const id = genId()
    const now = Date.now()
    // 先创建到后端（硬失败：API 失败则不创建内存态）
    await createSessionApi({ id })
    // API 成功后在内存中创建（UI 响应）
    const s: ChatSession = {
      id,
      title: DEFAULT_TITLE,
      createdAt: now,
      updatedAt: now,
      messages: [],
    }
    sessions.value.push(s)
    activeSessionId.value = id
  }

  async function persistSessionById(sessionId: string) {
    const session = sessions.value.find((s) => s.id === sessionId)
    if (!session) return
    session.updatedAt = Date.now()
    if (session.title === DEFAULT_TITLE) {
      const firstUser = session.messages.find((m) => m.role === 'user')
      if (firstUser) {
        const t = firstUser.content.slice(0, TITLE_MAX_LEN).trim()
        if (t) session.title = t
      }
    }
    // 全量同步到后端（标题 + 消息）
    await syncSession(session.id, {
      title: session.title,
      messages: session.messages,
    })
  }

  async function persistActiveSession() {
    const active = activeSession.value
    if (!active) return
    await persistSessionById(active.id)
  }

  function createSession() {
    // 草稿态：置空 activeId，首次发送时由 ensureActiveSession 真正创建
    activeSessionId.value = ''
  }

  function switchSession(id: string) {
    if (!sessions.value.some((s) => s.id === id)) return
    activeSessionId.value = id
  }

  async function deleteSession(id: string) {
    const idx = sessions.value.findIndex((s) => s.id === id)
    if (idx === -1) return
    // 先删除后端（硬失败）
    await deleteSessionApi(id)
    // API 成功后更新内存态
    sessions.value.splice(idx, 1)
    if (activeSessionId.value === id) {
      const rest = [...sessions.value].sort((a, b) => b.updatedAt - a.updatedAt)
      activeSessionId.value = rest[0]?.id || ''
    }
  }

  return {
    sessions: sorted,
    activeSessionId,
    activeSession,
    activeMessages,
    groupedSessions,
    sessionsLoaded,
    sessionsError,
    initSessions,
    createSession,
    ensureActiveSession,
    switchSession,
    deleteSession,
    persistActiveSession,
    persistSessionById,
  }
}
