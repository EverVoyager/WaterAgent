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
