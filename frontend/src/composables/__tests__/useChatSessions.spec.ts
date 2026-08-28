import { beforeEach, describe, expect, it, vi } from 'vitest'

// ====== mock @/api/sessions：所有测试共享的 API mock ======
// 每个 it 通过 vi.mocked() 重置返回值，确保测试隔离
vi.mock('@/api/sessions', () => ({
  listSessions: vi.fn(async () => []),
  createSession: vi.fn(async (req: { id: string }) => ({
    id: req.id,
    title: '新会话',
    createdAt: Date.now(),
    updatedAt: Date.now(),
    messages: [],
  })),
  getSession: vi.fn(async (id: string) => ({
    id,
    title: '新会话',
    createdAt: 0,
    updatedAt: 0,
    messages: [],
  })),
  syncSession: vi.fn(async (id: string, req: { title: string; messages: any[] }) => ({
    id,
    title: req.title,
    createdAt: 0,
    updatedAt: Date.now(),
    messages: req.messages,
  })),
  updateSessionTitle: vi.fn(async () => {}),
  deleteSession: vi.fn(async () => {}),
}))

import {
  groupSessions,
  type ChatSession,
} from '@/composables/useChatSessions'
import * as sessionsApi from '@/api/sessions'

function makeSession(partial: Partial<ChatSession>): ChatSession {
  return { id: 'x', title: '新会话', createdAt: 0, updatedAt: 0, messages: [], ...partial }
}

beforeEach(() => {
  vi.clearAllMocks()
  // 重置 listSessions 默认返回空数组
  vi.mocked(sessionsApi.listSessions).mockResolvedValue([])
  vi.resetModules()
})

describe('纯函数 groupSessions', () => {
  it('按今天/昨天/更早分组', () => {
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

  it('空列表返回空数组', () => {
    expect(groupSessions([], Date.now())).toEqual([])
  })
})

describe('useChatSessions 单例（后端 API 持久化）', () => {
  it('initSessions 调用 listSessions 并填充 sessions', async () => {
    vi.mocked(sessionsApi.listSessions).mockResolvedValueOnce([
      makeSession({ id: 's1', title: '历史会话', updatedAt: Date.now() }),
    ])
    const { useChatSessions } = await import('@/composables/useChatSessions')
    const s = useChatSessions()
    await s.initSessions()
    expect(sessionsApi.listSessions).toHaveBeenCalledOnce()
    expect(s.sessions.value).toHaveLength(1)
    expect(s.sessions.value[0].id).toBe('s1')
    expect(s.sessionsLoaded.value).toBe(true)
    expect(s.sessionsError.value).toBe('')
  })

  it('initSessions 失败时设置 sessionsError 并抛错，且不缓存失败状态（允许重试）', async () => {
    vi.mocked(sessionsApi.listSessions).mockRejectedValueOnce(new Error('后端不可用'))
    const { useChatSessions } = await import('@/composables/useChatSessions')
    const s = useChatSessions()
    await expect(s.initSessions()).rejects.toThrow('后端不可用')
    expect(s.sessionsError.value).toBe('后端不可用')
    // 失败不标记 loaded，且清空缓存的 rejected promise，下次调用可重试
    expect(s.sessionsLoaded.value).toBe(false)
    // 网络恢复后重试成功
    vi.mocked(sessionsApi.listSessions).mockResolvedValueOnce([])
    await s.initSessions()
    expect(s.sessionsLoaded.value).toBe(true)
  })

  it('ensureActiveSession 在无会话时创建并激活', async () => {
    const { useChatSessions } = await import('@/composables/useChatSessions')
    const s = useChatSessions()
    expect(s.activeSession.value).toBeNull()
    await s.ensureActiveSession()
    expect(sessionsApi.createSession).toHaveBeenCalledOnce()
    expect(s.activeSession.value).not.toBeNull()
    expect(s.activeMessages.value).toEqual([])
  })

  it('ensureActiveSession 已有激活会话时不重复创建', async () => {
    const { useChatSessions } = await import('@/composables/useChatSessions')
    const s = useChatSessions()
    await s.ensureActiveSession()
    const firstId = s.activeSessionId.value
    await s.ensureActiveSession()
    expect(sessionsApi.createSession).toHaveBeenCalledOnce()
    expect(s.activeSessionId.value).toBe(firstId)
  })

  it('ensureActiveSession 在 createSession API 失败时不创建内存态', async () => {
    vi.mocked(sessionsApi.createSession).mockRejectedValueOnce(new Error('创建失败'))
    const { useChatSessions } = await import('@/composables/useChatSessions')
    const s = useChatSessions()
    await expect(s.ensureActiveSession()).rejects.toThrow('创建失败')
    expect(s.activeSession.value).toBeNull()
    expect(s.sessions.value).toHaveLength(0)
  })

  it('persistActiveSession 用首条用户消息自动命名（前 20 字）并同步后端', async () => {
    const { useChatSessions } = await import('@/composables/useChatSessions')
    const s = useChatSessions()
    await s.ensureActiveSession()
    s.activeMessages.value.push({ role: 'user', content: '吴堡站当前水情如何？请详细说明一下情况' })
    await s.persistActiveSession()
    const expectedTitle = '吴堡站当前水情如何？请详细说明一下情况'.slice(0, 20)
    expect(s.activeSession.value!.title).toBe(expectedTitle)
    expect(sessionsApi.syncSession).toHaveBeenCalledOnce()
    const [id, req] = vi.mocked(sessionsApi.syncSession).mock.calls[0]
    expect(id).toBe(s.activeSessionId.value)
    expect(req.title).toBe(expectedTitle)
    expect(req.messages).toHaveLength(1)
  })

  it('persistActiveSession 无激活会话时静默返回', async () => {
    const { useChatSessions } = await import('@/composables/useChatSessions')
    const s = useChatSessions()
    await s.persistActiveSession()
    expect(sessionsApi.syncSession).not.toHaveBeenCalled()
  })

  it('createSession 进入草稿态（置空 activeId）', async () => {
    const { useChatSessions } = await import('@/composables/useChatSessions')
    const s = useChatSessions()
    await s.ensureActiveSession()
    s.createSession()
    expect(s.activeSession.value).toBeNull()
    expect(s.activeMessages.value).toEqual([])
  })

  it('switchSession 切换激活会话', async () => {
    const { useChatSessions } = await import('@/composables/useChatSessions')
    const s = useChatSessions()
    await s.ensureActiveSession()
    const firstId = s.activeSessionId.value
    s.createSession()
    await s.ensureActiveSession()
    expect(s.activeSessionId.value).not.toBe(firstId)
    s.switchSession(firstId)
    expect(s.activeSessionId.value).toBe(firstId)
  })

  it('switchSession 切换不存在的会话时不改变 activeId', async () => {
    const { useChatSessions } = await import('@/composables/useChatSessions')
    const s = useChatSessions()
    await s.ensureActiveSession()
    const currentId = s.activeSessionId.value
    s.switchSession('nonexistent')
    expect(s.activeSessionId.value).toBe(currentId)
  })

  it('deleteSession 删除当前会话后回退到最新会话', async () => {
    const { useChatSessions } = await import('@/composables/useChatSessions')
    const s = useChatSessions()
    await s.ensureActiveSession()
    const firstId = s.activeSessionId.value
    await s.persistActiveSession()
    s.createSession()
    await s.ensureActiveSession()
    const secondId = s.activeSessionId.value
    await s.deleteSession(secondId)
    expect(sessionsApi.deleteSession).toHaveBeenCalledWith(secondId)
    expect(s.activeSessionId.value).toBe(firstId)
    await s.deleteSession(firstId)
    expect(s.activeSessionId.value).toBe('')
    expect(s.sessions.value).toHaveLength(0)
  })

  it('deleteSession 删除非当前会话时不改变 activeId', async () => {
    const { useChatSessions } = await import('@/composables/useChatSessions')
    const s = useChatSessions()
    await s.ensureActiveSession()
    const activeId = s.activeSessionId.value
    s.createSession()
    await s.ensureActiveSession()
    const otherId = s.activeSessionId.value
    // 删除第一个（非当前激活）
    await s.deleteSession(activeId)
    expect(s.activeSessionId.value).toBe(otherId)
  })

  it('deleteSession API 失败时不更新内存态', async () => {
    vi.mocked(sessionsApi.deleteSession).mockRejectedValueOnce(new Error('删除失败'))
    const { useChatSessions } = await import('@/composables/useChatSessions')
    const s = useChatSessions()
    await s.ensureActiveSession()
    const id = s.activeSessionId.value
    await expect(s.deleteSession(id)).rejects.toThrow('删除失败')
    expect(s.sessions.value).toHaveLength(1)
  })
})
