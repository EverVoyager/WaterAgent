import { beforeEach, describe, expect, it, vi } from 'vitest'

// ====== mock queryAgentStream：默认 noop，不发起任何请求 ======
vi.mock('@/api/agent', () => ({
  queryAgentStream: vi.fn(() => new AbortController()),
}))

// ====== mock useToast ======
vi.mock('@/composables/useToast', () => ({
  useToast: () => ({ error: vi.fn(), success: vi.fn(), info: vi.fn() }),
}))

// ====== mock @/api/sessions：会话持久化已迁后端，测试中不发起真实请求 ======
vi.mock('@/api/sessions', () => ({
  listSessions: vi.fn(async () => []),
  createSession: vi.fn(async (req: { id: string }) => ({
    id: req.id,
    title: '新会话',
    createdAt: Date.now(),
    updatedAt: Date.now(),
    messages: [],
  })),
  syncSession: vi.fn(async (id: string, req: { title: string; messages: any[] }) => ({
    id,
    title: req.title,
    createdAt: 0,
    updatedAt: Date.now(),
    messages: req.messages,
  })),
  deleteSession: vi.fn(async () => {}),
  getSession: vi.fn(async (id: string) => ({
    id,
    title: '新会话',
    createdAt: 0,
    updatedAt: 0,
    messages: [],
  })),
  updateSessionTitle: vi.fn(async () => {}),
}))

beforeEach(() => {
  vi.resetModules()
  vi.clearAllMocks()
})

describe('纯函数', () => {
  it('levelDesc 返回等级描述，未知等级原样返回', async () => {
    const { levelDesc, LEVEL_DESC } = await import('@/composables/useAgentChat')
    expect(levelDesc('I')).toBe(LEVEL_DESC.I)
    expect(levelDesc('II')).toBe(LEVEL_DESC.II)
    expect(levelDesc('III')).toBe(LEVEL_DESC.III)
    expect(levelDesc('IV')).toBe(LEVEL_DESC.IV)
    expect(levelDesc('UNKNOWN')).toBe('UNKNOWN')
  })

  it('stepName 返回步骤中文名，未知步骤原样返回', async () => {
    const { stepName, STEP_NAME } = await import('@/composables/useAgentChat')
    expect(stepName('router')).toBe(STEP_NAME.router)
    expect(stepName('planner')).toBe(STEP_NAME.planner)
    expect(stepName('executor')).toBe(STEP_NAME.executor)
    expect(stepName('synthesizer')).toBe(STEP_NAME.synthesizer)
    expect(stepName('direct_chat')).toBe(STEP_NAME.direct_chat)
    expect(stepName('reflector')).toBe(STEP_NAME.reflector)
    expect(stepName('unknown_step')).toBe('unknown_step')
  })

  it('SUGGESTIONS 包含业务场景与闲聊示例', async () => {
    const { SUGGESTIONS } = await import('@/composables/useAgentChat')
    expect(SUGGESTIONS.length).toBeGreaterThanOrEqual(3)
    expect(SUGGESTIONS.some((s) => s.includes('水情'))).toBe(true)
    expect(SUGGESTIONS.some((s) => s.includes('预警'))).toBe(true)
  })

  it('LEVEL_DESC 覆盖 I/II/III/IV 四级', async () => {
    const { LEVEL_DESC } = await import('@/composables/useAgentChat')
    expect(Object.keys(LEVEL_DESC).sort()).toEqual(['I', 'II', 'III', 'IV'])
    expect(LEVEL_DESC.I).toContain('Ⅰ级')
    expect(LEVEL_DESC.IV).toContain('Ⅳ级')
  })

  it('STEP_NAME 覆盖所有推理阶段', async () => {
    const { STEP_NAME } = await import('@/composables/useAgentChat')
    const keys = ['router', 'planner', 'executor', 'synthesizer', 'direct_chat']
    for (const k of keys) {
      expect(STEP_NAME[k]).toBeTruthy()
    }
  })
})

describe('handleStreamEvent - 事件处理', () => {
  /** 准备一个空的 assistant 消息并获取其索引。 */
  async function setupAssistantMessage() {
    const { useAgentChat } = await import('@/composables/useAgentChat')
    const chat = useAgentChat()
    const messages = chat.messages
    messages.value.push({ role: 'user', content: '吴堡站水情' })
    messages.value.push({
      role: 'assistant',
      content: '',
      toolEvents: [],
      reasoningSteps: [],
      thinking: true,
      chainExpanded: true,
      reasoningExpanded: true,
    })
    const aiMsgIdx = messages.value.length - 1
    return { chat, messages, aiMsgIdx }
  }

  it('reasoning_step 事件被追加到 reasoningSteps', async () => {
    const { chat, messages, aiMsgIdx } = await setupAssistantMessage()
    chat.handleStreamEvent(
      { type: 'reasoning_step', step: 'router', phase: 'start', message: '识别意图...' },
      aiMsgIdx,
    )
    const msg = messages.value[aiMsgIdx]
    expect(msg.reasoningSteps).toHaveLength(1)
    expect(msg.reasoningSteps![0]).toMatchObject({
      step: 'router',
      phase: 'start',
      message: '识别意图...',
    })
    expect(msg.reasoningSteps![0].timestamp).toBeGreaterThan(0)
  })

  it('reasoning_step 缺省 step/phase/message 时有兜底值', async () => {
    const { chat, messages, aiMsgIdx } = await setupAssistantMessage()
    chat.handleStreamEvent({ type: 'reasoning_step' }, aiMsgIdx)
    const step = messages.value[aiMsgIdx].reasoningSteps![0]
    expect(step.step).toBe('router')
    expect(step.phase).toBe('start')
    expect(step.message).toBe('')
  })

  it('intent=chitchat 时折叠工具链展开', async () => {
    const { chat, messages, aiMsgIdx } = await setupAssistantMessage()
    chat.handleStreamEvent({ type: 'intent', intent: 'chitchat' }, aiMsgIdx)
    expect(messages.value[aiMsgIdx].chainExpanded).toBe(false)
  })

  it('intent=agent_task 不改变 chainExpanded', async () => {
    const { chat, messages, aiMsgIdx } = await setupAssistantMessage()
    chat.handleStreamEvent({ type: 'intent', intent: 'agent_task' }, aiMsgIdx)
    expect(messages.value[aiMsgIdx].chainExpanded).toBe(true)
  })

  it('tool_call 事件新增 running 状态的工具事件', async () => {
    const { chat, messages, aiMsgIdx } = await setupAssistantMessage()
    chat.handleStreamEvent(
      { type: 'tool_call', tool: 'get_hydrology', arguments: { station: '吴堡' }, round: 1 },
      aiMsgIdx,
    )
    const events = messages.value[aiMsgIdx].toolEvents!
    expect(events).toHaveLength(1)
    expect(events[0]).toMatchObject({
      tool: 'get_hydrology',
      arguments: { station: '吴堡' },
      round: 1,
      status: 'running',
    })
  })

  it('tool_call 缺省字段有兜底', async () => {
    const { chat, messages, aiMsgIdx } = await setupAssistantMessage()
    chat.handleStreamEvent({ type: 'tool_call' }, aiMsgIdx)
    const ev = messages.value[aiMsgIdx].toolEvents![0]
    expect(ev.tool).toBe('unknown')
    expect(ev.arguments).toEqual({})
    expect(ev.round).toBe(1)
    expect(ev.status).toBe('running')
  })

  it('tool_result 事件把最近一个 running 状态的同名工具更新为 done', async () => {
    const { chat, messages, aiMsgIdx } = await setupAssistantMessage()
    chat.handleStreamEvent(
      { type: 'tool_call', tool: 'get_hydrology', arguments: {}, round: 1 },
      aiMsgIdx,
    )
    chat.handleStreamEvent(
      {
        type: 'tool_result',
        tool: 'get_hydrology',
        result: { flow: 1234.5 },
        error: '',
        round: 1,
      },
      aiMsgIdx,
    )
    const ev = messages.value[aiMsgIdx].toolEvents![0]
    expect(ev.status).toBe('done')
    expect(ev.result).toEqual({ flow: 1234.5 })
    expect(ev.error).toBe('')
  })

  it('tool_result 带 error 时工具状态变 error', async () => {
    const { chat, messages, aiMsgIdx } = await setupAssistantMessage()
    chat.handleStreamEvent(
      { type: 'tool_call', tool: 'get_weather', arguments: {}, round: 1 },
      aiMsgIdx,
    )
    chat.handleStreamEvent(
      {
        type: 'tool_result',
        tool: 'get_weather',
        result: {},
        error: 'API timeout',
        round: 1,
      },
      aiMsgIdx,
    )
    const ev = messages.value[aiMsgIdx].toolEvents![0]
    expect(ev.status).toBe('error')
    expect(ev.error).toBe('API timeout')
  })

  it('tool_result 没有匹配 running 工具时不修改任何事件', async () => {
    const { chat, messages, aiMsgIdx } = await setupAssistantMessage()
    chat.handleStreamEvent(
      { type: 'tool_call', tool: 'get_hydrology', arguments: {}, round: 1 },
      aiMsgIdx,
    )
    chat.handleStreamEvent(
      { type: 'tool_result', tool: 'other_tool', result: {}, error: '', round: 1 },
      aiMsgIdx,
    )
    const ev = messages.value[aiMsgIdx].toolEvents![0]
    expect(ev.status).toBe('running') // 未变
    expect(ev.result).toBeUndefined()
  })

  it('synth_meta 事件初始化 response 并填充 warning_level/reasoning/actions', async () => {
    const { chat, messages, aiMsgIdx } = await setupAssistantMessage()
    chat.handleStreamEvent(
      {
        type: 'synth_meta',
        data: {
          warning_level: 'II',
          reasoning: '依据降雨+水情综合判定',
          actions: ['转移群众', '加强巡堤'],
        },
      },
      aiMsgIdx,
    )
    const resp = messages.value[aiMsgIdx].response!
    expect(resp.warning_level).toBe('II')
    expect(resp.reasoning).toBe('依据降雨+水情综合判定')
    expect(resp.actions).toEqual(['转移群众', '加强巡堤'])
    expect(resp.answer).toBe('')
  })

  it('synth_meta 二次推送覆盖已有 response 的等级字段', async () => {
    const { chat, messages, aiMsgIdx } = await setupAssistantMessage()
    chat.handleStreamEvent(
      { type: 'synth_meta', data: { warning_level: 'III', reasoning: 'r1', actions: [] } },
      aiMsgIdx,
    )
    chat.handleStreamEvent(
      { type: 'synth_meta', data: { warning_level: 'I', reasoning: 'r2', actions: ['a'] } },
      aiMsgIdx,
    )
    const resp = messages.value[aiMsgIdx].response!
    expect(resp.warning_level).toBe('I')
    expect(resp.reasoning).toBe('r2')
    expect(resp.actions).toEqual(['a'])
  })

  it('answer_delta 事件累加到 content', async () => {
    const { chat, messages, aiMsgIdx } = await setupAssistantMessage()
    chat.handleStreamEvent({ type: 'answer_delta', content: '吴堡站' }, aiMsgIdx)
    chat.handleStreamEvent({ type: 'answer_delta', content: '当前流量' }, aiMsgIdx)
    chat.handleStreamEvent({ type: 'answer_delta', content: '1234 m³/s' }, aiMsgIdx)
    expect(messages.value[aiMsgIdx].content).toBe('吴堡站当前流量1234 m³/s')
  })

  it('done 事件设置 response，关闭 thinking，loading 置 false', async () => {
    const { chat, messages, aiMsgIdx } = await setupAssistantMessage()
    const doneData = {
      answer: '最终答案',
      warning_level: 'III' as const,
      reasoning: 'r',
      actions: ['a1'],
      tool_calls: [],
      rounds: 2,
      intent: 'agent_task' as const,
    }
    chat.handleStreamEvent({ type: 'done', data: doneData }, aiMsgIdx)
    const msg = messages.value[aiMsgIdx]
    expect(msg.thinking).toBe(false)
    expect(msg.response).toEqual(doneData)
    expect(msg.content).toBe('最终答案') // 空 content 被覆盖
    expect(msg.chainExpanded).toBe(false)
    expect(msg.reasoningExpanded).toBe(false)
    expect(chat.loading.value).toBe(false)
  })

  it('done 事件不覆盖已流式累积的 content', async () => {
    const { chat, messages, aiMsgIdx } = await setupAssistantMessage()
    chat.handleStreamEvent({ type: 'answer_delta', content: '流式内容' }, aiMsgIdx)
    chat.handleStreamEvent(
      {
        type: 'done',
        data: {
          answer: '完整答案',
          warning_level: '',
          reasoning: '',
          actions: [],
          tool_calls: [],
          rounds: 0,
          intent: 'agent_task',
        },
      },
      aiMsgIdx,
    )
    // 已有流式内容时不被 done.data.answer 覆盖
    expect(messages.value[aiMsgIdx].content).toBe('流式内容')
  })

  it('error 事件设置错误文案，关闭 thinking，loading 置 false', async () => {
    const { chat, messages, aiMsgIdx } = await setupAssistantMessage()
    chat.handleStreamEvent({ type: 'error', message: 'Agent 崩溃' }, aiMsgIdx)
    const msg = messages.value[aiMsgIdx]
    expect(msg.thinking).toBe(false)
    expect(msg.content).toContain('Agent 崩溃')
    expect(chat.loading.value).toBe(false)
  })

  it('aiMsgIdx 越界时静默返回（不抛异常）', async () => {
    const { chat } = await setupAssistantMessage()
    expect(() =>
      chat.handleStreamEvent({ type: 'answer_delta', content: 'x' }, 9999),
    ).not.toThrow()
  })
})

describe('sendQuery - 发送流程', () => {
  it('空查询不发起请求', async () => {
    const { useAgentChat } = await import('@/composables/useAgentChat')
    const { queryAgentStream } = await import('@/api/agent')
    const chat = useAgentChat()
    chat.inputText.value = '   '
    await chat.sendQuery()
    expect(queryAgentStream).not.toHaveBeenCalled()
  })

  it('loading=true 时不重复发起请求', async () => {
    const { useAgentChat } = await import('@/composables/useAgentChat')
    const { queryAgentStream } = await import('@/api/agent')
    const chat = useAgentChat()
    chat.inputText.value = '问题'
    chat.loading.value = true
    await chat.sendQuery()
    expect(queryAgentStream).not.toHaveBeenCalled()
  })

  it('正常查询推送 user + assistant 消息并调用 queryAgentStream', async () => {
    const { useAgentChat } = await import('@/composables/useAgentChat')
    const { queryAgentStream } = await import('@/api/agent')
    const chat = useAgentChat()
    chat.inputText.value = '吴堡站水情'
    await chat.sendQuery()
    const messages = chat.messages.value
    expect(messages).toHaveLength(2)
    expect(messages[0].role).toBe('user')
    expect(messages[0].content).toBe('吴堡站水情')
    expect(messages[1].role).toBe('assistant')
    expect(messages[1].thinking).toBe(true)
    expect(chat.inputText.value).toBe('') // 清空输入
    expect(chat.loading.value).toBe(true)
    expect(queryAgentStream).toHaveBeenCalledOnce()
  })

  it('history 传给 queryAgentStream 不包含当前问题', async () => {
    const { useAgentChat } = await import('@/composables/useAgentChat')
    const { useChatSessions } = await import('@/composables/useChatSessions')
    const { queryAgentStream } = await import('@/api/agent')
    const chat = useAgentChat()
    // 必须先创建 active session，否则 activeMessages 返回临时空数组
    await useChatSessions().ensureActiveSession()
    // 先有一条历史
    chat.messages.value.push({ role: 'user', content: '上一轮问题' })
    chat.messages.value.push({ role: 'assistant', content: '上一轮回答' })
    chat.inputText.value = '本轮问题'
    await chat.sendQuery()
    const callArgs = (queryAgentStream as ReturnType<typeof vi.fn>).mock.calls[0][0]
    expect(callArgs.query).toBe('本轮问题')
    expect(callArgs.history).toEqual([
      { role: 'user', content: '上一轮问题' },
      { role: 'assistant', content: '上一轮回答' },
    ])
  })

  it('onError 回调把错误写入 assistant 消息并关闭 loading', async () => {
    const { useAgentChat } = await import('@/composables/useAgentChat')
    const { queryAgentStream } = await import('@/api/agent')
    ;(queryAgentStream as ReturnType<typeof vi.fn>).mockImplementationOnce(
      (_req: any, _onEvent: any, onError?: (e: Error) => void) => {
        // 同步触发错误
        setTimeout(() => onError?.(new Error('连接失败')), 0)
        return new AbortController()
      },
    )
    const chat = useAgentChat()
    chat.inputText.value = '问题'
    await chat.sendQuery()
    await new Promise((r) => setTimeout(r, 10))
    const last = chat.messages.value[chat.messages.value.length - 1]
    expect(last.thinking).toBe(false)
    expect(last.content).toContain('连接失败')
    expect(chat.loading.value).toBe(false)
  })
})

describe('stopQuery - 中断请求', () => {
  it('abort 当前请求并标记最后一条 assistant 消息为已中断', async () => {
    const { useAgentChat } = await import('@/composables/useAgentChat')
    const chat = useAgentChat()
    chat.inputText.value = '问题'
    await chat.sendQuery()
    expect(chat.loading.value).toBe(true)
    chat.stopQuery()
    expect(chat.loading.value).toBe(false)
    const last = chat.messages.value[chat.messages.value.length - 1]
    expect(last.thinking).toBe(false)
    expect(last.content).toBe('（已中断）')
  })

  it('无运行中请求时 stopQuery 不抛异常', async () => {
    const { useAgentChat } = await import('@/composables/useAgentChat')
    const chat = useAgentChat()
    expect(() => chat.stopQuery()).not.toThrow()
  })
})

describe('useSuggestion - 快捷提问', () => {
  it('设置 inputText 并发起查询', async () => {
    const { useAgentChat } = await import('@/composables/useAgentChat')
    const { queryAgentStream } = await import('@/api/agent')
    const chat = useAgentChat()
    chat.useSuggestion('吴堡站水情')
    // sendQuery 是 async，等待微任务
    await new Promise((r) => setTimeout(r, 0))
    expect(chat.inputText.value).toBe('') // sendQuery 会清空
    expect(queryAgentStream).toHaveBeenCalledOnce()
  })
})

describe('useAgentChat 单例', () => {
  it('多次调用返回同一状态实例', async () => {
    const { useAgentChat } = await import('@/composables/useAgentChat')
    const a = useAgentChat()
    const b = useAgentChat()
    // inputText 是共享的同一 ref
    a.inputText.value = 'shared'
    expect(b.inputText.value).toBe('shared')
    // loading 也是共享的
    a.loading.value = true
    expect(b.loading.value).toBe(true)
  })
})
