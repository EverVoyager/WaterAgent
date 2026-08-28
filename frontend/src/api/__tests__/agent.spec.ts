import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import {
  queryAgentStream,
  type AgentQueryRequest,
  type AgentStreamEvent,
} from '@/api/agent'

// ====== mock fetch + 计时器 ======

let fetchMock: ReturnType<typeof vi.fn>
let abortSignal: AbortSignal | undefined

/** 构造一个 mock 的 ReadableStream，按顺序推送预定义 chunk。 */
function makeReadableStream(chunks: string[]) {
  const encoder = new TextEncoder()
  let i = 0
  return new ReadableStream({
    pull(controller) {
      if (i < chunks.length) {
        controller.enqueue(encoder.encode(chunks[i++]))
      } else {
        controller.close()
      }
    },
  })
}

/** 构造一个永不关闭的 ReadableStream，用于超时测试。 */
function makeHangingStream() {
  return new ReadableStream({
    pull() {
      // 不 enqueue 也不 close，永久挂起
      return new Promise<void>(() => {})
    },
  })
}

function okResponse(stream: ReadableStream) {
  return new Response(stream, { status: 200, statusText: 'OK' })
}

beforeEach(() => {
  vi.useFakeTimers()
  fetchMock = vi.fn()
  vi.stubGlobal('fetch', fetchMock)
  abortSignal = undefined
  fetchMock.mockImplementation((_url: string, init: RequestInit) => {
    abortSignal = init.signal as AbortSignal
    return Promise.resolve(okResponse(makeReadableStream([])))
  })
})

afterEach(() => {
  vi.useRealTimers()
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

const baseReq: AgentQueryRequest = { query: '吴堡站水情' }

describe('queryAgentStream - 基础调用', () => {
  it('发起 POST 请求到 /agent/query/stream，body 为 JSON', async () => {
    queryAgentStream(baseReq, () => {})
    // 等待 fetch 的微任务执行
    await vi.runOnlyPendingTimersAsync()
    expect(fetchMock).toHaveBeenCalledOnce()
    const [url, init] = fetchMock.mock.calls[0]
    expect(url).toMatch(/\/agent\/query\/stream$/)
    expect(init.method).toBe('POST')
    expect(JSON.parse(init.body as string)).toEqual(baseReq)
    // 默认 baseURL = /api
    expect(url.startsWith('/api/') || url.includes('/api/')).toBe(true)
  })

  it('使用 VITE_API_BASE_URL 作为前缀', async () => {
    vi.stubEnv('VITE_API_BASE_URL', 'https://example.com/api')
    queryAgentStream(baseReq, () => {})
    await vi.runOnlyPendingTimersAsync()
    const [url] = fetchMock.mock.calls[0]
    expect(url).toBe('https://example.com/api/agent/query/stream')
    vi.unstubAllEnvs()
  })

  it('返回 AbortController，可主动中断', async () => {
    const ctrl = queryAgentStream(baseReq, () => {})
    expect(ctrl).toBeInstanceOf(AbortController)
    ctrl.abort()
    // signal 已 aborted
    expect(abortSignal?.aborted).toBe(true)
  })
})

describe('queryAgentStream - SSE 事件解析', () => {
  it('解析多个 data: 事件并回调 onEvent', async () => {
    const events: AgentStreamEvent[] = []
    const sseChunks = [
      'data: {"type":"intent","intent":"agent_task"}\n\n',
      'data: {"type":"answer_delta","content":"hello"}\n\n',
      'data: {"type":"done","data":{"answer":"hello"}}\n\n',
    ]
    fetchMock.mockImplementationOnce(() =>
      Promise.resolve(okResponse(makeReadableStream(sseChunks))),
    )
    queryAgentStream(baseReq, (ev) => events.push(ev))
    await vi.runOnlyPendingTimersAsync()
    expect(events).toHaveLength(3)
    expect(events[0]).toMatchObject({ type: 'intent', intent: 'agent_task' })
    expect(events[1]).toMatchObject({ type: 'answer_delta', content: 'hello' })
    expect(events[2].type).toBe('done')
  })

  it('跳过 comment 行（:keep-alive 心跳）', async () => {
    const events: AgentStreamEvent[] = []
    const sse = [
      ':keep-alive\n\n',
      'data: {"type":"answer_delta","content":"a"}\n\n',
    ]
    fetchMock.mockImplementationOnce(() =>
      Promise.resolve(okResponse(makeReadableStream(sse))),
    )
    queryAgentStream(baseReq, (ev) => events.push(ev))
    await vi.runOnlyPendingTimersAsync()
    expect(events).toHaveLength(1)
    expect(events[0].content).toBe('a')
  })

  it('跨 chunk 的事件能正确拼接（事件被 \n\n 切到两个 chunk）', async () => {
    const events: AgentStreamEvent[] = []
    const sse = [
      'data: {"type":"answer_delta","conten', // 不完整
      't":"x"}\n\n',
    ]
    fetchMock.mockImplementationOnce(() =>
      Promise.resolve(okResponse(makeReadableStream(sse))),
    )
    queryAgentStream(baseReq, (ev) => events.push(ev))
    await vi.runOnlyPendingTimersAsync()
    expect(events).toHaveLength(1)
    expect(events[0].content).toBe('x')
  })

  it('解析失败的事件不影响后续事件，仅打印警告', async () => {
    const events: AgentStreamEvent[] = []
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {})
    const sse = [
      'data: not-json\n\n',
      'data: {"type":"answer_delta","content":"ok"}\n\n',
    ]
    fetchMock.mockImplementationOnce(() =>
      Promise.resolve(okResponse(makeReadableStream(sse))),
    )
    queryAgentStream(baseReq, (ev) => events.push(ev))
    await vi.runOnlyPendingTimersAsync()
    expect(events).toHaveLength(1)
    expect(events[0].content).toBe('ok')
    expect(warnSpy).toHaveBeenCalled()
  })

  it('空 data 行被跳过', async () => {
    const events: AgentStreamEvent[] = []
    const sse = ['data:\n\n', 'data: {"type":"done","data":{}}\n\n']
    fetchMock.mockImplementationOnce(() =>
      Promise.resolve(okResponse(makeReadableStream(sse))),
    )
    queryAgentStream(baseReq, (ev) => events.push(ev))
    await vi.runOnlyPendingTimersAsync()
    expect(events).toHaveLength(1)
    expect(events[0].type).toBe('done')
  })
})

describe('queryAgentStream - 错误处理', () => {
  it('HTTP 非 2xx 触发 onError', async () => {
    const errors: Error[] = []
    fetchMock.mockImplementationOnce(() =>
      Promise.resolve(new Response(null, { status: 500, statusText: 'Internal Error' })),
    )
    queryAgentStream(baseReq, () => {}, (err) => errors.push(err))
    await vi.runOnlyPendingTimersAsync()
    expect(errors).toHaveLength(1)
    expect(errors[0].message).toContain('500')
  })

  it('fetch 抛异常（网络错误）触发 onError', async () => {
    const errors: Error[] = []
    fetchMock.mockImplementationOnce(() =>
      Promise.reject(new Error('network down')),
    )
    queryAgentStream(baseReq, () => {}, (err) => errors.push(err))
    await vi.runOnlyPendingTimersAsync()
    expect(errors).toHaveLength(1)
    expect(errors[0].message).toBe('network down')
  })

  it('AbortError 不触发 onError（主动中断）', async () => {
    const errors: Error[] = []
    const abortErr = new Error('aborted')
    abortErr.name = 'AbortError'
    fetchMock.mockImplementationOnce(() => Promise.reject(abortErr))
    queryAgentStream(baseReq, () => {}, (err) => errors.push(err))
    await vi.runOnlyPendingTimersAsync()
    expect(errors).toHaveLength(0)
  })
})

describe('queryAgentStream - 超时机制', () => {
  it('总超时 5 分钟后触发 onError 并 abort（静默超时也会先触发）', async () => {
    const errors: Error[] = []
    fetchMock.mockImplementationOnce((_url: string, init: RequestInit) => {
      abortSignal = init.signal as AbortSignal
      return Promise.resolve(okResponse(makeHangingStream()))
    })
    queryAgentStream(baseReq, () => {}, (err) => errors.push(err))
    // 推进 5 分钟，静默超时（60s）和总超时（5min）都会触发
    await vi.advanceTimersByTimeAsync(5 * 60 * 1000)
    // 至少有一个错误包含 '5 分钟'
    expect(errors.some((e) => e.message.includes('5 分钟'))).toBe(true)
    expect(abortSignal?.aborted).toBe(true)
  })

  it('静默超时 60 秒后触发 onError 并 abort', async () => {
    const errors: Error[] = []
    fetchMock.mockImplementationOnce((_url: string, init: RequestInit) => {
      abortSignal = init.signal as AbortSignal
      return Promise.resolve(okResponse(makeHangingStream()))
    })
    queryAgentStream(baseReq, () => {}, (err) => errors.push(err))
    // 仅推进 60 秒（远小于 5 分钟），只有静默超时会触发
    await vi.advanceTimersByTimeAsync(60 * 1000)
    expect(errors).toHaveLength(1)
    expect(errors[0].message).toContain('60 秒')
    expect(abortSignal?.aborted).toBe(true)
  })

  it('收到数据重置静默计时器，60 秒静默后才触发', async () => {
    const errors: Error[] = []
    const events: AgentStreamEvent[] = []
    // 构造一个发送一块数据后挂起的流（不 close，等待静默超时）
    const encoder = new TextEncoder()
    const hangingAfterData = new ReadableStream({
      pull(controller) {
        controller.enqueue(encoder.encode('data: {"type":"intent","intent":"agent_task"}\n\n'))
        // 发送一次后挂起，不再 enqueue 也不 close
        return new Promise<void>(() => {})
      },
    })
    fetchMock.mockImplementationOnce((_url: string, init: RequestInit) => {
      abortSignal = init.signal as AbortSignal
      return Promise.resolve(okResponse(hangingAfterData))
    })
    queryAgentStream(baseReq, (ev) => events.push(ev), (err) => errors.push(err))
    // 推进 50s，首包数据到达，重置静默计时
    await vi.advanceTimersByTimeAsync(50 * 1000)
    expect(events).toHaveLength(1)
    expect(errors).toHaveLength(0)
    // 首包后 50s 内不触发；再推进 11s（首包后 61s）应触发静默超时
    await vi.advanceTimersByTimeAsync(11 * 1000)
    expect(errors).toHaveLength(1)
    expect(errors[0].message).toContain('60 秒')
  })

  it('done 事件触发后清理计时器，不再触发超时', async () => {
    const errors: Error[] = []
    const sse = ['data: {"type":"done","data":{}}\n\n']
    fetchMock.mockImplementationOnce(() =>
      Promise.resolve(okResponse(makeReadableStream(sse))),
    )
    queryAgentStream(baseReq, () => {}, (err) => errors.push(err))
    await vi.advanceTimersByTimeAsync(1000) // 让流读取完成
    // 即使推进 6 分钟（超过总超时），不应触发 onError
    await vi.advanceTimersByTimeAsync(6 * 60 * 1000)
    expect(errors).toHaveLength(0)
  })
})
