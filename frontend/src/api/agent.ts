import http from './instance'

export interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
}

export interface ToolCallInfo {
  tool_name: string
  arguments: Record<string, any>
  result: Record<string, any>
  error: string
  round: number
}

export interface AgentQueryRequest {
  query: string
  system_prompt?: string
  history?: ChatMessage[]
}

export interface AgentQueryResponse {
  answer: string
  warning_level: 'I' | 'II' | 'III' | 'IV' | ''
  reasoning: string
  actions: string[]
  tool_calls: ToolCallInfo[]
  rounds: number
  intent: 'chitchat' | 'agent_task'
}

/** Agent 查询接口（非流式，保留兼容） */
export function queryAgent(req: AgentQueryRequest) {
  return http.post<AgentQueryResponse, AgentQueryResponse>('/agent/query', req)
}

// ====== SSE 流式接口 ======

/** 推理步骤 */
export interface ReasoningStep {
  step: 'router' | 'planner' | 'executor' | 'reflector' | 'synthesizer' | 'direct_chat'
  phase: 'start' | 'thinking' | 'decision' | 'done'
  message: string
  details?: Record<string, any>
}

/** 综合研判结构化结论（在 answer 流式输出前推送） */
export interface SynthMeta {
  warning_level: 'I' | 'II' | 'III' | 'IV' | ''
  reasoning: string
  actions: string[]
}

/** SSE 事件类型 */
export interface AgentStreamEvent {
  type:
    | 'reasoning_step'
    | 'intent'
    | 'tool_call'
    | 'tool_result'
    | 'synth_meta'
    | 'answer_delta'
    | 'done'
    | 'error'
  // reasoning_step
  step?: ReasoningStep['step']
  phase?: ReasoningStep['phase']
  message?: string
  details?: Record<string, any>
  // intent
  intent?: 'chitchat' | 'agent_task'
  // tool_call / tool_result
  tool?: string
  arguments?: Record<string, any>
  result?: Record<string, any>
  error?: string
  round?: number
  // synth_meta
  data?: AgentQueryResponse | SynthMeta
  // answer_delta
  content?: string
  // error
}

/**
 * Agent 流式查询（SSE）
 *
 * 使用 fetch + ReadableStream 解析 text/event-stream。
 * onEvent 每收到一个事件就被回调一次。
 *
 * P5 改进：
 * - 总超时 5 分钟（Agent 最长耗时预估 < 3 分钟，留余量）
 * - 静默超时 60 秒（无任何事件包括心跳时触发，提示连接异常）
 * - 后端每 15s 发 :keep-alive 心跳，前端解析时跳过 comment 行
 *
 * @returns AbortController 可用于中断请求
 */
export function queryAgentStream(
  req: AgentQueryRequest,
  onEvent: (event: AgentStreamEvent) => void,
  onError?: (err: Error) => void,
): AbortController {
  const controller = new AbortController()
  const baseURL = import.meta.env.VITE_API_BASE_URL || '/api'

  // 总超时：5 分钟（覆盖 planner+executor+synthesizer 全链路 + 余量）
  const TOTAL_TIMEOUT_MS = 5 * 60 * 1000
  // 静默超时：60 秒无任何数据（含心跳）则判定连接异常
  const SILENCE_TIMEOUT_MS = 60 * 1000

  let silenceTimer: ReturnType<typeof setTimeout> | null = null
  let totalTimer: ReturnType<typeof setTimeout> | null = null

  const clearTimers = () => {
    if (silenceTimer) clearTimeout(silenceTimer)
    if (totalTimer) clearTimeout(totalTimer)
    silenceTimer = null
    totalTimer = null
  }

  const resetSilenceTimer = () => {
    if (silenceTimer) clearTimeout(silenceTimer)
    silenceTimer = setTimeout(() => {
      console.warn('[SSE] 静默超时，中断连接')
      controller.abort()
      if (onError) onError(new Error('SSE 连接静默超时（60 秒无数据）'))
    }, SILENCE_TIMEOUT_MS)
  }

  totalTimer = setTimeout(() => {
    console.warn('[SSE] 总超时，中断连接')
    controller.abort()
    if (onError) onError(new Error('Agent 运行超时（5 分钟）'))
  }, TOTAL_TIMEOUT_MS)

  // 使用 fetch 而非 axios，因为 axios 不原生支持 stream
  fetch(`${baseURL}/agent/query/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
    signal: controller.signal,
  })
    .then(async (response) => {
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }
      const reader = response.body!.getReader()
      const decoder = new TextDecoder('utf-8')
      let buffer = ''

      // 收到首包，启动静默计时
      resetSilenceTimer()

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        // 任意数据到达，重置静默计时
        resetSilenceTimer()
        buffer += decoder.decode(value, { stream: true })

        // SSE 事件以 \n\n 分隔
        const events = buffer.split('\n\n')
        buffer = events.pop() || '' // 保留最后未完整的事件

        for (const rawEvent of events) {
          const line = rawEvent.trim()
          // P5：跳过后端心跳 comment（: 开头的行）
          if (!line.startsWith('data:')) continue
          const jsonStr = line.slice(5).trim()
          if (!jsonStr) continue
          try {
            const event = JSON.parse(jsonStr) as AgentStreamEvent
            onEvent(event)
            // done/error 事件后清理计时器
            if (event.type === 'done' || event.type === 'error') {
              clearTimers()
            }
          } catch (e) {
            console.warn('[SSE] 解析事件失败:', jsonStr, e)
          }
        }
      }
      clearTimers()
    })
    .catch((err) => {
      clearTimers()
      if (err.name === 'AbortError') return
      console.error('[SSE] 请求失败:', err)
      if (onError) onError(err)
    })

  return controller
}
