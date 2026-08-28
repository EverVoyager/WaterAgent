/**
 * Agent 对话状态管理 composable
 *
 * 消息数据接入 useChatSessions（后端 MySQL 持久化），
 * 本 composable 负责 SSE 流式交互与消息状态推进。
 * 模块级单例：多组件（AgentView / ChatInput / ...）共享同一状态。
 */
import { ref, type ComputedRef, type Ref } from 'vue'
import { useToast } from '@/composables/useToast'
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
const { sessions, activeSessionId, activeMessages, ensureActiveSession, persistActiveSession, persistSessionById } =
  useChatSessions()
const toast = useToast()

const inputText = ref('')
const loading = ref(false)
const userScrolledUp = ref(false)
let abortController: AbortController | null = null

/**
 * 流式目标：事件应写入的会话与消息索引。
 * 流式期间用户可能切换会话，activeMessages 会指向新会话，
 * 因此事件写入必须以 streamTarget 定位（per-conversation 隔离），
 * 否则会污染其它会话或丢失回答。
 */
let streamTarget: { sessionId: string; aiMsgIdx: number } | null = null

/** 从 streamTarget 定位消息（经 reactive sessions 取 proxy，保证流式更新触发视图） */
function findStreamMessage(target: { sessionId: string; aiMsgIdx: number } | null): Message | undefined {
  if (!target) return undefined
  const session = sessions.value.find((s) => s.id === target.sessionId)
  return session?.messages[target.aiMsgIdx]
}

/** 流结束后持久化目标会话（而非当前 active 会话），并清理 streamTarget */
function finishStream(target: { sessionId: string; aiMsgIdx: number } | null) {
  abortController = null
  streamTarget = null
  const persist = target ? persistSessionById(target.sessionId) : persistActiveSession()
  persist.catch((e) => toast.error(e?.message || '会话同步失败'))
}

async function sendQuery() {
  const q = inputText.value.trim()
  if (!q || loading.value) return

  try {
    await ensureActiveSession()
  } catch (e: any) {
    toast.error(e?.response?.data?.detail || e?.message || '创建会话失败')
    return
  }

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
  // 记录流式目标：后续事件始终写入该会话的消息，与用户当前浏览的会话解耦
  streamTarget = { sessionId: activeSessionId.value, aiMsgIdx }
  inputText.value = ''
  loading.value = true
  userScrolledUp.value = false

  abortController = queryAgentStream(
    { query: q, history },
    (event) => handleStreamEvent(event, aiMsgIdx),
    (err) => {
      const target = streamTarget
      const msg = findStreamMessage(target) ?? activeMessages.value[aiMsgIdx]
      if (msg) {
        msg.thinking = false
        // 已有部分流式内容时标记中断并保留内容，否则显示调用失败
        msg.content = msg.content
          ? `${msg.content}\n\n（连接中断）`
          : `调用失败：${err.message}`
      }
      loading.value = false
      finishStream(target)
      toast.error(err.message)
    },
  )
}

function handleStreamEvent(event: AgentStreamEvent, aiMsgIdx: number) {
  // 流式期间以 streamTarget 为准（会话可能已切换）；无 streamTarget 时
  // 退回当前会话索引（兼容直接调用）
  const aiMsg = findStreamMessage(streamTarget) ?? activeMessages.value[aiMsgIdx]
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
            citations: meta.citations || [],
            rounds: 0,
            intent: 'agent_task',
          } as AgentQueryResponse
        } else {
          aiMsg.response.warning_level = meta.warning_level || ''
          aiMsg.response.reasoning = meta.reasoning || ''
          aiMsg.response.actions = meta.actions || []
          aiMsg.response.citations = meta.citations || []
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
      finishStream(streamTarget)
      break

    case 'error':
      aiMsg.thinking = false
      aiMsg.content = `运行失败：${event.message}`
      loading.value = false
      finishStream(streamTarget)
      toast.error(event.message || 'Agent 运行失败')
      break
  }
}

function stopQuery() {
  if (abortController) {
    abortController.abort()
  }
  const target = streamTarget
  const last = findStreamMessage(target) ?? activeMessages.value[activeMessages.value.length - 1]
  if (last && last.role === 'assistant') {
    last.thinking = false
    if (!last.content) last.content = '（已中断）'
  }
  loading.value = false
  finishStream(target)
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
    useSuggestion,
    handleStreamEvent,
  }
}
