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
