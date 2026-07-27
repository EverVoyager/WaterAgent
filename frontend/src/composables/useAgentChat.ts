/**
 * M12：Agent 对话状态管理 composable
 *
 * 抽离自 AgentView.vue，分离业务状态与 UI 状态，便于测试和复用。
 * 参考 Vue 3 官方 composable 模式和 VueUse 设计。
 */
import { ref, type Ref } from 'vue'
import { ElMessage } from 'element-plus'
import {
  queryAgentStream,
  type AgentQueryResponse,
  type AgentStreamEvent,
  type ChatMessage,
  type ReasoningStep,
} from '@/api/agent'

// ====== 类型定义 ======

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

export interface UseAgentChatReturn {
  // 状态
  messages: Ref<Message[]>
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
  reflector: '信息评估',  // 兼容历史事件
  synthesizer: '综合研判',
  direct_chat: '对话生成',
}

export function levelDesc(level: string): string {
  return LEVEL_DESC[level] || level
}

export function stepName(step: string): string {
  return STEP_NAME[step] || step
}

// ====== composable 主函数 ======

export function useAgentChat(): UseAgentChatReturn {
  const inputText = ref('')
  const messages = ref<Message[]>([])
  const loading = ref(false)
  const userScrolledUp = ref(false)
  let abortController: AbortController | null = null

  async function sendQuery() {
    const q = inputText.value.trim()
    if (!q || loading.value) return

    const history: ChatMessage[] = messages.value.map((m) => ({
      role: m.role,
      content: m.content,
    }))

    messages.value.push({ role: 'user', content: q })
    messages.value.push({
      role: 'assistant',
      content: '',
      toolEvents: [],
      reasoningSteps: [],
      thinking: true,
      chainExpanded: true,
      reasoningExpanded: true,
      response: undefined,
    })
    const aiMsgIdx = messages.value.length - 1
    inputText.value = ''
    loading.value = true
    userScrolledUp.value = false

    abortController = queryAgentStream(
      { query: q, history },
      (event) => handleStreamEvent(event, aiMsgIdx),
      (err) => {
        const msg = messages.value[aiMsgIdx]
        if (msg) {
          msg.thinking = false
          msg.content = `调用失败：${err.message}`
        }
        loading.value = false
        ElMessage.error(err.message)
      },
    )
  }

  function handleStreamEvent(event: AgentStreamEvent, aiMsgIdx: number) {
    const aiMsg = messages.value[aiMsgIdx]
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

      case 'tool_result':
        {
          const events = aiMsg.toolEvents || []
          const last = [...events].reverse().find(
            (e) => e.tool === event.tool && e.status === 'running',
          )
          if (last) {
            last.result = event.result
            last.error = event.error
            last.status = event.error ? 'error' : 'done'
          }
        }
        break

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
        break

      case 'error':
        aiMsg.thinking = false
        aiMsg.content = `运行失败：${event.message}`
        loading.value = false
        ElMessage.error(event.message || 'Agent 运行失败')
        break
    }
  }

  function stopQuery() {
    if (abortController) {
      abortController.abort()
      abortController = null
    }
    const last = messages.value[messages.value.length - 1]
    if (last && last.role === 'assistant') {
      last.thinking = false
      if (!last.content) last.content = '（已中断）'
    }
    loading.value = false
  }

  function clearChat() {
    messages.value = []
    inputText.value = ''
    userScrolledUp.value = false
  }

  function useSuggestion(s: string) {
    inputText.value = s
    sendQuery()
  }

  return {
    messages,
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
