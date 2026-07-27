<template>
  <div class="agent-view">
    <!-- 顶部标题栏 -->
    <header class="agent-header">
      <div class="title-wrap">
        <div class="title-icon">
          <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M12 2L2 7l10 5 10-5-10-5z" />
            <path d="M2 17l10 5 10-5" />
            <path d="M2 12l10 5 10-5" />
          </svg>
        </div>
        <div>
          <h1>黄河吕梁段防汛预警智能体</h1>
          <p class="subtitle">LangGraph Agent · RAG · Qdrant · GIS · Function Calling</p>
        </div>
      </div>
    </header>

    <!-- 对话流 -->
    <main class="chat-stream" ref="streamRef" @scroll="onStreamScroll">
      <!-- 欢迎屏 -->
      <div v-if="!messages.length && !loading" class="welcome">
        <div class="welcome-icon">
          <svg viewBox="0 0 24 24" width="48" height="48" fill="none" stroke="currentColor" stroke-width="1.5">
            <path d="M12 2L2 7l10 5 10-5-10-5z" />
            <path d="M2 17l10 5 10-5" />
            <path d="M2 12l10 5 10-5" />
          </svg>
        </div>
        <h2>水卫 Agent 已就绪</h2>
        <p class="welcome-text">可以问我水情、降雨、径流预测、预警等级、法规预案等问题</p>
        <div class="suggestions">
          <button
            v-for="s in suggestions"
            :key="s"
            class="suggestion-chip"
            @click="useSuggestion(s)"
          >{{ s }}</button>
        </div>
      </div>

      <!-- 消息列表 -->
      <template v-for="(msg, idx) in messages" :key="idx">
        <!-- 用户消息 -->
        <div v-if="msg.role === 'user'" class="row user-row">
          <div class="avatar user-avatar">我</div>
          <div class="bubble user-bubble">{{ msg.content }}</div>
        </div>

        <!-- AI 消息 -->
        <div v-else class="row ai-row">
          <div class="avatar ai-avatar">
            <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2">
              <circle cx="12" cy="12" r="3" />
              <path d="M12 1v6m0 10v6m11-11h-6M7 12H1m17.5-7.5l-4.24 4.24M9.74 14.26L5.5 18.5m13 0l-4.24-4.24M9.74 9.74L5.5 5.5" />
            </svg>
          </div>
          <div class="ai-content">
            <!-- 推理过程时间线 -->
            <div v-if="msg.reasoningSteps && msg.reasoningSteps.length" class="reasoning-timeline">
              <div class="timeline-header" @click="msg.reasoningExpanded = !msg.reasoningExpanded">
                <svg
                  class="chevron"
                  :class="{ expanded: msg.reasoningExpanded }"
                  viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"
                >
                  <polyline points="9 18 15 12 9 6" />
                </svg>
                <span class="timeline-title">推理过程</span>
                <span class="timeline-count">{{ msg.reasoningSteps.length }} 步</span>
                <span v-if="msg.thinking" class="thinking-dot">
                  <span class="dot"></span><span class="dot"></span><span class="dot"></span>
                </span>
              </div>
              <div v-show="msg.reasoningExpanded" class="timeline-list">
                <div
                  v-for="(rs, ridx) in msg.reasoningSteps"
                  :key="ridx"
                  class="timeline-item"
                  :class="`step-${rs.step}`"
                >
                  <div class="timeline-dot" :class="`step-${rs.step}`">
                    <span v-if="rs.phase === 'done'" class="dot-icon">✓</span>
                    <span v-else-if="rs.phase === 'decision'" class="dot-icon">◆</span>
                    <span v-else class="dot-spinner"></span>
                  </div>
                  <div class="timeline-body">
                    <div class="timeline-step-name">{{ stepName(rs.step) }}</div>
                    <div class="timeline-message">{{ rs.message }}</div>
                  </div>
                </div>
              </div>
            </div>

            <!-- 工具调用链（思考链） -->
            <div v-if="msg.toolEvents && msg.toolEvents.length" class="thinking-chain">
              <div class="chain-header" @click="msg.chainExpanded = !msg.chainExpanded">
                <svg
                  class="chevron"
                  :class="{ expanded: msg.chainExpanded }"
                  viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"
                >
                  <polyline points="9 18 15 12 9 6" />
                </svg>
                <span>工具调用链 · {{ msg.toolEvents.length }} 步</span>
              </div>
              <div v-show="msg.chainExpanded" class="chain-list">
                <div
                  v-for="(ev, eidx) in msg.toolEvents"
                  :key="eidx"
                  class="chain-item"
                  :class="{ 'has-error': ev.error }"
                >
                  <div class="chain-item-head">
                    <span class="tool-badge" :class="ev.status">{{ ev.tool }}</span>
                    <span class="round-tag">R{{ ev.round }}</span>
                  </div>
                  <div class="chain-item-args">
                    <code>{{ JSON.stringify(ev.arguments) }}</code>
                  </div>
                  <div v-if="ev.error" class="chain-item-err">错误：{{ ev.error }}</div>
                </div>
              </div>
            </div>

            <!-- 答案文本（流式渲染） -->
            <div class="bubble ai-bubble">
              <span v-if="!msg.content && msg.thinking" class="typing-indicator">
                <span class="dot"></span><span class="dot"></span><span class="dot"></span>
              </span>
              <span v-else class="answer-text">{{ msg.content }}</span>
              <span v-if="msg.thinking && msg.content" class="cursor-blink">▊</span>
            </div>

            <!-- 预警等级横幅（agent_task 类型且有等级） -->
            <div
              v-if="msg.response && msg.response.intent === 'agent_task' && msg.response.warning_level"
              class="level-banner"
              :class="`level-${msg.response.warning_level}`"
            >
              <div class="level-left">
                <div class="level-tag">{{ msg.response.warning_level }} 级</div>
                <div class="level-desc">{{ levelDesc(msg.response.warning_level) }}</div>
              </div>
              <div class="level-rounds">轮次 {{ msg.response.rounds }}</div>
            </div>

            <!-- 应急措施 -->
            <div v-if="msg.response && msg.response.actions && msg.response.actions.length" class="meta-block">
              <div class="meta-title">应急措施</div>
              <ol class="actions-list">
                <li v-for="(a, ai) in msg.response.actions" :key="ai">{{ a }}</li>
              </ol>
            </div>
          </div>
        </div>
      </template>
    </main>

    <!-- 底部输入区 -->
    <footer class="input-area">
      <div class="input-wrap">
        <textarea
          ref="inputRef"
          v-model="inputText"
          class="input-box"
          placeholder="输入防汛相关问题，Ctrl+Enter 发送..."
          :disabled="loading"
          rows="2"
          @keydown.enter.ctrl.prevent="sendQuery"
        ></textarea>
        <div class="input-actions">
          <button v-if="loading" class="btn-stop" @click="stopQuery">
            <svg viewBox="0 0 24 24" width="14" height="14" fill="currentColor">
              <rect x="6" y="6" width="12" height="12" rx="2" />
            </svg>
            停止
          </button>
          <button v-else class="btn-send" :disabled="!inputText.trim()" @click="sendQuery">
            <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2">
              <line x1="22" y1="2" x2="11" y2="13" />
              <polygon points="22 2 15 22 11 13 2 9 22 2" />
            </svg>
            发送
          </button>
          <button class="btn-clear" @click="clearChat" :disabled="loading">
            清空
          </button>
        </div>
      </div>
      <div class="hint">Ctrl+Enter 发送 · 闲聊直接回复 · 业务问题走 LangGraph</div>
    </footer>
  </div>
</template>

<script setup lang="ts">
import { ref, watch } from 'vue'
import { useAgentChat, SUGGESTIONS, levelDesc, stepName } from '@/composables/useAgentChat'

// M12：业务状态从 composable 获取，组件只负责 UI 渲染
const {
  messages,
  loading,
  inputText,
  userScrolledUp,
  sendQuery,
  stopQuery,
  clearChat,
  useSuggestion,
} = useAgentChat()

// UI 状态保留在组件（与业务无关）
const streamRef = ref<HTMLElement | null>(null)
const inputRef = ref<HTMLTextAreaElement | null>(null)
let scrollRafId: number | null = null

function onStreamScroll() {
  if (!streamRef.value) return
  const { scrollTop, scrollHeight, clientHeight } = streamRef.value
  const atBottom = scrollHeight - scrollTop - clientHeight < 60
  userScrolledUp.value = !atBottom
}

function scrollToBottom(force = false) {
  if (scrollRafId !== null) return
  scrollRafId = requestAnimationFrame(() => {
    scrollRafId = null
    if (!streamRef.value) return
    if (!force && userScrolledUp.value) return
    streamRef.value.scrollTop = streamRef.value.scrollHeight
  })
}

// 监听消息变化自动滚动
watch(
  () => messages.value.length,
  () => scrollToBottom(true),
)

// 监听最后一条消息内容变化（流式更新时滚动）
watch(
  () => messages.value[messages.value.length - 1]?.content,
  () => scrollToBottom(),
)

// 暴露给模板的常量
const suggestions = SUGGESTIONS
</script>

<style scoped>
/* ===== 全局布局 ===== */
.agent-view {
  display: flex;
  flex-direction: column;
  height: 100%;
  max-width: 880px;
  margin: 0 auto;
  padding: 0 20px;
  box-sizing: border-box;
  overflow: hidden;
}

/* ===== 顶部标题栏 ===== */
.agent-header {
  flex-shrink: 0;
  padding: 16px 0 14px;
  border-bottom: 1px solid #eef0f3;
  background: #fff;
  z-index: 1;
}

.title-wrap {
  display: flex;
  align-items: center;
  gap: 14px;
}

.title-icon {
  width: 42px;
  height: 42px;
  border-radius: 10px;
  background: linear-gradient(135deg, #4f8cff 0%, #2c6fdd 100%);
  color: #fff;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 4px 12px rgba(79, 140, 255, 0.25);
}

.agent-header h1 {
  margin: 0;
  font-size: 18px;
  font-weight: 600;
  color: #1a1f36;
  letter-spacing: -0.3px;
}

.subtitle {
  margin: 2px 0 0;
  font-size: 12px;
  color: #8a94a6;
  letter-spacing: 0.2px;
}

/* ===== 对话流容器（独立滚动模块） ===== */
.chat-stream {
  flex: 1;
  min-height: 0;          /* 关键：flex column 子项需 min-height:0 才能正确收缩+滚动 */
  overflow-y: auto;
  padding: 20px 0 16px;
  scroll-behavior: auto;   /* 流式场景下 auto 比 smooth 更流畅 */
}

/* ===== 欢迎屏 ===== */
.welcome {
  text-align: center;
  padding: 60px 20px 40px;
}

.welcome-icon {
  width: 80px;
  height: 80px;
  margin: 0 auto 20px;
  border-radius: 20px;
  background: linear-gradient(135deg, #f0f5ff 0%, #e6efff 100%);
  color: #4f8cff;
  display: flex;
  align-items: center;
  justify-content: center;
}

.welcome h2 {
  margin: 0 0 8px;
  font-size: 22px;
  font-weight: 600;
  color: #1a1f36;
}

.welcome-text {
  margin: 0 0 24px;
  font-size: 14px;
  color: #6b7280;
}

.suggestions {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  justify-content: center;
}

.suggestion-chip {
  padding: 8px 16px;
  background: #fff;
  border: 1px solid #e5e7eb;
  border-radius: 18px;
  font-size: 13px;
  color: #4b5563;
  cursor: pointer;
  transition: all 0.15s ease;
}

.suggestion-chip:hover {
  background: #f8faff;
  border-color: #4f8cff;
  color: #4f8cff;
}

/* ===== 消息行 ===== */
.row {
  display: flex;
  gap: 12px;
  margin-bottom: 24px;
  animation: fade-in 0.3s ease;
}

@keyframes fade-in {
  from { opacity: 0; transform: translateY(6px); }
  to { opacity: 1; transform: translateY(0); }
}

.avatar {
  flex-shrink: 0;
  width: 32px;
  height: 32px;
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 12px;
  font-weight: 600;
  color: #fff;
}

.user-avatar {
  background: linear-gradient(135deg, #6b7280 0%, #4b5563 100%);
}

.ai-avatar {
  background: linear-gradient(135deg, #4f8cff 0%, #2c6fdd 100%);
}

/* ===== 用户气泡 ===== */
.user-row {
  flex-direction: row-reverse;
}

.bubble {
  max-width: 75%;
  padding: 10px 14px;
  border-radius: 12px;
  font-size: 14px;
  line-height: 1.65;
  word-break: break-word;
  white-space: pre-wrap;
}

.user-bubble {
  background: #f3f4f6;
  color: #1a1f36;
  border-top-right-radius: 4px;
}

/* ===== AI 气泡 ===== */
.ai-content {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.ai-bubble {
  background: transparent;
  color: #1a1f36;
  padding: 0;
  border-radius: 0;
  max-width: 100%;
}

.answer-text {
  font-size: 14px;
  line-height: 1.75;
  white-space: pre-wrap;
  word-break: break-word;
}

/* 打字指示器 */
.typing-indicator,
.thinking-dot {
  display: inline-flex;
  gap: 4px;
  align-items: center;
  padding: 4px 0;
}

.typing-indicator .dot,
.thinking-dot .dot {
  width: 6px;
  height: 6px;
  background: #9aa3b2;
  border-radius: 50%;
  animation: bounce 1.4s infinite ease-in-out;
}

.typing-indicator .dot:nth-child(2),
.thinking-dot .dot:nth-child(2) { animation-delay: 0.2s; }
.typing-indicator .dot:nth-child(3),
.thinking-dot .dot:nth-child(3) { animation-delay: 0.4s; }

@keyframes bounce {
  0%, 80%, 100% { transform: scale(0.6); opacity: 0.5; }
  40% { transform: scale(1); opacity: 1; }
}

.cursor-blink {
  display: inline-block;
  margin-left: 1px;
  color: #4f8cff;
  animation: blink 1s infinite;
}

@keyframes blink {
  0%, 50% { opacity: 1; }
  51%, 100% { opacity: 0; }
}

/* ===== 推理过程时间线 ===== */
.reasoning-timeline {
  background: linear-gradient(180deg, #fbfcfe 0%, #f8fafc 100%);
  border: 1px solid #eef0f3;
  border-radius: 10px;
  overflow: hidden;
}

.timeline-header {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 14px;
  cursor: pointer;
  font-size: 12px;
  color: #6b7280;
  user-select: none;
  transition: background 0.15s;
}

.timeline-header:hover {
  background: #f3f4f6;
}

.timeline-title {
  font-weight: 600;
  color: #4b5563;
}

.timeline-count {
  font-size: 11px;
  color: #9aa3b2;
  background: #eef0f3;
  padding: 1px 8px;
  border-radius: 10px;
}

.chevron {
  transition: transform 0.2s;
}

.chevron.expanded {
  transform: rotate(90deg);
}

.timeline-list {
  padding: 4px 14px 12px;
  display: flex;
  flex-direction: column;
  gap: 0;
  position: relative;
}

.timeline-item {
  display: flex;
  gap: 10px;
  padding: 8px 0;
  position: relative;
}

/* 连接线 */
.timeline-item:not(:last-child)::before {
  content: '';
  position: absolute;
  left: 7px;
  top: 22px;
  bottom: -4px;
  width: 2px;
  background: #e5e7eb;
}

.timeline-dot {
  flex-shrink: 0;
  width: 16px;
  height: 16px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 10px;
  font-weight: 700;
  color: #fff;
  z-index: 1;
  margin-top: 2px;
}

/* 不同 step 的颜色编码 */
.timeline-dot.step-router { background: #8b5cf6; }
.timeline-dot.step-planner { background: #4f8cff; }
.timeline-dot.step-executor { background: #10b981; }
.timeline-dot.step-reflector { background: #f59e0b; }
.timeline-dot.step-synthesizer { background: #ef4444; }
.timeline-dot.step-direct_chat { background: #6b7280; }

.dot-icon {
  font-size: 9px;
  line-height: 1;
}

.dot-spinner {
  width: 8px;
  height: 8px;
  border: 2px solid rgba(255, 255, 255, 0.4);
  border-top-color: #fff;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.timeline-body {
  flex: 1;
  min-width: 0;
}

.timeline-step-name {
  font-size: 12px;
  font-weight: 600;
  color: #1a1f36;
  margin-bottom: 2px;
}

.timeline-message {
  font-size: 12px;
  color: #6b7280;
  line-height: 1.5;
}

/* ===== 工具调用链（思考链） ===== */
.thinking-chain {
  background: #fafbfc;
  border: 1px solid #eef0f3;
  border-radius: 8px;
  overflow: hidden;
}

.chain-header {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  cursor: pointer;
  font-size: 12px;
  color: #6b7280;
  user-select: none;
  transition: background 0.15s;
}

.chain-header:hover {
  background: #f3f4f6;
}

.chain-list {
  padding: 4px 12px 10px;
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.chain-item {
  background: #fff;
  border: 1px solid #e5e7eb;
  border-radius: 6px;
  padding: 8px 10px;
  font-size: 12px;
}

.chain-item.has-error {
  border-color: #fecaca;
  background: #fef5f5;
}

.chain-item-head {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 4px;
}

.tool-badge {
  font-family: 'SF Mono', Monaco, Consolas, monospace;
  font-size: 12px;
  font-weight: 600;
  color: #4f8cff;
  background: #eef4ff;
  padding: 2px 8px;
  border-radius: 4px;
}

.tool-badge.done { color: #10b981; background: #ecfdf5; }
.tool-badge.error { color: #ef4444; background: #fef2f2; }
.tool-badge.running { color: #f59e0b; background: #fffbeb; }

.round-tag {
  font-size: 11px;
  color: #9aa3b2;
  background: #f3f4f6;
  padding: 1px 6px;
  border-radius: 3px;
}

.chain-item-args code {
  font-family: 'SF Mono', Monaco, Consolas, monospace;
  font-size: 11px;
  color: #6b7280;
  word-break: break-all;
}

.chain-item-err {
  margin-top: 4px;
  color: #ef4444;
  font-size: 11px;
}

/* ===== 预警等级横幅 ===== */
.level-banner {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 14px 18px;
  border-radius: 10px;
  color: #fff;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
  animation: banner-in 0.4s ease;
}

@keyframes banner-in {
  from { opacity: 0; transform: translateY(6px); }
  to { opacity: 1; transform: translateY(0); }
}

.level-banner.level-I { background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%); }
.level-banner.level-II { background: linear-gradient(135deg, #ea580c 0%, #f97316 100%); }
.level-banner.level-III { background: linear-gradient(135deg, #ca8a04 0%, #eab308 100%); }
.level-banner.level-IV { background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%); }

.level-left {
  display: flex;
  align-items: center;
  gap: 14px;
}

.level-tag {
  font-size: 26px;
  font-weight: 700;
  letter-spacing: -0.5px;
}

.level-desc {
  font-size: 14px;
  font-weight: 500;
  opacity: 0.95;
}

.level-rounds {
  font-size: 12px;
  background: rgba(255, 255, 255, 0.2);
  padding: 4px 10px;
  border-radius: 12px;
  backdrop-filter: blur(4px);
}

/* ===== 研判依据 / 应急措施 ===== */
.meta-block {
  padding: 12px 14px;
  background: #fafbfc;
  border: 1px solid #eef0f3;
  border-radius: 8px;
  animation: fade-in 0.3s ease;
}

.meta-title {
  font-size: 12px;
  font-weight: 600;
  color: #6b7280;
  margin-bottom: 6px;
  letter-spacing: 0.3px;
}

.meta-text {
  font-size: 13px;
  line-height: 1.7;
  color: #1a1f36;
}

.actions-list {
  margin: 0;
  padding-left: 20px;
  font-size: 13px;
  line-height: 1.8;
  color: #1a1f36;
}

/* ===== 底部输入区 ===== */
.input-area {
  flex-shrink: 0;
  padding: 12px 0 18px;
  border-top: 1px solid #eef0f3;
  background: #fff;
  z-index: 1;
}

.input-wrap {
  background: #fff;
  border: 1px solid #e5e7eb;
  border-radius: 12px;
  padding: 10px 12px;
  transition: border-color 0.15s, box-shadow 0.15s;
}

.input-wrap:focus-within {
  border-color: #4f8cff;
  box-shadow: 0 0 0 3px rgba(79, 140, 255, 0.1);
}

.input-box {
  width: 100%;
  border: none;
  outline: none;
  resize: none;
  font-family: inherit;
  font-size: 14px;
  line-height: 1.5;
  color: #1a1f36;
  background: transparent;
}

.input-box::placeholder {
  color: #9aa3b2;
}

.input-actions {
  display: flex;
  justify-content: flex-end;
  gap: 8px;
  margin-top: 8px;
}

.btn-send,
.btn-stop,
.btn-clear {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 6px 14px;
  border-radius: 8px;
  font-size: 13px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.15s;
  border: none;
}

.btn-send {
  background: #4f8cff;
  color: #fff;
}

.btn-send:hover:not(:disabled) {
  background: #3b7aed;
}

.btn-send:disabled {
  background: #c7d3e3;
  cursor: not-allowed;
}

.btn-stop {
  background: #fee2e2;
  color: #dc2626;
}

.btn-stop:hover {
  background: #fecaca;
}

.btn-clear {
  background: #f3f4f6;
  color: #6b7280;
}

.btn-clear:hover:not(:disabled) {
  background: #e5e7eb;
}

.hint {
  text-align: center;
  font-size: 11px;
  color: #9aa3b2;
  margin-top: 8px;
}
</style>
