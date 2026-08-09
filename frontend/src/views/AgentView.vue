<template>
  <div class="agent-view">
    <main ref="streamRef" class="chat-stream" @scroll="onStreamScroll">
      <!-- 欢迎屏 -->
      <div v-if="!messages.length && !loading" class="welcome">
        <div class="welcome-icon">
          <svg
            viewBox="0 0 24 24"
            width="40"
            height="40"
            fill="none"
            stroke="currentColor"
            stroke-width="1.5"
            stroke-linecap="round"
            stroke-linejoin="round"
          >
            <path d="M12 2.69l5.66 5.66a8 8 0 1 1-11.31 0z" />
          </svg>
        </div>
        <h1 class="welcome-title">水卫 · 黄河吕梁段防汛预警智能体</h1>
        <p class="welcome-sub">实时水情 / 降雨 / 径流预测 / 预警等级 / 法规预案</p>
        <div class="suggest-grid">
          <button
            v-for="s in suggestions"
            :key="s"
            class="suggest-card"
            type="button"
            @click="useSuggestion(s)"
          >
            {{ s }}
          </button>
        </div>
      </div>

      <!-- 消息列表 -->
      <ChatMessage v-for="(msg, i) in messages" :key="i" :msg="msg" />
    </main>

    <ChatInput v-model:text="inputText" :loading="loading" @send="sendQuery" @stop="stopQuery" />
  </div>
</template>

<script setup lang="ts">
import { ref, watch } from 'vue'
import ChatMessage from '@/components/ChatMessage.vue'
import ChatInput from '@/components/ChatInput.vue'
import { useAgentChat, SUGGESTIONS } from '@/composables/useAgentChat'
import { useChatSessions } from '@/composables/useChatSessions'

const { messages, loading, inputText, userScrolledUp, sendQuery, stopQuery, useSuggestion } =
  useAgentChat()
const { activeSessionId } = useChatSessions()

const streamRef = ref<HTMLElement | null>(null)
let scrollRafId: number | null = null

function onStreamScroll() {
  if (!streamRef.value) return
  const { scrollTop, scrollHeight, clientHeight } = streamRef.value
  userScrolledUp.value = scrollHeight - scrollTop - clientHeight >= 60
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

// 新消息到达：强制滚到底
watch(
  () => messages.value.length,
  () => scrollToBottom(true),
)

// 流式更新：跟随滚动（用户上翻时除外）
watch(
  () => messages.value[messages.value.length - 1]?.content,
  () => scrollToBottom(),
)

// 切换会话：滚到底
watch(activeSessionId, () => scrollToBottom(true))

const suggestions = SUGGESTIONS
</script>

<style scoped>
.agent-view {
  display: flex;
  flex-direction: column;
  height: 100%;
  overflow: hidden;
  background: var(--bg-base);
}

.chat-stream {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  padding: 24px 24px 8px;
  scroll-behavior: auto;
}

.chat-stream > * {
  max-width: 768px;
  margin-left: auto;
  margin-right: auto;
}

/* 消息组件自带 margin-bottom，此处仅约束列宽 */
.chat-stream :deep(.user-row),
.chat-stream :deep(.ai-row) {
  max-width: 768px;
  margin-left: auto;
  margin-right: auto;
}

/* ===== 欢迎屏 ===== */
.welcome {
  text-align: center;
  padding: 72px 20px 40px;
}

.welcome-icon {
  width: 72px;
  height: 72px;
  margin: 0 auto 18px;
  border-radius: 18px;
  background: var(--accent-soft);
  color: var(--accent);
  display: flex;
  align-items: center;
  justify-content: center;
}

.welcome-title {
  margin: 0 0 8px;
  font-size: 22px;
  font-weight: 650;
  color: var(--text-primary);
  letter-spacing: -0.3px;
}

.welcome-sub {
  margin: 0 0 28px;
  font-size: 13px;
  color: var(--text-secondary);
}

.suggest-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
  max-width: 560px;
  margin: 0 auto;
}

.suggest-card {
  padding: 13px 16px;
  background: var(--bg-elevated);
  border: 1px solid var(--border-default);
  border-radius: var(--radius-md);
  font-size: 13px;
  color: var(--text-secondary);
  text-align: left;
  cursor: pointer;
  transition:
    transform 0.15s ease,
    border-color 0.15s,
    box-shadow 0.15s;
}

.suggest-card:hover {
  transform: translateY(-2px);
  border-color: var(--accent);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);
  color: var(--text-primary);
}
</style>
