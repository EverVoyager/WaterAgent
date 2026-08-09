<template>
  <footer class="input-area">
    <div class="input-box">
      <textarea
        ref="textareaRef"
        class="input-el"
        :value="text"
        :disabled="loading"
        placeholder="输入防汛相关问题，Enter 发送…"
        rows="1"
        @input="onInput"
        @keydown.enter="onEnterKey"
      />
      <div class="input-bar">
        <span class="input-hint">Enter 发送 · Shift+Enter 换行</span>
        <button v-if="loading" class="send-btn stop" type="button" title="停止" @click="emit('stop')">
          <svg viewBox="0 0 24 24" width="13" height="13" fill="currentColor">
            <rect x="6" y="6" width="12" height="12" rx="2" />
          </svg>
        </button>
        <button
          v-else
          class="send-btn"
          type="button"
          :disabled="!text.trim()"
          title="发送"
          @click="emit('send')"
        >
          <svg
            viewBox="0 0 24 24"
            width="15"
            height="15"
            fill="none"
            stroke="currentColor"
            stroke-width="2"
            stroke-linecap="round"
            stroke-linejoin="round"
          >
            <line x1="12" y1="19" x2="12" y2="5" />
            <polyline points="5 12 12 5 19 12" />
          </svg>
        </button>
      </div>
    </div>
  </footer>
</template>

<script setup lang="ts">
import { nextTick, onMounted, ref, watch } from 'vue'

const props = defineProps<{ text: string; loading: boolean }>()
const emit = defineEmits<{ 'update:text': [value: string]; send: []; stop: [] }>()

const textareaRef = ref<HTMLTextAreaElement | null>(null)

function onInput(e: Event) {
  emit('update:text', (e.target as HTMLTextAreaElement).value)
}

function onEnterKey(e: KeyboardEvent) {
  if (e.isComposing) return // 中文输入法组词中不发送
  if (e.shiftKey) return // Shift+Enter 换行
  e.preventDefault()
  if (props.text.trim() && !props.loading) emit('send')
}

function resize() {
  const el = textareaRef.value
  if (!el) return
  el.style.height = 'auto'
  el.style.height = `${Math.min(el.scrollHeight, 200)}px`
}

watch(
  () => props.text,
  () => nextTick(resize),
)
onMounted(resize)
</script>

<style scoped>
.input-area {
  flex-shrink: 0;
  padding: 10px 24px 16px;
  background: var(--bg-base);
}

.input-box {
  max-width: 768px;
  margin: 0 auto;
  background: var(--bg-elevated);
  border: 1px solid var(--border-default);
  border-radius: var(--radius-lg);
  padding: 12px 14px 8px;
  transition:
    border-color 0.15s,
    box-shadow 0.15s;
}

.input-box:focus-within {
  border-color: var(--accent);
  box-shadow: 0 0 0 3px var(--accent-soft);
}

.input-el {
  display: block;
  width: 100%;
  border: none;
  outline: none;
  resize: none;
  font-family: inherit;
  font-size: 14px;
  line-height: 1.55;
  color: var(--text-primary);
  background: transparent;
  max-height: 200px;
}

.input-el::placeholder {
  color: var(--text-tertiary);
}

.input-el:disabled {
  opacity: 0.6;
}

.input-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-top: 6px;
}

.input-hint {
  font-size: 11px;
  color: var(--text-tertiary);
}

.send-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 30px;
  height: 30px;
  border: none;
  border-radius: 50%;
  background: var(--accent);
  color: #fff;
  cursor: pointer;
  transition:
    background-color 0.15s,
    transform 0.1s;
}

.send-btn:hover:not(:disabled) {
  background: var(--accent-hover);
}

.send-btn:active:not(:disabled) {
  transform: scale(0.94);
}

.send-btn:disabled {
  background: var(--border-strong);
  cursor: not-allowed;
}

.send-btn.stop {
  background: var(--error);
}
</style>
