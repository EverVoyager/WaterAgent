<template>
  <div class="toolchain">
    <button class="block-header" type="button" @click="emit('update:expanded', !expanded)">
      <svg
        class="chevron"
        :class="{ expanded }"
        viewBox="0 0 24 24"
        width="14"
        height="14"
        fill="none"
        stroke="currentColor"
        stroke-width="2"
      >
        <polyline points="9 18 15 12 9 6" />
      </svg>
      <span v-if="runningTool" class="spinner" />
      <span v-else class="status-ok">✓</span>
      <span class="header-text">{{ headerText }}</span>
    </button>
    <div v-show="expanded" class="tool-list">
      <div v-for="(ev, i) in events" :key="i" class="tool-item" :class="{ 'has-error': ev.error }">
        <div class="tool-line">
          <span class="tool-status">
            <span v-if="ev.status === 'running'" class="spinner" />
            <span v-else-if="ev.status === 'error'" class="status-err">✗</span>
            <span v-else class="status-ok">✓</span>
          </span>
          <code class="tool-call">{{ ev.tool }}({{ formatArgs(ev.arguments) }})</code>
          <span class="round-tag">R{{ ev.round }}</span>
        </div>
        <div v-if="ev.error" class="tool-err">错误：{{ ev.error }}</div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import type { ToolEvent } from '@/composables/useChatSessions'

const props = withDefaults(
  defineProps<{
    events: ToolEvent[]
    expanded?: boolean
  }>(),
  { expanded: false },
)

const emit = defineEmits<{ 'update:expanded': [value: boolean] }>()

const runningTool = computed(() => props.events.find((e) => e.status === 'running'))
const headerText = computed(() =>
  runningTool.value ? `正在调用 ${runningTool.value.tool}…` : `调用 ${props.events.length} 个工具`,
)

function formatArgs(args: Record<string, any>): string {
  return Object.entries(args)
    .map(([k, v]) => `${k}=${JSON.stringify(v)}`)
    .join(', ')
}
</script>

<style scoped>
.toolchain {
  border: 1px solid var(--border-default);
  border-radius: var(--radius-md);
  background: var(--bg-subtle);
  overflow: hidden;
}

.block-header {
  display: flex;
  align-items: center;
  gap: 8px;
  width: 100%;
  padding: 9px 14px;
  border: none;
  background: transparent;
  cursor: pointer;
  font-size: 12px;
  color: var(--text-secondary);
  user-select: none;
  text-align: left;
  transition: background-color 0.1s;
}

.block-header:hover {
  background: var(--accent-soft);
}

.chevron {
  flex-shrink: 0;
  color: var(--text-tertiary);
  transition: transform 0.2s;
}

.chevron.expanded {
  transform: rotate(90deg);
}

.spinner {
  flex-shrink: 0;
  width: 12px;
  height: 12px;
  border: 2px solid var(--border-strong);
  border-top-color: var(--accent);
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

.status-ok {
  font-size: 11px;
  color: var(--success);
}

.status-err {
  font-size: 11px;
  color: var(--error);
}

.header-text {
  font-weight: 500;
}

.tool-list {
  padding: 2px 14px 10px;
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.tool-item {
  background: var(--bg-elevated);
  border: 1px solid var(--border-default);
  border-radius: var(--radius-sm);
  padding: 7px 10px;
}

.tool-item.has-error {
  border-color: var(--error);
}

.tool-line {
  display: flex;
  align-items: center;
  gap: 8px;
  min-width: 0;
}

.tool-status {
  flex-shrink: 0;
  display: inline-flex;
  align-items: center;
}

.tool-call {
  flex: 1;
  min-width: 0;
  font-family: 'SF Mono', Monaco, Consolas, monospace;
  font-size: 12px;
  color: var(--text-secondary);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.round-tag {
  flex-shrink: 0;
  font-size: 11px;
  color: var(--text-tertiary);
  background: var(--bg-subtle);
  border: 1px solid var(--border-default);
  padding: 0 6px;
  border-radius: 4px;
}

.tool-err {
  margin-top: 4px;
  font-size: 11px;
  color: var(--error);
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}
</style>
