<template>
  <div class="reasoning-block">
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
      <span v-if="thinking" class="spinner" />
      <span v-else class="status-ok">✓</span>
      <span class="header-text">{{ headerText }}</span>
      <span class="step-count">{{ steps.length }} 步</span>
    </button>
    <div v-show="expanded" class="timeline">
      <div v-for="(rs, i) in steps" :key="i" class="timeline-item">
        <div class="dot">
          <span v-if="rs.phase === 'done'" class="dot-icon">✓</span>
          <span v-else-if="rs.phase === 'decision'" class="dot-icon">◆</span>
          <span v-else class="dot-spinner" />
        </div>
        <div class="item-body">
          <div class="item-name">{{ stepName(rs.step) }}</div>
          <div class="item-message">{{ rs.message }}</div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { stepName } from '@/composables/useAgentChat'
import type { ReasoningStepEntry } from '@/composables/useChatSessions'

const props = withDefaults(
  defineProps<{
    steps: ReasoningStepEntry[]
    thinking: boolean
    expanded?: boolean
  }>(),
  { expanded: false },
)

const emit = defineEmits<{ 'update:expanded': [value: boolean] }>()

const headerText = computed(() => {
  if (props.thinking && props.steps.length) {
    return stepName(props.steps[props.steps.length - 1].step)
  }
  return `已完成 ${props.steps.length} 步推理`
})
</script>

<style scoped>
.reasoning-block {
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
  flex-shrink: 0;
  font-size: 11px;
  color: var(--success);
}

.header-text {
  font-weight: 500;
}

.step-count {
  margin-left: auto;
  font-size: 11px;
  color: var(--text-tertiary);
  background: var(--bg-elevated);
  border: 1px solid var(--border-default);
  padding: 1px 8px;
  border-radius: 10px;
}

.timeline {
  padding: 2px 14px 12px;
}

.timeline-item {
  display: flex;
  gap: 10px;
  padding: 7px 0;
  position: relative;
}

.timeline-item:not(:last-child)::before {
  content: '';
  position: absolute;
  left: 6px;
  top: 22px;
  bottom: -4px;
  width: 2px;
  background: var(--border-default);
}

.dot {
  flex-shrink: 0;
  width: 14px;
  height: 14px;
  margin-top: 2px;
  border-radius: 50%;
  background: var(--text-tertiary);
  display: flex;
  align-items: center;
  justify-content: center;
  color: #fff;
  z-index: 1;
}

.dot-icon {
  font-size: 8px;
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

.item-body {
  flex: 1;
  min-width: 0;
}

.item-name {
  font-size: 12px;
  font-weight: 600;
  color: var(--text-primary);
  margin-bottom: 1px;
}

.item-message {
  font-size: 12px;
  color: var(--text-secondary);
  line-height: 1.5;
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}
</style>
