<template>
  <div class="warning-wrap">
    <div class="warning-card" :class="`lv-${response.warning_level}`">
      <span class="level-badge">{{ response.warning_level }} 级</span>
      <span class="level-desc">{{ levelDesc(response.warning_level) }}</span>
      <span class="rounds">轮次 {{ response.rounds }}</span>
    </div>
    <div v-if="response.actions && response.actions.length" class="actions-block">
      <div class="actions-title">应急措施</div>
      <ol class="actions-list">
        <li v-for="(a, i) in response.actions" :key="i">{{ a }}</li>
      </ol>
    </div>
  </div>
</template>

<script setup lang="ts">
import { levelDesc } from '@/composables/useAgentChat'
import type { AgentQueryResponse } from '@/api/agent'

defineProps<{ response: AgentQueryResponse }>()
</script>

<style scoped>
.warning-wrap {
  display: flex;
  flex-direction: column;
  gap: 10px;
  animation: fade-in 0.25s ease;
}

.warning-card {
  display: flex;
  align-items: center;
  gap: 14px;
  padding: 13px 16px;
  border-radius: var(--radius-md);
  border: 1px solid var(--border-default);
  border-left: 4px solid transparent;
}

.lv-I {
  border-left-color: var(--level-1);
  background: var(--level-1-soft);
}
.lv-I .level-badge {
  color: var(--level-1);
}
.lv-II {
  border-left-color: var(--level-2);
  background: var(--level-2-soft);
}
.lv-II .level-badge {
  color: var(--level-2);
}
.lv-III {
  border-left-color: var(--level-3);
  background: var(--level-3-soft);
}
.lv-III .level-badge {
  color: var(--level-3);
}
.lv-IV {
  border-left-color: var(--level-4);
  background: var(--level-4-soft);
}
.lv-IV .level-badge {
  color: var(--level-4);
}

.level-badge {
  font-size: 22px;
  font-weight: 700;
  letter-spacing: -0.5px;
}

.level-desc {
  font-size: 13px;
  font-weight: 500;
  color: var(--text-primary);
}

.rounds {
  margin-left: auto;
  font-size: 11px;
  color: var(--text-tertiary);
  background: var(--bg-elevated);
  border: 1px solid var(--border-default);
  padding: 2px 10px;
  border-radius: 10px;
}

.actions-block {
  border: 1px solid var(--border-default);
  border-radius: var(--radius-md);
  background: var(--bg-subtle);
  padding: 12px 16px;
}

.actions-title {
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.5px;
  text-transform: uppercase;
  color: var(--text-tertiary);
  margin-bottom: 6px;
}

.actions-list {
  padding-left: 18px;
  font-size: 13px;
  line-height: 1.8;
  color: var(--text-primary);
}

@keyframes fade-in {
  from {
    opacity: 0;
    transform: translateY(4px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}
</style>
