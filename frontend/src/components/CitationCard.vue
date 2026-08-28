<template>
  <div
    v-if="citations.length"
    class="cite-icon-wrapper"
    @mouseenter="visible = true"
    @mouseleave="visible = false"
  >
    <button class="cite-icon-btn" :class="{ active: visible || highlightId !== null }">
      <svg viewBox="0 0 16 16" width="14" height="14" fill="currentColor">
        <path d="M6.354 5.5H4a3 3 0 0 0 0 6h3a3 3 0 0 0 2.5-1.342l-.874-.583A2 2 0 0 1 7 10H4a2 2 0 1 1 0-4h2.354l.647-.647L6.354 5.5Zm3.292-2H12a3 3 0 0 1 0 6H9a3 3 0 0 1-2.5-1.342l.874-.583A2 2 0 0 0 9 8h3a2 2 0 1 0 0-4H9.646l-.647.647.647.647-.6.6Z"/>
      </svg>
      <span class="cite-badge">{{ citations.length }}</span>
    </button>
    <transition name="popover">
      <div v-if="visible || highlightId !== null" class="cite-popover">
        <div class="popover-title">参考来源</div>
        <a
          v-for="cite in citations"
          :key="cite.ref_id"
          :href="cite.url || undefined"
          target="_blank"
          rel="noopener noreferrer"
          class="popover-item"
          :class="{ highlighted: highlightId === cite.ref_id }"
          :title="cite.url"
        >
          <span class="item-num">[{{ cite.ref_id }}]</span>
          <span class="item-title">{{ cite.title || cite.url }}</span>
          <svg class="item-external" viewBox="0 0 16 16" width="11" height="11" fill="currentColor">
            <path d="M8.5 3H4a1 1 0 0 0-1 1v8a1 1 0 0 0 1 1h8a1 1 0 0 0 1-1V7.5h-1.5V11.5h-7v-7H8.5V3Zm4-1H10V3.5h1.79L7.65 7.65l1.06 1.06 4.14-4.15V6.5H14.5V2.5h-2Z"/>
          </svg>
        </a>
      </div>
    </transition>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import type { Citation } from '@/api/agent'

defineProps<{
  citations: Citation[]
}>()

const visible = ref(false)
const highlightId = ref<number | null>(null)

/** 高亮指定编号的引用项（由 ChatMessage 点击内联标记时调用） */
function highlight(refId: number) {
  highlightId.value = refId
  // 高亮 2 秒后自动清除（若鼠标未 hover 则 popover 也会消失）
  setTimeout(() => {
    if (highlightId.value === refId) highlightId.value = null
  }, 2500)
}

defineExpose({ highlight })
</script>

<style scoped>
.cite-icon-wrapper {
  position: relative;
  display: inline-block;
}

.cite-icon-btn {
  display: inline-flex;
  align-items: center;
  gap: 3px;
  padding: 2px 6px;
  border: 1px solid var(--border-default);
  border-radius: var(--radius-sm);
  background: var(--bg-subtle);
  color: var(--text-tertiary);
  cursor: pointer;
  transition: color 0.15s, border-color 0.15s, background 0.15s;
  line-height: 1;
}

.cite-icon-btn:hover,
.cite-icon-btn.active {
  color: var(--accent);
  border-color: var(--accent);
  background: var(--accent-soft);
}

.cite-badge {
  font-size: 10px;
  font-weight: 700;
  min-width: 14px;
  text-align: center;
}

.cite-popover {
  position: absolute;
  bottom: calc(100% + 6px);
  left: 0;
  z-index: 50;
  min-width: 240px;
  max-width: 380px;
  max-height: 320px;
  overflow-y: auto;
  padding: 8px;
  background: var(--bg-elevated);
  border: 1px solid var(--border-default);
  border-radius: var(--radius-md);
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
}

.popover-title {
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.5px;
  text-transform: uppercase;
  color: var(--text-tertiary);
  padding: 2px 6px 6px;
}

.popover-item {
  display: flex;
  align-items: center;
  gap: 5px;
  padding: 5px 6px;
  border-radius: var(--radius-sm);
  text-decoration: none;
  color: var(--text-secondary);
  transition: background 0.12s;
}

.popover-item:hover {
  background: var(--bg-subtle);
  color: var(--accent);
}

.popover-item.highlighted {
  background: var(--accent-soft);
  color: var(--accent);
}

.item-num {
  font-size: 11px;
  font-weight: 700;
  color: var(--accent);
  flex-shrink: 0;
}

.item-title {
  font-size: 12px;
  line-height: 1.4;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  flex: 1;
}

.item-external {
  flex-shrink: 0;
  opacity: 0.5;
}

/* 过渡动画 */
.popover-enter-active,
.popover-leave-active {
  transition: opacity 0.15s, transform 0.15s;
}

.popover-enter-from,
.popover-leave-to {
  opacity: 0;
  transform: translateY(4px);
}
</style>
