<template>
  <!-- 用户消息：右对齐胶囊 -->
  <div v-if="msg.role === 'user'" class="user-row">
    <div class="user-capsule">{{ msg.content }}</div>
  </div>

  <!-- AI 消息：无气泡流式排版 -->
  <div v-else class="ai-row">
    <ReasoningBlock
      v-if="msg.reasoningSteps && msg.reasoningSteps.length"
      :steps="msg.reasoningSteps"
      :thinking="!!msg.thinking"
      v-model:expanded="msg.reasoningExpanded"
    />
    <ToolChainBlock
      v-if="msg.toolEvents && msg.toolEvents.length"
      :events="msg.toolEvents"
      v-model:expanded="msg.chainExpanded"
    />
    <div class="answer">
      <span v-if="!msg.content && msg.thinking" class="typing"><i /><i /><i /></span>
      <template v-else>
        <span class="answer-text">
          <template v-for="(part, i) in parsedContent" :key="i">
            <span v-if="part.type === 'text'">{{ part.value }}</span>
            <sup
              v-else
              class="cite-ref"
              :class="{ active: hasCitation(part.refId) }"
              @click="onCiteClick(part.refId)"
            >[{{ part.refId }}]</sup>
          </template>
        </span>
        <CitationCard
          v-if="citations.length && !msg.thinking"
          ref="citationCardRef"
          :citations="citations"
          class="cite-inline"
        />
        <span v-if="msg.thinking" class="cursor">▏</span>
      </template>
    </div>
    <WarningCard v-if="showWarning && msg.response" :response="msg.response" />
  </div>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'
import ReasoningBlock from '@/components/ReasoningBlock.vue'
import ToolChainBlock from '@/components/ToolChainBlock.vue'
import WarningCard from '@/components/WarningCard.vue'
import CitationCard from '@/components/CitationCard.vue'
import type { Message } from '@/composables/useChatSessions'

const props = defineProps<{ msg: Message }>()

const showWarning = computed(
  () => props.msg.response?.intent === 'agent_task' && !!props.msg.response?.warning_level,
)

const citations = computed(() => props.msg.response?.citations || [])

const citationCardRef = ref<InstanceType<typeof CitationCard> | null>(null)

/** 解析 answer 中的 [编号] 标记，拆分为文本片段与引用标记 */
const parsedContent = computed(() => {
  const text = props.msg.content || ''
  if (!citations.value.length) return [{ type: 'text' as const, value: text }]
  const parts: Array<{ type: 'text'; value: string } | { type: 'cite'; refId: number }> = []
  const regex = /\[(\d+)\]/g
  let lastIndex = 0
  let match: RegExpExecArray | null
  while ((match = regex.exec(text)) !== null) {
    if (match.index > lastIndex) {
      parts.push({ type: 'text', value: text.slice(lastIndex, match.index) })
    }
    parts.push({ type: 'cite', refId: parseInt(match[1], 10) })
    lastIndex = regex.lastIndex
  }
  if (lastIndex < text.length) {
    parts.push({ type: 'text', value: text.slice(lastIndex) })
  }
  return parts
})

function hasCitation(refId: number): boolean {
  return citations.value.some((c) => c.ref_id === refId)
}

function onCiteClick(refId: number) {
  if (hasCitation(refId) && citationCardRef.value) {
    citationCardRef.value.highlight(refId)
  }
}
</script>

<style scoped>
.user-row {
  display: flex;
  justify-content: flex-end;
  margin-bottom: 20px;
  animation: fade-in 0.2s ease-out;
}

.user-capsule {
  max-width: 70%;
  padding: 9px 15px;
  background: var(--bg-subtle);
  border: 1px solid var(--border-default);
  border-radius: var(--radius-lg);
  font-size: 14px;
  line-height: 1.6;
  color: var(--text-primary);
  white-space: pre-wrap;
  word-break: break-word;
}

.ai-row {
  display: flex;
  flex-direction: column;
  gap: 10px;
  margin-bottom: 24px;
  animation: fade-in 0.2s ease-out;
}

.answer {
  font-size: 14px;
  line-height: 1.75;
  color: var(--text-primary);
}

.answer-text {
  white-space: pre-wrap;
  word-break: break-word;
}

.cite-inline {
  display: inline-flex;
  vertical-align: middle;
  margin-left: 6px;
}

.cite-ref {
  font-size: 11px;
  font-weight: 600;
  color: var(--text-tertiary);
  cursor: default;
  vertical-align: super;
  line-height: 0;
  padding: 0 1px;
}

.cite-ref.active {
  color: var(--accent);
  cursor: pointer;
  transition: color 0.15s;
}

.cite-ref.active:hover {
  color: var(--accent-hover);
}

.typing {
  display: inline-flex;
  gap: 4px;
  align-items: center;
  padding: 4px 0;
}

.typing i {
  width: 6px;
  height: 6px;
  background: var(--text-tertiary);
  border-radius: 50%;
  animation: bounce 1.4s infinite ease-in-out;
}

.typing i:nth-child(2) {
  animation-delay: 0.2s;
}
.typing i:nth-child(3) {
  animation-delay: 0.4s;
}

.cursor {
  display: inline-block;
  margin-left: 1px;
  color: var(--accent);
  animation: blink 1s infinite;
}

@keyframes bounce {
  0%,
  80%,
  100% {
    transform: scale(0.6);
    opacity: 0.5;
  }
  40% {
    transform: scale(1);
    opacity: 1;
  }
}

@keyframes blink {
  0%,
  50% {
    opacity: 1;
  }
  51%,
  100% {
    opacity: 0;
  }
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
