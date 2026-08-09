<template>
  <div class="health-view">
    <div class="health-card">
      <div class="card-head">
        <span class="status-dot" :class="dotClass" />
        <h2 class="card-title">后端服务健康检查</h2>
        <button class="refresh-btn" type="button" :disabled="loading" @click="fetchHealth">
          <svg
            viewBox="0 0 24 24"
            width="14"
            height="14"
            fill="none"
            stroke="currentColor"
            stroke-width="2"
            stroke-linecap="round"
            stroke-linejoin="round"
          >
            <polyline points="23 4 23 10 17 10" />
            <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10" />
          </svg>
          刷新
        </button>
      </div>

      <div v-if="loading && !healthData" class="skeleton-list">
        <div v-for="i in 4" :key="i" class="skeleton-row" />
      </div>

      <div v-else-if="healthData" class="info-rows">
        <div class="info-row">
          <span class="info-key">服务状态</span>
          <span class="info-value">
            <span class="status-tag" :class="healthData.status === 'ok' ? 'ok' : 'bad'">
              {{ healthData.status === 'ok' ? '运行中' : '异常' }}
            </span>
          </span>
        </div>
        <div class="info-row">
          <span class="info-key">服务名称</span>
          <span class="info-value">{{ healthData.service }}</span>
        </div>
        <div class="info-row">
          <span class="info-key">版本号</span>
          <span class="info-value mono">{{ healthData.version }}</span>
        </div>
        <div class="info-row">
          <span class="info-key">环境</span>
          <span class="info-value mono">{{ healthData.env }}</span>
        </div>
        <div class="info-row">
          <span class="info-key">时间戳</span>
          <span class="info-value">{{ formatTime(healthData.timestamp) }}</span>
        </div>
      </div>

      <div v-if="errorMsg" class="error-box">{{ errorMsg }}</div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { getHealth, type HealthResponse } from '@/api/health'

const healthData = ref<HealthResponse | null>(null)
const loading = ref(false)
const errorMsg = ref('')

const dotClass = computed(() => {
  if (errorMsg.value) return 'bad'
  if (healthData.value?.status === 'ok') return 'ok'
  if (healthData.value) return 'bad'
  return 'unknown'
})

async function fetchHealth() {
  loading.value = true
  errorMsg.value = ''
  try {
    healthData.value = await getHealth()
  } catch (e: any) {
    errorMsg.value = `后端连接失败：${e?.message || '未知错误'}`
    healthData.value = null
  } finally {
    loading.value = false
  }
}

function formatTime(ts?: string): string {
  if (!ts) return '-'
  try {
    return new Date(ts).toLocaleString('zh-CN', { timeZone: 'Asia/Shanghai' })
  } catch {
    return ts
  }
}

onMounted(fetchHealth)
</script>

<style scoped>
.health-view {
  height: 100%;
  overflow-y: auto;
  padding: 32px 24px;
  background: var(--bg-base);
}

.health-card {
  max-width: 640px;
  margin: 0 auto;
  background: var(--bg-elevated);
  border: 1px solid var(--border-default);
  border-radius: var(--radius-lg);
  padding: 20px 24px;
}

.card-head {
  display: flex;
  align-items: center;
  gap: 10px;
  padding-bottom: 16px;
  border-bottom: 1px solid var(--border-default);
}

.status-dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  flex-shrink: 0;
}

.status-dot.ok {
  background: var(--success);
  box-shadow: 0 0 0 3px var(--accent-soft);
  animation: pulse 2s infinite;
}

.status-dot.bad {
  background: var(--error);
}

.status-dot.unknown {
  background: var(--text-tertiary);
}

.card-title {
  flex: 1;
  font-size: 15px;
  font-weight: 600;
  color: var(--text-primary);
}

.refresh-btn {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 14px;
  border: 1px solid var(--border-default);
  border-radius: var(--radius-sm);
  background: var(--bg-elevated);
  color: var(--text-secondary);
  font-size: 12px;
  cursor: pointer;
  transition:
    border-color 0.15s,
    color 0.15s;
}

.refresh-btn:hover:not(:disabled) {
  border-color: var(--accent);
  color: var(--accent);
}

.refresh-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.skeleton-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
  padding: 16px 0 4px;
}

.skeleton-row {
  height: 18px;
  border-radius: 6px;
  background: linear-gradient(90deg, var(--bg-subtle) 25%, var(--border-default) 50%, var(--bg-subtle) 75%);
  background-size: 200% 100%;
  animation: shimmer 1.4s infinite;
}

.info-rows {
  display: flex;
  flex-direction: column;
  padding: 8px 0 4px;
}

.info-row {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 9px 0;
  border-bottom: 1px solid var(--border-default);
}

.info-row:last-child {
  border-bottom: none;
}

.info-key {
  width: 88px;
  flex-shrink: 0;
  font-size: 13px;
  color: var(--text-tertiary);
}

.info-value {
  font-size: 13px;
  color: var(--text-primary);
}

.mono {
  font-family: 'SF Mono', Monaco, Consolas, monospace;
}

.status-tag {
  display: inline-block;
  padding: 2px 10px;
  border-radius: 10px;
  font-size: 12px;
  font-weight: 500;
}

.status-tag.ok {
  color: var(--success);
  background: var(--level-4-soft);
}

.status-tag.bad {
  color: var(--error);
  background: var(--level-1-soft);
}

.error-box {
  margin-top: 14px;
  padding: 10px 14px;
  border: 1px solid var(--error);
  border-radius: var(--radius-sm);
  background: var(--level-1-soft);
  color: var(--error);
  font-size: 13px;
}

@keyframes pulse {
  0% {
    box-shadow: 0 0 0 0 rgba(22, 163, 74, 0.3);
  }
  70% {
    box-shadow: 0 0 0 6px rgba(22, 163, 74, 0);
  }
  100% {
    box-shadow: 0 0 0 0 rgba(22, 163, 74, 0);
  }
}

@keyframes shimmer {
  0% {
    background-position: 200% 0;
  }
  100% {
    background-position: -200% 0;
  }
}
</style>
