<template>
  <div class="health-view">
    <el-card shadow="hover">
      <template #header>
        <div class="card-header">
          <el-icon><Monitor /></el-icon>
          <span>后端服务健康检查</span>
        </div>
      </template>

      <el-skeleton :rows="4" :loading="loading" animated>
        <template #default>
          <el-descriptions :column="1" border>
            <el-descriptions-item label="服务状态">
              <el-tag
                :type="healthData?.status === 'ok' ? 'success' : 'danger'"
                effect="dark"
                size="default"
              >
                {{ healthData?.status === 'ok' ? '运行中' : '异常' }}
              </el-tag>
            </el-descriptions-item>
            <el-descriptions-item label="服务名称">
              {{ healthData?.service }}
            </el-descriptions-item>
            <el-descriptions-item label="版本号">
              {{ healthData?.version }}
            </el-descriptions-item>
            <el-descriptions-item label="环境">
              <el-tag :type="healthData?.env === 'production' ? 'warning' : 'info'" size="small">
                {{ healthData?.env }}
              </el-tag>
            </el-descriptions-item>
            <el-descriptions-item label="时间戳">
              {{ formatTime(healthData?.timestamp) }}
            </el-descriptions-item>
          </el-descriptions>

          <div class="actions">
            <el-button type="primary" :icon="Refresh" :loading="loading" @click="fetchHealth">
              刷新
            </el-button>
          </div>
        </template>
      </el-skeleton>

      <el-alert
        v-if="errorMsg"
        :title="errorMsg"
        type="error"
        show-icon
        :closable="false"
        class="error-alert"
      />
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { Monitor, Refresh } from '@element-plus/icons-vue'
import { getHealth, type HealthResponse } from '@/api/health'

const healthData = ref<HealthResponse | null>(null)
const loading = ref(false)
const errorMsg = ref('')

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

onMounted(() => {
  fetchHealth()
})
</script>

<style scoped>
.health-view {
  max-width: 720px;
  margin: 0 auto;
  padding: 24px;
  height: 100%;
  overflow-y: auto;
}

.card-header {
  display: flex;
  align-items: center;
  gap: 8px;
  font-weight: 600;
}

.actions {
  margin-top: 16px;
  display: flex;
  justify-content: flex-end;
}

.error-alert {
  margin-top: 16px;
}
</style>
