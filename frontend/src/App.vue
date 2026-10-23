<template>
  <div
    v-if="authorized"
    class="app-shell"
  >
    <AppSidebar />
    <div class="main-area">
      <router-view />
    </div>
    <ToastContainer />
  </div>
  <main
    v-else
    class="access-screen"
  >
    <form @submit.prevent="verifyAccess(key)">
      <h1>水卫 · WaterAgents</h1>
      <p
        v-if="checking"
        role="status"
      >
        正在连接服务…
      </p>
      <template v-else>
        <label for="access-key">访问钥匙</label>
        <input
          id="access-key"
          v-model="key"
          type="password"
          autocomplete="current-password"
        >
        <p
          v-if="error"
          role="alert"
        >
          {{ error }}
        </p>
        <button type="submit">
          进入
        </button>
      </template>
    </form>
  </main>
</template>

<script setup lang="ts">
import AppSidebar from '@/components/AppSidebar.vue'
import ToastContainer from '@/components/ToastContainer.vue'
import { useTheme } from '@/composables/useTheme'
import { onMounted, ref } from 'vue'
import { checkAccess } from '@/api/access'

const authorized = ref(false)
const checking = ref(true)
const key = ref('')
const error = ref('')

async function verifyAccess(value?: string) {
  checking.value = true
  error.value = ''
  try {
    await checkAccess(value)
    authorized.value = true
    key.value = ''
  } catch (e) {
    error.value = e instanceof Error ? e.message : '连接失败'
  } finally {
    checking.value = false
  }
}

onMounted(() => verifyAccess())

// 确保主题在应用启动时初始化
useTheme()
</script>

<style scoped>
.access-screen {
  display: grid;
  place-items: center;
  min-height: 100vh;
  background: var(--bg-base);
  color: var(--text-primary);
}

.access-screen form {
  display: grid;
  gap: 12px;
  width: min(360px, 90vw);
}

.access-screen input, .access-screen button {
  padding: 10px;
  font: inherit;
}

.app-shell {
  display: flex;
  height: 100vh;
  overflow: hidden;
  background: var(--bg-base);
  color: var(--text-primary);
}

.main-area {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
</style>
