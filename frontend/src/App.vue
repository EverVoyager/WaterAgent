<template>
  <div class="app-shell">
    <AppSidebar />
    <div class="main-area">
      <router-view />
    </div>
    <ToastContainer />
  </div>
</template>

<script setup lang="ts">
import AppSidebar from '@/components/AppSidebar.vue'
import ToastContainer from '@/components/ToastContainer.vue'
import { useTheme } from '@/composables/useTheme'

// 确保主题在应用启动时初始化
useTheme()
</script>

<style scoped>
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
