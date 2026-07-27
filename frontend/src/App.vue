<template>
  <el-config-provider :locale="zhCn">
    <div class="app-container">
      <el-container>
        <el-header class="app-header">
          <div class="logo">
            <el-icon :size="24"><Sunny /></el-icon>
            <span>黄河吕梁段防汛预警智能体</span>
          </div>
          <el-menu
            :default-active="activeRoute"
            mode="horizontal"
            background-color="transparent"
            text-color="#fff"
            active-text-color="#ffd04b"
            router
            class="nav-menu"
          >
            <el-menu-item index="/agent">智能研判</el-menu-item>
            <el-menu-item index="/health">服务健康</el-menu-item>
          </el-menu>
          <el-tag type="info" effect="plain" size="small">v0.1.0</el-tag>
        </el-header>
        <el-main class="app-main">
          <router-view />
        </el-main>
        <el-footer class="app-footer">
          <span>© 2026 Yellow River Lvliang Flood Warning Agent</span>
        </el-footer>
      </el-container>
    </div>
  </el-config-provider>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import zhCn from 'element-plus/es/locale/lang/zh-cn'

const route = useRoute()
const activeRoute = computed(() => route.path)
</script>

<style scoped>
.app-container {
  height: 100vh;
  overflow: hidden;
}

.app-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  background: linear-gradient(90deg, #2c3e50 0%, #4a6378 100%);
  color: #fff;
  padding: 0 24px;
  height: 60px;
}

.nav-menu {
  border-bottom: none;
  flex: 1;
  margin: 0 24px;
}

.nav-menu :deep(.el-menu-item) {
  border-bottom: none;
}

.logo {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 18px;
  font-weight: 600;
}

.app-main {
  padding: 0;
  overflow: hidden;
  max-width: 1200px;
  margin: 0 auto;
  width: 100%;
  height: calc(100vh - 60px - 48px);
}

.app-footer {
  text-align: center;
  color: #909399;
  font-size: 12px;
  height: 48px;
  line-height: 48px;
}
</style>
