<template>
  <aside class="sidebar" :class="{ collapsed }">
    <div class="side-head">
      <div v-if="!collapsed" class="brand">
        <svg
          class="brand-icon"
          viewBox="0 0 24 24"
          width="18"
          height="18"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
          stroke-linejoin="round"
        >
          <path d="M12 2.69l5.66 5.66a8 8 0 1 1-11.31 0z" />
        </svg>
        <span class="brand-name">水卫</span>
      </div>
      <button
        class="icon-btn"
        type="button"
        :title="collapsed ? '展开边栏' : '收起边栏'"
        @click="collapsed = !collapsed"
      >
        <svg
          v-if="collapsed"
          viewBox="0 0 24 24"
          width="16"
          height="16"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
          stroke-linejoin="round"
        >
          <polyline points="9 18 15 12 9 6" />
        </svg>
        <svg
          v-else
          viewBox="0 0 24 24"
          width="16"
          height="16"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
          stroke-linejoin="round"
        >
          <polyline points="15 18 9 12 15 6" />
        </svg>
      </button>
    </div>

    <button class="new-chat" type="button" title="新会话" @click="onNewChat">
      <svg
        viewBox="0 0 24 24"
        width="15"
        height="15"
        fill="none"
        stroke="currentColor"
        stroke-width="2"
        stroke-linecap="round"
      >
        <line x1="12" y1="5" x2="12" y2="19" />
        <line x1="5" y1="12" x2="19" y2="12" />
      </svg>
      <span v-if="!collapsed">新会话</span>
    </button>

    <nav v-if="!collapsed" class="session-list">
      <div v-for="g in groupedSessions" :key="g.label" class="session-group">
        <div class="group-label">{{ g.label }}</div>
        <button
          v-for="s in g.items"
          :key="s.id"
          class="session-item"
          :class="{ active: s.id === activeSessionId }"
          type="button"
          :title="s.title"
          @click="onSwitch(s.id)"
        >
          <span class="session-title">{{ s.title }}</span>
          <span class="session-del" title="删除会话" @click.stop="onDelete(s.id)">×</span>
        </button>
      </div>
      <div v-if="!groupedSessions.length" class="empty-hint">暂无历史会话</div>
    </nav>
    <div v-else class="side-spacer" />

    <div class="side-bottom">
      <router-link
        to="/agent"
        class="nav-item"
        :class="{ active: route.path === '/agent' }"
        title="智能研判"
      >
        <svg
          viewBox="0 0 24 24"
          width="16"
          height="16"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
          stroke-linejoin="round"
        >
          <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
        </svg>
        <span v-if="!collapsed">智能研判</span>
      </router-link>
      <router-link
        to="/health"
        class="nav-item"
        :class="{ active: route.path === '/health' }"
        title="服务健康"
      >
        <svg
          viewBox="0 0 24 24"
          width="16"
          height="16"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
          stroke-linejoin="round"
        >
          <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
        </svg>
        <span v-if="!collapsed">服务健康</span>
      </router-link>
      <router-link
        to="/skills"
        class="nav-item"
        :class="{ active: route.path === '/skills' }"
        title="技能管理"
      >
        <svg
          viewBox="0 0 24 24"
          width="16"
          height="16"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
          stroke-linejoin="round"
        >
          <path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z" />
        </svg>
        <span v-if="!collapsed">技能管理</span>
      </router-link>
      <button
        class="nav-item"
        type="button"
        :title="theme === 'dark' ? '切换到浅色模式' : '切换到深色模式'"
        @click="toggleTheme"
      >
        <svg
          v-if="theme === 'dark'"
          viewBox="0 0 24 24"
          width="16"
          height="16"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
          stroke-linejoin="round"
        >
          <circle cx="12" cy="12" r="5" />
          <line x1="12" y1="1" x2="12" y2="3" />
          <line x1="12" y1="21" x2="12" y2="23" />
          <line x1="4.22" y1="4.22" x2="5.64" y2="5.64" />
          <line x1="18.36" y1="18.36" x2="19.78" y2="19.78" />
          <line x1="1" y1="12" x2="3" y2="12" />
          <line x1="21" y1="12" x2="23" y2="12" />
          <line x1="4.22" y1="19.78" x2="5.64" y2="18.36" />
          <line x1="18.36" y1="5.64" x2="19.78" y2="4.22" />
        </svg>
        <svg
          v-else
          viewBox="0 0 24 24"
          width="16"
          height="16"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
          stroke-linejoin="round"
        >
          <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
        </svg>
        <span v-if="!collapsed">{{ theme === 'dark' ? '浅色模式' : '深色模式' }}</span>
      </button>
      <div v-if="!collapsed" class="version">v0.1.0</div>
    </div>
  </aside>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useChatSessions } from '@/composables/useChatSessions'
import { useTheme } from '@/composables/useTheme'

const route = useRoute()
const router = useRouter()
const { activeSessionId, groupedSessions, createSession, switchSession, deleteSession } =
  useChatSessions()
const { theme, toggleTheme } = useTheme()

const collapsed = ref(false)

function onNewChat() {
  createSession()
  if (route.path !== '/agent') router.push('/agent')
}

function onSwitch(id: string) {
  switchSession(id)
  if (route.path !== '/agent') router.push('/agent')
}

function onDelete(id: string) {
  deleteSession(id)
}
</script>

<style scoped>
.sidebar {
  width: var(--sidebar-width);
  flex-shrink: 0;
  height: 100vh;
  background: var(--bg-subtle);
  border-right: 1px solid var(--border-default);
  display: flex;
  flex-direction: column;
  padding: 12px 10px;
  gap: 8px;
  transition: width 0.2s ease;
  overflow: hidden;
}

.sidebar.collapsed {
  width: var(--sidebar-collapsed-width);
}

.side-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  min-height: 32px;
}

.collapsed .side-head {
  justify-content: center;
}

.brand {
  display: flex;
  align-items: center;
  gap: 8px;
  color: var(--accent);
  padding-left: 4px;
}

.brand-name {
  font-size: 15px;
  font-weight: 700;
  color: var(--text-primary);
  letter-spacing: 0.5px;
}

.icon-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 28px;
  border: none;
  border-radius: var(--radius-sm);
  background: transparent;
  color: var(--text-secondary);
  cursor: pointer;
  transition: background-color 0.1s;
}

.icon-btn:hover {
  background: var(--accent-soft);
  color: var(--text-primary);
}

.new-chat {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  width: 100%;
  padding: 8px 12px;
  border: 1px solid var(--border-default);
  border-radius: var(--radius-md);
  background: var(--bg-elevated);
  color: var(--text-primary);
  font-size: 13px;
  font-weight: 500;
  cursor: pointer;
  transition:
    border-color 0.15s,
    box-shadow 0.15s;
}

.new-chat:hover {
  border-color: var(--accent);
  box-shadow: 0 0 0 3px var(--accent-soft);
}

.session-list {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding: 2px;
}

.side-spacer {
  flex: 1;
}

.group-label {
  font-size: 11px;
  color: var(--text-tertiary);
  padding: 0 8px 4px;
}

.session-group {
  display: flex;
  flex-direction: column;
  gap: 1px;
}

.session-item {
  display: flex;
  align-items: center;
  gap: 6px;
  width: 100%;
  padding: 7px 8px;
  border: none;
  border-radius: var(--radius-sm);
  background: transparent;
  color: var(--text-secondary);
  font-size: 13px;
  text-align: left;
  cursor: pointer;
  transition: background-color 0.1s;
}

.session-item:hover {
  background: var(--accent-soft);
}

.session-item.active {
  background: var(--accent-soft);
  color: var(--text-primary);
  font-weight: 500;
}

.session-title {
  flex: 1;
  min-width: 0;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.session-del {
  flex-shrink: 0;
  width: 18px;
  height: 18px;
  display: none;
  align-items: center;
  justify-content: center;
  border-radius: 4px;
  color: var(--text-tertiary);
  font-size: 14px;
  line-height: 1;
}

.session-item:hover .session-del {
  display: inline-flex;
}

.session-del:hover {
  background: var(--bg-elevated);
  color: var(--error);
}

.empty-hint {
  padding: 12px 8px;
  font-size: 12px;
  color: var(--text-tertiary);
  text-align: center;
}

.side-bottom {
  display: flex;
  flex-direction: column;
  gap: 2px;
  border-top: 1px solid var(--border-default);
  padding-top: 8px;
}

.nav-item {
  display: flex;
  align-items: center;
  gap: 8px;
  width: 100%;
  padding: 7px 8px;
  border: none;
  border-radius: var(--radius-sm);
  background: transparent;
  color: var(--text-secondary);
  font-size: 13px;
  cursor: pointer;
  transition: background-color 0.1s;
}

.collapsed .nav-item {
  justify-content: center;
}

.nav-item:hover {
  background: var(--accent-soft);
  color: var(--text-primary);
}

.nav-item.active {
  color: var(--accent);
  font-weight: 500;
}

.version {
  padding: 6px 8px 0;
  font-size: 11px;
  color: var(--text-tertiary);
}
</style>
