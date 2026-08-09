/**
 * Toast 通知容器：右上角堆叠，自动消失，可手动关闭。
 *
 * 由 useToast 模块级单例驱动，挂载在 App.vue 顶层。
 * 替代 Element Plus ElMessage，无外部依赖。
 */
<script setup lang="ts">
import { useToast } from '@/composables/useToast'

const { toasts, remove } = useToast()
</script>

<template>
  <Teleport to="body">
    <div class="toast-container" role="status" aria-live="polite">
      <TransitionGroup name="toast">
        <div
          v-for="t in toasts"
          :key="t.id"
          :class="['toast', `toast--${t.type}`]"
          @click="remove(t.id)"
        >
          <span class="toast__bar" />
          <span class="toast__msg">{{ t.message }}</span>
        </div>
      </TransitionGroup>
    </div>
  </Teleport>
</template>

<style scoped>
.toast-container {
  position: fixed;
  top: 16px;
  right: 16px;
  z-index: 9999;
  display: flex;
  flex-direction: column;
  gap: 8px;
  pointer-events: none;
}

.toast {
  pointer-events: auto;
  display: flex;
  align-items: center;
  gap: 10px;
  min-width: 240px;
  max-width: 420px;
  padding: 10px 14px;
  background: var(--bg-elevated);
  border: 1px solid var(--border-default);
  border-radius: var(--radius-md);
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
  cursor: pointer;
  font-size: 14px;
  color: var(--text-primary);
  overflow: hidden;
}

.toast__bar {
  flex-shrink: 0;
  width: 3px;
  align-self: stretch;
  border-radius: 2px;
}

.toast--error .toast__bar {
  background: var(--error);
}
.toast--success .toast__bar {
  background: var(--success);
}
.toast--info .toast__bar {
  background: var(--accent);
}

.toast__msg {
  flex: 1;
  line-height: 1.5;
  word-break: break-word;
}

/* 进出动画 */
.toast-enter-active,
.toast-leave-active {
  transition: all 0.25s ease;
}
.toast-enter-from {
  opacity: 0;
  transform: translateX(20px);
}
.toast-leave-to {
  opacity: 0;
  transform: translateX(20px);
}
</style>
