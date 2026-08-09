/**
 * 轻量 toast 通知 composable（模块级单例）。
 *
 * 替代 Element Plus ElMessage，避免引入 ~300KB 全量 CSS。
 * 右上角堆叠，自动消失（默认 3s），可手动关闭。
 */
import { ref, readonly } from 'vue'

export type ToastType = 'error' | 'success' | 'info'

export interface Toast {
  id: number
  type: ToastType
  message: string
}

const toasts = ref<Toast[]>([])
let nextId = 0

function remove(id: number): void {
  const idx = toasts.value.findIndex((t) => t.id === id)
  if (idx !== -1) toasts.value.splice(idx, 1)
}

function show(message: string, type: ToastType = 'info', duration = 3000): void {
  const id = nextId++
  toasts.value.push({ id, type, message })
  if (duration > 0) {
    setTimeout(() => remove(id), duration)
  }
}

export function useToast() {
  return {
    toasts: readonly(toasts),
    error: (msg: string) => show(msg, 'error'),
    success: (msg: string) => show(msg, 'success'),
    info: (msg: string) => show(msg, 'info'),
    remove,
  }
}
