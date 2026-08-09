/**
 * 主题管理 composable：light/dark 切换 + localStorage 持久化 + 系统偏好兜底
 *
 * 模块级单例，html[data-theme] 驱动 theme.css 变量切换。
 */
import { ref, watchEffect, type Ref } from 'vue'

export type Theme = 'light' | 'dark'

export interface UseThemeReturn {
  theme: Ref<Theme>
  toggleTheme: () => void
  setTheme: (t: Theme) => void
}

const STORAGE_KEY = 'water-agents:theme'

function resolveInitialTheme(): Theme {
  try {
    const saved = localStorage.getItem(STORAGE_KEY)
    if (saved === 'light' || saved === 'dark') return saved
  } catch {
    // 存储不可用时静默降级
  }
  if (typeof window !== 'undefined' && window.matchMedia?.('(prefers-color-scheme: dark)').matches) {
    return 'dark'
  }
  return 'light'
}

const theme = ref<Theme>(resolveInitialTheme())

watchEffect(() => {
  document.documentElement.setAttribute('data-theme', theme.value)
  try {
    localStorage.setItem(STORAGE_KEY, theme.value)
  } catch {
    // quota 满等异常静默降级
  }
})

export function useTheme(): UseThemeReturn {
  function toggleTheme() {
    theme.value = theme.value === 'dark' ? 'light' : 'dark'
  }

  function setTheme(t: Theme) {
    theme.value = t
  }

  return { theme, toggleTheme, setTheme }
}
