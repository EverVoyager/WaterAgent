import { beforeEach, describe, expect, it, vi } from 'vitest'

const STORAGE_KEY = 'water-agents:theme'

function stubMatchMedia(dark: boolean) {
  Object.defineProperty(window, 'matchMedia', {
    writable: true,
    configurable: true,
    value: vi.fn().mockImplementation((query: string) => ({
      matches: dark,
      media: query,
      onchange: null,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      addListener: vi.fn(),
      removeListener: vi.fn(),
      dispatchEvent: vi.fn(),
    })),
  })
}

describe('useTheme', () => {
  beforeEach(() => {
    localStorage.clear()
    vi.resetModules()
  })

  it('无存储且系统浅色时默认 light', async () => {
    stubMatchMedia(false)
    const { useTheme } = await import('@/composables/useTheme')
    const { theme } = useTheme()
    expect(theme.value).toBe('light')
    expect(document.documentElement.getAttribute('data-theme')).toBe('light')
  })

  it('无存储且系统深色时默认 dark', async () => {
    stubMatchMedia(true)
    const { useTheme } = await import('@/composables/useTheme')
    expect(useTheme().theme.value).toBe('dark')
    expect(document.documentElement.getAttribute('data-theme')).toBe('dark')
  })

  it('localStorage 存储值优先于系统偏好', async () => {
    stubMatchMedia(true)
    localStorage.setItem(STORAGE_KEY, 'light')
    const { useTheme } = await import('@/composables/useTheme')
    expect(useTheme().theme.value).toBe('light')
  })

  it('toggleTheme 切换并写入 html[data-theme] 与 localStorage', async () => {
    stubMatchMedia(false)
    const { useTheme } = await import('@/composables/useTheme')
    const { theme, toggleTheme } = useTheme()
    expect(theme.value).toBe('light')
    toggleTheme()
    await Promise.resolve()
    expect(theme.value).toBe('dark')
    expect(document.documentElement.getAttribute('data-theme')).toBe('dark')
    expect(localStorage.getItem(STORAGE_KEY)).toBe('dark')
  })
})
