import { afterEach, expect, it, vi } from 'vitest'
import { createApp } from 'vue'
import App from '@/App.vue'

const { sidebarMounted } = vi.hoisted(() => ({ sidebarMounted: vi.fn() }))
vi.mock('@/components/AppSidebar.vue', () => ({
  default: { setup: () => { sidebarMounted(); return () => null } },
}))
vi.mock('@/components/ToastContainer.vue', () => ({ default: { render: () => null } }))

afterEach(() => {
  sessionStorage.clear()
  vi.unstubAllGlobals()
  vi.clearAllMocks()
})

it('waits for a valid key before mounting the workspace', async () => {
  const fetch = vi.fn().mockResolvedValueOnce(new Response('{}', { status: 401 }))
    .mockResolvedValueOnce(new Response('{"ok":true}'))
  vi.stubGlobal('fetch', fetch)
  const root = document.createElement('div')
  document.body.appendChild(root)
  const app = createApp(App)
  app.component('RouterView', { render: () => null })
  app.mount(root)
  try {
    await vi.waitFor(() => expect(root.querySelector('input')).not.toBeNull())
    expect(sidebarMounted).not.toHaveBeenCalled()
    const input = root.querySelector('input')!
    input.value = 'shared-secret'
    input.dispatchEvent(new Event('input', { bubbles: true }))
    root.querySelector('form')!.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }))
    await vi.waitFor(() => expect(sidebarMounted).toHaveBeenCalledOnce())
    expect(fetch.mock.calls[1][1].headers['X-API-Key']).toBe('shared-secret')
    expect(root.querySelector('form')).toBeNull()
  } finally {
    app.unmount()
    root.remove()
  }
})
