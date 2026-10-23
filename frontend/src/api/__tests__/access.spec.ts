import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { accessHeaders, checkAccess } from '@/api/access'
import { queryAgentStream } from '@/api/agent'

beforeEach(() => sessionStorage.clear())
afterEach(() => {
  sessionStorage.clear()
  vi.unstubAllGlobals()
})

it('stores only a validated key and sends it with later requests', async () => {
  const fetch = vi.fn().mockResolvedValue(new Response('{"ok":true}'))
  vi.stubGlobal('fetch', fetch)
  expect(accessHeaders()).toEqual({})
  await checkAccess('shared-secret')
  expect(accessHeaders()).toEqual({ 'X-API-Key': 'shared-secret' })
  await checkAccess()
  expect(fetch.mock.calls[1][1].headers).toEqual(accessHeaders())
})

it('rejects an invalid key and clears the old credential', async () => {
  const fetch = vi.fn().mockResolvedValueOnce(new Response('{}'))
    .mockResolvedValueOnce(new Response('{}', { status: 401 }))
  vi.stubGlobal('fetch', fetch)
  await checkAccess('old-key')
  await expect(checkAccess('wrong')).rejects.toThrow('访问钥匙')
  expect(accessHeaders()).toEqual({})
})

it('sends the validated key with SSE requests', async () => {
  const fetch = vi.fn().mockResolvedValueOnce(new Response('{}'))
    .mockResolvedValueOnce(new Response('data: {"type":"done","data":{}}\n\n'))
  vi.stubGlobal('fetch', fetch)
  await checkAccess('shared-secret')
  await new Promise<void>((resolve, reject) => {
    queryAgentStream({ query: 'test' }, (event) => {
      if (event.type === 'done') resolve()
    }, reject)
  })
  expect(fetch.mock.calls[1][1].headers['X-API-Key']).toBe('shared-secret')
})

it('sends the validated key through axios as well', async () => {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('{}')))
  await checkAccess('shared-secret')
  const { default: instance } = await import('@/api/instance')
  await instance.get('/sessions', {
    adapter: async (config) => {
      expect(config.headers.get('X-API-Key')).toBe('shared-secret')
      return { data: [], status: 200, statusText: 'OK', headers: {}, config }
    },
  })
})
