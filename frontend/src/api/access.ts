const STORAGE_KEY = 'wateragents-access-key'

export function accessHeaders(): Record<string, string> {
  const key = sessionStorage.getItem(STORAGE_KEY)
  return key ? { 'X-API-Key': key } : {}
}

export async function checkAccess(key?: string): Promise<void> {
  const baseURL = import.meta.env.VITE_API_BASE_URL || '/api'
  const response = await fetch(`${baseURL}/access`, {
    headers: key === undefined ? accessHeaders() : { 'X-API-Key': key },
  })
  if (!response.ok) {
    if (response.status === 401) sessionStorage.removeItem(STORAGE_KEY)
    throw new Error(response.status === 401 ? '请输入正确的访问钥匙' : '无法连接服务，请重试')
  }
  if (key !== undefined) sessionStorage.setItem(STORAGE_KEY, key)
}
