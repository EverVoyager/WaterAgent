/**
 * 会话管理 API 客户端（P1-b）
 *
 * 后端 MySQL 持久化，替代 localStorage。
 * 硬失败：后端 MySQL 不可用时返回 500 错误。
 */
import instance from './instance'
import type { Message } from '@/composables/useChatSessions'

export interface ChatSession {
  id: string
  title: string
  createdAt: number
  updatedAt: number
  messages: Message[]
}

export interface SessionListResponse {
  sessions: ChatSession[]
  total: number
}

export interface CreateSessionRequest {
  id: string
  title?: string
}

export interface SyncSessionRequest {
  title: string
  messages: Message[]
}

/**
 * 列出所有会话（含消息，前端启动时全量加载）
 */
export async function listSessions(): Promise<ChatSession[]> {
  const data = await instance.get<unknown, SessionListResponse>('/sessions')
  return data.sessions || []
}

/**
 * 创建会话
 */
export async function createSession(req: CreateSessionRequest): Promise<ChatSession> {
  return instance.post<unknown, ChatSession>('/sessions', req)
}

/**
 * 获取单个会话（含消息）
 */
export async function getSession(id: string): Promise<ChatSession> {
  return instance.get<unknown, ChatSession>(`/sessions/${id}`)
}

/**
 * 全量同步会话（标题 + 消息）
 * 用于 persistActiveSession：流式完成后一次性同步整个会话状态
 */
export async function syncSession(id: string, req: SyncSessionRequest): Promise<ChatSession> {
  return instance.put<unknown, ChatSession>(`/sessions/${id}`, req)
}

/**
 * 更新会话标题
 */
export async function updateSessionTitle(id: string, title: string): Promise<void> {
  await instance.patch(`/sessions/${id}`, { title })
}

/**
 * 删除会话（级联删除消息）
 */
export async function deleteSession(id: string): Promise<void> {
  await instance.delete(`/sessions/${id}`)
}
