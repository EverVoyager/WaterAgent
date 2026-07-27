import http from './instance'

export interface HealthResponse {
  status: string
  service: string
  version: string
  env: string
  timestamp: string
}

/** 健康检查接口 */
export function getHealth() {
  return http.get<HealthResponse, HealthResponse>('/health')
}
