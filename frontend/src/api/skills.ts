import http from './instance'

/** Skill 数据结构 */
export interface Skill {
  id: string
  name: string
  description: string
  instructions: string
  tool_names: string[]
  enabled: boolean
}

/** 内置工具信息 */
export interface BuiltinTool {
  name: string
  description: string
}

/** 创建/更新 Skill 的请求体 */
export interface SkillPayload {
  name: string
  description: string
  instructions: string
  tool_names: string[]
  enabled?: boolean
}

/** 获取内置工具列表 */
export function getBuiltinTools() {
  return http.get<BuiltinTool[], BuiltinTool[]>('/skills/tools')
}

/** 列出所有 Skill */
export function listSkills(enabledOnly = false) {
  return http.get<Skill[], Skill[]>('/skills', { params: { enabled_only: enabledOnly } })
}

/** 获取单个 Skill 详情 */
export function getSkill(name: string) {
  return http.get<Skill, Skill>(`/skills/${name}`)
}

/** 创建 Skill */
export function createSkill(payload: SkillPayload) {
  return http.post<Skill, Skill>('/skills', payload)
}

/** 更新 Skill */
export function updateSkill(name: string, payload: Partial<SkillPayload>) {
  return http.put<Skill, Skill>(`/skills/${name}`, payload)
}

/** 启用/禁用 Skill */
export function toggleSkill(name: string) {
  return http.patch<Skill, Skill>(`/skills/${name}/toggle`)
}

/** 删除 Skill */
export function deleteSkill(name: string) {
  return http.delete<void, void>(`/skills/${name}`)
}

/** Skill 包导入结果 */
export interface ImportResult {
  skill: Skill | null
  action: string // created / overwritten / renamed
  original_name: string
  final_name: string
  warnings: string[]
}

/** 导入 Skill 包（支持 .zip / .skill / .md） */
export function importSkillPackage(file: File, conflictStrategy: string = 'cancel') {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('conflict_strategy', conflictStrategy)
  return http.post<ImportResult, ImportResult>('/skills/import', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
}
