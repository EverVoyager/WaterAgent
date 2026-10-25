import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createApp, h, type App } from 'vue'
import * as api from '@/api/skills'
import { useSkills } from '@/composables/useSkills'
import SkillsView from '@/views/SkillsView.vue'

vi.mock('@/api/skills', () => ({
  listSkills: vi.fn(), getBuiltinTools: vi.fn(), createSkill: vi.fn(),
  updateSkill: vi.fn(), deleteSkill: vi.fn(), toggleSkill: vi.fn(),
  importSkillPackage: vi.fn(),
}))

const skill: api.Skill = {
  id: '1', name: 'flood', description: '汛情研判', instructions: '先核验水情',
  tool_names: ['get_weather'], enabled: true,
}
let app: App | undefined
let host: HTMLDivElement

function mountSkills() {
  let state!: ReturnType<typeof useSkills>
  app = createApp({ setup() { state = useSkills(); return () => h('div') } })
  app.mount(host)
  return state
}

beforeEach(() => {
  vi.resetAllMocks()
  vi.mocked(api.listSkills).mockResolvedValue([skill])
  vi.mocked(api.getBuiltinTools).mockResolvedValue([{ name: 'get_weather', description: '天气' }])
  host = document.createElement('div')
  document.body.append(host)
})

afterEach(() => {
  app?.unmount()
  host.remove()
  vi.useRealTimers()
})

describe('技能管理状态', () => {
  it('编辑时复制工具列表，保存只提交允许修改的字段', async () => {
    const state = mountSkills()
    await vi.waitFor(() => expect(state.skills.value).toHaveLength(1))
    state.openEdit(skill)
    state.form.value.tool_names.push('get_hydrology')
    expect(skill.tool_names).toEqual(['get_weather'])
    await state.onSave()
    expect(api.updateSkill).toHaveBeenCalledWith('flood', {
      description: skill.description, instructions: skill.instructions,
      tool_names: ['get_weather', 'get_hydrology'],
    })
    expect(state.showModal.value).toBe(false)
  })

  it('必填字段缺失时不发送创建请求', async () => {
    const state = mountSkills()
    state.openCreate()
    await state.onSave()
    expect(state.errorMsg.value).toContain('必填')
    expect(api.createSkill).not.toHaveBeenCalled()
  })

  it('导入冲突后可重命名重试，成功提示在卸载时清理计时器', async () => {
    const state = mountSkills()
    await vi.waitFor(() => expect(state.loading.value).toBe(false))
    vi.mocked(api.importSkillPackage)
      .mockRejectedValueOnce({ response: { data: { detail: "技能 'flood' 已存在" } } })
      .mockResolvedValueOnce({ skill, action: 'renamed', original_name: 'flood', final_name: 'flood_imported_1', warnings: [] })
    const file = new File(['skill'], 'SKILL.md')
    state.onFileSelected({ target: { files: [file], value: 'SKILL.md' } } as unknown as Event)
    await vi.waitFor(() => expect(state.showConflictModal.value).toBe(true))
    expect(state.conflictName.value).toBe('flood')
    vi.useFakeTimers()
    state.onConflictResolve('rename')
    await vi.waitFor(() => expect(state.importing.value).toBe(false))
    expect(api.importSkillPackage).toHaveBeenLastCalledWith(file, 'rename')
    expect(state.successMsg.value).toContain('flood_imported_1')
    expect(state.showConflictModal.value).toBe(false)
    expect(vi.getTimerCount()).toBe(1)
    app!.unmount()
    app = undefined
    expect(vi.getTimerCount()).toBe(0)
  })

  it('取消冲突后丢弃待导入文件', async () => {
    const state = mountSkills()
    vi.mocked(api.importSkillPackage).mockRejectedValueOnce({ response: { data: { detail: "技能 'flood' 已存在" } } })
    state.onFileSelected({ target: { files: [new File(['skill'], 'SKILL.md')], value: '' } } as unknown as Event)
    await vi.waitFor(() => expect(state.showConflictModal.value).toBe(true))
    state.onConflictCancel()
    state.onConflictResolve('overwrite')
    expect(api.importSkillPackage).toHaveBeenCalledTimes(1)
    expect(state.showConflictModal.value).toBe(false)
  })

  it('编辑表单的保存事件仍连接到创建接口', async () => {
    app = createApp(SkillsView)
    app.mount(host)
    await vi.waitFor(() => expect(host.querySelector('.skill-name')?.textContent).toBe('flood'))
    const create = Array.from(host.querySelectorAll('button')).find(b => b.textContent?.includes('新建技能'))!
    create.click()
    await vi.waitFor(() => expect(host.querySelector('.modal')).not.toBeNull())
    const inputs = host.querySelectorAll<HTMLInputElement | HTMLTextAreaElement>('.form-input')
    for (const [index, value] of ['new_skill', '新的触发条件', '新的行为指令'].entries()) {
      inputs[index].value = value
      inputs[index].dispatchEvent(new Event('input', { bubbles: true }))
    }
    host.querySelector<HTMLInputElement>('.tool-check input')!.click()
    host.querySelector<HTMLButtonElement>('.modal-foot .primary-btn')!.click()
    await vi.waitFor(() => expect(api.createSkill).toHaveBeenCalledWith({
      name: 'new_skill', description: '新的触发条件', instructions: '新的行为指令',
      tool_names: ['get_weather'], enabled: true,
    }))
    await vi.waitFor(() => expect(host.querySelector('.modal')).toBeNull())
  })
})
