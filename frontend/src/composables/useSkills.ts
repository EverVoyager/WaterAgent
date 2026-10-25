import { onMounted, onUnmounted, ref } from 'vue'
import {
  listSkills,
  createSkill,
  updateSkill,
  deleteSkill,
  toggleSkill,
  getBuiltinTools,
  importSkillPackage,
  type Skill,
  type BuiltinTool,
  type SkillPayload,
} from '@/api/skills'

export function useSkills() {
  const skills = ref<Skill[]>([])
  const builtinTools = ref<BuiltinTool[]>([])
  const loading = ref(false)
  const saving = ref(false)
  const importing = ref(false)
  const errorMsg = ref('')
  const successMsg = ref('')
  const showModal = ref(false)
  const editing = ref(false)

  // 导入相关
  const importFileInput = ref<HTMLInputElement | null>(null)
  const pendingFile = ref<File | null>(null)
  const showConflictModal = ref(false)
  const conflictName = ref('')
  let successTimer: ReturnType<typeof setTimeout> | null = null

  const emptyForm = (): SkillPayload => ({
    name: '',
    description: '',
    instructions: '',
    tool_names: [],
    enabled: true,
  })
  const form = ref<SkillPayload>(emptyForm())

  async function fetchSkills() {
    loading.value = true
    errorMsg.value = ''
    try {
      const [list, tools] = await Promise.all([listSkills(), getBuiltinTools()])
      skills.value = list
      builtinTools.value = tools
    } catch (e: any) {
      errorMsg.value = `加载失败：${e?.message || '未知错误'}`
    } finally {
      loading.value = false
    }
  }

  function openCreate() {
    editing.value = false
    form.value = emptyForm()
    showModal.value = true
  }

  function openEdit(s: Skill) {
    editing.value = true
    form.value = {
      name: s.name,
      description: s.description,
      instructions: s.instructions,
      tool_names: [...s.tool_names],
      enabled: s.enabled,
    }
    showModal.value = true
  }

  function closeModal() {
    showModal.value = false
  }

  async function onSave() {
    if (!form.value.name || !form.value.description || !form.value.instructions) {
      errorMsg.value = '请填写所有必填字段'
      return
    }
    saving.value = true
    errorMsg.value = ''
    try {
      if (editing.value) {
        await updateSkill(form.value.name, {
          description: form.value.description,
          instructions: form.value.instructions,
          tool_names: form.value.tool_names,
        })
      } else {
        await createSkill(form.value)
      }
      showModal.value = false
      await fetchSkills()
    } catch (e: any) {
      errorMsg.value = `保存失败：${e?.response?.data?.detail || e?.message || '未知错误'}`
    } finally {
      saving.value = false
    }
  }

  async function onToggle(s: Skill) {
    try {
      await toggleSkill(s.name)
      await fetchSkills()
    } catch (e: any) {
      errorMsg.value = `操作失败：${e?.message || '未知错误'}`
    }
  }

  async function onDelete(s: Skill) {
    if (!confirm(`确定删除技能 "${s.name}" 吗？`)) return
    try {
      await deleteSkill(s.name)
      await fetchSkills()
    } catch (e: any) {
      errorMsg.value = `删除失败：${e?.message || '未知错误'}`
    }
  }

  // ====== 导入技能包 ======

  function onImportClick() {
    importFileInput.value?.click()
  }

  function onFileSelected(e: Event) {
    const target = e.target as HTMLInputElement
    const file = target.files?.[0]
    if (!file) return
    // 校验扩展名
    const ext = file.name.split('.').pop()?.toLowerCase()
    if (!ext || !['zip', 'skill', 'md'].includes(ext)) {
      errorMsg.value = `不支持的文件类型: .${ext}，仅支持 .zip / .skill / .md`
      target.value = ''
      return
    }
    pendingFile.value = file
    target.value = '' // 重置以便重复选同一文件
    doImport('cancel')
  }

  async function doImport(strategy: string) {
    if (!pendingFile.value) return
    importing.value = true
    errorMsg.value = ''
    successMsg.value = ''
    try {
      const result = await importSkillPackage(pendingFile.value, strategy)
      showConflictModal.value = false

      // 构建成功提示
      const actionText: Record<string, string> = {
        created: '创建成功',
        overwritten: '已覆盖原技能',
        renamed: `已重命名为 ${result.final_name}`,
      }
      let msg = `技能 "${result.final_name}" ${actionText[result.action] || '导入成功'}`
      if (result.warnings.length) {
        msg += `（${result.warnings.join('；')}）`
      }
      successMsg.value = msg

      // 3 秒后自动清除提示
      if (successTimer) clearTimeout(successTimer)
      successTimer = setTimeout(() => { successMsg.value = '' }, 3000)

      pendingFile.value = null
      await fetchSkills()
    } catch (e: any) {
      const detail = e?.response?.data?.detail || ''
      if (detail.includes('已存在')) {
        // 冲突 → 弹出选择
        conflictName.value = detail.match(/'([^']+)'/)?.[1] || ''
        showConflictModal.value = true
      } else {
        errorMsg.value = `导入失败：${detail || e?.message || '未知错误'}`
        pendingFile.value = null
      }
    } finally {
      importing.value = false
    }
  }

  function onConflictResolve(strategy: string) {
    doImport(strategy)
  }

  function onConflictCancel() {
    showConflictModal.value = false
    pendingFile.value = null
  }

  onMounted(fetchSkills)

  onUnmounted(() => {
    if (successTimer) clearTimeout(successTimer)
  })

  return {
      skills, builtinTools, loading, saving, importing, errorMsg, successMsg,
      showModal, editing, form, importFileInput, showConflictModal, conflictName,
      openCreate, openEdit, closeModal, onSave, onToggle, onDelete,
      onImportClick, onFileSelected, onConflictResolve, onConflictCancel
  }
}
