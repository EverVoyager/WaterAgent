<template>
  <div class="skills-view">
    <div class="skills-card">
      <div class="card-head">
        <h2 class="card-title">技能管理</h2>
        <p class="card-desc">
          配置 Agent 的专业能力包（借鉴 Claude Skills 架构）。每个技能包含触发条件、行为指令和工具子集，
          Agent 运行时自动匹配并加载。
        </p>
        <div class="head-actions">
          <button class="secondary-btn" type="button" :disabled="importing" @click="onImportClick">
            <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="17 8 12 3 7 8" />
              <line x1="12" y1="3" x2="12" y2="15" />
            </svg>
            {{ importing ? '导入中...' : '导入技能包' }}
          </button>
          <button class="primary-btn" type="button" @click="openCreate">
            <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">
              <line x1="12" y1="5" x2="12" y2="19" />
              <line x1="5" y1="12" x2="19" y2="12" />
            </svg>
            新建技能
          </button>
        </div>
        <input
          id="import-file-input"
          ref="importFileInput"
          type="file"
          accept=".zip,.skill,.md"
          style="display:none"
          @change="onFileSelected"
        />
      </div>

      <div v-if="loading && !skills.length" class="skeleton-list">
        <div v-for="i in 3" :key="i" class="skeleton-row" />
      </div>

      <div v-else-if="skills.length" class="skill-list">
        <div v-for="s in skills" :key="s.id" class="skill-item" :class="{ disabled: !s.enabled }">
          <div class="skill-head">
            <div class="skill-name-row">
              <span class="skill-name">{{ s.name }}</span>
              <span class="status-tag" :class="s.enabled ? 'on' : 'off'">
                {{ s.enabled ? '启用' : '禁用' }}
              </span>
            </div>
            <div class="skill-actions">
              <button class="text-btn" type="button" @click="onToggle(s)"> {{ s.enabled ? '禁用' : '启用' }}</button>
              <button class="text-btn" type="button" @click="openEdit(s)">编辑</button>
              <button class="text-btn danger" type="button" @click="onDelete(s)">删除</button>
            </div>
          </div>
          <p class="skill-desc">{{ s.description }}</p>
          <div v-if="s.tool_names.length" class="tool-tags">
            <span v-for="t in s.tool_names" :key="t" class="tool-tag">{{ t }}</span>
          </div>
          <div v-else class="tool-tags">
            <span class="tool-tag all">全部工具</span>
          </div>
        </div>
      </div>

      <div v-else class="empty-state">
        <p>暂无技能配置</p>
        <p class="hint">点击"新建技能"创建第一个能力包</p>
      </div>

      <div v-if="errorMsg" class="error-box">{{ errorMsg }}</div>
      <div v-if="successMsg" class="success-box">{{ successMsg }}</div>
    </div>

    <!-- 冲突选择弹窗 -->
    <div v-if="showConflictModal" class="modal-overlay" @click.self="onConflictCancel">
      <div class="modal conflict-modal">
        <div class="modal-head">
          <h3>技能名冲突</h3>
          <button class="icon-btn" type="button" @click="onConflictCancel">×</button>
        </div>
        <div class="modal-body">
          <p class="conflict-text">
            技能 "<strong>{{ conflictName }}</strong>" 已存在，请选择处理方式：
          </p>
          <ul class="conflict-options">
            <li><strong>覆盖</strong>：删除原技能，用导入内容替换</li>
            <li><strong>重命名</strong>：自动添加 <code>_imported_N</code> 后缀</li>
            <li><strong>取消</strong>：放弃导入</li>
          </ul>
        </div>
        <div class="modal-foot">
          <button class="text-btn" type="button" @click="onConflictCancel">取消</button>
          <button class="secondary-btn" type="button" :disabled="importing" @click="onConflictResolve('rename')">
            {{ importing ? '导入中...' : '重命名导入' }}
          </button>
          <button class="primary-btn danger-btn" type="button" :disabled="importing" @click="onConflictResolve('overwrite')">
            覆盖
          </button>
        </div>
      </div>
    </div>

    <!-- 编辑/创建弹窗 -->
    <div v-if="showModal" class="modal-overlay" @click.self="closeModal">
      <div class="modal">
        <div class="modal-head">
          <h3>{{ editing ? '编辑技能' : '新建技能' }}</h3>
          <button class="icon-btn" type="button" @click="closeModal">×</button>
        </div>
        <div class="modal-body">
          <div class="form-row">
            <label>技能名 <span class="required">*</span></label>
            <input
              v-model="form.name"
              type="text"
              placeholder="flood_dispatch_analysis"
              :disabled="editing"
              class="form-input"
            />
            <small class="form-hint">snake_case，创建后不可修改</small>
          </div>
          <div class="form-row">
            <label>触发条件 <span class="required">*</span></label>
            <textarea
              v-model="form.description"
              placeholder="描述什么场景下应该启用此技能，Agent 据此做语义匹配"
              rows="2"
              class="form-input"
            />
          </div>
          <div class="form-row">
            <label>行为指令 <span class="required">*</span></label>
            <textarea
              v-model="form.instructions"
              placeholder="启用后注入 Agent 的完整指令，包括角色、工作流程、约束等"
              rows="8"
              class="form-input mono"
            />
          </div>
          <div class="form-row">
            <label>工具子集</label>
            <p class="form-hint">不选则使用全部内置工具</p>
            <div class="tool-checkboxes">
              <label v-for="t in builtinTools" :key="t.name" class="tool-check">
                <input
                  :checked="form.tool_names.includes(t.name)"
                  type="checkbox"
                  @change="toggleTool(t.name)"
                />
                <span class="tool-check-name">{{ t.name }}</span>
                <span class="tool-check-desc">{{ t.description }}</span>
              </label>
            </div>
          </div>
        </div>
        <div class="modal-foot">
          <button class="text-btn" type="button" @click="closeModal">取消</button>
          <button class="primary-btn" type="button" :disabled="saving" @click="onSave">
            {{ saving ? '保存中...' : '保存' }}
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { onMounted, ref } from 'vue'
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

function toggleTool(name: string) {
  const idx = form.value.tool_names.indexOf(name)
  if (idx >= 0) {
    form.value.tool_names.splice(idx, 1)
  } else {
    form.value.tool_names.push(name)
  }
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
</script>

<style scoped>
.skills-view {
  height: 100%;
  overflow-y: auto;
  padding: 32px 24px;
  background: var(--bg-base);
}

.skills-card {
  max-width: 820px;
  margin: 0 auto;
  background: var(--bg-elevated);
  border: 1px solid var(--border-default);
  border-radius: var(--radius-lg);
  padding: 20px 24px;
}

.card-head {
  padding-bottom: 16px;
  border-bottom: 1px solid var(--border-default);
}

.card-title {
  font-size: 15px;
  font-weight: 600;
  color: var(--text-primary);
  margin-bottom: 6px;
}

.card-desc {
  font-size: 12px;
  color: var(--text-tertiary);
  line-height: 1.6;
  margin-bottom: 14px;
}

.head-actions {
  display: flex;
  gap: 8px;
}

.primary-btn {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 7px 16px;
  border: none;
  border-radius: var(--radius-sm);
  background: var(--accent);
  color: #fff;
  font-size: 13px;
  font-weight: 500;
  cursor: pointer;
  transition: opacity 0.15s;
}

.primary-btn:hover:not(:disabled) {
  opacity: 0.9;
}

.primary-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.secondary-btn {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 7px 16px;
  border: 1px solid var(--border-default);
  border-radius: var(--radius-sm);
  background: var(--bg-base);
  color: var(--text-primary);
  font-size: 13px;
  font-weight: 500;
  cursor: pointer;
  transition: background-color 0.15s, border-color 0.15s;
}

.secondary-btn:hover:not(:disabled) {
  background: var(--accent-soft);
  border-color: var(--accent);
}

.secondary-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.danger-btn {
  background: var(--error);
}

.danger-btn:hover:not(:disabled) {
  opacity: 0.9;
}

.skeleton-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
  padding: 16px 0 4px;
}

.skeleton-row {
  height: 64px;
  border-radius: 8px;
  background: linear-gradient(90deg, var(--bg-subtle) 25%, var(--border-default) 50%, var(--bg-subtle) 75%);
  background-size: 200% 100%;
  animation: shimmer 1.4s infinite;
}

.skill-list {
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding: 16px 0 4px;
}

.skill-item {
  padding: 14px 16px;
  border: 1px solid var(--border-default);
  border-radius: var(--radius-md);
  background: var(--bg-subtle);
  transition: opacity 0.15s;
}

.skill-item.disabled {
  opacity: 0.55;
}

.skill-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 8px;
}

.skill-name-row {
  display: flex;
  align-items: center;
  gap: 10px;
}

.skill-name {
  font-size: 14px;
  font-weight: 600;
  color: var(--text-primary);
  font-family: 'SF Mono', Monaco, Consolas, monospace;
}

.status-tag {
  display: inline-block;
  padding: 1px 8px;
  border-radius: 10px;
  font-size: 11px;
  font-weight: 500;
}

.status-tag.on {
  color: var(--success);
  background: var(--level-4-soft);
}

.status-tag.off {
  color: var(--text-tertiary);
  background: var(--bg-elevated);
}

.skill-actions {
  display: flex;
  gap: 4px;
}

.text-btn {
  padding: 4px 10px;
  border: none;
  border-radius: var(--radius-sm);
  background: transparent;
  color: var(--text-secondary);
  font-size: 12px;
  cursor: pointer;
  transition: background-color 0.1s;
}

.text-btn:hover {
  background: var(--accent-soft);
  color: var(--text-primary);
}

.text-btn.danger:hover {
  color: var(--error);
}

.skill-desc {
  font-size: 13px;
  color: var(--text-secondary);
  line-height: 1.5;
  margin-bottom: 8px;
}

.tool-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
}

.tool-tag {
  padding: 2px 8px;
  border-radius: 4px;
  background: var(--bg-elevated);
  color: var(--text-tertiary);
  font-size: 11px;
  font-family: 'SF Mono', Monaco, Consolas, monospace;
}

.tool-tag.all {
  font-style: italic;
}

.empty-state {
  padding: 40px 0;
  text-align: center;
  color: var(--text-tertiary);
  font-size: 13px;
}

.empty-state .hint {
  font-size: 12px;
  margin-top: 6px;
}

.error-box {
  margin-top: 14px;
  padding: 10px 14px;
  border: 1px solid var(--error);
  border-radius: var(--radius-sm);
  background: var(--level-1-soft);
  color: var(--error);
  font-size: 13px;
}

.success-box {
  margin-top: 14px;
  padding: 10px 14px;
  border: 1px solid var(--success);
  border-radius: var(--radius-sm);
  background: var(--level-4-soft);
  color: var(--success);
  font-size: 13px;
  line-height: 1.5;
}

.conflict-modal {
  width: 90%;
  max-width: 460px;
}

.conflict-text {
  font-size: 13px;
  color: var(--text-primary);
  line-height: 1.6;
  margin-bottom: 12px;
}

.conflict-options {
  list-style: none;
  padding: 0;
  margin: 0;
  font-size: 12px;
  color: var(--text-secondary);
  line-height: 1.8;
}

.conflict-options li {
  padding-left: 14px;
  position: relative;
}

.conflict-options li::before {
  content: '•';
  position: absolute;
  left: 0;
  color: var(--text-tertiary);
}

.conflict-options code {
  padding: 1px 5px;
  border-radius: 3px;
  background: var(--bg-subtle);
  font-family: 'SF Mono', Monaco, Consolas, monospace;
  font-size: 11px;
}

/* Modal */
.modal-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.4);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 100;
}

.modal {
  width: 90%;
  max-width: 600px;
  max-height: 85vh;
  background: var(--bg-elevated);
  border: 1px solid var(--border-default);
  border-radius: var(--radius-lg);
  display: flex;
  flex-direction: column;
}

.modal-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px 20px;
  border-bottom: 1px solid var(--border-default);
}

.modal-head h3 {
  font-size: 15px;
  font-weight: 600;
  color: var(--text-primary);
}

.icon-btn {
  width: 28px;
  height: 28px;
  border: none;
  border-radius: var(--radius-sm);
  background: transparent;
  color: var(--text-secondary);
  font-size: 18px;
  cursor: pointer;
}

.icon-btn:hover {
  background: var(--accent-soft);
}

.modal-body {
  flex: 1;
  overflow-y: auto;
  padding: 16px 20px;
}

.form-row {
  margin-bottom: 14px;
}

.form-row label {
  display: block;
  font-size: 13px;
  font-weight: 500;
  color: var(--text-primary);
  margin-bottom: 6px;
}

.required {
  color: var(--error);
}

.form-input {
  width: 100%;
  padding: 8px 10px;
  border: 1px solid var(--border-default);
  border-radius: var(--radius-sm);
  background: var(--bg-base);
  color: var(--text-primary);
  font-size: 13px;
  font-family: inherit;
}

.form-input:focus {
  outline: none;
  border-color: var(--accent);
  box-shadow: 0 0 0 3px var(--accent-soft);
}

.form-input:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

textarea.form-input {
  resize: vertical;
  line-height: 1.5;
}

.mono {
  font-family: 'SF Mono', Monaco, Consolas, monospace;
  font-size: 12px;
}

.form-hint {
  display: block;
  font-size: 11px;
  color: var(--text-tertiary);
  margin-top: 4px;
}

.tool-checkboxes {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.tool-check {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 8px;
  border-radius: var(--radius-sm);
  background: var(--bg-base);
  cursor: pointer;
}

.tool-check input {
  margin: 0;
}

.tool-check-name {
  font-size: 12px;
  font-family: 'SF Mono', Monaco, Consolas, monospace;
  color: var(--text-primary);
  min-width: 140px;
}

.tool-check-desc {
  font-size: 12px;
  color: var(--text-tertiary);
}

.modal-foot {
  display: flex;
  justify-content: flex-end;
  gap: 8px;
  padding: 14px 20px;
  border-top: 1px solid var(--border-default);
}

@keyframes shimmer {
  0% { background-position: 200% 0; }
  100% { background-position: -200% 0; }
}
</style>
