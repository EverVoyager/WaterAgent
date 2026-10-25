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

    <SkillEditor
      v-if="showModal"
      v-model="form"
      :editing="editing"
      :saving="saving"
      :builtin-tools="builtinTools"
      @cancel="closeModal"
      @save="onSave"
    />
  </div>
</template>

<script setup lang="ts">
import SkillEditor from '@/components/SkillEditor.vue'
import { useSkills } from '@/composables/useSkills'

const {
  skills, builtinTools, loading, saving, importing, errorMsg, successMsg,
  showModal, editing, form, importFileInput, showConflictModal, conflictName,
  openCreate, openEdit, closeModal, onSave, onToggle, onDelete,
  onImportClick, onFileSelected, onConflictResolve, onConflictCancel
} = useSkills()
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

@keyframes shimmer {
  0% { background-position: 200% 0; }
  100% { background-position: -200% 0; }
}
</style>

<style scoped src="@/styles/skill-controls.css"></style>
