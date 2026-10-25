<template>
  <div class="modal-overlay" @click.self="$emit('cancel')">
    <div class="modal">
      <div class="modal-head">
        <h3>{{ editing ? '编辑技能' : '新建技能' }}</h3>
        <button class="icon-btn" type="button" @click="$emit('cancel')">×</button>
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
        <button class="text-btn" type="button" @click="$emit('cancel')">取消</button>
        <button class="primary-btn" type="button" :disabled="saving" @click="$emit('save')">
          {{ saving ? '保存中...' : '保存' }}
        </button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import type { BuiltinTool, SkillPayload } from '@/api/skills'

defineProps<{ editing: boolean; saving: boolean; builtinTools: BuiltinTool[] }>()
defineEmits<{ save: []; cancel: [] }>()
const form = defineModel<SkillPayload>({ required: true })

function toggleTool(name: string) {
  const selected = form.value.tool_names
  const index = selected.indexOf(name)
  if (index >= 0) selected.splice(index, 1)
  else selected.push(name)
}
</script>

<style scoped src="@/styles/skill-controls.css"></style>

<style scoped>
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
</style>
