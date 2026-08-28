# TDD 证据报告 · Codex 风格前端重设计

- 日期：2026-08-09
- 规格：`docs/superpowers/specs/2026-08-09-codex-style-frontend-redesign-design.md`
- 计划：`docs/superpowers/plans/2026-08-09-codex-style-frontend-redesign.md`
- 分支：`feat/codex-style-frontend`

## 执行说明（重要）

执行过程中检测到工作区内出现并行实现（非本会话主线程写入，内容经逐文件核对与计划完全一致）。因此各任务的 RED/GREEN 证据按**实际观测**记录如下，未观测到的 RED 不虚构。

## 测试规格与证据

| # | 保证项 | 测试位置 | 类型 | 结果 | 证据 |
|---|--------|----------|------|------|------|
| 1 | 无存储且系统浅色时默认 light | `useTheme.spec.ts` | unit | PASS | 首跑 RED（模块不存在）→ 实现后 GREEN |
| 2 | 无存储且系统深色时默认 dark | `useTheme.spec.ts` | unit | PASS | 同上 |
| 3 | localStorage 存储值优先于系统偏好 | `useTheme.spec.ts` | unit | PASS | 同上 |
| 4 | toggleTheme 切换并写 html[data-theme] + localStorage | `useTheme.spec.ts` | unit | PASS | 同上 |
| 5 | loadSessions 空/损坏/非法项容错 | `useChatSessions.spec.ts` 纯函数 ×3 | unit | PASS | 首次观测即 GREEN（并行实现先于测试执行落盘） |
| 6 | groupSessions 今天/昨天/更早分组 + 空列表 | `useChatSessions.spec.ts` ×2 | unit | PASS | 同上 |
| 7 | ensureActiveSession 创建并激活 | `useChatSessions.spec.ts` | unit | PASS | 同上 |
| 8 | persistActiveSession 自动命名（前 20 字）并写盘 | `useChatSessions.spec.ts` | unit | PASS | 同上 |
| 9 | createSession 草稿态 | `useChatSessions.spec.ts` | unit | PASS | 同上 |
| 10 | switchSession 切换并持久化 activeId | `useChatSessions.spec.ts` | unit | PASS | 同上 |
| 11 | deleteSession 删除当前后回退最新/清空 | `useChatSessions.spec.ts` | unit | PASS | 同上 |
| 12 | 启动时 activeId 失效回退 | `useChatSessions.spec.ts` | unit | PASS | 同上 |
| 13 | 全部源码通过 vue-tsc strict + vite build | `npm run build` | 构建 | PASS | `built in 8.27s`，bundle 1187kB→108kB |
| 14 | 全部测试通过 | `npm run test` | 回归 | PASS | 15/15（2 文件） |

## 验证命令与最终输出（提交后状态复跑）

```
cd frontend && npm run build   # vue-tsc -b && vite build → built in 8.27s ✓
cd frontend && npm run test    # Test Files 2 passed (2), Tests 15 passed (15) ✓
```

## 覆盖与已知缺口

- 组件层（6 个 SFC）与视图层无单元测试：前端此前无组件测试基建，本次按规格以 `vue-tsc` 严格类型检查 + 构建 + 人工走查作为验证手段。组件为纯渲染层，交互逻辑（会话、主题、流式状态机）均在有测试的 composable 层。
- 人工走查清单（规格 §12）需在 dev server 下由人逐项确认：`cd frontend && npm run dev`。
- RED 证据缺口：Task 2（useChatSessions）因并行实现先于测试执行落盘，未观测到 RED；其测试在首次运行时即 11/11 通过。

## 提交序列（本分支）

1. `0b1898f` feat(frontend): 主题系统 + 会话存储 + vitest 测试基建
2. `7c0aa78` refactor(frontend): useAgentChat 接入会话存储并提升为模块级单例
3. `ea04b32` feat(frontend): Codex 风格对话组件（推理块/工具链/预警卡/消息/输入区/边栏）
4. `77a663a` feat(frontend): 边栏布局 App Shell 与视图重写（无气泡流式对话）
