import { createRouter, createWebHistory, type RouteRecordRaw } from 'vue-router'

const routes: RouteRecordRaw[] = [
  {
    path: '/',
    redirect: '/agent',
  },
  {
    path: '/health',
    name: 'Health',
    component: () => import('@/views/HealthView.vue'),
    meta: { title: '服务健康' },
  },
  {
    path: '/agent',
    name: 'Agent',
    component: () => import('@/views/AgentView.vue'),
    meta: { title: '防汛预警 Agent' },
  },
  {
    path: '/skills',
    name: 'Skills',
    component: () => import('@/views/SkillsView.vue'),
    meta: { title: '技能管理' },
  },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

export default router
