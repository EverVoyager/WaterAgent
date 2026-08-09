import { test, expect } from '@playwright/test'

test.describe('Smoke', () => {
  test('首页加载并显示关键元素', async ({ page }) => {
    await page.goto('/')

    // 页面标题正确
    await expect(page).toHaveTitle('黄河吕梁段防汛预警智能体')

    // 侧边栏存在
    await expect(page.locator('.sidebar')).toBeVisible()

    // 欢迎屏标题存在（/ 会重定向到 /agent，展示欢迎屏）
    await expect(page.locator('.welcome-title')).toContainText('水卫')

    // 输入框存在
    await expect(page.getByPlaceholder('输入防汛相关问题，Enter 发送…')).toBeVisible()
  })

  test('可直接访问 /agent 路由', async ({ page }) => {
    await page.goto('/agent')

    await expect(page).toHaveURL(/\/agent$/)

    // 智能研判视图渲染：输入框可用
    const input = page.getByPlaceholder('输入防汛相关问题，Enter 发送…')
    await expect(input).toBeVisible()
    await expect(input).toBeEnabled()

    // 侧边栏“智能研判”导航项存在
    await expect(page.getByTitle('智能研判')).toBeVisible()
  })
})
