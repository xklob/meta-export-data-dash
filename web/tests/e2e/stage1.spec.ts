import { expect, type Page, test } from '@playwright/test'
import AxeBuilder from '@axe-core/playwright'
import { fixtures } from '../../src/test/fixtures'

async function mockStageOneApi(page: Page) {
  await page.route('**/api/v1/**', async (route) => {
    const request = route.request()
    const path = new URL(request.url()).pathname
    const key = path.split('/').at(-1) as keyof typeof fixtures

    if (!(key in fixtures)) {
      await route.fulfill({ status: 404, json: { detail: 'Not found' } })
      return
    }

    if (key === 'settings' && request.method() === 'PATCH') {
      const update = request.postDataJSON() as { timezone: string; cloud_enabled: boolean }
      await route.fulfill({ json: { ...fixtures.settings, ...update } })
      return
    }

    await route.fulfill({ json: fixtures[key] })
  })
}

test.beforeEach(async ({ page }) => {
  await mockStageOneApi(page)
})

test('Playwright artifact creation is private by default', () => {
  expect(process.umask()).toBe(0o077)
})

test('desktop Stage 1 user journey is source-backed and navigable', async ({ page }) => {
  await page.goto('/')

  await expect(page.getByRole('heading', { name: 'Archive control center' })).toBeVisible()
  await expect(page.getByText('240')).toBeVisible()
  await expect(page.getByText('Cloud off')).toBeVisible()
  await expect(page.getByRole('link', { name: 'Conversations' })).toHaveAttribute('href', '/conversations')
  await page.screenshot({ path: 'test-results/stage1-home-desktop.png', fullPage: true })

  await page.getByRole('link', { name: 'Data Coverage' }).click()
  await expect(page.getByRole('heading', { name: 'Data coverage' })).toBeVisible()
  await page.getByRole('button', { name: 'Unavailable' }).click()
  await expect(page.locator('main').getByText('Friends')).toBeVisible()
  await expect(page.locator('main').getByText('Messages')).toHaveCount(0)

  await page.getByRole('link', { name: 'Integrity' }).click()
  await expect(page.getByRole('heading', { name: 'Integrity' })).toBeVisible()
  await expect(page.getByText('Five message threads begin after segment one.')).toBeVisible()

  await page.getByRole('link', { name: 'Jobs' }).click()
  await expect(page.getByRole('heading', { name: 'Jobs' })).toBeVisible()
  await expect(page.getByText('Hashing source files')).toBeVisible()
  await expect(page.getByRole('progressbar', { name: 'Archive Inventory progress' })).toHaveAttribute('aria-valuenow', '96')

  await page.getByRole('link', { name: 'Settings' }).click()
  await expect(page.getByRole('heading', { name: 'Settings' })).toBeVisible()
  await expect(page.getByText('Archive Owner')).toBeVisible()
  await expect(page.getByRole('button', { name: /Approval required/ })).toBeDisabled()
})

test('settings update keeps cloud state conspicuous', async ({ page }) => {
  await page.goto('/settings')
  await page.getByRole('checkbox', { name: 'Allow cloud-assisted features' }).check()
  await expect(page.getByText('Cloud access will be enabled after you save.')).toBeVisible()
  await page.getByRole('button', { name: 'Save settings' }).click()
  await expect(page.getByText('Cloud on')).toBeVisible()
  await expect(page.getByText('Cloud access enabled')).toBeVisible()
})

test('mobile navigation opens, moves focusable routes, and closes after selection', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await page.goto('/')

  const menuButton = page.getByRole('button', { name: 'Open navigation' })
  await expect(menuButton).toHaveAttribute('aria-expanded', 'false')
  await expect(page.locator('#primary-navigation')).toHaveCSS('visibility', 'hidden')
  await menuButton.click()
  await expect(menuButton).toHaveAttribute('aria-expanded', 'true')
  await expect(page.locator('#primary-navigation')).toHaveClass(/sidebar--open/)
  await expect(page.locator('#primary-navigation')).toHaveCSS('transform', 'matrix(1, 0, 0, 1, 0, 0)')
  await expect(page.locator('#primary-navigation')).toHaveAttribute('role', 'dialog')
  await expect(page.getByRole('button', { name: 'Close navigation', exact: true }).first()).toBeFocused()
  await page.keyboard.press('Shift+Tab')
  await expect(page.locator('#primary-navigation a[href]').last()).toBeFocused()
  await page.keyboard.press('Tab')
  await expect(page.getByRole('button', { name: 'Close navigation', exact: true }).first()).toBeFocused()
  await page.keyboard.press('Escape')
  await expect(menuButton).toBeFocused()
  await expect(page.locator('#primary-navigation')).toHaveCSS('visibility', 'hidden')
  await menuButton.click()
  await page.screenshot({ path: 'test-results/stage1-navigation-mobile.png', fullPage: true })

  await page.getByRole('link', { name: 'Data Coverage' }).click()
  await expect(page.getByRole('heading', { name: 'Data coverage' })).toBeVisible()
  await expect(page.getByText('Each category includes source counts, normalized logical counts, and its last-checked date.')).toBeVisible()
  const coverageTable = page.getByRole('region', { name: 'Coverage categories table' })
  expect(await coverageTable.evaluate((element) => element.scrollWidth <= element.clientWidth)).toBe(true)
  const firstCoverageBadge = coverageTable.locator('.status-badge').first()
  const badgeBox = await firstCoverageBadge.boundingBox()
  expect(badgeBox).not.toBeNull()
  expect((badgeBox?.x ?? 0) + (badgeBox?.width ?? 0)).toBeLessThanOrEqual(390)
  await expect(page.locator('#primary-navigation')).not.toHaveClass(/sidebar--open/)
  await expect(page.getByText('Cloud off')).toBeVisible()
})

test('keyboard and SPA navigation announce each route and preserve a useful focus target', async ({ page }) => {
  await page.goto('/')
  await expect(page).toHaveTitle('Home · Meta Archive Explorer')

  await page.keyboard.press('Tab')
  const skipLink = page.getByRole('link', { name: 'Skip to main content' })
  await expect(skipLink).toBeFocused()
  await page.keyboard.press('Enter')
  const main = page.getByRole('main')
  await expect(main).toBeFocused()
  const mainBox = await main.boundingBox()
  const topbarBox = await page.locator('.topbar').boundingBox()
  expect(mainBox).not.toBeNull()
  expect(topbarBox).not.toBeNull()
  expect(mainBox?.y ?? 0).toBeGreaterThanOrEqual(
    (topbarBox?.y ?? 0) + (topbarBox?.height ?? 0) - 1,
  )

  await page.getByRole('link', { name: 'Data Coverage' }).click()
  await expect(page.getByRole('heading', { name: 'Data coverage' })).toBeVisible()
  await expect(page).toHaveTitle('Data Coverage · Meta Archive Explorer')
  await expect(page.getByRole('main')).toBeFocused()
  await expect(page.getByTestId('route-announcement')).toHaveText('Data Coverage page loaded')
})

test('mobile skip navigation leaves the main heading below the sticky header', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await page.goto('/')
  await page.keyboard.press('Tab')
  await expect(page.getByRole('link', { name: 'Skip to main content' })).toBeFocused()
  await page.keyboard.press('Enter')

  const main = page.getByRole('main')
  await expect(main).toBeFocused()
  const mainBox = await main.boundingBox()
  const topbarBox = await page.locator('.topbar').boundingBox()
  expect(mainBox).not.toBeNull()
  expect(topbarBox).not.toBeNull()
  expect(mainBox?.y ?? 0).toBeGreaterThanOrEqual(
    (topbarBox?.y ?? 0) + (topbarBox?.height ?? 0) - 1,
  )
  await expect(page.getByRole('heading', { name: 'Archive control center' })).toBeInViewport()
})

test('reduced-motion preference collapses interface transitions', async ({ page }) => {
  await page.emulateMedia({ reducedMotion: 'reduce' })
  await page.goto('/')
  const durations = await page.getByRole('link', { name: 'Data Coverage' }).evaluate((element) =>
    getComputedStyle(element).transitionDuration.split(',').map((value) => Number.parseFloat(value)),
  )
  expect(durations.length).toBeGreaterThan(0)
  expect(durations.every((duration) => duration <= 0.001)).toBe(true)
})

test('Stage 1 and not-found pages are overflow-free and axe-clean at desktop and mobile sizes', async ({ page }) => {
  test.setTimeout(60_000)
  for (const viewport of [{ width: 1440, height: 1000 }, { width: 390, height: 844 }]) {
    await page.setViewportSize(viewport)
    for (const path of ['/', '/coverage', '/integrity', '/jobs', '/settings', '/not-a-real-route']) {
      await page.goto(path)
      await expect(page.locator('main h1')).toBeVisible()
      if (path === '/integrity') {
        const checks = page.locator('.integrity-checks')
        await expect(checks).not.toHaveAttribute('open', '')
        await expect(checks.getByText('Review all 2 detailed checks')).toBeVisible()
      }
      const widths = await page.locator('body').evaluate((element) => ({ scroll: element.scrollWidth, client: element.clientWidth }))
      expect(widths.scroll, `${path} at ${viewport.width}px`).toBeLessThanOrEqual(widths.client + 1)
      const result = await new AxeBuilder({ page })
        .withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'wcag22a', 'wcag22aa'])
        .analyze()
      expect(result.violations, `${path} at ${viewport.width}px: ${result.violations.map((violation) => violation.id).join(', ')}`).toEqual([])
      expect(result.incomplete, `${path} at ${viewport.width}px: incomplete ${result.incomplete.map((issue) => issue.id).join(', ')}`).toEqual([])
    }
  }
})

test('empty job history explains how operations appear', async ({ page }) => {
  await page.route(/\/api\/v1\/jobs(?:\?.*)?$/, (route) =>
    route.fulfill({ json: { items: [], page: { total: 0, filtered_total: 0 } } }),
  )
  await page.goto('/jobs')

  await expect(page.getByRole('heading', { name: 'Jobs' })).toBeVisible()
  await expect(page.getByText('No archive jobs yet')).toBeVisible()
  await expect(page.getByText('Operations started from the CLI will appear here automatically.')).toBeVisible()
})

test('Stage 1 home has deterministic visual and accessibility baselines', async ({ page }) => {
  await page.emulateMedia({ reducedMotion: 'reduce' })
  await page.goto('/')
  const main = page.getByRole('main')
  await expect(main.getByRole('heading', { name: 'Archive control center' })).toBeVisible()
  await expect(main).toHaveScreenshot('stage1-home-main.png', { animations: 'disabled' })
  await expect(main).toMatchAriaSnapshot({ name: 'stage1-home-main.aria.yml' })
})

test('every remaining Stage 1 route and the not-found view has deterministic visual and ARIA baselines', async ({ page }) => {
  test.setTimeout(60_000)
  await page.emulateMedia({ reducedMotion: 'reduce' })
  await page.clock.setFixedTime(new Date('2026-07-20T22:11:00Z'))
  const routes = [
    { path: '/coverage', slug: 'coverage' },
    { path: '/integrity', slug: 'integrity' },
    { path: '/jobs', slug: 'jobs' },
    { path: '/settings', slug: 'settings' },
    { path: '/not-a-real-route', slug: 'not-found' },
  ]
  for (const route of routes) {
    await page.goto(route.path)
    const main = page.getByRole('main')
    await expect(main.getByRole('heading', { level: 1 })).toBeVisible()
    await expect(main).toHaveScreenshot(`stage1-${route.slug}-main.png`, { animations: 'disabled' })
    await expect(main).toMatchAriaSnapshot({ name: `stage1-${route.slug}-main.aria.yml` })
  }
})
