import { expect, test } from '@playwright/test'

const message = {
  kind: 'message',
  star_target: { target_type: 'message', target_id: 'message-one' },
  star: { id: 'star-one', starred_at: '2026-07-22T20:00:00Z' },
  title: 'A message worth keeping',
  subtitle: 'Alex Example · Direct message',
  excerpt: 'A bounded fixture excerpt.',
  timestamp_ms: 1_700_000_000_000,
  jump: {
    href: '/conversations/conv-one?around=message-one#message-message-one',
    conversation_id: 'conv-one',
    message_id: 'message-one',
  },
}

const media = {
  kind: 'media',
  star_target: { target_type: 'blob', target_id: 'a'.repeat(64) },
  star: { id: 'star-two', starred_at: '2026-07-21T20:00:00Z' },
  title: 'sunset.jpg',
  subtitle: 'Alex Example',
  excerpt: null,
  timestamp_ms: 1_600_000_000_000,
  jump: {
    href: '/conversations/conv-one?around=message-two#message-message-two',
    conversation_id: 'conv-one',
    message_id: 'message-two',
  },
  media_scope: 'unique',
  media: {
    id: 'attachment-two',
    view_key: `blob:${'a'.repeat(64)}`,
    type: 'photo',
    kind: 'photos',
    filename: 'sunset.jpg',
    availability: 'missing',
    blob_sha256: 'a'.repeat(64),
    unknown_date: false,
    occurrence_count: 2,
  },
}

test('starred library filters mixed targets and removes one through its canonical endpoint', async ({ page }) => {
  let retained = [message, media]
  await page.route('**/api/v1/**', async (route) => {
    const request = route.request()
    const url = new URL(request.url())
    if (url.pathname === '/api/v1/health') {
      await route.fulfill({ json: { status: 'ok', catalog_ready: true, version: 'fixture' } })
      return
    }
    if (url.pathname === '/api/v1/settings') {
      await route.fulfill({ json: { timezone: 'America/New_York', cloud_enabled: false } })
      return
    }
    if (url.pathname === '/api/v1/stars/message/message-one' && request.method() === 'DELETE') {
      retained = retained.filter((item) => item.star_target.target_id !== 'message-one')
      await route.fulfill({ status: 204 })
      return
    }
    if (url.pathname === '/api/v1/stars') {
      const category = url.searchParams.get('category') ?? 'all'
      const items = category === 'media' ? retained.filter((item) => item.kind === 'media') : retained
      await route.fulfill({ json: {
        items,
        page: { limit: 50, total: items.length, next_cursor: null },
        category,
        direction: url.searchParams.get('direction') ?? 'desc',
        facets: {
          all: retained.length,
          conversations: 0,
          messages: retained.some((item) => item.kind === 'message') ? 1 : 0,
          media: 1,
          activity: 0,
          people: 0,
        },
      } })
      return
    }
    await route.fulfill({ status: 404, json: { detail: 'Not found' } })
  })

  await page.goto('/starred')
  await expect(page.getByRole('heading', { name: 'Starred' })).toBeVisible()
  await expect(page.getByRole('banner').getByText('Personal library')).toBeVisible()
  await expect(page.getByRole('link', { name: 'Starred' })).toHaveAttribute('aria-current', 'page')
  await expect(page.getByText('A message worth keeping')).toBeVisible()
  await expect(page.getByText('sunset.jpg')).toBeVisible()

  const deleteRequest = page.waitForRequest((request) =>
    request.method() === 'DELETE'
      && new URL(request.url()).pathname === '/api/v1/stars/message/message-one')
  await page.getByRole('button', { name: 'Unstar A message worth keeping' }).click()
  await deleteRequest
  await expect(page.getByRole('heading', { name: 'A message worth keeping' })).toHaveCount(0)
  await expect(page.getByRole('button', { name: 'Unstar sunset.jpg' })).toBeFocused()

  await page.getByRole('button', { name: 'Media' }).click()
  await expect(page.locator('.status-badge', { hasText: 'Unique File' })).toBeVisible()
  await expect(page.getByText('sunset.jpg')).toBeVisible()
})
