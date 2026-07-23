import { expect, type Page, test } from '@playwright/test'
import AxeBuilder from '@axe-core/playwright'
import { fixtures } from '../../src/test/fixtures'

const SAVED_CONVERSATION_VIEWS_KEY = 'meta-archive:conversation-views:v1'

const linkedMediaAttachments = fixtures.media.items
  .filter((item) => item.conversation && item.message_id)
  .map((item) => ({
    id: item.id,
    kind: item.kind,
    filename: item.filename,
    media_type: item.media_type,
    size_bytes: item.size_bytes,
    timestamp_ms: item.timestamp_ms,
    availability: item.availability,
    blob_sha256: item.blob_sha256,
    blob_href: item.blob_href,
    is_external_reference: item.is_external_reference,
    message_id: item.message_id,
    jump: item.jump,
  }))

function conversationMessagesWithMediaOccurrences() {
  return fixtures.conversationMessages.items.map((message) => ({
    ...message,
    sender: message.sender === 'Archive Owner' ? 'Archive Owner' : message.sender,
    attachments: [
      ...message.attachments,
      ...linkedMediaAttachments
        .filter((attachment) => attachment.message_id === message.id)
        .map(({ message_id: _messageId, ...attachment }) => attachment),
    ],
  }))
}

async function mockStageTwoApi(page: Page) {
  await page.route('**/api/v1/**', async (route) => {
    const request = route.request()
    const url = new URL(request.url())
    const path = url.pathname
    if (path === '/api/v1/health') return route.fulfill({ json: fixtures.health })
    if (path === '/api/v1/settings') return route.fulfill({ json: { ...fixtures.settings, owner_identity: 'Archive Owner', archive_root_display: 'fixture-export' } })
    if (path === '/api/v1/conversation-corrections') return route.fulfill({ json: { items: [], identity_revision: 'a'.repeat(64), page: { limit: 100, offset: 0, total: 0, next_offset: null }, notice: 'Direct histories are provisional.' } })
    if (path === '/api/v1/conversation-corrections/split') return route.fulfill({ status: 201, json: { correction_id: 'correction-split', action: 'split', status: 'active', identity_revision: 'b'.repeat(64), redirect_conversation_id: 'conv-alex', effect: { segments: 1 } } })
    if (path === '/api/v1/conversations') return route.fulfill({ json: fixtures.conversations })
    if (path === '/api/v1/conversations/conv-alex') {
      return route.fulfill({ json: {
        ...fixtures.conversationDetail,
        segments: fixtures.conversationDetail.segments.map((segment) => ({
          ...segment,
          participants: segment.participants.map((participant) => participant.is_owner
            ? { ...participant, name: 'Archive Owner' }
            : participant),
        })),
      } })
    }
    if (path.endsWith('/messages/date-jump')) {
      return route.fulfill({ json: {
        requested_timestamp_ms: Number(url.searchParams.get('at_ms')),
        target_timestamp_ms: fixtures.conversationMessages.items[49]?.timestamp_ms,
        jump: { href: '/conversations/conv-alex?around=message-050#message-message-050', conversation_id: 'conv-alex', message_id: 'message-050', fragment: 'message-message-050' },
      } })
    }
    if (path.endsWith('/message-search')) {
      const match = fixtures.conversationMessages.items[0]
      return route.fulfill({ json: { items: match ? [{ ...match, matched_fields: ['content'] }] : [], page: { limit: 40, total: match ? 1 : 0, next_cursor: null }, query: url.searchParams.get('q') } })
    }
    if (path.endsWith('/advanced')) {
      return route.fulfill({ json: { message_id: 'message-001', has_sensitive_details: true, sensitive: { ip: '192.0.2.10', payment_info: { amount: 'fixture-only' } }, notice: 'Advanced details can contain private network or payment data.' } })
    }
    if (path === '/api/v1/conversations/conv-alex/messages') {
      const around = url.searchParams.get('around')
      return route.fulfill({ json: {
        ...fixtures.conversationMessages,
        items: conversationMessagesWithMediaOccurrences(),
        focus: around ? { href: `/conversations/conv-alex?around=${around}`, message_id: around, fragment: `message-${around}` } : null,
      } })
    }
    if (path.startsWith('/api/v1/media/blobs/')) {
      const repeated = { ...fixtures.media.items[0], occurrence_count: 2 }
      return route.fulfill({ json: { blob_sha256: repeated.blob_sha256, items: [repeated, { ...repeated, id: 'media-photo-2', view_key: 'occurrence:media-photo-2' }], page: { limit: 100, total: 2, next_cursor: null } } })
    }
    if (path === '/api/v1/media') {
      if (url.searchParams.get('view') === 'unique') {
        const repeated = { ...fixtures.media.items[0], view_key: `blob:${'a'.repeat(64)}`, occurrence_count: 2, occurrences_href: `/api/v1/media/blobs/${'a'.repeat(64)}/occurrences` }
        return route.fulfill({ json: { ...fixtures.media, view: 'unique', items: [repeated, ...fixtures.media.items.slice(1, 5)], page: { limit: 60, total: 5, next_cursor: null } } })
      }
      return route.fulfill({ json: fixtures.media })
    }
    if (path.startsWith('/api/v1/blobs/')) {
      return route.fulfill({
        status: 200,
        contentType: 'image/png',
        body: Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=', 'base64'),
      })
    }
    return route.fulfill({ status: 404, json: { detail: 'Not found' } })
  })
}

async function prepareFullPageScreenshot(page: Page) {
  await page.evaluate(() => {
    window.scrollTo(0, 0)
    if (document.activeElement instanceof HTMLElement) document.activeElement.blur()
  })
}

test.beforeEach(async ({ page }) => {
  await mockStageTwoApi(page)
})

test('conversation dashboard filters and opens a bounded reader', async ({ page }) => {
  await page.goto('/conversations')
  await expect(page.getByRole('heading', { name: 'Conversations' })).toBeVisible()
  await expect(page.getByText('4', { exact: true })).toBeVisible()
  await expect(page.getByRole('link', { name: 'Alex Example' })).toBeVisible()
  await expect(page.getByRole('table', { name: 'Conversations' })).toHaveAttribute('aria-rowcount', '5')
  const conversationTableWidths = await page.getByRole('table', { name: 'Conversations' }).evaluate((element) => ({ client: element.clientWidth, scroll: element.scrollWidth }))
  expect(conversationTableWidths.scroll).toBeLessThanOrEqual(conversationTableWidths.client + 1)
  const coverageBadgeBounds = await page.locator('.conversation-table__row [role="cell"]:last-child .status-badge').evaluateAll((badges) => badges.map((badgeElement) => {
    const badge = badgeElement.getBoundingClientRect()
    const cell = badgeElement.closest('[role="cell"]')?.getBoundingClientRect()
    const viewport = badgeElement.closest('[role="table"]')?.querySelector('.conversation-table__viewport')?.getBoundingClientRect()
    return {
      badge: { left: badge.left, right: badge.right, top: badge.top, bottom: badge.bottom },
      cell: cell ? { left: cell.left, right: cell.right, top: cell.top, bottom: cell.bottom } : null,
      viewport: viewport ? { left: viewport.left, right: viewport.right } : null,
    }
  }))
  expect(coverageBadgeBounds.length).toBeGreaterThan(0)
  for (const bounds of coverageBadgeBounds) {
    expect(bounds.cell).not.toBeNull()
    expect(bounds.viewport).not.toBeNull()
    expect(bounds.badge.left).toBeGreaterThanOrEqual(bounds.cell!.left - 1)
    expect(bounds.badge.right).toBeLessThanOrEqual(bounds.cell!.right + 1)
    expect(bounds.badge.top).toBeGreaterThanOrEqual(bounds.cell!.top - 1)
    expect(bounds.badge.bottom).toBeLessThanOrEqual(bounds.cell!.bottom + 1)
    expect(bounds.badge.left).toBeGreaterThanOrEqual(bounds.viewport!.left - 1)
    expect(bounds.badge.right).toBeLessThanOrEqual(bounds.viewport!.right + 1)
  }

  const filteredRequest = page.waitForRequest((request) => {
    const url = new URL(request.url())
    return url.pathname === '/api/v1/conversations'
      && url.searchParams.get('conversation_type') === 'group'
      && url.searchParams.get('min_photos') === '10'
      && url.searchParams.get('max_audio') === '500'
      && url.searchParams.get('sort') === 'photos'
  })
  await page.getByRole('combobox', { name: 'Type', exact: true }).selectOption('group')
  await page.getByLabel('Minimum photos').fill('10')
  await page.getByLabel('Maximum audio files').fill('500')
  await page.getByRole('combobox', { name: 'Sort by', exact: true }).selectOption('photos')
  await page.getByRole('button', { name: 'Apply filters' }).click()
  await filteredRequest

  await page.screenshot({ path: 'test-results/stage2-conversations-desktop.png', fullPage: true })
  await page.getByRole('link', { name: 'Alex Example' }).click()
  await expect(page.getByRole('heading', { name: 'Alex Example' })).toBeVisible()
  const feed = page.getByRole('feed', { name: 'Conversation messages' })
  await expect(feed).toHaveAttribute('data-loaded-message-count', '140')
  expect(Number(await feed.getAttribute('data-rendered-message-count'))).toBeLessThan(30)
  await expect(feed.getByRole('article').first()).toHaveAttribute('aria-setsize', '140')
  const virtualRows = await feed.evaluate((viewport) => {
    const bounds = viewport.getBoundingClientRect()
    return [...viewport.querySelectorAll('.message-timeline__item')].map((row) => {
      const rowBounds = row.getBoundingClientRect()
      return {
        outside: rowBounds.bottom <= bounds.top || rowBounds.top >= bounds.bottom,
        hidden: row.getAttribute('aria-hidden') === 'true',
      }
    })
  })
  expect(virtualRows.filter((row) => row.outside).every((row) => row.hidden)).toBe(true)
})

test('conversation saved views recover malformed storage and persist across reload', async ({ page }) => {
  await page.goto('/conversations')
  await page.evaluate((key) => localStorage.setItem(key, '{malformed-json'), SAVED_CONVERSATION_VIEWS_KEY)
  await page.reload()

  const savedViews = page.getByRole('combobox', { name: 'Open a saved view' })
  await expect(savedViews.getByRole('option', { name: 'Saved views (0)' })).toBeAttached()
  await page.evaluate((key) => localStorage.setItem(key, '[null]'), SAVED_CONVERSATION_VIEWS_KEY)
  await page.reload()
  await expect(page.getByRole('combobox', { name: 'Open a saved view' }).getByRole('option', { name: 'Saved views (0)' })).toBeAttached()

  await page.getByLabel('Conversation title').fill('Fixture family')
  await page.getByRole('combobox', { name: 'Type', exact: true }).selectOption('group')
  await page.getByLabel('Minimum messages').fill('100')
  await page.getByLabel('Active after').fill('2022-01-01')
  await page.getByLabel('Active before').fill('2024-12-31')
  await page.getByLabel('Attachments').selectOption('yes')
  await page.getByLabel('Minimum photos').fill('10')
  await page.getByRole('combobox', { name: 'Sort by', exact: true }).selectOption('photos')
  await page.getByLabel('Direction').selectOption('asc')
  const filteredRequest = page.waitForRequest((request) => {
    const url = new URL(request.url())
    return url.pathname === '/api/v1/conversations'
      && url.searchParams.get('q') === 'Fixture family'
      && url.searchParams.get('conversation_type') === 'group'
      && url.searchParams.get('min_messages') === '100'
      && url.searchParams.has('activity_from_ms')
      && url.searchParams.has('activity_to_ms')
      && url.searchParams.get('has_attachments') === 'true'
      && url.searchParams.get('min_photos') === '10'
      && url.searchParams.get('sort') === 'photos'
      && url.searchParams.get('direction') === 'asc'
  })
  await page.getByRole('button', { name: 'Apply filters' }).click()
  await filteredRequest

  await page.getByRole('button', { name: 'Save view' }).click()
  await page.getByLabel('View name').fill('Media-heavy groups')
  await page.getByRole('button', { name: 'Save current filters' }).click()
  await expect(savedViews.getByRole('option', { name: 'Saved views (1)' })).toBeAttached()
  await expect(savedViews.getByRole('option', { name: 'Media-heavy groups' })).toBeAttached()
  expect(await page.evaluate((key) => JSON.parse(localStorage.getItem(key) ?? '[]'), SAVED_CONVERSATION_VIEWS_KEY)).toEqual([
    expect.objectContaining({
      name: 'Media-heavy groups',
      view: expect.objectContaining({
        q: 'Fixture family',
        conversationType: 'group',
        minMessages: '100',
        activityFrom: '2022-01-01',
        activityTo: '2024-12-31',
        hasAttachments: 'yes',
        minPhotos: '10',
        sort: 'photos',
        direction: 'asc',
      }),
    }),
  ])

  await page.reload()
  const persistedViews = page.getByRole('combobox', { name: 'Open a saved view' })
  await expect(persistedViews.getByRole('option', { name: 'Media-heavy groups' })).toBeAttached()
  const restoredRequest = page.waitForRequest((request) => {
    const url = new URL(request.url())
    return url.pathname === '/api/v1/conversations'
      && url.searchParams.get('q') === 'Fixture family'
      && url.searchParams.get('conversation_type') === 'group'
      && url.searchParams.get('min_messages') === '100'
      && url.searchParams.has('activity_from_ms')
      && url.searchParams.has('activity_to_ms')
      && url.searchParams.get('has_attachments') === 'true'
      && url.searchParams.get('min_photos') === '10'
      && url.searchParams.get('sort') === 'photos'
      && url.searchParams.get('direction') === 'asc'
  })
  await persistedViews.selectOption({ label: 'Media-heavy groups' })
  await restoredRequest
  await expect(page.getByLabel('Conversation title')).toHaveValue('Fixture family')
  await expect(page.getByRole('combobox', { name: 'Type', exact: true })).toHaveValue('group')
  await expect(page.getByLabel('Minimum messages')).toHaveValue('100')
  await expect(page.getByLabel('Active after')).toHaveValue('2022-01-01')
  await expect(page.getByLabel('Active before')).toHaveValue('2024-12-31')
  await expect(page.getByLabel('Attachments')).toHaveValue('yes')
  await expect(page.getByLabel('Minimum photos')).toHaveValue('10')
  await expect(page.getByRole('combobox', { name: 'Sort by', exact: true })).toHaveValue('photos')
  await expect(page.getByLabel('Direction')).toHaveValue('asc')
})

test('conversation dashboard serializes every filter and sort control and supports header sorting', async ({ page }) => {
  test.setTimeout(60_000)
  await page.goto('/conversations')
  await page.getByLabel('Conversation title').fill('Fixture family')
  await page.getByRole('combobox', { name: 'Type', exact: true }).selectOption('group')
  await page.getByRole('combobox', { name: 'Completeness', exact: true }).selectOption('incomplete')
  await page.getByRole('textbox', { name: 'Source category', exact: true }).fill('archived_threads')
  await page.getByLabel('Minimum people').fill('2')
  await page.getByLabel('Maximum people').fill('12')
  await page.getByLabel('Minimum messages').fill('10')
  await page.getByLabel('Maximum messages').fill('50000')
  await page.getByLabel('Active after').fill('2022-01-01')
  await page.getByLabel('Active before').fill('2024-12-31')
  await page.getByLabel('Attachments').selectOption('yes')
  await page.getByLabel('Minimum photos').fill('1')
  await page.getByLabel('Maximum photos').fill('2000')
  await page.getByLabel('Minimum videos').fill('2')
  await page.getByLabel('Maximum videos').fill('400')
  await page.getByLabel('Minimum audio files').fill('3')
  await page.getByLabel('Maximum audio files').fill('500')
  await page.getByRole('combobox', { name: 'Sort by', exact: true }).selectOption('first_activity')
  await page.getByLabel('Direction').selectOption('asc')
  const fullFilterRequest = page.waitForRequest((request) => {
    const url = new URL(request.url())
    return url.pathname === '/api/v1/conversations' && url.searchParams.get('sort') === 'first_activity'
  })
  await page.getByRole('button', { name: 'Apply filters' }).click()
  const fullQuery = new URL((await fullFilterRequest).url()).searchParams
  expect(Object.fromEntries(fullQuery)).toMatchObject({
    q: 'Fixture family', conversation_type: 'group', completeness: 'incomplete', source_category: 'archived_threads',
    min_participants: '2', max_participants: '12', min_messages: '10', max_messages: '50000', has_attachments: 'true',
    min_photos: '1', max_photos: '2000', min_videos: '2', max_videos: '400', min_audio: '3', max_audio: '500',
    sort: 'first_activity', direction: 'asc', limit: '75',
  })
  expect(Number(fullQuery.get('activity_from_ms'))).toBeGreaterThan(0)
  expect(Number(fullQuery.get('activity_to_ms'))).toBeGreaterThan(Number(fullQuery.get('activity_from_ms')))

  for (const sort of ['last_activity', 'title', 'type', 'source_category', 'participants', 'messages', 'sent', 'received', 'attachments', 'unique_attachments', 'photos', 'videos', 'audio', 'attachment_bytes', 'active_days', 'completeness']) {
    const requestPromise = page.waitForRequest((request) => {
      const url = new URL(request.url())
      return url.pathname === '/api/v1/conversations' && url.searchParams.get('sort') === sort
    })
    await page.getByRole('combobox', { name: 'Sort by', exact: true }).selectOption(sort)
    await page.getByRole('button', { name: 'Apply filters' }).click()
    await requestPromise
  }

  const titleDescending = page.waitForRequest((request) => {
    const url = new URL(request.url())
    return url.pathname === '/api/v1/conversations' && url.searchParams.get('sort') === 'title' && url.searchParams.get('direction') === 'desc'
  })
  await page.getByRole('button', { name: 'Sort by Conversation' }).click()
  await titleDescending
  await page.getByRole('button', { name: 'Sort by Conversation' }).click()
  await expect(page.getByRole('columnheader').first()).toHaveAttribute('aria-sort', 'ascending')
})

test('conversation dashboard loads another cursor page and recovers from an empty filtered state', async ({ page }) => {
  await page.route(/\/api\/v1\/conversations\?.*/, async (route) => {
    const url = new URL(route.request().url())
    if (url.searchParams.get('q') === 'no fixture matches') {
      return route.fulfill({ json: { ...fixtures.conversations, items: [], page: { limit: 75, total: 0, next_cursor: null } } })
    }
    if (url.searchParams.get('cursor') === 'conversation-page-two') {
      const item = { ...fixtures.conversations.items[1], id: 'conv-page-two', title: 'Page Two Group', href: '/conversations/conv-page-two' }
      return route.fulfill({ json: { ...fixtures.conversations, items: [item], page: { limit: 75, total: 5, next_cursor: null } } })
    }
    return route.fulfill({ json: { ...fixtures.conversations, page: { limit: 75, total: 5, next_cursor: 'conversation-page-two' } } })
  })
  await page.goto('/conversations')
  await expect(page.getByText('4 of 5 conversations loaded')).toBeVisible()
  const secondPage = page.waitForRequest((request) => request.url().includes('cursor=conversation-page-two'))
  await page.getByRole('button', { name: 'Load more conversations' }).click()
  await secondPage
  await expect(page.getByText('5 of 5 conversations loaded')).toBeVisible()
  await expect(page.getByRole('link', { name: 'Page Two Group' })).toBeVisible()

  await page.getByLabel('Conversation title').fill('no fixture matches')
  await page.getByRole('button', { name: 'Apply filters' }).click()
  await expect(page.getByText('No conversations match', { exact: true })).toBeVisible()
  await page.getByRole('button', { name: 'Reset filters' }).click()
  await expect(page.getByRole('link', { name: 'Alex Example' })).toBeVisible()
})

test('conversation reader supports search, date jump, and explicit sensitive reveal', async ({ page }) => {
  await page.goto('/conversations/conv-alex?around=message-002#message-message-002')
  await expect(page.getByRole('heading', { name: 'Alex Example' })).toBeVisible()
  await expect(page.getByText('192.0.2.10')).toHaveCount(0)

  await page.getByRole('button', { name: 'Show advanced details' }).first().click()
  await expect(page.getByText('192.0.2.10')).toBeVisible()
  await page.getByRole('button', { name: 'Hide details' }).click()
  await expect(page.getByText('192.0.2.10')).toHaveCount(0)

  await page.getByRole('button', { name: 'Search this conversation' }).click()
  await page.getByRole('searchbox', { name: 'Search words in this history' }).fill('Français ABC')
  await page.getByRole('region', { name: 'Search this conversation' }).getByRole('button', { name: 'Search', exact: true }).click()
  await expect(page.getByText(/1 matches for/)).toBeVisible()

  const dateRequest = page.waitForRequest((request) => request.url().includes('/messages/date-jump?at_ms='))
  await page.getByLabel('Jump to date').fill('2023-01-02')
  await page.getByRole('button', { name: 'Jump', exact: true }).click()
  await dateRequest
  await expect(page).toHaveURL(/around=message-050/)
  await page.screenshot({ path: 'test-results/stage2-reader-desktop.png', fullPage: true })
})

test('conversation reader covers both edges, both cursor directions, message kinds, search filters, and result jumps', async ({ page }) => {
  test.setTimeout(60_000)
  await page.goto('/conversations/conv-alex?around=message-004#message-message-004')
  await expect(page.getByText('Call · 1m 32s')).toBeVisible()
  for (const message of [
    { id: 'message-005', text: 'Payment activity — details hidden.' },
    { id: 'message-006', text: 'This message was unsent.' },
    { id: 'message-007', text: 'Message body is unavailable in the export.' },
  ]) {
    await page.goto(`/conversations/conv-alex?around=${message.id}#message-${message.id}`)
    await expect(page.getByText(message.text)).toBeVisible()
  }

  const beginningRequest = page.waitForRequest((request) => {
    const url = new URL(request.url())
    return url.pathname.endsWith('/conversations/conv-alex/messages') && url.searchParams.get('start') === 'earliest'
  })
  await page.getByRole('button', { name: 'Beginning' }).click()
  await beginningRequest
  const olderRequest = page.waitForRequest((request) => request.url().includes('/conversations/conv-alex/messages?') && request.url().includes('cursor=older-page'))
  await page.getByRole('button', { name: 'Load older now' }).click()
  await olderRequest
  const newerRequest = page.waitForRequest((request) => request.url().includes('/conversations/conv-alex/messages?') && request.url().includes('cursor=newer-page'))
  await page.getByRole('button', { name: 'Load newer now' }).click()
  await newerRequest

  await page.goto('/conversations/conv-alex?around=message-002#message-message-002')
  const latestRequest = page.waitForRequest((request) => {
    const url = new URL(request.url())
    return url.pathname.endsWith('/conversations/conv-alex/messages') && url.searchParams.get('start') === 'latest' && !url.searchParams.has('around')
  })
  await page.getByRole('button', { name: 'Latest' }).click()
  await latestRequest

  await page.getByRole('button', { name: 'Search this conversation' }).click()
  await page.getByRole('searchbox', { name: 'Search words in this history' }).fill('Français ABC')
  await page.getByLabel('Message kind').fill('generic')
  await page.getByLabel('Sender').fill('Alex Example')
  const searchRequest = page.waitForRequest((request) => {
    const url = new URL(request.url())
    return url.pathname.endsWith('/message-search') && url.searchParams.get('kind') === 'generic' && url.searchParams.get('sender') === 'Alex Example'
  })
  await page.getByRole('region', { name: 'Search this conversation' }).getByRole('button', { name: 'Search', exact: true }).click()
  await searchRequest
  const result = page.locator('.thread-search-results li').first().getByRole('button')
  await expect(result).toContainText('Jump to message')
  await result.click()
  await expect(page).toHaveURL(/around=message-001#message-message-001/)
  await expect(page.locator('#message-message-001')).toBeFocused()
})

test('conversation and media failures retry safely, and media-only threads never invent messages', async ({ page }) => {
  let conversationFailures = 0
  await page.route(/\/api\/v1\/conversations\?.*/, (route) => {
    conversationFailures += 1
    if (conversationFailures <= 2) return route.fulfill({ status: 503, json: { detail: 'Fixture conversation catalog temporarily unavailable.' } })
    return route.fulfill({ json: fixtures.conversations })
  })
  await page.goto('/conversations')
  await expect(page.getByRole('alert')).toContainText('Fixture conversation catalog temporarily unavailable.')
  await page.getByRole('button', { name: 'Try again' }).click()
  await expect(page.getByRole('link', { name: 'Alex Example' })).toBeVisible()

  let mediaFailures = 0
  await page.route(/\/api\/v1\/media\?.*/, (route) => {
    mediaFailures += 1
    if (mediaFailures <= 2) return route.fulfill({ status: 503, json: { detail: 'Fixture media catalog temporarily unavailable.' } })
    return route.fulfill({ json: fixtures.media })
  })
  await page.goto('/media')
  await expect(page.getByRole('alert')).toContainText('Fixture media catalog temporarily unavailable.')
  await page.getByRole('button', { name: 'Try again' }).click()
  await expect(page.getByText('date-night.jpg')).toBeVisible()

  const mediaOnlyConversation = fixtures.conversations.items.find((item) => item.id === 'conv-media-only')!
  await page.route('**/api/v1/conversations/conv-media-only', (route) => route.fulfill({ json: { conversation: mediaOnlyConversation, segments: [] } }))
  await page.route('**/api/v1/conversations/conv-media-only/messages*', (route) => route.fulfill({ json: { items: [], page: { limit: 100, total: 0, before_cursor: null, after_cursor: null }, focus: null } }))
  await page.goto('/conversations/conv-media-only')
  await expect(page.getByText('No messages in this conversation', { exact: true })).toBeVisible()
  await expect(page.getByText(/media-only source thread/)).toBeVisible()
  await expect(page.locator('.reader-heading__title .status-badge').first()).toHaveText(/Media Only/i)
})

test('conversation date jump preserves a visible local error when no retained message can be resolved', async ({ page }) => {
  await page.route('**/api/v1/conversations/conv-alex/messages/date-jump*', (route) => route.fulfill({ json: { requested_timestamp_ms: 1_672_617_600_000, target_timestamp_ms: null, jump: null } }))
  await page.goto('/conversations/conv-alex')
  await page.getByLabel('Jump to date').fill('2023-01-02')
  await page.getByRole('button', { name: 'Jump', exact: true }).click()
  await expect(page.getByRole('alert')).toHaveText('This conversation has no message near that date.')
  await expect(page).toHaveURL('/conversations/conv-alex')
})

test('direct-history identity corrections preserve segment-level intent in the request', async ({ page }) => {
  await page.goto('/conversations/conv-alex')
  await page.getByText('Direct-history identity').click()
  await expect(page.getByText('This history is provisional.')).toBeVisible()
  await page.getByRole('checkbox', { name: /archived_threads\/alex_example/i }).check()
  await page.getByLabel('New history title').fill('Alex archived history')
  await page.getByLabel('Reason (optional, saved to audit)').fill('Reviewed source boundaries')
  const correctionRequest = page.waitForRequest((request) => request.url().endsWith('/api/v1/conversation-corrections/split'))
  await page.getByRole('button', { name: 'Split selected segments' }).click()
  const request = await correctionRequest
  expect(request.postDataJSON()).toMatchObject({
    conversation_id: 'conv-alex',
    segment_ids: ['segment-old'],
    title: 'Alex archived history',
    identity_revision: 'a'.repeat(64),
    reason: 'Reviewed source boundaries',
  })
  await prepareFullPageScreenshot(page)
  await page.screenshot({ path: 'test-results/stage2-identity-correction.png', fullPage: true })
})

test('direct-history merge reports a stale revision and succeeds after an explicit reload', async ({ page }) => {
  let revision = 'a'.repeat(64)
  let merged = false
  const mergeRequests: unknown[] = []
  const candidateRequests: Record<string, string>[] = []
  const correctionItem = {
    id: 'correction-merge-recovered',
    action: 'merge',
    status: 'active',
    reason: 'Recovered merge accepted',
    request: {},
    before: {},
    after: {},
    effect: { conversations: 2 },
    created_at: '2026-07-20T10:00:00Z',
    reverted_at: null,
    can_revert: true,
    segments: [],
    audits: [],
  }
  const target = {
    ...fixtures.conversations.items[0],
    id: 'conv-jordan',
    title: 'Jordan Example',
    href: '/conversations/conv-jordan',
  }

  await page.route(/\/api\/v1\/conversation-corrections(?:\?.*)?$/, (route) => route.fulfill({
    json: {
      items: merged ? [correctionItem] : [],
      identity_revision: revision,
      page: { limit: 100, offset: 0, total: merged ? 1 : 0, next_offset: null },
      notice: 'Direct histories are provisional.',
    },
  }))
  await page.route(/\/api\/v1\/conversations\?.*$/, async (route) => {
    const url = new URL(route.request().url())
    if (url.searchParams.get('q') !== 'Jordan Example') return route.fallback()
    candidateRequests.push(Object.fromEntries(url.searchParams))
    return route.fulfill({
      json: {
        ...fixtures.conversations,
        items: [target],
        page: { limit: 75, total: 1, next_cursor: null },
        sort: { field: 'title', direction: 'asc' },
      },
    })
  })
  await page.route('**/api/v1/conversation-corrections/merge', async (route) => {
    mergeRequests.push(route.request().postDataJSON())
    if (mergeRequests.length === 1) {
      revision = 'b'.repeat(64)
      return route.fulfill({
        status: 409,
        json: { detail: 'Conversation identity changed while you were reviewing it. Reload and try again.' },
      })
    }
    merged = true
    return route.fulfill({
      status: 201,
      json: {
        correction_id: correctionItem.id,
        action: 'merge',
        status: 'active',
        identity_revision: 'c'.repeat(64),
        redirect_conversation_id: 'conv-alex',
        effect: correctionItem.effect,
      },
    })
  })

  const submitMerge = async (reason: string) => {
    await page.getByText('Direct-history identity').click()
    await page.getByRole('button', { name: 'Merge history' }).click()
    await page.getByLabel('Find target history').fill('Jordan Example')
    await page.getByRole('button', { name: 'Find direct histories' }).click()
    await page.getByRole('radio', { name: /Jordan Example/ }).check()
    await page.getByLabel('Reason (optional, saved to audit)').fill(reason)
    await page.getByRole('button', { name: 'Merge into selected history' }).click()
  }

  await page.goto('/conversations/conv-alex')
  await submitMerge('Reviewed matching direct histories')
  await expect(page.getByRole('alert')).toHaveText('Conversation identity changed while you were reviewing it. Reload and try again.')

  await page.reload()
  await submitMerge('Recovered merge accepted')
  await expect(page.getByRole('alert')).toHaveCount(0)
  await expect(page.getByRole('region', { name: 'Correction history' }).getByText('Recovered merge accepted')).toBeVisible()
  expect(candidateRequests).toEqual([
    { q: 'Jordan Example', conversation_type: 'direct', sort: 'title', direction: 'asc', limit: '75' },
    { q: 'Jordan Example', conversation_type: 'direct', sort: 'title', direction: 'asc', limit: '75' },
  ])
  expect(mergeRequests).toEqual([
    {
      conversation_ids: ['conv-alex', 'conv-jordan'],
      target_conversation_id: 'conv-jordan',
      identity_revision: 'a'.repeat(64),
      reason: 'Reviewed matching direct histories',
    },
    {
      conversation_ids: ['conv-alex', 'conv-jordan'],
      target_conversation_id: 'conv-jordan',
      identity_revision: 'b'.repeat(64),
      reason: 'Recovered merge accepted',
    },
  ])
})

test('direct-history correction history reverts with the current identity revision', async ({ page }) => {
  let reverted = false
  const revision = 'd'.repeat(64)
  const revertRequests: unknown[] = []
  const correction = {
    id: 'correction-active-merge',
    action: 'merge',
    status: reverted ? 'reverted' : 'active',
    reason: 'These fixture histories belong together.',
    request: {},
    before: {},
    after: {},
    effect: { conversations: 2 },
    created_at: '2026-07-20T10:00:00Z',
    reverted_at: null,
    can_revert: true,
    segments: [],
    audits: [],
  }
  await page.route(/\/api\/v1\/conversation-corrections(?:\?.*)?$/, (route) => route.fulfill({
    json: {
      items: [{ ...correction, status: reverted ? 'reverted' : 'active', can_revert: !reverted, reverted_at: reverted ? '2026-07-20T10:05:00Z' : null }],
      identity_revision: revision,
      page: { limit: 100, offset: 0, total: 1, next_offset: null },
      notice: 'Direct histories are provisional.',
    },
  }))
  await page.route('**/api/v1/conversation-corrections/correction-active-merge/revert', async (route) => {
    revertRequests.push(route.request().postDataJSON())
    reverted = true
    return route.fulfill({
      json: {
        correction_id: 'correction-active-merge',
        action: 'revert',
        status: 'reverted',
        identity_revision: 'e'.repeat(64),
        redirect_conversation_id: 'conv-alex',
        effect: { conversations: 2 },
      },
    })
  })

  await page.goto('/conversations/conv-alex')
  await page.getByText('Direct-history identity').click()
  const history = page.getByRole('region', { name: 'Correction history' })
  await expect(history.getByText('These fixture histories belong together.')).toBeVisible()
  await history.getByRole('button', { name: 'Undo this correction' }).click()
  await expect(history.locator('.status-badge')).toHaveText('Reverted')
  await expect(history.getByRole('button', { name: 'Undo this correction' })).toHaveCount(0)
  expect(revertRequests).toEqual([{ identity_revision: revision }])
})

test('media gallery reconciles unique files, placeholders, and occurrence jumps', async ({ page }) => {
  await page.goto('/media')
  await expect(page.getByRole('heading', { name: 'Media' })).toBeVisible()
  await expect(page.getByText('File missing from export')).toBeVisible()
  await expect(page.getByText('orphan.bin').last()).toBeVisible()
  await expect(page.getByText('Unknown date').first()).toBeVisible()
  const gallery = page.getByRole('list', { name: 'Media gallery' })
  await expect(gallery).toHaveAttribute('data-loaded-item-count', '6')
  await expect(gallery.getByRole('listitem').first()).toHaveAttribute('aria-posinset', '1')
  await expect(gallery.getByRole('listitem').first()).toHaveAttribute('aria-setsize', '6')
  const visibility = page.getByRole('group', { name: 'Media visibility' })
  await expect(visibility.getByRole('button', { name: 'Resolved first' })).toHaveAttribute('aria-pressed', 'true')
  await expect(page.getByText(/Resolved files appear first within the loaded results/)).toBeVisible()
  await visibility.getByRole('button', { name: 'Show all' }).click()
  await expect(visibility.getByRole('button', { name: 'Show all' })).toHaveAttribute('aria-pressed', 'true')
  const missingRequest = page.waitForRequest((request) => new URL(request.url()).searchParams.get('availability') === 'missing')
  await visibility.getByRole('button', { name: 'Missing only' }).click()
  await missingRequest
  await expect(visibility.getByRole('button', { name: 'Missing only' })).toHaveAttribute('aria-pressed', 'true')
  await visibility.getByRole('button', { name: 'Resolved first' }).click()
  await expect(page.getByText('Why these files have no source jump')).toBeVisible()
  const activityCard = page.locator('.media-card').filter({ hasText: 'birthday.mp4' })
  const activityJump = activityCard.getByRole('link', { name: 'Jump to activity' })
  await expect(activityCard).not.toContainText('Unlinked')
  await expect(activityCard).toContainText('Birthday album')
  await expect(activityJump).toHaveAttribute('href', fixtures.media.items[1].jump.href)
  await expect(page.getByText('library/media/unlinked/unknown-date/unlinked--orphan--media-unlinked.bin')).toBeVisible()
  await expect(page.getByRole('button', { name: 'Copy library path' })).toBeVisible()
  await expect(page.getByRole('link', { name: 'Jump to message' }).first()).toHaveAttribute('href', /around=message-002/)

  await page.getByRole('button', { name: 'Unique files' }).click()
  await expect(page.getByText('2 occurrences')).toBeVisible()
  await page.locator('.media-card__actions').getByRole('button', { name: 'Occurrences' }).click()
  await expect(page.getByRole('dialog', { name: 'Every occurrence' })).toBeVisible()
  await expect(page.getByText('2 retained occurrences')).toBeVisible()
  await page.getByRole('button', { name: 'Close occurrences' }).click()

  const mediaRequest = page.waitForRequest((request) => {
    const url = new URL(request.url())
    return url.pathname === '/api/v1/media' && url.searchParams.get('conversation_q') === 'Alex' && url.searchParams.get('type') === 'audio'
  })
  await page.getByLabel('Conversation title').fill('Alex')
  await page.getByRole('combobox', { name: 'Media type', exact: true }).selectOption('audio')
  await page.getByRole('button', { name: 'Apply filters' }).click()
  await mediaRequest
  await page.getByRole('button', { name: 'Table' }).click()
  await expect(page.getByRole('table', { name: 'Media files' })).toBeVisible()
  await expect(page.getByRole('table', { name: 'Media files' })).toHaveAttribute('aria-rowcount', '6')
  await page.screenshot({ path: 'test-results/stage2-media-desktop.png', fullPage: true })
})

test('activity-linked media jumps to its exact retained timeline source', async ({ page }) => {
  await page.goto('/media')
  const activityCard = page.locator('.media-card').filter({ hasText: 'birthday.mp4' })
  await expect(activityCard).not.toContainText('Unlinked')
  await activityCard.getByRole('link', { name: 'Jump to activity' }).click()
  await expect(page).toHaveURL(
    /\/timeline\?around=activity%3Aactivity-album#activity-activity-album$/,
  )
})

test('media catalog serializes every detailed filter and every occurrence and unique-file sort', async ({ page }) => {
  test.setTimeout(60_000)
  await page.goto('/media')
  await page.getByRole('searchbox', { name: 'Filename', exact: true }).fill('fixture')
  await page.getByRole('combobox', { name: 'Media type', exact: true }).selectOption('document')
  await page.getByRole('combobox', { name: 'Availability', exact: true }).selectOption('missing')
  await page.getByLabel('Attachment kind').fill('files')
  await page.getByLabel('Conversation title').fill('Alex')
  await page.getByLabel('Conversation type').selectOption('direct')
  await page.getByLabel('Source category').fill('archived_threads')
  await page.getByLabel('From date').fill('2022-01-01')
  await page.getByLabel('To date').fill('2024-12-31')
  await page.getByLabel('Minimum size (MB)').fill('0.25')
  await page.getByLabel('Maximum size (MB)').fill('20')
  await page.getByRole('combobox', { name: 'Sort by', exact: true }).selectOption('size')
  await page.getByLabel('Direction').selectOption('asc')
  const fullFilterRequest = page.waitForRequest((request) => {
    const url = new URL(request.url())
    return url.pathname === '/api/v1/media' && url.searchParams.get('sort') === 'size' && url.searchParams.get('direction') === 'asc'
  })
  await page.getByRole('button', { name: 'Apply filters' }).click()
  const query = new URL((await fullFilterRequest).url()).searchParams
  expect(Object.fromEntries(query)).toMatchObject({
    view: 'occurrences', q: 'fixture', type: 'document', availability: 'missing', kind: 'files', conversation_q: 'Alex', conversation_type: 'direct', source_category: 'archived_threads', min_size: '262144', max_size: '20971520', sort: 'size', direction: 'asc', limit: '60',
  })
  expect(Number(query.get('from_ms'))).toBeGreaterThan(0)
  expect(Number(query.get('to_ms'))).toBeGreaterThan(Number(query.get('from_ms')))

  for (const sort of ['date', 'type', 'filename', 'conversation', 'availability']) {
    const requestPromise = page.waitForRequest((request) => {
      const url = new URL(request.url())
      return url.pathname === '/api/v1/media' && url.searchParams.get('sort') === sort && url.searchParams.get('view') === 'occurrences'
    })
    await page.getByRole('combobox', { name: 'Sort by', exact: true }).selectOption(sort)
    await page.getByRole('button', { name: 'Apply filters' }).click()
    await requestPromise
  }
  const uniqueRequest = page.waitForRequest((request) => request.url().includes('/api/v1/media?') && request.url().includes('view=unique'))
  await page.getByRole('button', { name: 'Unique files' }).click()
  await uniqueRequest
  const occurrenceSort = page.waitForRequest((request) => request.url().includes('/api/v1/media?') && request.url().includes('sort=occurrences'))
  await page.getByRole('combobox', { name: 'Sort by', exact: true }).selectOption('occurrences')
  await page.getByRole('button', { name: 'Apply filters' }).click()
  await occurrenceSort
})

test('media facets, exact-conversation clearing, pagination, and empty recovery remain interactive', async ({ page }) => {
  await page.route(/\/api\/v1\/media\?.*/, async (route) => {
    const url = new URL(route.request().url())
    if (url.searchParams.get('q') === 'no media matches') {
      return route.fulfill({ json: { ...fixtures.media, items: [], page: { limit: 60, total: 0, next_cursor: null } } })
    }
    if (url.searchParams.get('cursor') === 'media-page-two') {
      const item = { ...fixtures.media.items[1], id: 'media-page-two', view_key: 'occurrence:media-page-two', filename: 'page-two-video.mp4' }
      return route.fulfill({ json: { ...fixtures.media, items: [item], page: { limit: 60, total: 7, next_cursor: null } } })
    }
    return route.fulfill({ json: { ...fixtures.media, page: { limit: 60, total: 7, next_cursor: 'media-page-two' } } })
  })
  await page.goto('/media?conversation=conv-alex')
  await expect(page.getByText(/Exact conversation filter:/)).toBeVisible()
  await page.getByRole('button', { name: 'Clear exact conversation filter' }).click()
  const clearRequest = page.waitForRequest((request) => {
    const url = new URL(request.url())
    return url.pathname === '/api/v1/media' && !url.searchParams.has('conversation_id')
  })
  await page.getByRole('button', { name: 'Apply filters' }).click()
  await clearRequest

  for (const label of ['Photo', 'Video', 'Audio', 'Gif', 'Sticker', 'Document', 'Other']) {
    const chip = page.getByRole('button', { name: new RegExp(`^${label} `) })
    await chip.click()
    await expect(chip).toHaveAttribute('aria-pressed', 'true')
    await chip.click()
    await expect(chip).toHaveAttribute('aria-pressed', 'false')
  }
  for (const label of ['Resolved', 'Missing', 'Unlinked', 'External']) {
    const chip = page.getByRole('region', { name: 'Media facets' }).getByRole('button', { name: new RegExp(`^${label} `) })
    await chip.click()
    await expect(chip).toHaveAttribute('aria-pressed', 'true')
    await chip.click()
    await expect(chip).toHaveAttribute('aria-pressed', 'false')
  }

  const secondPage = page.waitForRequest((request) => request.url().includes('cursor=media-page-two'))
  await page.getByRole('button', { name: 'Load more media' }).click()
  await secondPage
  await expect(page.getByText('page-two-video.mp4')).toBeVisible()
  await expect(page.getByText('7 of 7 occurrences loaded')).toBeVisible()

  await page.getByRole('searchbox', { name: 'Filename', exact: true }).fill('no media matches')
  await page.getByRole('button', { name: 'Apply filters' }).click()
  await expect(page.getByText('No media matches', { exact: true })).toBeVisible()
  await page.getByRole('button', { name: 'Reset filters' }).click()
  await expect(page.getByText('date-night.jpg')).toBeVisible()
})

test('media occurrence dialog traps focus, closes with Escape, and returns focus to its trigger', async ({ page }) => {
  await page.goto('/media')
  await page.getByRole('button', { name: 'Unique files' }).click()
  const trigger = page.locator('.media-card__actions').getByRole('button', { name: 'Occurrences' })
  await trigger.focus()
  await trigger.click()
  const dialog = page.getByRole('dialog', { name: 'Every occurrence' })
  await expect(dialog).toBeVisible()
  await expect(page.getByRole('button', { name: 'Close occurrences' })).toBeFocused()
  await page.keyboard.press('Shift+Tab')
  await expect(dialog.getByRole('link', { name: 'Jump to message' }).last()).toBeFocused()
  await page.keyboard.press('Tab')
  await expect(page.getByRole('button', { name: 'Close occurrences' })).toBeFocused()
  await page.keyboard.press('Escape')
  await expect(dialog).toHaveCount(0)
  await expect(trigger).toBeFocused()
})

test('media jumps land on and focus the exact retained occurrence, including missing files', async ({ page }) => {
  await page.goto('/media')
  const resolvedCard = page.locator('.media-card').filter({ hasText: 'date-night.jpg' })
  await resolvedCard.getByRole('link', { name: 'Jump to message' }).click()
  await expect(page).toHaveURL(/\/conversations\/conv-alex\?around=message-002&media=media-photo#message-message-002/)
  const resolvedOccurrence = page.locator('#attachment-media-photo')
  await expect(resolvedOccurrence).toHaveClass(/message-attachment--focused/)
  await expect(resolvedOccurrence).toBeFocused()
  await expect(resolvedOccurrence.getByRole('img', { name: 'date-night.jpg' })).toBeVisible()

  await page.goto('/media')
  const missingCard = page.locator('.media-card').filter({ hasText: 'missing-photo.jpg' })
  await missingCard.getByRole('link', { name: 'Jump to message' }).click()
  await expect(page).toHaveURL(/\/conversations\/conv-alex\?around=message-101&media=media-missing#message-message-101/)
  const missingOccurrence = page.locator('#attachment-media-missing')
  await expect(missingOccurrence).toHaveClass(/message-attachment--focused/)
  await expect(missingOccurrence).toBeFocused()
  await expect(missingOccurrence.getByRole('img', { name: /missing-photo\.jpg: File missing from export/ })).toBeVisible()
})

test('Stage 2 remains usable on a narrow mobile viewport', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await page.goto('/media')
  await expect(page.getByRole('heading', { name: 'Media' })).toBeVisible()
  await expect(page.locator('.media-card').first()).toBeVisible()
  const mediaViewport = page.getByRole('list', { name: 'Media gallery' })
  await expect(page.locator('#media-results-scroll-hint')).toBeVisible()
  await expect(mediaViewport).toHaveAttribute('aria-describedby', 'media-results-scroll-hint')
  expect(await mediaViewport.evaluate((element) => getComputedStyle(element).scrollbarGutter)).toContain('stable')
  const bodyWidth = await page.locator('body').evaluate((element) => ({ scroll: element.scrollWidth, client: element.clientWidth }))
  expect(bodyWidth.scroll).toBeLessThanOrEqual(bodyWidth.client + 1)
  await page.screenshot({ path: 'test-results/stage2-media-mobile.png', fullPage: true })

  await page.getByRole('button', { name: 'Open navigation' }).click()
  await page.getByRole('link', { name: 'Conversations' }).click()
  await expect(page.getByRole('heading', { name: 'Conversations' })).toBeVisible()
  await expect(page.locator('#primary-navigation')).not.toHaveClass(/sidebar--open/)
  await expect(page.getByRole('link', { name: 'Alex Example' })).toBeVisible()
  const conversationFilterToggle = page.getByRole('button', { name: 'Show detailed filters' })
  await expect(conversationFilterToggle).toHaveAttribute('aria-expanded', 'false')
  await expect(page.locator('#conversation-detailed-filters')).toBeHidden()
  const conversationViewport = page.locator('.conversation-table__viewport')
  await expect(page.locator('#conversation-results-scroll-hint')).toBeVisible()
  await expect(conversationViewport).toHaveAttribute('aria-describedby', 'conversation-results-scroll-hint')
  expect(await conversationViewport.evaluate((element) => getComputedStyle(element).scrollbarGutter)).toContain('stable')
  const resultsTop = await page.locator('.conversation-results').evaluate((element) => element.getBoundingClientRect().top + window.scrollY)
  expect(resultsTop).toBeLessThan(1_100)
  await conversationFilterToggle.click()
  await expect(page.locator('#conversation-detailed-filters')).toBeVisible()
  await expect(page.getByLabel('Minimum photos')).toBeVisible()
  const conversationBodyWidth = await page.locator('body').evaluate((element) => ({ scroll: element.scrollWidth, client: element.clientWidth }))
  expect(conversationBodyWidth.scroll).toBeLessThanOrEqual(conversationBodyWidth.client + 1)
  await page.screenshot({ path: 'test-results/stage2-conversations-mobile.png', fullPage: true })

  await page.getByRole('link', { name: 'Alex Example' }).click()
  const readerMobileLayout = await page.getByRole('feed', { name: 'Conversation messages' }).evaluate((feed) => {
    const edgeControl = feed.previousElementSibling?.querySelector('button')
    const firstMessage = feed.querySelector('.message-row')
    if (!(edgeControl instanceof HTMLElement) || !(firstMessage instanceof HTMLElement)) {
      throw new Error('Expected the reader edge control and a rendered message')
    }
    const controlBounds = edgeControl.getBoundingClientRect()
    const feedBounds = feed.getBoundingClientRect()
    const feedStyle = getComputedStyle(feed)
    const messageStyle = getComputedStyle(firstMessage)
    return {
      controlBottom: controlBounds.bottom,
      feedTop: feedBounds.top,
      scrollPaddingTop: Number.parseFloat(feedStyle.scrollPaddingTop),
      messageScrollMarginTop: Number.parseFloat(messageStyle.scrollMarginTop),
    }
  })
  expect(readerMobileLayout.controlBottom).toBeLessThanOrEqual(readerMobileLayout.feedTop + 1)
  expect(readerMobileLayout.scrollPaddingTop).toBeGreaterThan(0)
  expect(readerMobileLayout.messageScrollMarginTop).toBeGreaterThan(0)
  await page.getByText('Direct-history identity').click()
  await expect(page.getByRole('button', { name: 'Split selected segments' })).toBeVisible()
  const correctionBodyWidth = await page.locator('body').evaluate((element) => ({ scroll: element.scrollWidth, client: element.clientWidth }))
  expect(correctionBodyWidth.scroll).toBeLessThanOrEqual(correctionBodyWidth.client + 1)
  await prepareFullPageScreenshot(page)
  await page.screenshot({ path: 'test-results/stage2-identity-correction-mobile.png', fullPage: true })
})

test('Stage 2 core pages are overflow-free and axe-clean at desktop and mobile sizes', async ({ page }) => {
  test.setTimeout(60_000)
  for (const viewport of [{ width: 1440, height: 1000 }, { width: 390, height: 844 }]) {
    await page.setViewportSize(viewport)
    for (const path of ['/conversations', '/conversations/conv-alex', '/media']) {
      await page.goto(path)
      await expect(page.locator('main h1')).toBeVisible()
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

test('Stage 2 direct-history reader has deterministic visual and accessibility baselines', async ({ page }) => {
  await page.emulateMedia({ reducedMotion: 'reduce' })
  await page.goto('/conversations/conv-alex')
  await page.getByText('Direct-history identity').click()
  const main = page.getByRole('main')
  const pageContent = page.locator('.conversation-reader-page')
  await expect(main.getByText('This history is provisional.')).toBeVisible()
  await page.addStyleTag({ content: '.topbar { visibility: hidden !important; }' })
  await expect(pageContent).toHaveScreenshot('stage2-direct-history-main.png', { animations: 'disabled', timeout: 15_000 })
  await expect(pageContent).toMatchAriaSnapshot({ name: 'stage2-direct-history-main.aria.yml' })
})

test('conversation dashboard and media gallery have deterministic visual and ARIA baselines', async ({ page }) => {
  await page.emulateMedia({ reducedMotion: 'reduce' })
  for (const route of [
    { path: '/conversations', slug: 'conversations' },
    { path: '/media', slug: 'media' },
  ]) {
    await page.goto(route.path)
    await page.addStyleTag({ content: '.topbar { visibility: hidden !important; }' })
    const main = page.getByRole('main')
    await expect(main.getByRole('heading', { level: 1 })).toBeVisible()
    if (route.path === '/conversations') await expect(main.getByRole('link', { name: 'Alex Example' })).toBeVisible()
    if (route.path === '/media') await expect(main.getByText('date-night.jpg')).toBeVisible()
    await expect(main).toHaveScreenshot(`stage2-${route.slug}-main.png`, { animations: 'disabled', timeout: 15_000 })
    await expect(main).toMatchAriaSnapshot({ name: `stage2-${route.slug}-main.aria.yml` })
  }
})
