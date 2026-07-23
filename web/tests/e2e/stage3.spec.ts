import { expect, type Page, test } from '@playwright/test'
import AxeBuilder from '@axe-core/playwright'

const searchResult = {
  id: 'window:argument-one',
  window_id: 'argument-one',
  entity_type: 'message_window',
  source_id: 'message-one',
  rank: 1,
  score: 0.83,
  score_kind: 'cosine_similarity',
  timestamp_ms: 1_704_153_600_000,
  excerpt: 'Alex: This keeps happening.\nCasey: I disagree, and I need you to listen.',
  sender: null,
  conversation: { id: 'conv-alex', title: 'Alex Example' },
  participants: ['alex example', 'casey example'],
  source_category: 'inbox',
  coverage: {
    key: 'messages',
    label: 'Messages',
    status: 'complete',
    raw_count: 1,
    logical_count: 1,
    description: 'Fixture message coverage is complete.',
    source_paths: { fixture: true },
    absence_is_not_zero_history: false,
  },
  confidence: null,
  confidence_kind: 'not_applicable',
  confidence_calibrated: false,
  availability: 'resolved',
  deletion_state: 'retained',
  provenance: {
    entity_type: 'message_window',
    source_id: 'message-one',
    window_id: 'argument-one',
    source_category: 'inbox',
    representation: 'semantic_window',
  },
  window: { start_message_id: 'message-one', end_message_id: 'message-six', message_ids: ['message-one', 'message-six'], turn_count: 6, first_timestamp_ms: 1_704_153_600_000, last_timestamp_ms: 1_704_153_900_000 },
  jump: { href: '/conversations/conv-alex?around=message-one#message-message-one', conversation_id: 'conv-alex', message_id: 'message-one', fragment: 'message-message-one' },
}

function resultForSearch(url: URL) {
  const mode = url.searchParams.get('mode') ?? 'keyword'
  const entityType = url.searchParams.get('entity_type') ?? 'all'
  const scoreKind = mode === 'exact' ? 'exact' : mode === 'keyword' ? 'bm25' : mode === 'hybrid' ? 'rrf' : 'cosine_similarity'
  if (entityType === 'audio_transcript') {
    return {
      ...searchResult,
      id: 'audio:transcript-one',
      window_id: null,
      entity_type: 'audio_transcript',
      source_id: 'transcript-one',
      score_kind: scoreKind,
      excerpt: 'Fixture voice note about repairing a disagreement.',
      sender: 'Alex Example',
      coverage: { ...searchResult.coverage, key: 'audio_transcriptions', label: 'Audio transcriptions' },
      confidence: 0.91,
      confidence_kind: 'audio_decoder_score',
      provenance: {
        entity_type: 'audio_transcript',
        source_id: 'transcript-one',
        window_id: null,
        source_category: 'inbox',
        representation: 'derived_text',
      },
      window: null,
      occurrences: [{ occurrence_id: 'occurrence-one', conversation_id: 'conv-alex', conversation_title: 'Alex Example', message_id: 'message-002', timestamp_ms: searchResult.timestamp_ms, source_category: 'inbox', start_ms: 1_250, href: '/conversations/conv-alex?around=message-002&media=occurrence-one&t=1250#message-message-002' }],
      occurrence_page: { limit: 1, offset: 0, returned: 1, total: 2, next_offset: 1 },
    }
  }
  if (entityType === 'attachment_text') {
    return {
      ...searchResult,
      id: 'attachment-text:document-one',
      window_id: null,
      entity_type: 'attachment_text',
      source_id: 'document-one',
      score_kind: scoreKind,
      excerpt: 'Fixture extracted document text about resolving a tense exchange.',
      sender: null,
      coverage: { ...searchResult.coverage, key: 'attachment_text', label: 'Attachment text' },
      provenance: {
        entity_type: 'attachment_text',
        source_id: 'document-one',
        window_id: null,
        source_category: 'inbox',
        representation: 'derived_text',
      },
      window: null,
    }
  }
  if (entityType === 'activity') {
    return {
      ...searchResult,
      id: 'activity:comment-one',
      window_id: null,
      entity_type: 'activity',
      source_id: 'comment-one',
      score_kind: scoreKind,
      excerpt: 'Fixture comment about resolving a disagreement.',
      sender: 'Alex Example',
      subtype: 'group_comment',
      coverage: { ...searchResult.coverage, key: 'posts', label: 'Posts and comments' },
      provenance: { entity_type: 'activity', source_id: 'comment-one', window_id: null, source_category: 'groups', representation: 'source_record' },
      window: null,
      jump: { href: '/timeline?around=activity:comment-one#activity-comment-one', fragment: 'activity-comment-one' },
    }
  }
  return { ...searchResult, score_kind: scoreKind }
}

async function mockStageThreeApi(page: Page) {
  const saved: Array<{ id: string; name: string; query: Record<string, unknown> }> = []
  const feedback: Array<{ id: string; query_sha256: string; query: string; search_mode: string; result_id: string; label: string; evaluation_split?: string; notes?: string; filter_scope?: Record<string, unknown>; created_at: string }> = []
  const audits: unknown[] = []
  await page.route('**/api/v1/**', async (route) => {
    const request = route.request()
    const url = new URL(request.url())
    const path = url.pathname
    if (path === '/api/v1/health') return route.fulfill({ json: { status: 'ok', catalog_ready: true, catalog_state: 'ready' } })
    if (path === '/api/v1/settings') return route.fulfill({ json: { timezone: 'America/New_York', cloud_enabled: true, image_semantic_stage_enabled: false } })
    if (path === '/api/v1/search/status') return route.fulfill({ json: { ready: true, active: { id: 'semantic-one', status: 'ready', model_name: 'fixture/embedding', model_revision: 'fixture-v1', dimensions: 4, window_version: 'v1', window_count: 25, audit: {} }, latest_job: null } })
    if (path === '/api/v1/search') {
      const query = url.searchParams.get('q') ?? ''
      const cursor = url.searchParams.get('cursor')
      const empty = query === 'nothing in fixture'
      const paged = query === 'paged search'
      const result = { ...resultForSearch(url), ...(cursor ? { id: 'window:argument-two', window_id: 'argument-two', source_id: 'message-two', excerpt: 'A second fixture result.', rank: 2 } : {}) }
      return route.fulfill({ json: {
        items: empty ? [] : [result],
        page: { limit: 50, returned: empty ? 0 : 1, candidate_count: empty ? 0 : paged ? 2 : 1, result_cap: 1000, next_cursor: paged && !cursor ? 'search-page-two' : null },
        query,
        mode: url.searchParams.get('mode'),
        filters: {
          entity_type: url.searchParams.get('entity_type'),
          conversation_id: url.searchParams.get('conversation_id'),
          participant: url.searchParams.get('participant'),
          source_category: url.searchParams.get('source_category'),
          from_ms: url.searchParams.get('from_ms'),
          to_ms: url.searchParams.get('to_ms'),
        },
        index_config: { id: 'semantic-one', model_name: 'fixture/embedding' },
      } })
    }
    if (path === '/api/v1/saved-searches' && request.method() === 'GET') return route.fulfill({ json: { items: saved } })
    if (path === '/api/v1/saved-searches' && request.method() === 'POST') {
      const body = request.postDataJSON() as { name: string; query: Record<string, unknown> }
      const item = { id: 'saved-one', ...body }
      saved.push(item)
      return route.fulfill({ status: 201, json: item })
    }
    if (path.startsWith('/api/v1/saved-searches/') && request.method() === 'DELETE') {
      const id = decodeURIComponent(path.split('/').at(-1) ?? '')
      const index = saved.findIndex((item) => item.id === id)
      if (index >= 0) saved.splice(index, 1)
      return route.fulfill({ status: 204, body: '' })
    }
    if (path === '/api/v1/search/feedback' && request.method() === 'GET') return route.fulfill({ json: { items: feedback, page: { limit: 20, returned: feedback.length, next_cursor: feedback.length && !url.searchParams.has('cursor') ? 'evaluation-page-two' : null } } })
    if (path === '/api/v1/search/feedback') {
      const body = request.postDataJSON() as { query: string; search_mode: string; result_id: string; label: string; evaluation_split?: string; notes?: string }
      const item = { id: 'feedback-one', query_sha256: 'hash', ...body, created_at: '2026-07-20T10:00:00Z' }
      feedback.splice(0, feedback.length, item)
      return route.fulfill({ status: 201, json: item })
    }
    if (path.startsWith('/api/v1/search/feedback/') && request.method() === 'PATCH') {
      const id = decodeURIComponent(path.split('/').at(-1) ?? '')
      const body = request.postDataJSON() as { label: string; evaluation_split?: string | null; notes?: string | null }
      const item = feedback.find((entry) => entry.id === id)
      if (!item) return route.fulfill({ status: 404, json: { detail: 'Evaluation not found' } })
      Object.assign(item, body, { evaluation_split: body.evaluation_split ?? undefined, notes: body.notes ?? undefined })
      return route.fulfill({ json: item })
    }
    if (path.startsWith('/api/v1/search/feedback/') && request.method() === 'DELETE') {
      const id = decodeURIComponent(path.split('/').at(-1) ?? '')
      const index = feedback.findIndex((item) => item.id === id)
      if (index >= 0) feedback.splice(index, 1)
      return route.fulfill({ status: 204, body: '' })
    }
    if (path === '/api/v1/search/derived-occurrences') return route.fulfill({ json: { items: [{ occurrence_id: 'occurrence-two', conversation_id: 'conv-family', conversation_title: 'Family Plans', message_id: 'family-22', timestamp_ms: searchResult.timestamp_ms + 1_000, source_category: 'inbox', start_ms: 2_500, href: '/conversations/conv-family?around=family-22&media=occurrence-two&t=2500#message-family-22' }], page: { limit: 20, offset: 1, returned: 1, total: 2, next_offset: null } } })
    if (path === '/api/v1/rag/audits') {
      const secondPage = url.searchParams.get('cursor') === 'rag-audit-page-two'
      const items = secondPage ? [{ id: 'audit-older', status: 'completed', model: 'gpt-4.1-mini', query_sha256: 'older-hash', search_mode: 'semantic', retrieved_ids: ['window:older-result'], citation_ids: [], usage: { input_tokens: 8, output_tokens: 2, total_tokens: 10 }, created_at: '2026-07-19T10:00:00Z' }] : audits
      return route.fulfill({ json: { items, page: { limit: 20, returned: items.length, next_cursor: !secondPage && audits.length ? 'rag-audit-page-two' : null } } })
    }
    if (path === '/api/v1/rag/answer') {
      const body = request.postDataJSON() as { question: string; search_mode: string }
      audits.splice(0, audits.length, { id: 'audit-one', status: 'completed', model: 'gpt-5.6-sol', query_sha256: 'hash', search_mode: body.search_mode, retrieved_ids: [searchResult.id], citation_ids: [searchResult.id], remote_response_id: 'response-local-audit-one', usage: { input_tokens: 20, output_tokens: 7, total_tokens: 27 }, created_at: '2026-07-20T10:00:00Z' })
      return route.fulfill({ json: { answer: 'The cited exchange appears to be a disagreement.', citation_ids: [searchResult.id], insufficient_evidence: false, sources: [searchResult], audit_id: 'audit-one', model: 'gpt-5.6-sol' } })
    }
    return route.fulfill({ status: 404, json: { detail: `Unhandled ${request.method()} ${path}` } })
  })
}

test.beforeEach(async ({ page }) => {
  await mockStageThreeApi(page)
})

test('pristine search stays idle until a nonempty query is submitted', async ({ page }) => {
  let searchRequests = 0
  page.on('request', (request) => {
    if (new URL(request.url()).pathname === '/api/v1/search') searchRequests += 1
  })

  await page.goto('/search')
  await expect(page.getByRole('heading', { name: 'Search the archive' })).toBeVisible()
  await expect(page.getByText('Searching the local archive')).toHaveCount(0)
  expect(searchRequests).toBe(0)
})

test('semantic search, feedback, saved search, RAG, and jumps form one traceable workflow', async ({ page }) => {
  await page.goto('/search')
  await expect(page.getByRole('heading', { name: 'Search the archive' })).toBeVisible()
  await expect(page.locator('.topbar__title > span')).toHaveText('Stage 3')
  await expect(page.getByText('fixture/embedding · 25 searchable windows')).toBeVisible()
  await page.getByText('Semantic', { exact: true }).click()
  await page.getByLabel('What do you want to find?').fill('argument')
  await page.getByText('Scope and date filters').click()
  await page.getByLabel('Record type').selectOption('message')
  await page.getByLabel('Participant or actor').fill('Alex Example')

  const searchRequest = page.waitForRequest((request) => {
    const url = new URL(request.url())
    return url.pathname === '/api/v1/search' && url.searchParams.get('mode') === 'semantic' && url.searchParams.get('participant') === 'Alex Example'
  })
  await page.getByRole('button', { name: 'Search archive' }).click()
  await searchRequest
  await expect(page.getByText(/This keeps happening/)).toBeVisible()
  await expect(page.getByRole('link', { name: 'Jump to source' })).toHaveAttribute('href', searchResult.jump.href)

  await page.getByLabel('Saved search name').fill('Arguments with Alex')
  await page.getByRole('button', { name: 'Save' }).click()
  await expect(page.getByText('Arguments with Alex')).toBeVisible()
  await page.getByRole('button', { name: 'Mark result 1 relevant' }).click()
  await expect(page.getByText('Marked Relevant')).toBeVisible()

  await page.getByRole('button', { name: 'Generate cited answer' }).click()
  await expect(page.getByText('The cited exchange appears to be a disagreement.')).toBeVisible()
  const answer = page.locator('.rag-answer')
  await expect(answer.getByRole('link', { name: 'Alex Example' })).toHaveAttribute('href', searchResult.jump.href)
  await expect(answer.getByText(/audit audit-one/)).toBeVisible()
  const ragPanel = page.locator('.rag-panel')
  await ragPanel.getByText(/Local RAG audit history/).click()
  await ragPanel.getByText('Source IDs and usage').click()
  const detail = ragPanel.locator('.rag-audit-detail')
  await expect(detail.getByText(searchResult.id)).toHaveCount(2)
  await expect(detail.getByText('Input Tokens')).toBeVisible()
  await expect(detail.getByText('20')).toBeVisible()
  const auditPages = ragPanel.getByRole('navigation', { name: 'RAG audit history pages' })
  await auditPages.getByRole('button', { name: 'Next page' }).click()
  await expect(ragPanel.getByText(/gpt-4.1-mini/)).toBeVisible()
  await auditPages.getByRole('button', { name: 'Previous page' }).click()
  await expect(ragPanel.locator('.rag-audits').getByText(/gpt-5.6-sol/)).toBeVisible()
  await page.screenshot({ path: 'test-results/stage3-search-desktop.png', fullPage: true })
})

test('all search modes and derived record filters preserve the exact request and result contract', async ({ page }) => {
  await page.goto('/search')
  await expect(page.getByRole('heading', { name: 'Search the archive' })).toBeVisible()
  await expect(page.getByText('fixture/embedding · 25 searchable windows')).toBeVisible()
  await page.getByLabel('What do you want to find?').fill('argument')
  await page.getByText('Scope and date filters').click()

  const cases = [
    { label: 'Keyword', mode: 'keyword', entityType: 'all', resultType: 'message_window', excerpt: 'This keeps happening', confidence: null, representation: 'semantic_window' },
    { label: 'Exact literal', mode: 'exact', entityType: 'message', resultType: 'message_window', excerpt: 'This keeps happening', confidence: null, representation: 'semantic_window' },
    { label: 'Semantic', mode: 'semantic', entityType: 'audio_transcript', resultType: 'audio_transcript', excerpt: 'Fixture voice note', confidence: 0.91, representation: 'derived_text' },
    { label: 'Hybrid', mode: 'hybrid', entityType: 'attachment_text', resultType: 'attachment_text', excerpt: 'Fixture extracted document text', confidence: null, representation: 'derived_text' },
    { label: 'Keyword', mode: 'keyword', entityType: 'activity', resultType: 'activity', excerpt: 'Fixture comment', confidence: null, representation: 'source_record' },
  ] as const

  for (const searchCase of cases) {
    const mode = page.getByRole('radio', { name: new RegExp(`^${searchCase.label}`) })
    await expect(mode).toBeEnabled()
    await mode.check({ force: true })
    await page.getByLabel('Record type').selectOption(searchCase.entityType)
    const responsePromise = page.waitForResponse((response) => {
      const url = new URL(response.url())
      return url.pathname === '/api/v1/search'
        && url.searchParams.get('mode') === searchCase.mode
        && url.searchParams.get('entity_type') === searchCase.entityType
    })
    await page.getByRole('button', { name: 'Search archive' }).click()
    const response = await responsePromise
    const requestUrl = new URL(response.request().url())
    expect(Object.fromEntries(requestUrl.searchParams)).toEqual({
      q: 'argument',
      mode: searchCase.mode,
      entity_type: searchCase.entityType,
      limit: '50',
    })

    const payload = await response.json() as { items: Array<Record<string, unknown>> }
    expect(payload.items[0]).toMatchObject({
      entity_type: searchCase.resultType,
      coverage: { status: 'complete', absence_is_not_zero_history: false },
      confidence: searchCase.confidence,
      confidence_kind: searchCase.confidence == null ? 'not_applicable' : 'audio_decoder_score',
      confidence_calibrated: false,
      availability: 'resolved',
      deletion_state: 'retained',
      provenance: { representation: searchCase.representation },
    })

    const result = page.locator('.search-result')
    await expect(result).toContainText(searchCase.excerpt)
    await expect(result).toContainText('Coverage: Complete')
    await expect(result).toContainText(searchCase.confidence == null
      ? 'Record confidence: not applicable'
      : 'Audio decoder score: 91.0% · uncalibrated')
  }
})

test('search serializes every scope and date filter, paginates, and follows an activity source jump', async ({ page }) => {
  await page.goto('/search')
  await page.getByLabel('What do you want to find?').fill('paged search')
  await page.getByText('Scope and date filters').click()
  await page.getByLabel('Record type').selectOption('activity')
  await page.getByLabel('Participant or actor').fill('Alex Example')
  await page.getByLabel('Conversation ID').fill('conv-alex')
  await page.getByLabel('Source category').fill('groups')
  await page.getByLabel('From').fill('2023-01-01')
  await page.getByLabel('Through').fill('2024-12-31')
  const firstRequest = page.waitForRequest((request) => {
    const url = new URL(request.url())
    return url.pathname === '/api/v1/search' && url.searchParams.get('q') === 'paged search'
  })
  await page.getByRole('button', { name: 'Search archive' }).click()
  const query = new URL((await firstRequest).url()).searchParams
  expect(Object.fromEntries(query)).toMatchObject({ q: 'paged search', mode: 'keyword', entity_type: 'activity', conversation_id: 'conv-alex', participant: 'Alex Example', source_category: 'groups', limit: '50' })
  expect(Number(query.get('from_ms'))).toBeGreaterThan(0)
  expect(Number(query.get('to_ms'))).toBeGreaterThan(Number(query.get('from_ms')))
  await expect(page.getByText('Fixture comment about resolving a disagreement.')).toBeVisible()
  await expect(page.getByRole('link', { name: 'Jump to source' })).toHaveAttribute('href', '/timeline?around=activity:comment-one#activity-comment-one')
  const secondPage = page.waitForRequest((request) => request.url().includes('/api/v1/search?') && request.url().includes('cursor=search-page-two'))
  await page.getByRole('button', { name: 'Load more results' }).click()
  await secondPage
  await expect(page.getByText('A second fixture result.')).toBeVisible()
  await expect(page.locator('.search-result')).toHaveCount(2)
  await page.getByRole('link', { name: 'Jump to source' }).first().click()
  await expect(page).toHaveURL(/\/timeline\?around=activity(?::|%3A)comment-one#activity-comment-one/)
})

test('saved searches and retrieval evaluations can be created, edited, paged, and deleted', async ({ page }) => {
  await page.route(/\/api\/v1\/search\/feedback\?.*/, async (route) => {
    const url = new URL(route.request().url())
    if (url.searchParams.get('cursor') === 'evaluation-page-two') return route.fulfill({ json: { items: [], page: { limit: 20, returned: 0, next_cursor: null } } })
    return route.fallback()
  })
  await page.goto('/search')
  await page.getByLabel('What do you want to find?').fill('argument')
  await page.getByRole('button', { name: 'Search archive' }).click()
  await expect(page.getByText(/This keeps happening/)).toBeVisible()

  await page.getByLabel('Saved search name').fill('Argument review')
  await page.getByRole('button', { name: 'Save', exact: true }).click()
  await expect(page.getByText('Argument review')).toBeVisible()
  const deleteSaved = page.waitForRequest((request) => request.url().endsWith('/saved-searches/saved-one') && request.method() === 'DELETE')
  await page.getByRole('button', { name: 'Delete saved search Argument review' }).click()
  await deleteSaved
  await expect(page.getByText('Argument review')).toHaveCount(0)

  const evaluation = page.locator('.search-result__evaluation')
  await evaluation.getByText('Evaluation details').click()
  await evaluation.getByRole('combobox', { name: 'Dataset split' }).selectOption('validation')
  await evaluation.getByRole('textbox', { name: 'Notes' }).fill('Fixture relevance note')
  const createFeedback = page.waitForRequest((request) => request.url().endsWith('/search/feedback') && request.method() === 'POST')
  await page.getByRole('button', { name: 'Mark result 1 uncertain' }).click()
  expect((await createFeedback).postDataJSON()).toMatchObject({ label: 'uncertain', evaluation_split: 'validation', notes: 'Fixture relevance note' })
  await page.getByText('Retrieval evaluation set').click()
  const row = page.locator('.search-evaluations li').first()
  await expect(row).toContainText('argument')
  const evaluationPage = page.waitForRequest((request) => request.url().includes('/search/feedback?') && request.url().includes('cursor=evaluation-page-two'))
  await page.getByRole('button', { name: 'Load more evaluations' }).click()
  await evaluationPage
  await row.getByRole('combobox', { name: 'Evaluation label for feedback-one' }).selectOption('relevant')
  await row.getByRole('combobox', { name: 'Evaluation split for feedback-one' }).selectOption('test')
  await row.getByRole('textbox', { name: 'Evaluation notes for feedback-one' }).fill('Updated fixture note')
  const updateFeedback = page.waitForRequest((request) => request.url().endsWith('/search/feedback/feedback-one') && request.method() === 'PATCH')
  await row.getByRole('button', { name: 'Save changes' }).click()
  expect((await updateFeedback).postDataJSON()).toEqual({ label: 'relevant', evaluation_split: 'test', notes: 'Updated fixture note' })
  const deleteFeedback = page.waitForRequest((request) => request.url().endsWith('/search/feedback/feedback-one') && request.method() === 'DELETE')
  await page.getByRole('button', { name: `Delete evaluation for ${searchResult.id}` }).click()
  await deleteFeedback
  await expect(page.getByText('No relevance judgements recorded yet.')).toBeVisible()
})

test('derived audio search reveals every occurrence through bounded occurrence paging', async ({ page }) => {
  await page.goto('/search')
  await page.getByRole('radio', { name: /^Semantic/ }).check({ force: true })
  await page.getByLabel('What do you want to find?').fill('voice disagreement')
  await page.getByText('Scope and date filters').click()
  await page.getByLabel('Record type').selectOption('audio_transcript')
  await page.getByRole('button', { name: 'Search archive' }).click()
  await expect(page.getByText('Fixture voice note about repairing a disagreement.')).toBeVisible()
  await page.getByText('2 source locations').click()
  await expect(page.getByRole('link', { name: 'Jump to this occurrence' })).toHaveCount(1)
  const more = page.waitForRequest((request) => request.url().includes('/search/derived-occurrences?') && request.url().includes('offset=1'))
  await page.getByRole('button', { name: 'Show more source locations' }).click()
  await more
  await expect(page.getByRole('link', { name: 'Jump to this occurrence' })).toHaveCount(2)
  await expect(page.getByRole('link', { name: 'Jump to this occurrence' }).last()).toHaveAttribute('href', /conv-family/)
})

test('search empty, error, and cloud-disabled states remain explicit and recoverable', async ({ page }) => {
  await page.route('**/api/v1/settings', (route) => route.fulfill({ json: { timezone: 'America/New_York', cloud_enabled: false, image_semantic_stage_enabled: false } }))
  await page.route(/\/api\/v1\/search\?.*/, (route) => {
    const url = new URL(route.request().url())
    if (url.searchParams.get('q') === 'fixture error') return route.fulfill({ status: 503, json: { detail: 'Fixture search failed safely.' } })
    return route.fallback()
  })
  await page.goto('/search')
  await expect(page.locator('.rag-panel__disabled')).toContainText('Cloud processing is off.')
  await expect(page.getByRole('button', { name: 'Generate cited answer' })).toHaveCount(0)
  await page.getByLabel('What do you want to find?').fill('nothing in fixture')
  await page.getByRole('button', { name: 'Search archive' }).click()
  await expect(page.getByText('No matching archive records', { exact: true })).toBeVisible()
  await page.getByLabel('What do you want to find?').fill('fixture error')
  await page.getByRole('button', { name: 'Search archive' }).click()
  await expect(page.getByRole('alert')).toContainText('Fixture search failed safely.')
  await expect(page.getByRole('button', { name: 'Try again' })).toBeVisible()
})

test('search remains readable, keyboard reachable, and overflow-free on mobile', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await page.goto('/search')
  await page.getByLabel('What do you want to find?').fill('argument')
  await page.getByRole('button', { name: 'Search archive' }).click()
  await expect(page.getByText(/This keeps happening/)).toBeVisible()
  const widths = await page.locator('body').evaluate((element) => ({ scroll: element.scrollWidth, client: element.clientWidth }))
  expect(widths.scroll).toBeLessThanOrEqual(widths.client + 1)
  await page.getByRole('button', { name: 'Mark result 1 relevant' }).focus()
  await expect(page.getByRole('button', { name: 'Mark result 1 relevant' })).toBeFocused()
  await page.screenshot({ path: 'test-results/stage3-search-mobile.png', fullPage: true })
})

test('search is overflow-free and axe-clean with populated results at desktop and mobile sizes', async ({ page }) => {
  for (const viewport of [{ width: 1440, height: 1000 }, { width: 390, height: 844 }]) {
    await page.setViewportSize(viewport)
    await page.goto('/search')
    await page.getByLabel('What do you want to find?').fill('argument')
    await page.getByRole('button', { name: 'Search archive' }).click()
    await expect(page.getByText(/This keeps happening/)).toBeVisible()
    const widths = await page.locator('body').evaluate((element) => ({ scroll: element.scrollWidth, client: element.clientWidth }))
    expect(widths.scroll, `search at ${viewport.width}px`).toBeLessThanOrEqual(widths.client + 1)
    const result = await new AxeBuilder({ page })
      .withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'wcag22a', 'wcag22aa'])
      .analyze()
    expect(result.violations, `${viewport.width}px: ${result.violations.map((violation) => `${violation.id}: ${violation.nodes.length}`).join(', ')}`).toEqual([])
    expect(result.incomplete, `${viewport.width}px: incomplete ${result.incomplete.map((issue) => `${issue.id}: ${issue.nodes.length}`).join(', ')}`).toEqual([])
  }
})

test('Stage 3 search has deterministic visual and accessibility baselines', async ({ page }) => {
  await page.emulateMedia({ reducedMotion: 'reduce' })
  await page.goto('/search')
  await expect(page.getByText('fixture/embedding · 25 searchable windows')).toBeVisible()
  const semanticMode = page.getByRole('radio', { name: /^Semantic/ })
  await expect(semanticMode).toBeEnabled()
  await semanticMode.check({ force: true })
  await page.getByLabel('What do you want to find?').fill('argument')
  await page.getByRole('button', { name: 'Search archive' }).click()
  const main = page.getByRole('main')
  const pageContent = page.locator('.search-page')
  await expect(main.getByText(/This keeps happening/)).toBeVisible()
  await page.addStyleTag({ content: '.topbar { visibility: hidden !important; }' })
  await expect(pageContent).toHaveScreenshot('stage3-search-main.png', { animations: 'disabled', timeout: 15_000 })
  await expect(pageContent).toMatchAriaSnapshot({ name: 'stage3-search-main.aria.yml' })
})
