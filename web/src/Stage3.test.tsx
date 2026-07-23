import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { createMemoryHistory, createRouter, RouterProvider } from '@tanstack/react-router'
import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { routeTree } from './router'

const result = {
  id: 'window:argument-one',
  window_id: 'argument-one',
  entity_type: 'message_window',
  source_id: 'message-one',
  rank: 1,
  score: 0.83,
  score_kind: 'cosine_similarity',
  timestamp_ms: 1_704_153_600_000,
  excerpt: 'Alex: This keeps happening.\nArchive Owner: I disagree, and I need you to listen.',
  sender: null,
  conversation: { id: 'conv-alex', title: 'Alex Example' },
  participants: ['alex example', 'archive owner'],
  source_category: 'inbox',
  coverage: { status: 'available', absence_is_not_zero_history: false },
  confidence: null,
  confidence_kind: 'not_applicable',
  confidence_calibrated: false,
  availability: 'available',
  deletion_state: 'retained',
  provenance: {
    entity_type: 'message_window',
    source_id: 'message-one',
    window_id: 'argument-one',
    source_category: 'inbox',
    representation: 'semantic_window',
  },
  window: {
    start_message_id: 'message-one',
    end_message_id: 'message-six',
    message_ids: ['message-one', 'message-six'],
    turn_count: 6,
    first_timestamp_ms: 1_704_153_600_000,
    last_timestamp_ms: 1_704_153_900_000,
  },
  jump: {
    href: '/conversations/conv-alex?around=message-one#message-message-one',
    conversation_id: 'conv-alex',
    message_id: 'message-one',
    fragment: 'message-message-one',
  },
}

const acceptanceEvaluation = {
  evaluator_version: 'semantic-ir-acceptance-v1',
  evaluated_at: '2026-07-20T10:00:00Z',
  passed: true,
  decision: 'accepted',
  automatic_known_item: {
    benchmark_kind: 'automatic_known_item',
    status: 'passed',
    passed: true,
    used_for_activation: true,
    metrics: { query_count: 25, k: 10, recall_at_k: 1, mrr: 1, ndcg_at_k: 1 },
  },
  held_out_feedback: {
    benchmark_kind: 'held_out_user_feedback',
    status: 'not_available',
    passed: null,
    used_for_activation: false,
    metrics: { query_count: 0, k: 10, recall_at_k: 0, mrr: 0, ndcg_at_k: 0 },
  },
}

const rejectedEvaluation = {
  ...acceptanceEvaluation,
  passed: false,
  decision: 'rejected; an existing ready index, when present, remains active',
  automatic_known_item: {
    ...acceptanceEvaluation.automatic_known_item,
    status: 'failed',
    passed: false,
    metrics: {
      ...acceptanceEvaluation.automatic_known_item.metrics,
      recall_at_k: 0.5,
      mrr: 0.25,
      ndcg_at_k: 0.3,
    },
  },
}

function installFetchMock({
  ready = true,
  cloud = false,
  rejected = false,
  stale = false,
  feedbackCount = 0,
  withOccurrences = false,
}: {
  ready?: boolean
  cloud?: boolean
  rejected?: boolean
  stale?: boolean
  feedbackCount?: number
  withOccurrences?: boolean
} = {}) {
  const savedItems: { id: string; name: string; query: Record<string, unknown> }[] = []
  const feedbackItems: Record<string, unknown>[] = Array.from({ length: feedbackCount }, (_, index) => ({
    id: `feedback-${index}`,
    query_sha256: `hash-${index}`,
    query: `query ${index}`,
    search_mode: index % 2 ? 'hybrid' : 'semantic',
    filter_scope: index % 2 ? { entity_type: 'message', conversation_id: 'conv-alex' } : {},
    result_id: `window:result-${index}`,
    label: 'relevant',
    evaluation_split: 'test',
    notes: `note ${index}`,
    created_at: '2026-07-20T10:00:00Z',
  }))
  const ragAuditItems: Record<string, unknown>[] = []
  const olderRagAudit = {
    id: 'audit-older',
    status: 'completed',
    model: 'gpt-4.1-mini',
    query_sha256: 'older-hash',
    search_mode: 'semantic',
    retrieved_ids: ['window:older-result'],
    citation_ids: [],
    usage: { input_tokens: 8, output_tokens: 2, total_tokens: 10 },
    created_at: '2026-07-19T10:00:00Z',
  }
  const effectiveReady = stale ? false : ready
  const allOccurrences = Array.from({ length: 26 }, (_, index) => ({
    occurrence_id: `occurrence-${index}`,
    conversation_id: 'conv-alex',
    conversation_title: `Conversation ${index + 1}`,
    message_id: `message-${index}`,
    timestamp_ms: result.timestamp_ms + index,
    source_category: 'inbox',
    start_ms: index * 1_000,
    href: `/conversations/conv-alex?around=message-${index}&media=occurrence-${index}#message-message-${index}`,
  }))
  const searchResult = withOccurrences ? {
    ...result,
    occurrences: allOccurrences.slice(0, 20),
    occurrence_page: { limit: 20, offset: 0, returned: 20, total: allOccurrences.length, next_offset: 20 },
  } : result
  const mock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = new URL(String(input), 'http://localhost')
    const method = (init?.method ?? 'GET').toUpperCase()
    if (url.pathname === '/api/v1/health') return Response.json({ status: 'ok', catalog_ready: true, catalog_state: 'ready' })
    if (url.pathname === '/api/v1/settings') return Response.json({ timezone: 'America/New_York', cloud_enabled: cloud, image_semantic_stage_enabled: false })
    if (url.pathname === '/api/v1/search/status' && method === 'GET') return Response.json({ ready: effectiveReady, needs_rebuild: stale, stale_reason: stale ? { reason: 'Searchable audio transcript text changed', source_kind: 'audio_transcription', source_id: 'transcript-one' } : null, stale_config: stale ? { id: 'semantic-one', status: 'stale', model_name: 'fixture/embedding', model_revision: 'fixture-v1', window_count: 25 } : null, active: effectiveReady ? { id: 'semantic-one', status: 'ready', model_name: 'fixture/embedding', model_revision: 'fixture-v1', dimensions: 4, window_version: 'v1', window_count: 25, audit: {} } : null, acceptance: effectiveReady ? acceptanceEvaluation : null, latest_evaluation: rejected ? rejectedEvaluation : effectiveReady ? acceptanceEvaluation : null, latest_candidate: rejected ? { id: 'semantic-rejected', status: 'rejected', active: false, window_count: 25, evaluation: rejectedEvaluation } : effectiveReady ? { id: 'semantic-one', status: 'ready', active: true, window_count: 25, evaluation: acceptanceEvaluation } : null, latest_job: rejected ? { id: 'job-rejected', status: 'failed', stage: 'Semantic index rejected by retrieval quality gate' } : null })
    if (url.pathname === '/api/v1/search/index' && method === 'POST') return Response.json({ job_id: 'job-one', status: 'queued', status_href: '/api/v1/search/status' }, { status: 202 })
    if (url.pathname === '/api/v1/search' && method === 'GET') return Response.json({ items: [searchResult], page: { limit: 50, returned: 1, candidate_count: 1, result_cap: 1000, next_cursor: null }, query: url.searchParams.get('q'), mode: url.searchParams.get('mode'), filters: { entity_type: url.searchParams.get('entity_type') ?? 'all' }, index_config: effectiveReady ? { id: 'semantic-one', model_name: 'fixture/embedding' } : null })
    if (url.pathname === '/api/v1/search/derived-occurrences' && method === 'GET') {
      const offset = Number(url.searchParams.get('offset') ?? 0)
      const limit = Number(url.searchParams.get('limit') ?? 20)
      const items = allOccurrences.slice(offset, offset + limit)
      return Response.json({ items, page: { limit, offset, returned: items.length, total: allOccurrences.length, next_offset: offset + items.length < allOccurrences.length ? offset + items.length : null } })
    }
    if (url.pathname === '/api/v1/saved-searches' && method === 'GET') return Response.json({ items: savedItems })
    if (url.pathname === '/api/v1/saved-searches' && method === 'POST') {
      const body = JSON.parse(String(init?.body)) as { name: string; query: Record<string, unknown> }
      const item = { id: 'saved-one', ...body }
      savedItems.push(item)
      return Response.json(item, { status: 201 })
    }
    if (url.pathname === '/api/v1/search/feedback' && method === 'GET') {
      const limit = Number(url.searchParams.get('limit') ?? 20)
      const offset = url.searchParams.get('cursor') ? 20 : 0
      const items = feedbackItems.slice(offset, offset + limit)
      return Response.json({ items, page: { limit, returned: items.length, next_cursor: offset + limit < feedbackItems.length ? 'next-feedback-page' : null } })
    }
    if (url.pathname === '/api/v1/search/feedback' && method === 'POST') {
      const body = JSON.parse(String(init?.body)) as { query: string; search_mode: string; result_id: string; label: string; evaluation_split?: string; notes?: string }
      const item = { id: 'feedback-one', query_sha256: 'hash', ...body, created_at: '2026-07-20T10:00:00Z' }
      feedbackItems.unshift(item)
      return Response.json(item, { status: 201 })
    }
    if (url.pathname.startsWith('/api/v1/search/feedback/') && method === 'PATCH') {
      const id = decodeURIComponent(url.pathname.split('/').at(-1)!)
      const body = JSON.parse(String(init?.body)) as Record<string, unknown>
      const index = feedbackItems.findIndex((item) => item.id === id)
      const item = { ...(feedbackItems[index] ?? { id }), ...body }
      if (index >= 0) feedbackItems[index] = item
      return Response.json(item)
    }
    if (url.pathname.startsWith('/api/v1/search/feedback/') && method === 'DELETE') return new Response(null, { status: 204 })
    if (url.pathname === '/api/v1/rag/audits' && method === 'GET') {
      const secondPage = url.searchParams.get('cursor') === 'rag-audit-page-two'
      const items = secondPage ? [olderRagAudit] : ragAuditItems
      return Response.json({ items, page: { limit: 20, returned: items.length, next_cursor: !secondPage && ragAuditItems.length ? 'rag-audit-page-two' : null } })
    }
    if (url.pathname === '/api/v1/rag/answer' && method === 'POST') {
      const body = JSON.parse(String(init?.body)) as { question: string; search_mode: string }
      ragAuditItems.splice(0, ragAuditItems.length, { id: 'audit-one', status: 'completed', model: 'gpt-5.6-sol', query_sha256: 'hash', search_mode: body.search_mode, retrieved_ids: [result.id], citation_ids: [result.id], remote_response_id: 'response-local-audit-one', usage: { input_tokens: 20, output_tokens: 7, total_tokens: 27 }, created_at: '2026-07-20T10:00:00Z' })
      return Response.json({ answer: 'The cited exchange appears to be a disagreement.', citation_ids: [result.id], insufficient_evidence: false, sources: [result], audit_id: 'audit-one', model: 'gpt-5.6-sol' })
    }
    return Response.json({ detail: `Unhandled ${method} ${url.pathname}` }, { status: 404 })
  })
  vi.stubGlobal('fetch', mock)
  return mock
}

async function renderSearch() {
  const history = createMemoryHistory({ initialEntries: ['/search'] })
  const router = createRouter({ routeTree, history })
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  await router.load()
  return render(<QueryClientProvider client={queryClient}><RouterProvider router={router} /></QueryClientProvider>)
}

describe('Stage 3 search experience', () => {
  beforeEach(() => {
    document.cookie = 'meta_archive_csrf=stage3-test; path=/'
  })

  it('keeps the pristine search page idle until a nonempty query is submitted', async () => {
    const fetchMock = installFetchMock()
    await renderSearch()

    expect(await screen.findByRole('heading', { name: 'Search the archive' })).toBeVisible()
    expect(screen.queryByText('Searching the local archive')).not.toBeInTheDocument()
    expect(
      fetchMock.mock.calls.some(
        ([input]) => new URL(String(input), 'http://localhost').pathname === '/api/v1/search',
      ),
    ).toBe(false)
  })

  it('runs semantic search with scoped filters and a source jump', async () => {
    const fetchMock = installFetchMock()
    const user = userEvent.setup()
    await renderSearch()

    expect(await screen.findByRole('heading', { name: 'Search the archive' })).toBeVisible()
    expect(await screen.findByText('fixture/embedding · 25 searchable windows')).toBeVisible()
    expect((await screen.findByText('Semantic retrieval quality gate:')).closest('p')).toHaveTextContent(
      'Retrieval gate passed · known-item Recall@10 1.000 · MRR 1.000 · nDCG@10 1.000 · no held-out feedback yet',
    )
    await user.click(screen.getByText('Semantic').closest('label')!)
    await user.type(screen.getByLabelText('What do you want to find?'), 'argument')
    await user.click(screen.getByText('Scope and date filters'))
    await user.selectOptions(screen.getByLabelText('Record type'), 'message')
    await user.type(screen.getByLabelText('Participant or actor'), 'Alex Example')
    await user.click(screen.getByRole('button', { name: 'Search archive' }))

    expect(await screen.findByText(/This keeps happening/)).toBeVisible()
    expect(screen.getByRole('link', { name: 'Jump to source' })).toHaveAttribute('href', result.jump.href)
    const searchCall = fetchMock.mock.calls.map(([input]) => new URL(String(input), 'http://localhost')).find((url) => url.pathname === '/api/v1/search')
    expect(searchCall?.searchParams.get('mode')).toBe('semantic')
    expect(searchCall?.searchParams.get('entity_type')).toBe('message')
    expect(searchCall?.searchParams.get('participant')).toBe('Alex Example')
  })

  it('keeps the prior ready index visible when a replacement candidate is rejected', async () => {
    installFetchMock({ rejected: true })
    await renderSearch()

    expect(await screen.findByText('fixture/embedding · 25 searchable windows')).toBeVisible()
    expect(screen.getByText('Semantic retrieval quality gate:').closest('p')).toHaveTextContent(
      'Retrieval gate failed · known-item Recall@10 0.500 · MRR 0.250 · nDCG@10 0.300',
    )
    expect(screen.getByText(/The prior ready index remains active/)).toBeVisible()
    expect(screen.getByRole('button', { name: 'Retry semantic index build' })).toBeEnabled()
  })

  it('saves repeatable queries and records relevance feedback', async () => {
    const fetchMock = installFetchMock()
    const user = userEvent.setup()
    await renderSearch()

    await user.type(await screen.findByLabelText('What do you want to find?'), 'fixture')
    await user.click(screen.getByRole('button', { name: 'Search archive' }))
    expect(await screen.findByText(/This keeps happening/)).toBeVisible()
    await user.type(screen.getByLabelText('Saved search name'), 'Important exchanges')
    await user.click(screen.getByRole('button', { name: 'Save' }))
    expect(await screen.findByText('Important exchanges')).toBeVisible()

    await user.click(screen.getByText('Evaluation details'))
    await user.selectOptions(screen.getByLabelText('Dataset split'), 'test')
    await user.type(screen.getByLabelText('Notes'), 'Held-out derived-search check')
    await user.click(screen.getByRole('button', { name: 'Mark result 1 uncertain' }))
    expect(await screen.findByText('Marked Uncertain')).toBeVisible()
    expect((await screen.findAllByDisplayValue('Held-out derived-search check')).length).toBeGreaterThanOrEqual(1)
    const writes = fetchMock.mock.calls.filter(([, init]) => ['POST', 'PATCH', 'DELETE'].includes(String(init?.method)))
    expect(writes.every(([, init]) => (init?.headers as Record<string, string>)['X-CSRF-Token'] === 'stage3-test')).toBe(true)
    expect(writes.some(([input]) => new URL(String(input), 'http://localhost').pathname === '/api/v1/search/feedback')).toBe(true)
    const feedbackWrite = writes.find(([input]) => new URL(String(input), 'http://localhost').pathname === '/api/v1/search/feedback')
    expect(JSON.parse(String(feedbackWrite?.[1]?.body))).toMatchObject({ label: 'uncertain', evaluation_split: 'test', notes: 'Held-out derived-search check', filter_scope: { entity_type: 'all' } })
  })

  it('keeps RAG visibly cloud-gated and renders only cited sources when enabled', async () => {
    const fetchMock = installFetchMock({ cloud: true })
    const user = userEvent.setup()
    await renderSearch()
    await user.type(await screen.findByLabelText('What do you want to find?'), 'argument')
    await user.click(screen.getByText('Scope and date filters'))
    await user.selectOptions(screen.getByLabelText('Record type'), 'attachment_text')
    await user.click(screen.getByRole('button', { name: 'Search archive' }))
    await screen.findByText(/This keeps happening/)

    const rag = screen.getByRole('heading', { name: 'Ask across retrieved excerpts' }).closest('section')!
    expect(within(rag).getByText('Cloud Enabled')).toBeVisible()
    await user.click(within(rag).getByRole('button', { name: 'Generate cited answer' }))
    expect(await within(rag).findByText('The cited exchange appears to be a disagreement.')).toBeVisible()
    expect(within(rag.querySelector('.rag-answer')!).getByText(/gpt-5.6-sol/)).toBeVisible()
    expect(within(rag).getByRole('link', { name: 'Alex Example' })).toHaveAttribute('href', result.jump.href)
    await user.click(within(rag).getByText(/Local RAG audit history/))
    expect(await within(rag).findByText(/1 retrieved · 1 cited/)).toBeVisible()
    await user.click(within(rag).getByText('Source IDs and usage'))
    const auditDetail = within(rag).getByText('Source IDs and usage').closest('details')!
    expect(within(auditDetail).getAllByText(result.id)).toHaveLength(2)
    expect(within(auditDetail).getByText('Input Tokens')).toBeVisible()
    expect(within(auditDetail).getByText('20')).toBeVisible()
    expect(within(auditDetail).getByText('Total Tokens')).toBeVisible()
    expect(within(auditDetail).getByText('27')).toBeVisible()
    const auditPages = within(rag).getByRole('navigation', { name: 'RAG audit history pages' })
    await user.click(within(auditPages).getByRole('button', { name: 'Next page' }))
    expect(await within(rag).findByText(/gpt-4.1-mini/)).toBeVisible()
    expect(within(auditPages).getByRole('button', { name: 'Previous page' })).toHaveAttribute('aria-disabled', 'false')
    const ragWrite = fetchMock.mock.calls.find(([input]) => new URL(String(input), 'http://localhost').pathname === '/api/v1/rag/answer')
    expect(JSON.parse(String(ragWrite?.[1]?.body))).toMatchObject({ entity_type: 'attachment_text' })
  })

  it('disables semantic modes until a local index is explicitly queued', async () => {
    const fetchMock = installFetchMock({ ready: false })
    const user = userEvent.setup()
    await renderSearch()
    expect(await screen.findByRole('radio', { name: /Semantic/ })).toBeDisabled()
    await user.click(screen.getByRole('button', { name: 'Build local semantic index' }))
    expect(fetchMock.mock.calls.some(([input, init]) => new URL(String(input), 'http://localhost').pathname === '/api/v1/search/index' && init?.method === 'POST')).toBe(true)
  })

  it('explains a stale semantic index and offers an explicit rebuild', async () => {
    const fetchMock = installFetchMock({ stale: true })
    const user = userEvent.setup()
    await renderSearch()

    expect(await screen.findByText(/prior semantic index is stale/i)).toHaveTextContent('Searchable audio transcript text changed')
    expect(screen.getByRole('radio', { name: /Semantic/ })).toBeDisabled()
    await user.click(screen.getByRole('button', { name: 'Rebuild local semantic index' }))
    expect(fetchMock.mock.calls.some(([input, init]) => new URL(String(input), 'http://localhost').pathname === '/api/v1/search/index' && init?.method === 'POST')).toBe(true)
  })

  it('paginates and edits the local retrieval evaluation set', async () => {
    const fetchMock = installFetchMock({ feedbackCount: 21 })
    const user = userEvent.setup()
    await renderSearch()

    expect(screen.queryByText('query 19')).not.toBeInTheDocument()
    await user.click(screen.getByText('Retrieval evaluation set'))
    expect(await screen.findByText('query 19')).toBeVisible()
    expect(screen.queryByText('query 20')).not.toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Load more evaluations' }))
    const lastQuery = await screen.findByText('query 20')
    const row = lastQuery.closest('li')!
    await user.selectOptions(within(row).getByLabelText('Evaluation label for feedback-20'), 'not_relevant')
    await user.selectOptions(within(row).getByLabelText('Evaluation split for feedback-20'), 'validation')
    const notes = within(row).getByLabelText('Evaluation notes for feedback-20')
    await user.clear(notes)
    await user.type(notes, 'reviewed after scoped search')
    await user.click(within(row).getByRole('button', { name: 'Save changes' }))

    const patch = fetchMock.mock.calls.find(([input, init]) => new URL(String(input), 'http://localhost').pathname.endsWith('/search/feedback/feedback-20') && init?.method === 'PATCH')
    expect(JSON.parse(String(patch?.[1]?.body))).toEqual({ label: 'not_relevant', evaluation_split: 'validation', notes: 'reviewed after scoped search' })
  })

  it('shows every derived media occurrence with a source jump', async () => {
    const fetchMock = installFetchMock({ withOccurrences: true })
    const user = userEvent.setup()
    await renderSearch()
    await user.type(await screen.findByLabelText('What do you want to find?'), 'fixture')
    await user.click(screen.getByRole('button', { name: 'Search archive' }))
    await user.click(await screen.findByText('26 source locations'))

    expect(screen.getAllByRole('link', { name: 'Jump to this occurrence' })).toHaveLength(5)
    expect(screen.queryByText('Conversation 21')).not.toBeInTheDocument()
    for (let index = 0; index < 3; index += 1) {
      await user.click(screen.getByRole('button', { name: 'Show more source locations' }))
    }
    await user.click(screen.getByRole('button', { name: 'Show more source locations' }))
    expect(await screen.findByText('Conversation 21')).toBeVisible()
    await user.click(screen.getByRole('button', { name: 'Show more source locations' }))
    expect(await screen.findByText('Conversation 26')).toBeVisible()
    expect(screen.getAllByRole('link', { name: 'Jump to this occurrence' })).toHaveLength(26)
    expect(screen.getAllByRole('link', { name: 'Jump to this occurrence' })[25]).toHaveAttribute('href', expect.stringContaining('media=occurrence-25'))
    expect(fetchMock.mock.calls.some(([input]) => new URL(String(input), 'http://localhost').pathname === '/api/v1/search/derived-occurrences')).toBe(true)
  })
})
