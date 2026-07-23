import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { createMemoryHistory, createRouter, RouterProvider } from '@tanstack/react-router'
import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { routeTree } from './router'
import type { DeletionOperation } from './lib/contracts'

const operationId = '11111111-1111-4111-8111-111111111111'
const olderOperationId = '22222222-2222-4222-8222-222222222222'
const selectionRevision = 'd'.repeat(64)

function installFetchMock({ committed = false, restoreRebuildRequired = false, conflictGatePassed = false, statusGate, selectionGate, selectionFailsOnce = false, secondSelectionGate, detailFailsOnce = false, purgePreviewGateAfterFirst }: { committed?: boolean; restoreRebuildRequired?: boolean; conflictGatePassed?: boolean; statusGate?: Promise<void>; selectionGate?: Promise<void>; selectionFailsOnce?: boolean; secondSelectionGate?: Promise<void>; detailFailsOnce?: boolean; purgePreviewGateAfterFirst?: Promise<void> } = {}) {
  const rebuildSummary = { required: true as const, reason: 'later_quarantine' as const, product_kinds: ['search_windows', 'semantic_index', 'nlp_hourly_rollups', 'conflict_aggregates', 'topic_clusters'] as const, invalidated_by_count: 2 }
  const operations: DeletionOperation[] = committed || restoreRebuildRequired ? [{ id: operationId, status: restoreRebuildRequired ? 'restored' : 'committed', item_count: 2, impact: { selected: { message: 1, attachment: 1 } }, selection_sha256: selectionRevision, ...(restoreRebuildRequired ? { restore_rebuild_required: { ...rebuildSummary, product_kinds: [...rebuildSummary.product_kinds] } } : {}), created_at: '2026-07-20T10:00:00Z', updated_at: '2026-07-20T10:00:00Z' }] : []
  const olderOperation: DeletionOperation = { id: olderOperationId, status: 'committed', item_count: 5, impact: { selected: { message: 5 } }, selection_sha256: selectionRevision, created_at: '2026-06-20T10:00:00Z', updated_at: '2026-06-20T10:00:00Z' }
  let conflictLabelWrites = 0
  let selectionFailures = 0
  let previewWrites = 0
  let detailFailures = 0
  let purgePreviewWrites = 0
  const conflictLabels = ['direct conflict', 'calm disagreement', 'banter', 'third-party conflict', 'no conflict', 'uncertain'] as const
  const emptyLabelCounts = Object.fromEntries(conflictLabels.map((label) => [label, conflictGatePassed ? 28 : label === 'banter' ? 1 : 0]))
  const reviewItem = {
    window_id: 'conflict-one', conversation_id: 'conversation-one', evaluation_split: 'test', first_message_id: 'message-one', last_message_id: 'message-two', first_timestamp_ms: 1_752_000_000_000, last_timestamp_ms: 1_752_000_030_000, turn_count: 2,
    text_display: 'Alex: I do not agree with that.\nArchive Owner: This is getting tense.', current_manual_label: null,
    jump: { href: '/conversations/conversation-one?around=message-one#message-message-one', label: 'Open conversation at this window' },
  }
  const mock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = new URL(String(input), 'http://localhost')
    const method = (init?.method ?? 'GET').toUpperCase()
    if (url.pathname === '/api/v1/health') return Response.json({ status: 'ok', catalog_ready: true, catalog_state: 'ready' })
    if (url.pathname === '/api/v1/settings') return Response.json({ timezone: 'America/New_York', cloud_enabled: false, image_semantic_stage_enabled: false })
    if (url.pathname === '/api/v1/enrichment/status') {
      if (statusGate) await statusGate
      return Response.json({
      audio_transcriptions: { completed: 1 }, attachment_text_extractions: { completed: 1 }, active_nlp_annotations: { sentiment: 1 }, active_nlp_hourly_rollups: { sentiment: 1 },
      audio_transcription_selection: { current: { pipeline_version: 'audio-v1', model_name: 'faster-whisper', model_revision: 'small' }, current_record_count: 1, historical_record_count: 1 },
      attachment_text_selection: { current: { pipeline_version: 'extract-v1', extractor_name: 'safe-text', extractor_revision: 'revision-one' }, current_record_count: 1, historical_record_count: 1 },
      nlp_models: { sentiment: { model_name: 'fixture/sentiment', model_revision: 'revision-one', language_scope: 'English', warnings: ['Model estimates can be wrong.'] } },
      attachment_sandbox: { available: true, backend: 'bubblewrap', network: 'unshared' }, attachment_limits: { cpu_seconds: 60 },
      conflict: conflictGatePassed
        ? { status: 'enabled', reason: null, sample_count: 500, aggregation_enabled: true, relationship_ranking_enabled: false }
        : { status: 'disabled_until_calibrated', reason: '499 more labels required.', sample_count: 1, aggregation_enabled: false, relationship_ranking_enabled: false },
      conflict_aggregates: { available: false, status: 'unavailable', reason: conflictGatePassed ? 'Ready to build the gated corpus.' : 'Calibration gate has not passed.', gate: {}, run: null, job: null, warnings: [], relationship_ranking_enabled: false, relationship_dimensions: [] },
        jobs: [],
      })
    }
    if (url.pathname === '/api/v1/enrichment/transcriptions/jobs' && method === 'POST') return Response.json({ job_id: 'audio-job', status: 'queued', status_href: '/api/v1/enrichment/status' }, { status: 202 })
    if (url.pathname === '/api/v1/enrichment/attachment-text/jobs' && method === 'POST') return Response.json({ job_id: 'text-job', status: 'queued', status_href: '/api/v1/enrichment/status' }, { status: 202 })
    if (url.pathname === '/api/v1/enrichment/nlp/jobs' && method === 'POST') return Response.json({ job_id: 'nlp-job', status: 'queued', status_href: '/api/v1/enrichment/status' }, { status: 202 })
    if (url.pathname === '/api/v1/enrichment/conflict/aggregate-jobs' && method === 'POST') return Response.json({ job_id: 'conflict-aggregate-job', status: 'queued', status_href: '/api/v1/enrichment/conflict/status' }, { status: 202 })
    if (url.pathname === '/api/v1/enrichment/conflict/review/status') return Response.json({
      labels: conflictLabels, minimum_label_count: 500, eligible_window_count: 900, manual_label_count: conflictGatePassed ? 504 : 1, unlabelled_window_count: conflictGatePassed ? 396 : 899,
      evaluation_split_algorithm: { version: 'conversation-sha256-v1', ratios: { train: 70, validation: 15, test: 15 } }, by_label: emptyLabelCounts,
      by_split: Object.fromEntries(['train', 'validation', 'test'].map((split) => [split, { eligible_window_count: 300, manual_label_count: conflictGatePassed ? 168 : split === 'test' ? 1 : 0, unlabelled_window_count: conflictGatePassed ? 132 : split === 'test' ? 299 : 300, by_label: emptyLabelCounts }])),
      validation: { eligible_for_calibration: conflictGatePassed, conversation_disjoint: true, stratified: conflictGatePassed, exact_split_assignment: true, missing_conversation_count: 0, split_assignment_mismatch_count: 0, leaked_conversation_count: 0, missing_strata: conflictGatePassed ? [] : [{ evaluation_split: 'train', label: 'direct conflict' }], blockers: conflictGatePassed ? [] : [{ code: 'insufficient_manual_labels', message: 'At least 500 genuine manual labels are required.', count: 499 }] },
      relationship_ranking_enabled: false, warning: 'Manual conflict review must never be used to rank people or relationships.',
    })
    if (url.pathname === '/api/v1/enrichment/conflict/review/queue') return Response.json({ items: [reviewItem], page: { limit: 12, returned: 1, total: 899, next_after: 'conflict-one' }, filters: { label: url.searchParams.get('label'), evaluation_split: url.searchParams.get('split') }, navigation: { next_unlabelled_href: '/api/v1/enrichment/conflict/review/next?split=all&after=conflict-one' } })
    if (url.pathname === '/api/v1/enrichment/conflict/review/next') return Response.json({ item: { ...reviewItem, window_id: 'conflict-two', first_message_id: 'message-three', jump: { href: '/conversations/conversation-one?around=message-three#message-message-three', label: 'Open conversation at this window' } }, wrapped: false, navigation: { next_unlabelled_href: '/api/v1/enrichment/conflict/review/next?split=all&after=conflict-two' } })
    if (url.pathname === '/api/v1/enrichment/conflict/calibrations' && method === 'POST') return Response.json({ id: 'calibration-one', status: 'calibrated', sample_count: 504, aggregation_enabled: true, direct_conflict_precision: .8, macro_f1: .75, evaluation_split_algorithm: { version: 'conversation-sha256-v1', ratios: { train: 70, validation: 15, test: 15 } }, relationship_ranking_enabled: false }, { status: 201 })
    if (url.pathname === '/api/v1/enrichment/transcriptions') {
      const history = url.searchParams.get('version') === 'history'
      return Response.json({
      items: [{
        id: history ? 'transcript-history-one' : 'transcript-one', blob_sha256: 'a'.repeat(64), pipeline_version: history ? 'audio-v0' : 'audio-v1', is_current_version: !history, status: 'completed', model_name: 'faster-whisper', model_revision: history ? 'tiny-legacy' : 'small', device: 'cpu', compute_type: 'int8', language: 'en', language_probability: .94, duration_ms: 3000, text: 'A retained audio message.', segment_count: 1, warnings: ['Decoder scores are not calibrated probabilities.'], provenance: { generated_locally: true }, error: null,
        segments: [{ id: 'segment-one', ordinal: 0, start_ms: 0, end_ms: 3000, text: 'A retained audio message.', confidence: .91, jumps: Array.from({ length: 6 }, (_, index) => ({ occurrence_id: `attachment-${index + 1}`, conversation_id: 'conversation-one', message_id: `message-${index + 1}`, href: `/conversations/conversation-one?around=message-${index + 1}#message-message-${index + 1}` })) }],
      }],
      page: { limit: 25, offset: 0, next_offset: null },
      selection: { scope: history ? 'history' : 'current', pipeline_version: history ? null : 'audio-v1', model_name: history ? null : 'faster-whisper', model_revision: history ? null : 'small', current: { pipeline_version: 'audio-v1', model_name: 'faster-whisper', model_revision: 'small' } },
    })
    }
    if (url.pathname === '/api/v1/enrichment/attachment-text') {
      const history = url.searchParams.get('version') === 'history'
      const fullText = `Safely extracted text.${'x'.repeat(25_000)}`
      return Response.json({ items: [{ id: history ? 'extract-history-one' : 'extract-one', blob_sha256: 'b'.repeat(64), pipeline_version: history ? 'extract-v0' : 'extract-v1', extractor_name: 'safe-text', extractor_revision: history ? 'legacy-revision' : 'revision-one', is_current_version: !history, status: 'completed', filename: 'notes.txt', media_type: 'text/plain', text_preview: fullText.slice(0, 16_000), preview_character_count: 16_000, preview_truncated: true, preview_limit: 16_000, character_count: 25_022, truncated: false, index_eligible: true, sandboxed: true, sandbox_status: 'available', limits: {}, warnings: [], provenance: {}, error: null, jumps: [{ occurrence_id: 'attachment-two', href: '/media?occurrence=attachment-two' }] }], page: { limit: 25, offset: 0, next_offset: null }, selection: { scope: history ? 'history' : 'current', pipeline_version: history ? null : 'extract-v1', extractor_name: history ? null : 'safe-text', extractor_revision: history ? null : 'revision-one', current: { pipeline_version: 'extract-v1', extractor_name: 'safe-text', extractor_revision: 'revision-one' } } })
    }
    if (url.pathname === '/api/v1/enrichment/nlp') return Response.json({ items: [{ id: 'nlp-one', target_type: 'message', target_id: 'message-one', task: 'sentiment', model_name: 'fixture/sentiment', model_revision: 'revision-one', predicted_label: 'positive', confidence: .8, probabilities: { positive: .8, neutral: .15, negative: .05 }, probability_activation: 'softmax', attribution: { model_attribution: 'Fixture model' }, warnings: ['Model estimates can be wrong.'], provenance: { input_truncated: false } }], page: { limit: 50, offset: 0, next_offset: null } })
    if (url.pathname === '/api/v1/enrichment/conflict/search') return Response.json({
      items: [{
        id: 'window:conflict-one', window_id: 'conflict-one', entity_type: 'message_window', source_id: 'message-one', rank: 1, score: 2.125, score_kind: 'nli_entailment_logit', timestamp_ms: 1_752_000_000_000, excerpt: 'Alex: I do not agree with that.\nArchive Owner: This is getting tense.', conversation: { id: 'conversation-one', title: 'Alex' }, participants: ['Alex', 'Archive Owner'], source_category: 'inbox', window: { message_ids: ['message-one'], turn_count: 2 }, jump: { href: '/conversations/conversation-one?around=message-one#message-message-one' },
        conflict_scoring: { target_label: 'direct conflict', predicted_label: 'direct conflict', scores: { 'direct conflict': 2.125, 'calm disagreement': 1.25, banter: -1.5, 'third-party conflict': -0.7, 'no conflict': -2.25, uncertain: 0.1 }, score_kind: 'nli_entailment_logit', model_name: 'tasksource/ModernBERT-base-nli', model_revision: 'de4ab7e77845098b7fab7f6ab9d370ddff27b19c', quantization: 'dynamic_int8', ranking_feature_only: true, aggregation_gate_bypassed: false, warnings: ['Uncalibrated ranking feature.'], provenance: { generated_locally: true } },
      }],
      page: { limit: 25, returned: 1, candidate_count: 4 }, query: 'argument or tense disagreement', target_label: 'direct conflict', retrieval_config: {}, conflict_model: { status: 'loaded_fixture' }, aggregation: { status: 'disabled_until_calibrated', sample_count: 1, aggregation_enabled: false },
    })
    if (url.pathname === '/api/v1/enrichment/conflict/labels' && method === 'POST') {
      const body = JSON.parse(String(init?.body)) as { window_id: string; label: string; notes?: string }
      conflictLabelWrites += 1
      return Response.json({ id: 'label-one', window_id: body.window_id, label: body.label, notes: body.notes ?? null, evaluation_split: 'test', split_assignment: { version: 'conversation-sha256-v1', source: 'conversation_id' }, updated: conflictLabelWrites > 1, calibration_invalidated: true }, { status: 201 })
    }
    if (url.pathname === '/api/v1/deletions' && method === 'GET') {
      const secondPage = url.searchParams.get('cursor') === 'deletion-page-two'
      const items = secondPage && operations.length ? [olderOperation] : operations
      return Response.json({
        items,
        page: {
          total: operations.length ? 2 : 0,
          limit: 50,
          next_cursor: !secondPage && operations.length ? 'deletion-page-two' : null,
        },
      })
    }
    if (url.pathname === '/api/v1/deletions/preview' && method === 'POST') {
      previewWrites += 1
      const second = Boolean(secondSelectionGate && previewWrites > 1)
      const nextOperationId = second ? olderOperationId : operationId
      const nextRevision = second ? 'e'.repeat(64) : selectionRevision
      operations.splice(0, operations.length, { id: nextOperationId, status: 'previewed', item_count: 2, impact: { selected: { message: 1, attachment: 1 }, attachments: { exclusive_blobs: 1 } }, selection_sha256: nextRevision, created_at: '2026-07-20T10:00:00Z', updated_at: '2026-07-20T10:00:00Z' })
      return Response.json({ operation_id: nextOperationId, commit_token: 'commit-token-with-sufficient-length', catalog_revision: 'c'.repeat(64), selection_sha256: nextRevision, impact: operations[0]!.impact }, { status: 201 })
    }
    if (url.pathname.match(/^\/api\/v1\/deletions\/[^/]+\/selection$/) && method === 'GET') {
      if (selectionGate) await selectionGate
      if (url.pathname.includes(olderOperationId) && secondSelectionGate) await secondSelectionGate
      if (selectionFailsOnce && selectionFailures++ === 0) return Response.json({ detail: { code: 'deletion_selection_unavailable', message: 'Selection preview temporarily unavailable' } }, { status: 503 })
      const secondPage = url.searchParams.get('cursor') === 'selection-page-two'
      const selectedOperation = operations[0]
      const quarantined = selectedOperation?.status === 'committed'
      const items = secondPage ? [{
        entity_type: 'attachment', entity_id: 'attachment-one', reason: 'message_cascade', label: 'photo.jpg', label_truncated: false, snippet: null, snippet_truncated: false, preview_limit: 160, timestamp_ms: 1_752_000_030_000, resource_kind: 'photos', availability: 'resolved', media_type: 'image/jpeg', size_bytes: 2048, quarantined,
        jump: quarantined ? null : { href: '/conversations/conversation-one?around=message-one&media=attachment-one#message-message-one', label: 'Jump to attachment in conversation' },
      }] : [{
        entity_type: 'message', entity_id: 'message-one', reason: 'direct', label: 'Message from Archive Owner', label_truncated: false, snippet: 'abcd1234 with enough context to verify the exact selection.', snippet_truncated: false, preview_limit: 160, timestamp_ms: 1_752_000_000_000, resource_kind: 'generic', availability: null, media_type: null, size_bytes: null, quarantined,
        jump: quarantined ? null : { href: '/conversations/conversation-one?around=message-one#message-message-one', label: 'Jump to message' },
      }]
      return Response.json({ operation_id: url.pathname.split('/')[4], revision: url.searchParams.get('revision'), status: quarantined ? 'committed' : 'previewed', items, page: { total: 2, limit: 25, next_cursor: secondPage ? null : 'selection-page-two' }, privacy: { snippet_limit: 160, label_limit: 96, raw_fields_exposed: false } })
    }
    if (url.pathname === `/api/v1/deletions/${operationId}` && method === 'GET') {
      if (detailFailsOnce && detailFailures++ === 0) return Response.json({ detail: 'Operation detail temporarily unavailable' }, { status: 503 })
      return Response.json({
      ...operations[0],
      selector: { text: 'abcd1234' },
      purge_summary: operations[0]?.status === 'committed' ? {
        scope: 'workspace',
        status: { operation: 'committed', publication_action: 'commit', publication_phase: 'published' },
        counts: { normalized_rows: 2, workspace_files: 3, workspace_bytes: 2048, affected_messages: 2 },
        digests: { logical_restore_state_sha256: 'f'.repeat(64) },
        timestamps: { planned_at: '2026-07-20T10:05:00Z' },
        logical_restore_state: { topic_terms: ['UI_PRIVATE_CANARY'] },
      } : null,
      audits: restoreRebuildRequired ? [{
        id: 'audit-rebuild',
        action: 'restore',
        scope: 'logical_quarantine',
        outcome: 'completed',
        created_at: '2026-07-20T10:07:00Z',
        detail: {
          counts: { affected_messages: 2 },
          digests: {},
          status: {},
          restore_rebuild_required: { ...rebuildSummary, product_kinds: [...rebuildSummary.product_kinds], private_diagnostic: 'UI_PRIVATE_REBUILD_CANARY' },
        },
      }] : operations[0]?.status === 'committed' ? [{
        id: 'audit-one',
        action: 'commit',
        scope: 'logical_quarantine',
        outcome: 'completed',
        created_at: '2026-07-20T10:04:00Z',
        detail: {
          counts: { affected_messages: 2 },
          digests: { catalog_revision: 'c'.repeat(64) },
          status: {},
          logical_restore_state: { topic_terms: ['UI_PRIVATE_AUDIT_CANARY'] },
        },
      }] : [],
      })
    }
    if (url.pathname === `/api/v1/deletions/${olderOperationId}` && method === 'GET') return Response.json({ ...olderOperation, selector: { text: 'older exact selection' }, audits: [] })
    if (url.pathname === `/api/v1/deletions/${operationId}/commit` && method === 'POST') {
      operations[0]!.status = 'committed'
      return Response.json({ operation_id: operationId, status: 'committed', affected: { message: 2 }, audit_id: 'audit-one' })
    }
    if (url.pathname === `/api/v1/deletions/${operationId}/restore` && method === 'POST') return Response.json({ operation_id: operationId, status: 'restored', affected: { message: 2 }, audit_id: 'audit-two' })
    if (url.pathname === `/api/v1/deletions/${operationId}/purge/preview` && method === 'POST') {
      purgePreviewWrites += 1
      if (purgePreviewWrites > 1 && purgePreviewGateAfterFirst) await purgePreviewGateAfterFirst
      return Response.json({ operation_id: operationId, scope: 'workspace', executable: true, blockers: [], purge_token: 'purge-token-with-sufficient-length', purge_revision: 'e'.repeat(64), impact: { workspace_files: 2, blobs: 1 } })
    }
    if (url.pathname === `/api/v1/deletions/${operationId}/purge` && method === 'POST') return Response.json({ operation_id: operationId, status: 'purged', affected: { message: 2, blobs: 1 }, audit_id: 'audit-three' })
    return Response.json({ detail: `Unhandled ${method} ${url.pathname}` }, { status: 404 })
  })
  vi.stubGlobal('fetch', mock)
  return mock
}

async function renderAt(path: string) {
  const history = createMemoryHistory({ initialEntries: [path] })
  const router = createRouter({ routeTree, history })
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity }, mutations: { retry: false } } })
  await router.load()
  return render(<QueryClientProvider client={queryClient}><RouterProvider router={router} /></QueryClientProvider>)
}

describe('Stage 3 enrichment browsing', () => {
  it('renders the enrichment page identity while status data is still loading', async () => {
    let releaseStatus!: () => void
    const statusGate = new Promise<void>((resolve) => { releaseStatus = resolve })
    installFetchMock({ statusGate })
    await renderAt('/enrichment')

    expect(await screen.findByRole('heading', { name: 'Transcripts and language signals' })).toBeVisible()
    expect(screen.getByText('Loading local enrichment status')).toBeVisible()
    expect(screen.queryByRole('tab', { name: 'Overview' })).not.toBeInTheDocument()

    releaseStatus()
    expect(await screen.findByRole('tab', { name: 'Overview' })).toBeVisible()
  })

  it('implements one-tab-stop keyboard navigation with labelled tab panels', async () => {
    installFetchMock()
    const user = userEvent.setup()
    await renderAt('/enrichment')

    const overview = await screen.findByRole('tab', { name: 'Overview' })
    const transcripts = screen.getByRole('tab', { name: 'Audio transcripts' })
    const conflict = screen.getByRole('tab', { name: 'Conflict review' })
    expect(overview).toHaveAttribute('aria-controls', 'enrichment-panel-overview')
    expect(overview).toHaveAttribute('tabindex', '0')
    expect(transcripts).toHaveAttribute('tabindex', '-1')
    expect(document.getElementById('enrichment-panel-overview')).not.toHaveAttribute('hidden')
    expect(document.getElementById('enrichment-panel-transcripts')).toHaveAttribute('hidden')

    overview.focus()
    await user.keyboard('{ArrowRight}')
    await waitFor(() => expect(transcripts).toHaveFocus())
    expect(transcripts).toHaveAttribute('aria-selected', 'true')
    expect(transcripts).toHaveAttribute('tabindex', '0')
    expect(overview).toHaveAttribute('tabindex', '-1')
    expect(document.getElementById('enrichment-panel-transcripts')).not.toHaveAttribute('hidden')

    await user.keyboard('{End}')
    await waitFor(() => expect(conflict).toHaveFocus())
    expect(conflict).toHaveAttribute('aria-selected', 'true')
    await user.keyboard('{Home}')
    await waitFor(() => expect(overview).toHaveFocus())
    expect(overview).toHaveAttribute('aria-selected', 'true')
  })

  it('browses transcripts, extracted text, and model probabilities with source jumps', async () => {
    const fetchMock = installFetchMock()
    const user = userEvent.setup()
    await renderAt('/enrichment')
    expect(await screen.findByRole('heading', { name: 'Transcripts and language signals' })).toBeVisible()
    expect(await screen.findByText('499 more labels required.')).toBeVisible()
    await user.click(screen.getByRole('button', { name: 'Transcribe audio' }))
    expect(fetchMock.mock.calls.some(([input, init]) => String(input).includes('/enrichment/transcriptions/jobs') && init?.method === 'POST')).toBe(true)
    await user.click(screen.getByRole('tab', { name: 'Audio transcripts' }))
    expect((await screen.findAllByText('A retained audio message.')).length).toBeGreaterThan(0)
    expect(screen.getByText('Current configured results only')).toBeVisible()
    expect(fetchMock.mock.calls.some(([input]) => String(input).includes('/enrichment/transcriptions?') && String(input).includes('version=current'))).toBe(true)
    expect(screen.getByRole('link', { name: 'Jump to source 1' })).toHaveAttribute('href', expect.stringContaining('around=message-1'))
    expect(screen.queryByRole('link', { name: 'Jump to source 5' })).not.toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Next occurrences' }))
    expect(screen.getByRole('link', { name: 'Jump to source 5' })).toHaveAttribute('href', expect.stringContaining('around=message-5'))
    await user.selectOptions(screen.getByLabelText('Transcript version scope'), 'history')
    expect(await screen.findByText('Version history (read-only)')).toBeVisible()
    expect(screen.getByText('Historical')).toBeVisible()
    expect(fetchMock.mock.calls.some(([input]) => String(input).includes('/enrichment/transcriptions?') && String(input).includes('version=history'))).toBe(true)
    await user.click(screen.getByRole('tab', { name: 'Extracted text' }))
    expect(await screen.findByText(/Safely extracted text/)).toBeVisible()
    expect(screen.getByText(/Showing a bounded 16,000-character preview/)).toBeVisible()
    expect(screen.getByText(/separate from extraction safety truncation/)).toBeVisible()
    expect(document.querySelector('.extracted-text')?.textContent).toHaveLength(16_000)
    expect(fetchMock.mock.calls.some(([input]) => String(input).includes('/enrichment/attachment-text?') && String(input).includes('version=current'))).toBe(true)
    await user.selectOptions(screen.getByLabelText('Extraction version scope'), 'history')
    expect(await screen.findByText('Version history (read-only)')).toBeVisible()
    expect(fetchMock.mock.calls.some(([input]) => String(input).includes('/enrichment/attachment-text?') && String(input).includes('version=history'))).toBe(true)
    await user.click(screen.getByRole('tab', { name: 'Language signals' }))
    expect(await screen.findByText('80.0%')).toBeVisible()
    expect(screen.queryByText(/NaN/)).not.toBeInTheDocument()
    expect(screen.getByText('fixture/sentiment · revision-one')).toBeVisible()
    await user.click(screen.getByText('Model attribution, warnings, and provenance'))
    expect(screen.getByText('Model estimates can be wrong.')).toBeVisible()
  })

  it('reranks conflict windows locally and records a six-way human evaluation label', async () => {
    const fetchMock = installFetchMock()
    document.cookie = 'meta_archive_csrf=stage3-test; path=/'
    const user = userEvent.setup()
    await renderAt('/enrichment')
    expect(await screen.findByRole('heading', { name: 'Transcripts and language signals' })).toBeVisible()
    await user.click(await screen.findByRole('tab', { name: 'Conflict review' }))
    expect(await screen.findByRole('heading', { name: 'Human calibration progress' })).toBeVisible()
    expect(screen.getByRole('button', { name: 'Run held-out calibration' })).toBeDisabled()
    expect(screen.getByText('Manual labels by deterministic conversation-disjoint split')).toBeVisible()
    await user.click(screen.getByRole('button', { name: 'Search conflict windows' }))
    expect(await screen.findByText('nli_entailment_logit: 2.125')).toBeVisible()
    expect(screen.getByText('Ranking feature only · never used to rank people or relationships')).toBeVisible()
    await user.click(screen.getByText('All six uncalibrated ranking features'))
    expect(screen.getByText('tasksource/ModernBERT-base-nli', { exact: false })).toBeVisible()
    await user.selectOptions(screen.getByLabelText('Human label for result 1'), 'banter')
    await user.type(screen.getByLabelText('Notes for result 1'), 'Reviewed in context')
    await user.click(screen.getByRole('button', { name: 'Save human label' }))
    const updateButton = await screen.findByRole('button', { name: 'Update human label' })
    expect(updateButton).toBeEnabled()
    await user.selectOptions(screen.getByLabelText('Human label for result 1'), 'calm disagreement')
    await user.click(updateButton)
    await waitFor(() => expect(fetchMock.mock.calls.filter(([input, init]) => String(input).includes('/enrichment/conflict/labels') && init?.method === 'POST')).toHaveLength(2))
    const labelWrites = fetchMock.mock.calls.filter(([input, init]) => String(input).includes('/enrichment/conflict/labels') && init?.method === 'POST')
    expect(JSON.parse(String(labelWrites[0]?.[1]?.body))).toMatchObject({ window_id: 'conflict-one', label: 'banter', notes: 'Reviewed in context' })
    expect(JSON.parse(String(labelWrites[0]?.[1]?.body))).not.toHaveProperty('evaluation_split')
    expect(JSON.parse(String(labelWrites[1]?.[1]?.body))).toMatchObject({ label: 'calm disagreement' })
    expect((labelWrites[0]?.[1]?.headers as Record<string, string>)['X-CSRF-Token']).toBe('stage3-test')
  })

  it('navigates the unlabelled queue and runs calibration only when server eligibility passes', async () => {
    const fetchMock = installFetchMock({ conflictGatePassed: true })
    document.cookie = 'meta_archive_csrf=stage3-test; path=/'
    const user = userEvent.setup()
    await renderAt('/enrichment')
    await user.click(await screen.findByRole('tab', { name: 'Conflict review' }))
    const calibration = await screen.findByRole('button', { name: 'Run held-out calibration' })
    expect(calibration).toBeEnabled()
    await user.click(screen.getByRole('button', { name: 'Next unlabelled window' }))
    expect(await screen.findByRole('heading', { name: 'Next unlabelled window' })).toBeVisible()
    expect(screen.getByLabelText('Manual label for review window conflict-two')).toBeVisible()
    await user.click(calibration)
    await waitFor(() => expect(fetchMock.mock.calls.some(([input, init]) => String(input).includes('/enrichment/conflict/calibrations') && init?.method === 'POST')).toBe(true))
    expect(await screen.findByText(/Calibration recorded: direct-conflict precision 80.0%/)).toBeVisible()
  })

  it('starts full-corpus conflict analytics only after the calibration gate passes', async () => {
    const fetchMock = installFetchMock({ conflictGatePassed: true })
    document.cookie = 'meta_archive_csrf=stage3-test; path=/'
    const user = userEvent.setup()
    await renderAt('/enrichment')
    const trigger = await screen.findByRole('button', { name: 'Build conflict aggregates' })
    expect(trigger).toBeEnabled()
    await user.click(trigger)
    const request = fetchMock.mock.calls.find(([input, init]) => String(input).includes('/enrichment/conflict/aggregate-jobs') && init?.method === 'POST')
    expect(request).toBeDefined()
    expect(JSON.parse(String(request?.[1]?.body))).toEqual({ batch_size: 32 })
  })
})

describe('Stage 5 deletion workflow', () => {
  beforeEach(() => { document.cookie = 'meta_archive_csrf=stage5-test; path=/' })

  it('requires a preview and exact operation confirmation before quarantine commit', async () => {
    const fetchMock = installFetchMock()
    const user = userEvent.setup()
    await renderAt('/deletion')
    expect(await screen.findByRole('heading', { name: 'Deletion and quarantine' })).toBeVisible()
    const previewButton = screen.getByRole('button', { name: 'Preview selection' })
    expect(previewButton).toBeDisabled()
    await user.type(screen.getByLabelText('Text to match'), 'abcd1234')
    expect(previewButton).toBeEnabled()
    await user.click(previewButton)
    expect(await screen.findByRole('heading', { name: '2. Review quarantine impact' })).toBeVisible()
    expect(await screen.findByRole('heading', { name: 'Exact records selected' })).toBeVisible()
    expect(await screen.findByText('abcd1234 with enough context to verify the exact selection.')).toBeVisible()
    expect(screen.getByRole('link', { name: 'Jump to message' })).toHaveAttribute('href', '/conversations/conversation-one?around=message-one#message-message-one')
    const selectionPages = screen.getByRole('navigation', { name: `Exact selection for operation ${operationId} pages` })
    await user.click(within(selectionPages).getByRole('button', { name: 'Next page' }))
    expect(await screen.findByText('photo.jpg')).toBeVisible()
    expect(screen.getByText(/2.00 KiB/)).toBeVisible()
    const commit = screen.getByRole('button', { name: 'Commit to quarantine' })
    expect(commit).toBeDisabled()
    await user.type(screen.getByLabelText('Type the operation ID to commit logical quarantine'), operationId)
    await user.click(commit)
    await waitFor(() => expect(fetchMock.mock.calls.some(([input, init]) => String(input).includes(`/${operationId}/commit`) && init?.method === 'POST')).toBe(true))
    expect(await screen.findByText(/Quarantined records are hidden from browser pages/)).toBeVisible()
    expect(screen.queryByRole('link', { name: 'Jump to message' })).not.toBeInTheDocument()
    expect(fetchMock.mock.calls.filter(([input]) => String(input).includes(`/${operationId}/selection?`)).length).toBeGreaterThanOrEqual(3)
    const writes = fetchMock.mock.calls.filter(([, init]) => init?.method === 'POST')
    expect(writes.every(([, init]) => (init?.headers as Record<string, string>)['X-CSRF-Token'] === 'stage5-test')).toBe(true)
  })

  it('separates restore from a second preview-gated irreversible workspace purge', async () => {
    const fetchMock = installFetchMock({ committed: true })
    const user = userEvent.setup()
    await renderAt('/deletion')
    const operations = await screen.findByRole('list', { name: 'Deletion operations' })
    const operationButton = within(operations).getByRole('button')
    expect(within(operations).getByRole('listitem')).toContainElement(operationButton)
    await user.click(operationButton)
    expect(await screen.findByText('Selector and audit details')).toBeVisible()
    await user.click(screen.getByText('Selector and audit details'))
    expect(screen.getByText('Audit data is a public allowlist of counts, digests, status codes, and timestamps. Private restore state, paths, terms, tokens, and recovery internals are never rendered.')).toBeVisible()
    expect(screen.getByText('Catalog revision:')).toBeVisible()
    expect(screen.queryByText('UI_PRIVATE_AUDIT_CANARY')).not.toBeInTheDocument()
    const restore = screen.getByRole('button', { name: 'Restore quarantine' })
    expect(restore).toBeDisabled()
    await user.click(screen.getByText('Safe purge summary'))
    expect(screen.getByText('Normalized rows')).toBeVisible()
    expect(screen.getByText('2.00 KiB')).toBeVisible()
    expect(screen.queryByText('UI_PRIVATE_CANARY')).not.toBeInTheDocument()
    expect(screen.getByLabelText('Purge scope')).toHaveValue('workspace')
    expect(screen.getByText(/Workspace scope is the public default/)).toBeVisible()
    await user.type(screen.getByLabelText('Type the operation ID to authorize an action'), operationId)
    await user.click(screen.getByRole('button', { name: 'Preview hard purge' }))
    expect(await screen.findByRole('heading', { name: 'Workspace hard purge' })).toBeVisible()
    await user.type(screen.getByLabelText('Type the operation ID to authorize an action'), operationId)
    await user.click(screen.getByRole('button', { name: 'Permanently execute hard purge' }))
    await waitFor(() => expect(fetchMock.mock.calls.some(([input, init]) => String(input).includes(`/${operationId}/purge`) && !String(input).includes('/preview') && init?.method === 'POST')).toBe(true))
  })

  it('keeps commit controls behind the selection loading and retry states', async () => {
    let releaseSelection!: () => void
    const selectionGate = new Promise<void>((resolve) => { releaseSelection = resolve })
    installFetchMock({ selectionGate })
    const user = userEvent.setup()
    await renderAt('/deletion')

    await user.type(await screen.findByLabelText('Text to match'), 'abcd1234')
    await user.click(screen.getByRole('button', { name: 'Preview selection' }))
    expect(await screen.findByText('Loading the exact selected records for this operation…')).toBeVisible()
    expect(screen.queryByLabelText('Type the operation ID to commit logical quarantine')).not.toBeInTheDocument()
    releaseSelection()
    expect(await screen.findByLabelText('Type the operation ID to commit logical quarantine')).toBeVisible()
  })

  it('never shows a prior operation selection or commit control while a replacement preview loads', async () => {
    let releaseSecondSelection!: () => void
    const secondSelectionGate = new Promise<void>((resolve) => { releaseSecondSelection = resolve })
    installFetchMock({ secondSelectionGate })
    const user = userEvent.setup()
    await renderAt('/deletion')

    const textInput = await screen.findByLabelText('Text to match')
    await user.type(textInput, 'abcd1234')
    await user.click(screen.getByRole('button', { name: 'Preview selection' }))
    expect(await screen.findByText('abcd1234 with enough context to verify the exact selection.')).toBeVisible()
    expect(await screen.findByLabelText('Type the operation ID to commit logical quarantine')).toBeVisible()

    await user.clear(textInput)
    await user.type(textInput, 'replacement')
    await user.click(screen.getByRole('button', { name: 'Preview selection' }))
    expect(await screen.findByText('Loading the exact selected records for this operation…')).toBeVisible()
    expect(screen.queryByText('abcd1234 with enough context to verify the exact selection.')).not.toBeInTheDocument()
    expect(screen.queryByLabelText('Type the operation ID to commit logical quarantine')).not.toBeInTheDocument()
    releaseSecondSelection()
    expect(await screen.findByLabelText('Type the operation ID to commit logical quarantine')).toBeVisible()
    expect(screen.getAllByText(new RegExp(olderOperationId)).length).toBeGreaterThan(0)
  })

  it('fails closed when exact selection details cannot load and retries in place', async () => {
    installFetchMock({ selectionFailsOnce: true })
    const user = userEvent.setup()
    await renderAt('/deletion')

    await user.type(await screen.findByLabelText('Text to match'), 'abcd1234')
    await user.click(screen.getByRole('button', { name: 'Preview selection' }))
    expect(await screen.findByText('Selection preview temporarily unavailable')).toBeVisible()
    expect(screen.queryByLabelText('Type the operation ID to commit logical quarantine')).not.toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Try again' }))
    expect(await screen.findByLabelText('Type the operation ID to commit logical quarantine')).toBeVisible()
  })

  it('warns that normalized-only purge leaves re-importable workspace source data', async () => {
    installFetchMock({ committed: true })
    const user = userEvent.setup()
    await renderAt('/deletion')
    await user.click(within(await screen.findByRole('list', { name: 'Deletion operations' })).getByRole('button'))

    await user.selectOptions(await screen.findByLabelText('Purge scope'), 'normalized')
    const warning = screen.getByText(/Normalized-only purge leaves matching records/)
    expect(warning).toHaveAttribute('role', 'alert')
    expect(warning).toHaveTextContent(/re-import can restore them/)
  })

  it('clears stale purge plans and locks every destructive action while replacement planning runs', async () => {
    let releasePurgePreview!: () => void
    const purgePreviewGateAfterFirst = new Promise<void>((resolve) => { releasePurgePreview = resolve })
    installFetchMock({ committed: true, purgePreviewGateAfterFirst })
    const user = userEvent.setup()
    await renderAt('/deletion')
    await user.click(within(await screen.findByRole('list', { name: 'Deletion operations' })).getByRole('button'))
    const confirmation = await screen.findByLabelText('Type the operation ID to authorize an action')
    await user.type(confirmation, operationId)
    await user.click(screen.getByRole('button', { name: 'Preview hard purge' }))
    expect(await screen.findByRole('heading', { name: 'Workspace hard purge' })).toBeVisible()
    await user.type(confirmation, operationId)

    await user.click(screen.getByRole('button', { name: 'Preview hard purge' }))
    expect(screen.queryByRole('heading', { name: 'Workspace hard purge' })).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Restore quarantine' })).toBeDisabled()
    expect(screen.getByRole('button', { name: 'Preview hard purge' })).toBeDisabled()
    expect(screen.getByLabelText('Purge scope')).toBeDisabled()
    expect(within(screen.getByRole('list', { name: 'Deletion operations' })).getByRole('button')).toBeDisabled()
    expect(screen.getByRole('button', { name: 'Refresh history' })).toBeDisabled()
    expect(within(screen.getByRole('navigation', { name: 'Deletion operation history pages' })).getByRole('button', { name: 'Next page' })).toHaveAttribute('aria-disabled', 'true')
    releasePurgePreview()
    expect(await screen.findByRole('heading', { name: 'Workspace hard purge' })).toBeVisible()
  })

  it('shows an operation-detail error and retries without losing the selected operation', async () => {
    installFetchMock({ committed: true, detailFailsOnce: true })
    const user = userEvent.setup()
    await renderAt('/deletion')
    await user.click(within(await screen.findByRole('list', { name: 'Deletion operations' })).getByRole('button'))
    expect(await screen.findByText('Operation detail temporarily unavailable')).toBeVisible()
    await user.click(screen.getByRole('button', { name: 'Try again' }))
    expect(await screen.findByText('Safe purge summary')).toBeVisible()
    expect(screen.getByRole('button', { name: 'Restore quarantine' })).toBeDisabled()
  })

  it('labels incompatible restore analytics as rebuild-required, not exact or current', async () => {
    installFetchMock({ restoreRebuildRequired: true })
    const user = userEvent.setup()
    await renderAt('/deletion')
    await user.click(within(await screen.findByRole('list', { name: 'Deletion operations' })).getByRole('button'))

    await screen.findAllByText('Derived analytics rebuild required')
    const notice = document.querySelector('.deletion-detail > .restore-rebuild-notice')
    expect(notice).toBeVisible()
    expect(within(notice as HTMLElement).getByText(/intentionally not presented as exact or current until rebuilt/)).toBeVisible()
    expect(within(notice as HTMLElement).getByText(/later quarantine that remains active/)).toBeVisible()
    expect(within(notice as HTMLElement).getByText('Search windows')).toBeVisible()
    expect(within(notice as HTMLElement).getByText('Semantic index')).toBeVisible()
    expect(within(notice as HTMLElement).getByText('NLP hourly rollups')).toBeVisible()
    expect(within(notice as HTMLElement).getByText('Conflict aggregates')).toBeVisible()
    expect(within(notice as HTMLElement).getByText('Topic clusters')).toBeVisible()
    expect(within(notice as HTMLElement).getByText('Detected invalidating operations: 2')).toBeVisible()
    await user.click(screen.getByText('Selector and audit details'))
    expect(screen.getAllByText('Derived analytics rebuild required')).toHaveLength(2)
    expect(screen.queryByText('UI_PRIVATE_REBUILD_CANARY')).not.toBeInTheDocument()
  })

  it('pages through older deletion operations and keeps their restore controls reachable', async () => {
    const fetchMock = installFetchMock({ committed: true })
    const user = userEvent.setup()
    await renderAt('/deletion')

    const pages = await screen.findByRole('navigation', { name: 'Deletion operation history pages' })
    expect(screen.getByText(operationId)).toBeVisible()
    await user.click(within(await screen.findByRole('list', { name: 'Deletion operations' })).getByRole('button'))
    await user.selectOptions(await screen.findByLabelText('Purge scope'), 'normalized')
    expect(screen.getByLabelText('Purge scope')).toHaveValue('normalized')
    await user.click(within(pages).getByRole('button', { name: 'Next page' }))
    const operations = await screen.findByRole('list', { name: 'Deletion operations' })
    expect(within(operations).getByText(olderOperationId)).toBeVisible()
    await user.click(within(operations).getByRole('button'))
    expect(await screen.findByText('Selector and audit details')).toBeVisible()
    expect(screen.getByRole('button', { name: 'Restore quarantine' })).toBeDisabled()
    expect(screen.getByLabelText('Purge scope')).toHaveValue('workspace')
    expect(fetchMock.mock.calls.some(([input]) => new URL(String(input), 'http://localhost').pathname === `/api/v1/deletions/${olderOperationId}`)).toBe(true)
    await user.click(within(pages).getByRole('button', { name: 'Previous page' }))
    expect(await screen.findByText(operationId)).toBeVisible()
  })

  it('makes attachment keyword deletion fail closed and explains conversation cascading', async () => {
    installFetchMock()
    const user = userEvent.setup()
    await renderAt('/deletion')

    const attachment = await screen.findByRole('checkbox', { name: 'Attachment occurrences' })
    await user.click(attachment)
    expect(attachment).toBeChecked()
    await user.selectOptions(screen.getByLabelText('Match mode'), 'keyword')
    expect(attachment).not.toBeChecked()
    expect(attachment).toBeDisabled()
    expect(screen.getByText(/Selecting Whole conversations finds conversations with a matching retained message, then cascades/)).toBeVisible()
    expect(screen.getByText(/Attachment-occurrence selection is unavailable in keyword mode/)).toBeVisible()
  })
})
