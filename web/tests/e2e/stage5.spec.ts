import { expect, type Page, test } from '@playwright/test'
import AxeBuilder from '@axe-core/playwright'
import { fixtures } from '../../src/test/fixtures'

const operationId = '11111111-1111-4111-8111-111111111111'
const olderOperationId = '22222222-2222-4222-8222-222222222222'
const audioSha256 = 'c'.repeat(64)
const conflictLabels = ['direct conflict', 'calm disagreement', 'banter', 'third-party conflict', 'no conflict', 'uncertain'] as const
const emptyConflictCounts = Object.fromEntries(conflictLabels.map((label) => [label, label === 'banter' ? 1 : 0]))
const conflictReviewItem = {
  window_id: 'conflict-review-one',
  conversation_id: 'conversation-one',
  evaluation_split: 'test',
  first_message_id: 'message-one',
  last_message_id: 'message-two',
  first_timestamp_ms: 1_752_000_000_000,
  last_timestamp_ms: 1_752_000_030_000,
  turn_count: 2,
  text_display: 'Alex: I do not agree with that.\nArchive Owner: This is getting tense.',
  current_manual_label: null,
  jump: { href: '/conversations/conversation-one?around=message-one#message-message-one', label: 'Open conversation at this window' },
}

function silentWave(durationSeconds = 4, sampleRate = 8_000) {
  const bytesPerSample = 2
  const dataSize = durationSeconds * sampleRate * bytesPerSample
  const wave = Buffer.alloc(44 + dataSize)
  wave.write('RIFF', 0)
  wave.writeUInt32LE(36 + dataSize, 4)
  wave.write('WAVE', 8)
  wave.write('fmt ', 12)
  wave.writeUInt32LE(16, 16)
  wave.writeUInt16LE(1, 20)
  wave.writeUInt16LE(1, 22)
  wave.writeUInt32LE(sampleRate, 24)
  wave.writeUInt32LE(sampleRate * bytesPerSample, 28)
  wave.writeUInt16LE(bytesPerSample, 32)
  wave.writeUInt16LE(16, 34)
  wave.write('data', 36)
  wave.writeUInt32LE(dataSize, 40)
  return wave
}

function enrichmentDestinationMessages() {
  return fixtures.conversationMessages.items.map((message) => {
    if (message.id !== 'message-100') return message
    return {
      ...message,
      attachments: [{
        id: 'media-document',
        kind: 'files',
        filename: 'notes.txt',
        media_type: 'text/plain',
        size_bytes: 330_000,
        timestamp_ms: message.timestamp_ms,
        availability: 'resolved',
        blob_sha256: 'd'.repeat(64),
        blob_href: `/api/v1/blobs/${'d'.repeat(64)}`,
        is_external_reference: false,
        jump: { href: '/conversations/conv-alex?around=message-100&media=media-document#message-message-100', conversation_id: 'conv-alex', message_id: 'message-100', fragment: 'message-message-100' },
      }],
    }
  })
}

async function mockApi(page: Page) {
  let operation = { id: operationId, status: 'committed', item_count: 2, impact: { selected: { message: 2 }, attachments: { exclusive_blobs: 1 } }, created_at: '2026-07-20T10:00:00Z', updated_at: '2026-07-20T10:00:00Z' }
  const olderOperation = { id: olderOperationId, status: 'committed', item_count: 5, impact: { selected: { message: 5 } }, created_at: '2026-06-20T10:00:00Z', updated_at: '2026-06-20T10:00:00Z' }
  await page.route('**/api/v1/**', async (route) => {
    const request = route.request()
    const url = new URL(request.url())
    const path = url.pathname
    if (path === '/api/v1/health') return route.fulfill({ json: { status: 'ok', catalog_ready: true, catalog_state: 'ready' } })
    if (path === '/api/v1/settings') return route.fulfill({ json: { timezone: 'America/New_York', cloud_enabled: false, image_semantic_stage_enabled: false } })
    if (path === '/api/v1/enrichment/status') return route.fulfill({ json: {
      audio_transcriptions: { completed: 1 },
      attachment_text_extractions: { completed: 1 },
      active_nlp_annotations: { sentiment: 1 },
      active_nlp_hourly_rollups: { sentiment: 1 },
      audio_transcription_selection: { current: { pipeline_version: 'audio-v1', model_name: 'faster-whisper', model_revision: 'small' }, current_record_count: 1, historical_record_count: 1 },
      attachment_text_selection: { current: { pipeline_version: 'extract-v1', extractor_name: 'safe-text', extractor_revision: 'revision-one' }, current_record_count: 1, historical_record_count: 1 },
      nlp_models: { sentiment: { model_name: 'fixture/sentiment', model_revision: 'revision-one', language_scope: 'English', warnings: [] } },
      attachment_sandbox: { available: true, backend: 'bubblewrap' },
      attachment_limits: {},
      conflict: { status: 'disabled_until_calibrated', reason: '499 more labels required.', sample_count: 1, aggregation_enabled: false },
      conflict_aggregates: { available: false, status: 'unavailable', reason: 'Calibration gate has not passed.', gate: {}, run: null, job: null, warnings: [], relationship_ranking_enabled: false, relationship_dimensions: [] },
      jobs: [],
    } })
    if (path === '/api/v1/enrichment/transcriptions/jobs' || path === '/api/v1/enrichment/attachment-text/jobs' || path === '/api/v1/enrichment/nlp/jobs') {
      return route.fulfill({ status: 202, json: { job_id: `fixture-${path.split('/').at(-2)}-job`, status: 'queued', status_href: '/api/v1/enrichment/status' } })
    }
    if (path === '/api/v1/enrichment/transcriptions') {
      const history = url.searchParams.get('version') === 'history'
      const offset = Number(url.searchParams.get('offset') ?? 0)
      return route.fulfill({ json: {
        items: [{ id: history ? 'transcript-history-one' : `transcript-${offset || 1}`, blob_sha256: audioSha256, pipeline_version: history ? 'audio-v0' : 'audio-v1', is_current_version: !history, status: 'completed', model_name: 'faster-whisper', model_revision: history ? 'tiny-legacy' : 'small', device: 'cpu', compute_type: 'int8', language: 'en', language_probability: .94, duration_ms: 4000, text: 'A retained audio message.', segment_count: 1, warnings: [], provenance: {}, error: null, segments: [{ id: 'segment-one', ordinal: 0, start_ms: 1250, end_ms: 3000, text: 'A retained audio message.', confidence: .91, jumps: [{ occurrence_id: 'attachment-audio', conversation_id: 'conv-alex', message_id: 'message-002', href: '/conversations/conv-alex?around=message-002&media=attachment-audio&t=1250#message-message-002' }] }] }],
        page: { limit: 25, offset, next_offset: offset === 0 ? 25 : null },
        selection: { scope: history ? 'history' : 'current', current: { pipeline_version: 'audio-v1', model_name: 'faster-whisper', model_revision: 'small' } },
      } })
    }
    if (path === '/api/v1/enrichment/attachment-text') {
      const history = url.searchParams.get('version') === 'history'
      const offset = Number(url.searchParams.get('offset') ?? 0)
      return route.fulfill({ json: {
        items: [{ id: history ? 'extract-history-one' : `extract-${offset || 1}`, blob_sha256: 'd'.repeat(64), pipeline_version: history ? 'extract-v0' : 'extract-v1', extractor_name: 'safe-text', extractor_revision: history ? 'legacy-revision' : 'revision-one', is_current_version: !history, status: 'completed', filename: 'notes.txt', media_type: 'text/plain', text_preview: 'Safely extracted fixture text.', preview_character_count: 30, preview_truncated: false, preview_limit: 16_000, character_count: 30, truncated: false, index_eligible: true, sandboxed: true, sandbox_status: 'available', limits: {}, warnings: [], provenance: {}, error: null, jumps: [{ occurrence_id: 'media-document', conversation_id: 'conv-alex', message_id: 'message-100', href: '/conversations/conv-alex?around=message-100&media=media-document#message-message-100' }] }],
        page: { limit: 25, offset, next_offset: offset === 0 ? 25 : null },
        selection: { scope: history ? 'history' : 'current', current: { pipeline_version: 'extract-v1', extractor_name: 'safe-text', extractor_revision: 'revision-one' } },
      } })
    }
    if (path === '/api/v1/enrichment/nlp') return route.fulfill({ json: { items: [{ id: 'nlp-one', target_type: 'message', target_id: 'message-one', task: url.searchParams.get('task') === 'all' ? 'sentiment' : url.searchParams.get('task'), model_name: 'fixture/sentiment', model_revision: 'revision-one', predicted_label: 'positive', confidence: .8, probabilities: { positive: .8, neutral: .15, negative: .05 }, probability_activation: 'softmax', attribution: { model_attribution: 'Fixture model' }, warnings: [], provenance: {} }], page: { limit: 50, offset: Number(url.searchParams.get('offset') ?? 0), next_offset: null } } })
    if (path === '/api/v1/enrichment/conflict/review/status') return route.fulfill({ json: {
      labels: conflictLabels,
      minimum_label_count: 500,
      eligible_window_count: 900,
      manual_label_count: 1,
      unlabelled_window_count: 899,
      evaluation_split_algorithm: { version: 'conversation-sha256-v1', ratios: { train: 70, validation: 15, test: 15 } },
      by_label: emptyConflictCounts,
      by_split: Object.fromEntries(['train', 'validation', 'test'].map((split) => [split, { eligible_window_count: 300, manual_label_count: split === 'test' ? 1 : 0, unlabelled_window_count: split === 'test' ? 299 : 300, by_label: emptyConflictCounts }])),
      validation: { eligible_for_calibration: false, conversation_disjoint: true, stratified: false, exact_split_assignment: true, missing_conversation_count: 0, split_assignment_mismatch_count: 0, leaked_conversation_count: 0, missing_strata: [{ evaluation_split: 'train', label: 'direct conflict' }], blockers: [{ code: 'insufficient_manual_labels', message: 'At least 500 genuine manual labels are required.', count: 499 }] },
      relationship_ranking_enabled: false,
      warning: 'Manual conflict review must never be used to rank people or relationships.',
    } })
    if (path === '/api/v1/enrichment/conflict/review/queue') return route.fulfill({ json: { items: [conflictReviewItem], page: { limit: 12, returned: 1, total: 899, next_after: url.searchParams.has('after') ? null : 'conflict-review-one' }, filters: { label: url.searchParams.get('label'), evaluation_split: url.searchParams.get('split') }, navigation: { next_unlabelled_href: '/api/v1/enrichment/conflict/review/next' } } })
    if (path === '/api/v1/enrichment/conflict/review/next') return route.fulfill({ json: { item: { ...conflictReviewItem, window_id: 'conflict-review-two' }, wrapped: false, navigation: { next_unlabelled_href: '/api/v1/enrichment/conflict/review/next?after=conflict-review-two' } } })
    if (path === '/api/v1/enrichment/conflict/search') return route.fulfill({ json: { items: [{ id: 'window:conflict-one', window_id: 'conflict-one', entity_type: 'message_window', source_id: 'message-one', rank: 1, score: 2.125, score_kind: 'nli_entailment_logit', excerpt: 'Alex: I do not agree with that.\nArchive Owner: This is getting tense.', conversation: { id: 'conversation-one', title: 'Alex' }, participants: ['Alex', 'Archive Owner'], window: { message_ids: ['message-one'], turn_count: 2 }, jump: { href: '/conversations/conversation-one?around=message-one#message-message-one' }, conflict_scoring: { target_label: 'direct conflict', predicted_label: 'direct conflict', scores: { 'direct conflict': 2.125, 'calm disagreement': 1.25, banter: -1.5, 'third-party conflict': -.7, 'no conflict': -2.25, uncertain: .1 }, score_kind: 'nli_entailment_logit', model_name: 'tasksource/ModernBERT-base-nli', model_revision: 'de4ab7e77845098b7fab7f6ab9d370ddff27b19c', quantization: 'dynamic_int8', ranking_feature_only: true, aggregation_gate_bypassed: false, warnings: [], provenance: { generated_locally: true } } }], page: { limit: 25, returned: 1, candidate_count: 4 }, query: 'argument or tense disagreement', target_label: 'direct conflict', conflict_model: { status: 'loaded_fixture' }, aggregation: { status: 'disabled_until_calibrated', sample_count: 1, aggregation_enabled: false } } })
    if (path === '/api/v1/enrichment/conflict/labels') return route.fulfill({ status: 201, json: { id: 'label-one', window_id: request.postDataJSON().window_id, label: request.postDataJSON().label, notes: request.postDataJSON().notes ?? null, evaluation_split: 'test', split_assignment: { version: 'conversation-sha256-v1', source: 'conversation_id' }, updated: false, calibration_invalidated: true } })
    if (path === '/api/v1/deletions' && request.method() === 'GET') {
      const secondPage = url.searchParams.get('cursor') === 'deletion-page-two'
      return route.fulfill({ json: { items: [secondPage ? olderOperation : operation], page: { total: 2, limit: 50, next_cursor: secondPage ? null : 'deletion-page-two' } } })
    }
    if (path === '/api/v1/deletions/preview') return route.fulfill({ status: 201, json: { operation_id: operationId, commit_token: 'commit-token-with-sufficient-length', catalog_revision: 'c'.repeat(64), selection_sha256: 'd'.repeat(64), impact: operation.impact } })
    if (path === `/api/v1/deletions/${operationId}/selection` && request.method() === 'GET') return route.fulfill({ json: {
      operation_id: operationId,
      revision: url.searchParams.get('revision'),
      status: 'previewed',
      items: [{
        entity_type: 'message', entity_id: 'message-one', reason: 'direct', label: 'Message from Archive Owner', label_truncated: false, snippet: 'abcd1234', snippet_truncated: false, preview_limit: 160, timestamp_ms: 1_752_000_000_000, resource_kind: 'generic', availability: null, media_type: null, size_bytes: null, quarantined: false,
        jump: { href: '/conversations/conversation-one?around=message-one#message-message-one', label: 'Jump to message' },
      }],
      page: { total: 1, limit: 25, next_cursor: null },
      privacy: { snippet_limit: 160, label_limit: 96, raw_fields_exposed: false },
    } })
    if (path === `/api/v1/deletions/${operationId}` && request.method() === 'GET') return route.fulfill({ json: { ...operation, selector: { text: 'abcd1234' }, audits: [] } })
    if (path === `/api/v1/deletions/${olderOperationId}` && request.method() === 'GET') return route.fulfill({ json: { ...olderOperation, selector: { text: 'older exact selection' }, audits: [] } })
    if (path === `/api/v1/deletions/${operationId}/commit`) { operation = { ...operation, status: 'committed' }; return route.fulfill({ json: { operation_id: operationId, status: 'committed', affected: { message: 2 }, audit_id: 'audit-one' } }) }
    if (path === `/api/v1/deletions/${operationId}/restore`) { operation = { ...operation, status: 'restored', updated_at: '2026-07-20T10:05:00Z' }; return route.fulfill({ json: { operation_id: operationId, status: 'restored', affected: { message: 2 }, audit_id: 'audit-two' } }) }
    if (path === `/api/v1/deletions/${operationId}/purge/preview`) return route.fulfill({ json: { operation_id: operationId, scope: request.postDataJSON().scope, executable: true, blockers: [], purge_token: 'purge-token-with-sufficient-length', purge_revision: 'e'.repeat(64), impact: { workspace_files: 2, blobs: 1 } } })
    if (path === `/api/v1/deletions/${operationId}/purge`) { operation = { ...operation, status: 'purged', updated_at: '2026-07-20T10:06:00Z' }; return route.fulfill({ json: { operation_id: operationId, status: 'purged', affected: { message: 2, blobs: 1 }, audit_id: 'audit-three' } }) }
    if (path === '/api/v1/conversations/conv-alex') return route.fulfill({ json: fixtures.conversationDetail })
    if (path === '/api/v1/conversations/conv-alex/messages') {
      const around = url.searchParams.get('around')
      return route.fulfill({ json: { ...fixtures.conversationMessages, items: enrichmentDestinationMessages(), focus: around ? { href: `/conversations/conv-alex?around=${around}`, message_id: around, fragment: `message-${around}` } : null } })
    }
    if (path === '/api/v1/conversation-corrections') return route.fulfill({ json: { items: [], identity_revision: 'a'.repeat(64), page: { limit: 100, offset: 0, total: 0, next_offset: null }, notice: 'Direct histories are provisional.' } })
    if (path === `/api/v1/blobs/${audioSha256}`) {
      const body = silentWave()
      return route.fulfill({
        status: 206,
        contentType: 'audio/wav',
        headers: {
          'Accept-Ranges': 'bytes',
          'Content-Range': `bytes 0-${body.length - 1}/${body.length}`,
        },
        body,
      })
    }
    return route.fulfill({ status: 404, json: { detail: `Unhandled ${request.method()} ${path}` } })
  })
}

test.beforeEach(async ({ page }) => { await mockApi(page) })

test('enrichment tabs expose panels and support standard arrow, Home, and End keys', async ({ page }) => {
  await page.goto('/enrichment')
  const overview = page.getByRole('tab', { name: 'Overview' })
  const transcripts = page.getByRole('tab', { name: 'Audio transcripts' })
  const conflict = page.getByRole('tab', { name: 'Conflict review' })

  await overview.focus()
  await page.keyboard.press('ArrowRight')
  await expect(transcripts).toBeFocused()
  await expect(transcripts).toHaveAttribute('aria-selected', 'true')
  await expect(page.getByRole('tabpanel', { name: 'Audio transcripts' })).toBeVisible()
  await page.keyboard.press('End')
  await expect(conflict).toBeFocused()
  await expect(conflict).toHaveAttribute('aria-selected', 'true')
  await page.keyboard.press('Home')
  await expect(overview).toBeFocused()
  await expect(overview).toHaveAttribute('aria-selected', 'true')
})

test('derived audio remains source-traceable and locally qualified', async ({ page }) => {
  await page.goto('/enrichment')
  await expect(page.getByRole('heading', { name: 'Transcripts and language signals' })).toBeVisible()
  await expect(page.locator('.topbar__title > span')).toHaveText('Stage 3')
  await expect(page.getByText('499 more labels required.')).toBeVisible()
  await page.getByRole('tab', { name: 'Audio transcripts' }).click()
  await expect(page.getByText('A retained audio message.').first()).toBeVisible()
  await page.getByText('Timestamped segments', { exact: true }).click()
  const jump = page.getByRole('link', { name: 'Jump to source' })
  await expect(jump).toHaveAttribute('href', '/conversations/conv-alex?around=message-002&media=attachment-audio&t=1250#message-message-002')
  await page.screenshot({ path: 'test-results/stage3-enrichment-desktop.png', fullPage: true })
  await jump.click()
  await expect(page).toHaveURL(/\/conversations\/conv-alex\?around=message-002&media=attachment-audio&t=1250#message-message-002/)
  const occurrence = page.locator('#attachment-attachment-audio')
  await expect(occurrence).toHaveClass(/message-attachment--focused/)
  await expect(occurrence).toBeFocused()
  const player = occurrence.locator('audio')
  await expect(player).toHaveAttribute('data-seek-ms', '1250')
  await expect.poll(() => player.evaluate((element) => (element as HTMLAudioElement).currentTime)).toBeGreaterThan(1.2)
  expect(await player.evaluate((element) => (element as HTMLAudioElement).currentTime)).toBeLessThan(1.3)
})

test('extracted attachment text jumps to its exact retained file occurrence', async ({ page }) => {
  await page.goto('/enrichment')
  await page.getByRole('tab', { name: 'Extracted text' }).click()
  await expect(page.getByText('Safely extracted fixture text.')).toBeVisible()
  const jump = page.getByRole('link', { name: 'Jump to source' })
  await expect(jump).toHaveAttribute('href', '/conversations/conv-alex?around=message-100&media=media-document#message-message-100')
  await jump.click()
  await expect(page).toHaveURL(/\/conversations\/conv-alex\?around=message-100&media=media-document#message-message-100/)
  const occurrence = page.locator('#attachment-media-document')
  await expect(occurrence).toHaveClass(/message-attachment--focused/)
  await expect(occurrence).toBeFocused()
  await expect(occurrence.getByRole('link', { name: /notes\.txt/ })).toBeVisible()
})

test('enrichment launches bounded jobs and browses every result selector and page', async ({ page }) => {
  test.setTimeout(60_000)
  await page.goto('/enrichment')

  for (const job of [
    { button: 'Transcribe audio', endpoint: '/enrichment/transcriptions/jobs' },
    { button: 'Extract attachment text', endpoint: '/enrichment/attachment-text/jobs' },
    { button: 'Annotate searchable text', endpoint: '/enrichment/nlp/jobs' },
  ]) {
    const requestPromise = page.waitForRequest((request) => request.url().includes(job.endpoint) && request.method() === 'POST')
    await page.getByRole('button', { name: job.button }).click()
    await requestPromise
  }
  await expect(page.getByRole('button', { name: 'Build conflict aggregates' })).toBeDisabled()

  await page.getByRole('tab', { name: 'Audio transcripts' }).click()
  await expect(page.getByText('Current configured results only')).toBeVisible()
  for (const status of ['processing', 'failed', 'completed']) {
    const requestPromise = page.waitForRequest((request) => {
      const url = new URL(request.url())
      return url.pathname === '/api/v1/enrichment/transcriptions' && url.searchParams.get('status') === status
    })
    await page.getByRole('combobox', { name: 'Status' }).selectOption(status)
    await requestPromise
  }
  await page.getByRole('combobox', { name: 'Status' }).selectOption('all')
  await expect(page.getByRole('combobox', { name: 'Status' })).toHaveValue('all')
  const historyRequest = page.waitForRequest((request) => request.url().includes('/enrichment/transcriptions?') && request.url().includes('version=history'))
  await page.getByLabel('Transcript version scope').selectOption('history')
  await historyRequest
  await expect(page.getByText('Version history (read-only)')).toBeVisible()
  const nextTranscriptPage = page.waitForRequest((request) => request.url().includes('/enrichment/transcriptions?') && request.url().includes('offset=25'))
  await page.getByRole('button', { name: 'Next', exact: true }).click()
  await nextTranscriptPage
  await expect(page.getByText('Starting at 26')).toBeVisible()
  await page.getByRole('button', { name: 'Previous', exact: true }).click()
  await expect(page.getByText('Starting at 1')).toBeVisible()

  await page.getByRole('tab', { name: 'Extracted text' }).click()
  for (const status of ['failed', 'excluded', 'unsupported', 'limit_exceeded', 'completed']) {
    const requestPromise = page.waitForRequest((request) => {
      const url = new URL(request.url())
      return url.pathname === '/api/v1/enrichment/attachment-text' && url.searchParams.get('status') === status
    })
    await page.getByRole('combobox', { name: 'Status' }).selectOption(status)
    await requestPromise
  }
  await page.getByRole('combobox', { name: 'Status' }).selectOption('all')
  await expect(page.getByRole('combobox', { name: 'Status' })).toHaveValue('all')
  const extractionHistoryRequest = page.waitForRequest((request) => request.url().includes('/enrichment/attachment-text?') && request.url().includes('version=history'))
  await page.getByLabel('Extraction version scope').selectOption('history')
  await extractionHistoryRequest
  await expect(page.getByText('Version history (read-only)')).toBeVisible()

  await page.getByRole('tab', { name: 'Language signals' }).click()
  for (const task of ['sentiment', 'emotion', 'toxicity']) {
    const requestPromise = page.waitForRequest((request) => {
      const url = new URL(request.url())
      return url.pathname === '/api/v1/enrichment/nlp' && url.searchParams.get('task') === task
    })
    await page.getByRole('combobox', { name: 'Task' }).selectOption(task)
    await requestPromise
  }
  await page.getByRole('combobox', { name: 'Task' }).selectOption('all')
  await expect(page.getByRole('combobox', { name: 'Task' })).toHaveValue('all')
  await expect(page.getByText('80.0%')).toBeVisible()
})

test('conflict workbench covers deterministic queue filters, paging, discovery, and server-assigned labels', async ({ page }) => {
  await page.goto('/enrichment')
  await page.getByRole('tab', { name: 'Conflict review' }).click()
  await expect(page.getByRole('heading', { name: 'Human calibration progress' })).toBeVisible()
  await expect(page.getByRole('button', { name: 'Run held-out calibration' })).toBeDisabled()
  await expect(page.getByText('At least 500 genuine manual labels are required.')).toBeVisible()

  const filteredQueue = page.waitForRequest((request) => {
    const url = new URL(request.url())
    return url.pathname === '/api/v1/enrichment/conflict/review/queue'
      && url.searchParams.get('label') === 'banter'
      && url.searchParams.get('split') === 'test'
  })
  await page.getByRole('combobox', { name: 'Manual label state' }).selectOption('banter')
  await page.getByRole('combobox', { name: 'Deterministic split' }).selectOption('test')
  await filteredQueue
  const queueNext = page.waitForRequest((request) => request.url().includes('/enrichment/conflict/review/queue?') && request.url().includes('after=conflict-review-one'))
  await page.getByRole('button', { name: 'Next queue page' }).click()
  await queueNext
  const nextReview = page.waitForRequest((request) => request.url().includes('/enrichment/conflict/review/next?') && request.url().includes('split=test'))
  await page.getByRole('button', { name: 'Next unlabelled window' }).click()
  await nextReview
  await expect(page.getByRole('heading', { name: 'Next unlabelled window' })).toBeVisible()

  await page.getByLabel('Rank for label').selectOption('calm disagreement')
  await page.getByLabel('Participant contains').fill('Alex')
  await page.getByLabel('Conversation ID').fill('conversation-one')
  const discoveryRequest = page.waitForRequest((request) => {
    const url = new URL(request.url())
    return url.pathname === '/api/v1/enrichment/conflict/search'
      && url.searchParams.get('label') === 'calm disagreement'
      && url.searchParams.get('participant') === 'Alex'
      && url.searchParams.get('conversation_id') === 'conversation-one'
  })
  await page.getByRole('button', { name: 'Search conflict windows' }).click()
  await discoveryRequest
  await expect(page.getByText('nli_entailment_logit: 2.125')).toBeVisible()
  await page.getByText('All six uncalibrated ranking features').click()
  await expect(page.getByText(/tasksource\/ModernBERT-base-nli/)).toBeVisible()
  await page.getByLabel('Human label for result 1').selectOption('banter')
  await page.getByLabel('Notes for result 1').fill('Reviewed in context')
  const labelRequest = page.waitForRequest((request) => request.url().endsWith('/enrichment/conflict/labels') && request.method() === 'POST')
  await page.getByRole('button', { name: 'Save human label' }).click()
  const labelWrite = await labelRequest
  expect(labelWrite.postDataJSON()).toEqual({ window_id: 'conflict-one', label: 'banter', notes: 'Reviewed in context' })
  await expect(page.getByRole('button', { name: 'Update human label' })).toBeEnabled()
  await expect(page.getByText(/Saved to the deterministic Test split/)).toBeVisible()
  await page.screenshot({ path: 'test-results/stage3-conflict-review-desktop.png', fullPage: true })
})

test('restore executes only after exact confirmation and updates the operation outcome', async ({ page }) => {
  await page.goto('/deletion')
  await page.getByRole('list', { name: 'Deletion operations' }).getByRole('button').first().click()
  await page.getByLabel('Type the operation ID to authorize an action').fill(operationId)
  const restoreRequest = page.waitForRequest((request) => request.url().endsWith(`/${operationId}/restore`) && request.method() === 'POST')
  await page.getByRole('button', { name: 'Restore quarantine' }).click()
  const request = await restoreRequest
  expect(request.postDataJSON()).toEqual({ confirmation: operationId })
  await expect(page.locator('.deletion-detail__header .status-badge')).toHaveText('Restored')
  await expect(page.getByRole('button', { name: 'Restore quarantine' })).toHaveCount(0)
})

test('older deletion operations remain reachable through cursor history', async ({ page }) => {
  await page.goto('/deletion')
  const pages = page.getByRole('navigation', { name: 'Deletion operation history pages' })
  await expect(page.getByText(operationId)).toBeVisible()
  await pages.getByRole('button', { name: 'Next page' }).click()
  const operations = page.getByRole('list', { name: 'Deletion operations' })
  await expect(operations.getByText(olderOperationId)).toBeVisible()
  await operations.getByRole('button').click()
  await expect(page.getByText('Selector and audit details')).toBeVisible()
  await pages.getByRole('button', { name: 'Previous page' }).click()
  await expect(page.getByText(operationId)).toBeVisible()
})

test('deletion requires preview and exact confirmations before executing workspace purge', async ({ page }) => {
  await page.goto('/deletion')
  await expect(page.locator('.topbar__title > span')).toHaveText('Stage 5')
  const preview = page.getByRole('button', { name: 'Preview selection' })
  await expect(preview).toBeDisabled()
  await page.getByLabel('Text to match').fill('abcd1234')
  await preview.click()
  const commitConfirmation = page.getByLabel('Type the operation ID to commit logical quarantine')
  await expect(commitConfirmation).toBeVisible()
  const commit = page.getByRole('button', { name: 'Commit to quarantine' })
  await expect(commit).toBeDisabled()
  await commitConfirmation.fill(operationId)
  await commit.click()
  await page.getByRole('list', { name: 'Deletion operations' }).getByRole('button').first().click()
  await page.getByLabel('Type the operation ID to authorize an action').fill(operationId)
  await page.getByLabel('Purge scope').selectOption('workspace')
  await page.getByRole('button', { name: 'Preview hard purge' }).click()
  await expect(page.getByRole('heading', { name: 'Workspace hard purge' })).toBeVisible()
  await expect(page.getByRole('button', { name: 'Permanently execute hard purge' })).toBeDisabled()
  await page.getByLabel('Type the operation ID to authorize an action').fill(operationId)
  const execute = page.getByRole('button', { name: 'Permanently execute hard purge' })
  await expect(execute).toBeEnabled()
  await page.screenshot({ path: 'test-results/stage5-deletion-desktop.png', fullPage: true })
  const purgeRequest = page.waitForRequest((request) => request.url().endsWith(`/${operationId}/purge`) && request.method() === 'POST')
  await execute.click()
  const request = await purgeRequest
  expect(request.postDataJSON()).toEqual({
    scope: 'workspace',
    purge_token: 'purge-token-with-sufficient-length',
    purge_revision: 'e'.repeat(64),
    confirmation: operationId,
  })
  await expect(page.locator('.deletion-detail__header .status-badge')).toHaveText('Purged')
  await expect(page.getByRole('button', { name: 'Restore quarantine' })).toHaveCount(0)
})

test('date-only deletion preview sends inclusive local dates without inventing a text selector', async ({ page }) => {
  await page.goto('/deletion')
  await page.getByLabel('Inclusive start date').fill('2023-06-01')
  await page.getByLabel('Inclusive end date').fill('2025-06-30')
  const previewRequest = page.waitForRequest((request) => request.url().endsWith('/deletions/preview') && request.method() === 'POST')
  await page.getByRole('button', { name: 'Preview selection' }).click()
  const request = await previewRequest
  expect(request.postDataJSON()).toEqual({
    entity_types: ['message', 'activity'],
    text_mode: 'exact',
    case_sensitive: true,
    whole_field: false,
    text_variant: 'display',
    from_date: '2023-06-01',
    to_date: '2025-06-30',
    timezone: 'America/New_York',
    conversation_ids: [],
    media_types: [],
    availability: [],
    entity_ids: {},
    allow_all: false,
  })
  await expect(page.getByRole('heading', { name: '2. Review quarantine impact' })).toBeVisible()
})

test('deletion selector exposes raw exact, keyword, attachment, participant, and unbounded safety modes', async ({ page }) => {
  await page.goto('/deletion')
  await page.getByLabel('Whole conversations').check()
  await page.getByLabel('Attachment occurrences').check()
  await page.getByLabel('Text to match').fill('Exact Fixture')
  await page.getByLabel('Text representation').selectOption('raw')
  await page.getByLabel('Case-sensitive').uncheck()
  await page.getByLabel('Whole field only').check()
  await page.getByLabel('Participant label contains').fill('Alex')
  await page.getByLabel('Conversation IDs').fill('conv-alex, conv-family\nconv-alex')
  await page.getByLabel('Photo').check()
  await page.getByLabel('Missing').check()
  const exactRequest = page.waitForRequest((request) => request.url().endsWith('/deletions/preview') && request.method() === 'POST')
  await page.getByRole('button', { name: 'Preview selection' }).click()
  expect((await exactRequest).postDataJSON()).toMatchObject({
    entity_types: ['message', 'activity', 'conversation', 'attachment'],
    text: 'Exact Fixture',
    text_mode: 'exact',
    case_sensitive: false,
    whole_field: true,
    text_variant: 'raw',
    participant: 'Alex',
    conversation_ids: ['conv-alex', 'conv-family'],
    media_types: ['photo'],
    availability: ['missing'],
  })

  await page.reload()
  await page.getByLabel('Text to match').fill('fixture phrase')
  await page.getByLabel('Match mode').selectOption('keyword')
  await expect(page.getByText(/Keyword mode uses normalized FTS/)).toBeVisible()
  await expect(page.getByLabel('Text representation')).toBeDisabled()
  const keywordRequest = page.waitForRequest((request) => request.url().endsWith('/deletions/preview') && request.method() === 'POST')
  await page.getByRole('button', { name: 'Preview selection' }).click()
  expect((await keywordRequest).postDataJSON()).toMatchObject({ text_mode: 'keyword', case_sensitive: false, whole_field: false, text_variant: 'display' })

  await page.reload()
  await page.getByLabel('Allow an unbounded preview').check()
  await expect(page.getByRole('button', { name: 'Preview selection' })).toBeDisabled()
  await page.getByLabel('Type PREVIEW ENTIRE ARCHIVE').fill('PREVIEW ENTIRE ARCHIVE')
  const allRequest = page.waitForRequest((request) => request.url().endsWith('/deletions/preview') && request.method() === 'POST')
  await page.getByRole('button', { name: 'Preview selection' }).click()
  expect((await allRequest).postDataJSON()).toMatchObject({ allow_all: true })
})

test('stale deletion commits, blocked purges, and API failures stay non-destructive and visible', async ({ page }) => {
  await page.route(`**/api/v1/deletions/${operationId}/commit`, (route) => route.fulfill({ status: 409, json: { detail: 'Catalog changed since preview. Re-run preview.' } }))
  await page.goto('/deletion')
  await page.getByLabel('Text to match').fill('abcd1234')
  await page.getByRole('button', { name: 'Preview selection' }).click()
  await page.getByLabel('Type the operation ID to commit logical quarantine').fill(operationId)
  await page.getByRole('button', { name: 'Commit to quarantine' }).click()
  await expect(page.getByRole('alert').filter({ hasText: 'Catalog changed since preview' })).toBeVisible()
  await expect(page.getByRole('heading', { name: '2. Review quarantine impact' })).toBeVisible()

  await page.unroute(`**/api/v1/deletions/${operationId}/commit`)
  await page.route(`**/api/v1/deletions/${operationId}/purge/preview`, (route) => route.fulfill({ json: { operation_id: operationId, scope: 'workspace', executable: false, blockers: ['A newer catalog revision exists. Preview again.'], purge_token: null, purge_revision: 'f'.repeat(64), impact: { normalized_rows: 2 } } }))
  await page.reload()
  await page.getByRole('list', { name: 'Deletion operations' }).getByRole('button').first().click()
  await page.getByLabel('Type the operation ID to authorize an action').fill(operationId)
  await page.getByRole('button', { name: 'Preview hard purge' }).click()
  await expect(page.getByText('A newer catalog revision exists. Preview again.')).toBeVisible()
  await expect(page.getByRole('button', { name: 'Permanently execute hard purge' })).toBeDisabled()

  await page.unroute(`**/api/v1/deletions/${operationId}/purge/preview`)
  await page.route('**/api/v1/deletions/preview', (route) => route.fulfill({ status: 500, json: { detail: { code: 'deletion_preview_failed', message: 'Fixture preview service failed safely.' } } }))
  await page.reload()
  await page.getByLabel('Text to match').fill('failure case')
  await page.getByRole('button', { name: 'Preview selection' }).click()
  await expect(page.getByRole('alert').filter({ hasText: 'Fixture preview service failed safely.' })).toBeVisible()
  await expect(page.getByRole('heading', { name: '2. Review quarantine impact' })).toHaveCount(0)
})

test('new stage pages are overflow-free and axe-clean at desktop and mobile sizes', async ({ page }) => {
  for (const viewport of [{ width: 1440, height: 1000 }, { width: 390, height: 844 }]) {
    await page.setViewportSize(viewport)
    for (const path of ['/enrichment', '/deletion']) {
      await page.goto(path)
      await expect(page.locator('h1')).toBeVisible()
      if (path === '/enrichment' && viewport.width === 390) {
        await expect(page.getByText(/Swipe to see more views/)).toBeVisible()
      }
      if (viewport.width === 390) {
        const noticeIcon = page.locator(path === '/enrichment' ? '.enrichment-safety > svg' : '.deletion-warning > svg')
        await expect(noticeIcon).toBeVisible()
        const iconMetrics = await noticeIcon.evaluate((element) => {
          const bounds = element.getBoundingClientRect()
          return { flexShrink: getComputedStyle(element).flexShrink, width: bounds.width, height: bounds.height }
        })
        expect(iconMetrics.flexShrink).toBe('0')
        expect(iconMetrics.width).toBeGreaterThanOrEqual(16)
        expect(iconMetrics.height).toBeGreaterThanOrEqual(16)
      }
      const widths = await page.locator('body').evaluate((element) => ({ scroll: element.scrollWidth, client: element.clientWidth }))
      expect(widths.scroll, `${path} at ${viewport.width}px`).toBeLessThanOrEqual(widths.client + 1)
      const result = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'wcag22a', 'wcag22aa']).analyze()
      expect(result.violations, `${path} at ${viewport.width}px: ${result.violations.map((violation) => `${violation.id}: ${violation.nodes.length}`).join(', ')}`).toEqual([])
      expect(result.incomplete, `${path} at ${viewport.width}px: incomplete ${result.incomplete.map((issue) => `${issue.id}: ${issue.nodes.length}`).join(', ')}`).toEqual([])
    }
  }
})

test('Stage 5 deletion has deterministic visual and accessibility baselines', async ({ page }) => {
  await page.emulateMedia({ reducedMotion: 'reduce' })
  await page.goto('/deletion')
  const main = page.getByRole('main')
  await expect(main.getByRole('heading', { name: 'Deletion and quarantine' })).toBeVisible()
  await expect(page).toHaveScreenshot('stage5-deletion-viewport.png', { animations: 'disabled' })
  await expect(main).toMatchAriaSnapshot({ name: 'stage5-deletion-main.aria.yml' })
})

test('Stage 3 enrichment has deterministic visual and accessibility baselines', async ({ page }) => {
  await page.emulateMedia({ reducedMotion: 'reduce' })
  await page.goto('/enrichment')
  const main = page.getByRole('main')
  await expect(main.getByRole('heading', { name: 'Transcripts and language signals' })).toBeVisible()
  await expect(main).toHaveScreenshot('stage3-enrichment-main.png', { animations: 'disabled', timeout: 15_000 })
  await expect(main).toMatchAriaSnapshot({ name: 'stage3-enrichment-main.aria.yml' })
})
