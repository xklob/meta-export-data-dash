import { keepPreviousData, useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { AudioLines, FileText, Languages, LockKeyhole, MessageSquareText, Play, Search, ShieldCheck, Tags } from 'lucide-react'
import { type KeyboardEvent, useMemo, useState } from 'react'
import { api } from '../lib/api'
import type { ConflictEvaluationSplit, ConflictLabel, ConflictReviewItem, ConflictReviewLabelFilter, ConflictReviewStatus, ConflictSearchResult, EnrichmentVersionSelection } from '../lib/contracts'
import { formatCount, formatSeconds, sentenceCase } from '../lib/format'
import { EmptyState, ErrorState, MetricCard, PageHeading, PageLoading, SectionHeader, StatusBadge } from '../components/ui'
import './EnrichmentPage.css'

type View = 'overview' | 'transcripts' | 'documents' | 'nlp' | 'conflict'

const views: { id: View; label: string }[] = [
  { id: 'overview', label: 'Overview' },
  { id: 'transcripts', label: 'Audio transcripts' },
  { id: 'documents', label: 'Extracted text' },
  { id: 'nlp', label: 'Language signals' },
  { id: 'conflict', label: 'Conflict review' },
]

const conflictLabels: ConflictLabel[] = ['direct conflict', 'calm disagreement', 'banter', 'third-party conflict', 'no conflict', 'uncertain']

function total(values: Record<string, number> | undefined) {
  return Object.values(values ?? {}).reduce((sum, value) => sum + value, 0)
}

function Pager({ offset, limit, nextOffset, onChange }: { offset: number; limit: number; nextOffset?: number | null; onChange: (offset: number) => void }) {
  return (
    <div className="enrichment-pager" role="navigation" aria-label="Results pagination">
      <button className="button button--secondary" type="button" disabled={offset === 0} onClick={() => onChange(Math.max(0, offset - limit))}>Previous</button>
      <span>Starting at {formatCount(offset + 1)}</span>
      <button className="button button--secondary" type="button" disabled={nextOffset == null} onClick={() => onChange(nextOffset ?? offset)}>Next</button>
    </div>
  )
}

function JumpLinks({ jumps }: { jumps: { href: string; occurrence_id: string }[] }) {
  const pageSize = 4
  const [offset, setOffset] = useState(0)
  if (!jumps.length) return <span className="enrichment-muted">No retained source occurrence</span>
  const page = jumps.slice(offset, offset + pageSize)
  return (
    <div className="enrichment-jumps">
      {page.map((jump, index) => <a className="button button--secondary" href={jump.href} key={jump.occurrence_id}>Jump to source{jumps.length > 1 ? ` ${offset + index + 1}` : ''}</a>)}
      {jumps.length > pageSize ? <div className="enrichment-jumps__pager" role="navigation" aria-label="Attachment occurrence pages"><button className="button button--secondary" type="button" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - pageSize))}>Previous occurrences</button><span>{formatCount(offset + 1)}–{formatCount(Math.min(offset + pageSize, jumps.length))} of {formatCount(jumps.length)}</span><button className="button button--secondary" type="button" disabled={offset + pageSize >= jumps.length} onClick={() => setOffset(offset + pageSize)}>Next occurrences</button></div> : null}
    </div>
  )
}

function RecordMetadata({ warnings, provenance, title = 'Warnings and provenance' }: { warnings: string[]; provenance: Record<string, unknown>; title?: string }) {
  if (!warnings.length && !Object.keys(provenance).length) return null
  return (
    <details className="enrichment-record__metadata">
      <summary>{title}</summary>
      {warnings.length ? <ul>{warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul> : null}
      {Object.keys(provenance).length ? <dl>{Object.entries(provenance).map(([key, value]) => <div key={key}><dt>{sentenceCase(key)}</dt><dd>{typeof value === 'object' ? JSON.stringify(value) : String(value)}</dd></div>)}</dl> : null}
    </details>
  )
}

const MAX_EXTRACTED_TEXT_RENDER_CHARACTERS = 16_000

function ExtractedTextPreview({
  textPreview,
  characterCount,
  previewTruncated,
  previewLimit,
}: {
  textPreview?: string | null
  characterCount?: number | null
  previewTruncated: boolean
  previewLimit: number
}) {
  if (!textPreview) return <p className="enrichment-muted">No bounded text preview is available for this record.</p>
  const visible = textPreview.slice(0, MAX_EXTRACTED_TEXT_RENDER_CHARACTERS)
  const totalCharacters = Math.max(characterCount ?? 0, textPreview.length)
  const bounded = previewTruncated || textPreview.length > visible.length || totalCharacters > visible.length
  return (
    <div className="extracted-text-preview">
      <pre className="extracted-text">{visible}</pre>
      {bounded ? <p role="note">Showing a bounded {formatCount(visible.length)}-character preview of {formatCount(totalCharacters)} extracted characters (preview limit: {formatCount(Math.min(previewLimit, MAX_EXTRACTED_TEXT_RENDER_CHARACTERS))}). This display limit is separate from extraction safety truncation; use local search to inspect matching sections.</p> : null}
    </div>
  )
}

function VersionSelectionSummary({ selection, kind }: { selection: EnrichmentVersionSelection; kind: 'transcript' | 'extraction' }) {
  const currentIdentity = kind === 'transcript'
    ? `${selection.current.model_name ?? 'Unknown model'} · ${selection.current.model_revision ?? 'unknown revision'}`
    : `${selection.current.extractor_name ?? 'Unknown extractor'} · ${selection.current.extractor_revision ?? 'unknown revision'}`
  return (
    <div className="enrichment-version-selection" role="note">
      <StatusBadge status={selection.scope === 'current' ? 'current' : 'history'} />
      <div>
        <strong>{selection.scope === 'current' ? 'Current configured results only' : 'Version history (read-only)'}</strong>
        <span>Current pipeline: <code>{selection.current.pipeline_version}</code> · {currentIdentity}</span>
        {selection.scope === 'history' ? <small>Historical records remain preserved and are labelled individually below; they are not presented as current results.</small> : null}
      </div>
    </div>
  )
}

function ConflictResultCard({ item }: { item: ConflictSearchResult }) {
  const queryClient = useQueryClient()
  const [label, setLabel] = useState<ConflictLabel | ''>('')
  const [notes, setNotes] = useState('')
  const mutation = useMutation({
    mutationFn: () => api.createConflictLabel({
      window_id: item.window_id!,
      label: label as ConflictLabel,
      ...(notes.trim() ? { notes: notes.trim() } : {}),
    }),
    onSuccess: async () => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ['enrichment-status'] }),
        queryClient.invalidateQueries({ queryKey: ['conflict-review-status'] }),
        queryClient.invalidateQueries({ queryKey: ['conflict-review-queue'] }),
      ])
    },
  })
  return (
    <article className="conflict-result">
      <header><div><span className="conflict-result__rank">#{item.rank}</span><StatusBadge status={item.conflict_scoring.predicted_label} /><strong>{item.conversation?.title ?? 'Unknown conversation'}</strong></div><span>{item.conflict_scoring.score_kind}: {item.score.toFixed(3)}</span></header>
      <pre>{item.excerpt}</pre>
      <details><summary>All six uncalibrated ranking features</summary><dl>{conflictLabels.map((candidate) => <div key={candidate}><dt>{sentenceCase(candidate)}</dt><dd>{item.conflict_scoring.scores[candidate].toFixed(3)}</dd></div>)}</dl><p>{item.conflict_scoring.model_name} · <code>{item.conflict_scoring.model_revision}</code> · {item.conflict_scoring.quantization}</p></details>
      <div className="conflict-result__review">
        <label className="filter-field"><span>Human label</span><select aria-label={`Human label for result ${item.rank}`} value={label} onChange={(event) => setLabel(event.target.value as ConflictLabel | '')}><option value="">Choose after reviewing</option>{conflictLabels.map((candidate) => <option value={candidate} key={candidate}>{sentenceCase(candidate)}</option>)}</select></label>
        <label className="filter-field conflict-result__notes"><span>Optional notes</span><input aria-label={`Notes for result ${item.rank}`} value={notes} onChange={(event) => setNotes(event.target.value)} /></label>
        <button className="button button--primary" type="button" disabled={mutation.isPending || !label} onClick={() => mutation.mutate()}><Tags size={15} aria-hidden="true" />{mutation.isPending ? 'Saving…' : mutation.data ? 'Update human label' : 'Save human label'}</button>
      </div>
      {mutation.data ? <p className="enrichment-success" role="status">Saved to the deterministic {sentenceCase(mutation.data.evaluation_split)} split. You can edit and save again.</p> : null}
      {mutation.isError ? <p className="enrichment-error" role="alert">{mutation.error.message}</p> : null}
      <footer><span>Ranking feature only · never used to rank people or relationships</span>{item.jump ? <a className="button button--secondary" href={item.jump.href}>Jump to conversation</a> : null}</footer>
    </article>
  )
}

function ConflictReviewCard({ item, onSaved }: { item: ConflictReviewItem; onSaved: () => Promise<void> }) {
  const [label, setLabel] = useState<ConflictLabel | ''>(item.current_manual_label?.label ?? '')
  const [notes, setNotes] = useState(item.current_manual_label?.notes ?? '')
  const mutation = useMutation({
    mutationFn: () => api.createConflictLabel({
      window_id: item.window_id,
      label: label as ConflictLabel,
      ...(notes.trim() ? { notes: notes.trim() } : {}),
    }),
    onSuccess: onSaved,
  })
  return (
    <article className="conflict-review-card">
      <header><div><StatusBadge status={item.evaluation_split} /><strong>{item.current_manual_label ? sentenceCase(item.current_manual_label.label) : 'Unlabelled window'}</strong><span>{formatCount(item.turn_count)} turns</span></div><a className="button button--secondary" href={item.jump.href}>{item.jump.label}</a></header>
      <pre>{item.text_display}</pre>
      <div className="conflict-result__review">
        <label className="filter-field"><span>Manual six-way label</span><select aria-label={`Manual label for review window ${item.window_id}`} value={label} onChange={(event) => setLabel(event.target.value as ConflictLabel | '')}><option value="">Choose after reviewing</option>{conflictLabels.map((candidate) => <option value={candidate} key={candidate}>{sentenceCase(candidate)}</option>)}</select></label>
        <label className="filter-field conflict-result__notes"><span>Review notes</span><input aria-label={`Notes for review window ${item.window_id}`} value={notes} onChange={(event) => setNotes(event.target.value)} /></label>
        <button className="button button--primary" type="button" disabled={!label || mutation.isPending} onClick={() => mutation.mutate()}><Tags size={15} aria-hidden="true" />{mutation.isPending ? 'Saving…' : item.current_manual_label || mutation.data ? 'Update review' : 'Save review'}</button>
      </div>
      {mutation.data ? <p className="enrichment-success" role="status">Saved in the server-assigned {sentenceCase(mutation.data.evaluation_split)} split. Editing remains available.</p> : null}
      {mutation.isError ? <p className="enrichment-error" role="alert">{mutation.error.message}</p> : null}
      <footer>Split assigned from the conversation ID · no people, conversations, or relationships are ranked</footer>
    </article>
  )
}

function ConflictReviewMatrix({ status }: { status: ConflictReviewStatus }) {
  const splits: ConflictEvaluationSplit[] = ['train', 'validation', 'test']
  return (
    <div className="conflict-review-matrix-wrap">
      <table className="data-table conflict-review-matrix">
        <caption>Manual labels by deterministic conversation-disjoint split</caption>
        <thead><tr><th>Human label</th>{splits.map((split) => <th key={split}>{sentenceCase(split)}</th>)}<th>Total</th></tr></thead>
        <tbody>{status.labels.map((label) => <tr key={label}><th scope="row">{sentenceCase(label)}</th>{splits.map((split) => <td key={split} className="number-cell">{formatCount(status.by_split[split].by_label[label] ?? 0)}</td>)}<td className="number-cell">{formatCount(status.by_label[label] ?? 0)}</td></tr>)}</tbody>
      </table>
    </div>
  )
}

export function EnrichmentPage() {
  const queryClient = useQueryClient()
  const [view, setView] = useState<View>('overview')
  const [statusFilter, setStatusFilter] = useState('all')
  const [taskFilter, setTaskFilter] = useState('all')
  const [offset, setOffset] = useState(0)
  const [transcriptVersion, setTranscriptVersion] = useState<'current' | 'history'>('current')
  const [extractionVersion, setExtractionVersion] = useState<'current' | 'history'>('current')
  const [conflictDraft, setConflictDraft] = useState({ q: 'argument or tense disagreement', label: 'direct conflict' as ConflictLabel, participant: '', conversationId: '' })
  const [conflictApplied, setConflictApplied] = useState<typeof conflictDraft | null>(null)
  const [reviewLabel, setReviewLabel] = useState<ConflictReviewLabelFilter>('unlabelled')
  const [reviewSplit, setReviewSplit] = useState<'all' | ConflictEvaluationSplit>('all')
  const [reviewAfter, setReviewAfter] = useState<string | undefined>()
  const [focusedReviewItem, setFocusedReviewItem] = useState<ConflictReviewItem | null>(null)
  const status = useQuery({
    queryKey: ['enrichment-status'],
    queryFn: api.enrichmentStatus,
    refetchInterval: (query) => query.state.data?.conflict_aggregates?.status === 'building' ? 1_500 : 15_000,
  })
  const transcripts = useQuery({
    queryKey: ['audio-transcriptions', statusFilter, transcriptVersion, offset],
    queryFn: () => api.audioTranscriptions(statusFilter, offset, 25, transcriptVersion),
    enabled: view === 'transcripts',
  })
  const documents = useQuery({
    queryKey: ['attachment-text', statusFilter, extractionVersion, offset],
    queryFn: () => api.attachmentTextExtractions(statusFilter, offset, 25, extractionVersion),
    enabled: view === 'documents',
  })
  const annotations = useQuery({
    queryKey: ['nlp-annotations', taskFilter, offset],
    queryFn: () => api.nlpAnnotations(taskFilter, offset),
    enabled: view === 'nlp',
    placeholderData: keepPreviousData,
  })
  const conflict = useQuery({
    queryKey: ['conflict-search', conflictApplied],
    queryFn: () => api.conflictSearch(conflictApplied!),
    enabled: view === 'conflict' && conflictApplied != null,
  })
  const conflictReviewStatus = useQuery({
    queryKey: ['conflict-review-status'],
    queryFn: api.conflictReviewStatus,
    enabled: view === 'conflict',
  })
  const conflictReviewQueue = useQuery({
    queryKey: ['conflict-review-queue', reviewLabel, reviewSplit, reviewAfter],
    queryFn: () => api.conflictReviewQueue({ label: reviewLabel, split: reviewSplit, after: reviewAfter }),
    enabled: view === 'conflict',
    placeholderData: keepPreviousData,
  })
  const refreshStatus = async () => { await queryClient.invalidateQueries({ queryKey: ['enrichment-status'] }) }
  const refreshConflictReview = async () => {
    await Promise.all([
      queryClient.invalidateQueries({ queryKey: ['conflict-review-status'] }),
      queryClient.invalidateQueries({ queryKey: ['conflict-review-queue'] }),
      queryClient.invalidateQueries({ queryKey: ['enrichment-status'] }),
    ])
  }
  const startAudio = useMutation({ mutationFn: () => api.startAudioTranscriptionJob(true), onSuccess: refreshStatus })
  const startDocuments = useMutation({ mutationFn: () => api.startAttachmentTextJob(true), onSuccess: refreshStatus })
  const startNlp = useMutation({ mutationFn: () => api.startNlpAnnotationJob(), onSuccess: refreshStatus })
  const startConflictAggregate = useMutation({ mutationFn: () => api.startConflictAggregateJob(), onSuccess: refreshStatus })
  const nextConflictReview = useMutation({
    mutationFn: () => api.nextConflictReviewItem({ split: reviewSplit, after: focusedReviewItem?.window_id }),
    onSuccess: (data) => setFocusedReviewItem(data.item ?? null),
  })
  const createConflictCalibration = useMutation({
    mutationFn: api.createConflictCalibration,
    onSuccess: refreshConflictReview,
  })
  const runningJobs = useMemo(() => status.data?.jobs.filter((job) => ['queued', 'running', 'processing'].includes(job.status)).length ?? 0, [status.data])
  const activeKinds = new Set(status.data?.jobs.filter((job) => ['queued', 'running', 'processing'].includes(job.status)).map((job) => job.kind) ?? [])

  const selectView = (next: View) => {
    setView(next)
    setOffset(0)
    setStatusFilter('all')
    if (next !== 'conflict') setConflictApplied(null)
  }
  const moveTabFocus = (event: KeyboardEvent<HTMLButtonElement>, current: View) => {
    const currentIndex = views.findIndex((item) => item.id === current)
    let nextIndex: number | undefined
    if (event.key === 'ArrowRight' || event.key === 'ArrowDown') nextIndex = (currentIndex + 1) % views.length
    if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') nextIndex = (currentIndex - 1 + views.length) % views.length
    if (event.key === 'Home') nextIndex = 0
    if (event.key === 'End') nextIndex = views.length - 1
    if (nextIndex == null) return
    event.preventDefault()
    const next = views[nextIndex]!.id
    selectView(next)
    requestAnimationFrame(() => document.getElementById(`enrichment-tab-${next}`)?.focus())
  }

  const pageHeading = <PageHeading eyebrow="Stage 3 · Local processing" title="Transcripts and language signals" description="Browse locally generated audio transcripts, safely extracted attachment text, and model-derived annotations with source jumps and model provenance." actions={<a className="button button--secondary" href="/jobs">View background jobs</a>} />

  if (status.isPending) return <div className="page-stack enrichment-page">{pageHeading}<PageLoading label="Loading local enrichment status" /></div>
  if (status.isError || !status.data) return <div className="page-stack enrichment-page">{pageHeading}<ErrorState error={status.error} onRetry={() => void status.refetch()} /></div>
  const conflictLabelCount = status.data.conflict.sample_count ?? status.data.conflict.label_count ?? 0
  const conflictReason = status.data.conflict.reason ?? status.data.conflict.warning ?? 'Aggregate conflict charts remain disabled until calibration requirements are met.'
  const aggregate = status.data.conflict_aggregates
  const aggregateRunning = activeKinds.has('conflict_aggregates') || aggregate?.status === 'building'
  const canStartAggregate = status.data.conflict.aggregation_enabled === true && !aggregateRunning && !aggregate?.available
  const aggregateReason = status.data.conflict.aggregation_enabled === true ? aggregate?.reason ?? conflictReason : conflictReason

  return (
    <div className="page-stack enrichment-page">
      {pageHeading}

      <div className="enrichment-safety" role="note"><LockKeyhole size={18} aria-hidden="true" /><div><strong>Local by default</strong><span>These derived records stay on this machine. Model labels are estimates—not facts about people or relationships.</span></div></div>

      <div className="horizontal-tabs enrichment-tab-overflow">
        <div className="tab-list horizontal-tabs__scroller" role="tablist" aria-label="Enrichment views" aria-orientation="horizontal">
          {views.map((item) => <button key={item.id} id={`enrichment-tab-${item.id}`} role="tab" aria-selected={view === item.id} aria-controls={`enrichment-panel-${item.id}`} tabIndex={view === item.id ? 0 : -1} className={view === item.id ? 'tab-button tab-button--active' : 'tab-button'} type="button" onClick={() => selectView(item.id)} onKeyDown={(event) => moveTabFocus(event, item.id)}>{item.label}</button>)}
        </div>
        <span className="horizontal-tabs__hint" aria-hidden="true">Swipe to see more views →</span>
      </div>

      {view === 'overview' ? (
        <div id="enrichment-panel-overview" role="tabpanel" aria-labelledby="enrichment-tab-overview" tabIndex={0} className="enrichment-tab-panel">
          <section className="metric-grid" aria-label="Enrichment totals">
            <MetricCard icon={AudioLines} label="Audio transcriptions" value={formatCount(total(status.data.audio_transcriptions))} detail={status.data.audio_transcription_selection ? `${formatCount(status.data.audio_transcription_selection.current_record_count)} current · ${formatCount(status.data.audio_transcription_selection.historical_record_count)} historical` : `${formatCount(status.data.audio_transcriptions.completed ?? 0)} completed`} />
            <MetricCard icon={FileText} label="Attachment extractions" value={formatCount(total(status.data.attachment_text_extractions))} detail={status.data.attachment_text_selection ? `${formatCount(status.data.attachment_text_selection.current_record_count)} current · ${formatCount(status.data.attachment_text_selection.historical_record_count)} historical` : `${formatCount(status.data.attachment_text_extractions.completed ?? 0)} searchable`} />
            <MetricCard icon={Languages} label="Active NLP annotations" value={formatCount(total(status.data.active_nlp_annotations))} detail={`${formatCount(total(status.data.active_nlp_hourly_rollups))} hourly rollups`} />
            <MetricCard icon={MessageSquareText} label="Processing jobs" value={formatCount(runningJobs)} detail={`${formatCount(status.data.jobs.length)} recent jobs`} tone={runningJobs ? 'warning' : 'positive'} />
          </section>

          <section className="panel enrichment-job-launcher">
            <SectionHeader title="Run local enrichment" description="Start or resume bounded background jobs in this localhost process. Audio uses faster-whisper small on CPU INT8; attachment text stays inside Bubblewrap; language labels remain fallible local estimates." />
            <div>
              <button className="button button--secondary" type="button" disabled={startAudio.isPending || activeKinds.has('audio_transcription')} onClick={() => startAudio.mutate()}><Play size={14} aria-hidden="true" />{activeKinds.has('audio_transcription') ? 'Audio job running' : 'Transcribe audio'}</button>
              <button className="button button--secondary" type="button" disabled={startDocuments.isPending || activeKinds.has('attachment_text')} onClick={() => startDocuments.mutate()}><Play size={14} aria-hidden="true" />{activeKinds.has('attachment_text') ? 'Extraction running' : 'Extract attachment text'}</button>
              <button className="button button--secondary" type="button" disabled={startNlp.isPending || activeKinds.has('nlp_annotations')} onClick={() => startNlp.mutate()}><Play size={14} aria-hidden="true" />{activeKinds.has('nlp_annotations') ? 'NLP job running' : 'Annotate searchable text'}</button>
              <button className="button button--secondary" type="button" disabled={!canStartAggregate || startConflictAggregate.isPending} title={status.data.conflict.aggregation_enabled ? 'Build resumable full-corpus time aggregates' : 'Pass the held-out calibration gate first'} onClick={() => startConflictAggregate.mutate()}><Play size={14} aria-hidden="true" />{aggregateRunning ? 'Conflict aggregate running' : aggregate?.available ? 'Conflict aggregates ready' : 'Build conflict aggregates'}</button>
            </div>
            {startAudio.isError || startDocuments.isError || startNlp.isError || startConflictAggregate.isError ? <p className="enrichment-error" role="alert">{startAudio.error?.message ?? startDocuments.error?.message ?? startNlp.error?.message ?? startConflictAggregate.error?.message}</p> : null}
          </section>

          <section className="panel enrichment-grid-section">
            <SectionHeader title="Model and sandbox provenance" description="Every annotation retains its exact model revision and configuration. Attachment extraction fails closed if its local sandbox is unavailable." />
            <div className="enrichment-model-grid">
              {Object.entries(status.data.nlp_models).map(([task, model]) => <article key={task}><div><strong>{sentenceCase(task)}</strong><StatusBadge status="available" /></div><p>{model.model_name}</p><code>{model.model_revision}</code><small>{model.language_scope}</small>{model.warnings.map((warning) => <span className="enrichment-warning" key={warning}>{warning}</span>)}</article>)}
              <article><div><strong>Attachment sandbox</strong><StatusBadge status={status.data.attachment_sandbox.available === false ? 'unavailable' : 'available'} /></div><p>Network-isolated, bounded extraction</p><code>{String(status.data.attachment_sandbox.backend ?? 'Bubblewrap + prlimit')}</code><small>{String(status.data.attachment_sandbox.reason ?? 'Limits are recorded with every extraction.')}</small></article>
            </div>
          </section>

          <section className="panel enrichment-conflict">
            <SectionHeader title="Conflict calibration and aggregate gate" description="Conflict labels remain a search aid until at least 500 user-labelled windows and the held-out precision/F1 gates pass." action={<StatusBadge status={aggregate?.status ?? status.data.conflict.status} />} />
            <div className="enrichment-conflict__body"><ShieldCheck size={22} aria-hidden="true" /><div><strong>{formatCount(conflictLabelCount)} labelled windows</strong><p>{aggregate?.available ? `${formatCount(aggregate.run?.scored_count)} classified windows are published into time-only analytics.` : aggregateReason}</p>{aggregateRunning && aggregate?.job ? <p role="status">{aggregate.job.stage ?? 'Scoring'} · {formatCount(aggregate.job.progress_current)} of {formatCount(aggregate.job.progress_total)}</p> : null}<span>Relationship and conversation ranking: disabled</span></div></div>
          </section>
        </div>
      ) : <div id="enrichment-panel-overview" role="tabpanel" aria-labelledby="enrichment-tab-overview" hidden />}

      {view === 'transcripts' ? (
        <section id="enrichment-panel-transcripts" role="tabpanel" aria-labelledby="enrichment-tab-transcripts" tabIndex={0} className="panel">
          <SectionHeader title="Audio transcripts" description="Timestamped segments generated by the local faster-whisper worker. Jumps return to every retained occurrence of the audio." action={<div className="enrichment-section-filters"><label className="filter-field"><span>Status</span><select value={statusFilter} onChange={(event) => { setStatusFilter(event.target.value); setOffset(0) }}><option value="all">All</option><option value="completed">Completed</option><option value="processing">Processing</option><option value="failed">Failed</option></select></label><label className="filter-field"><span>Pipeline results</span><select aria-label="Transcript version scope" value={transcriptVersion} onChange={(event) => { setTranscriptVersion(event.target.value as 'current' | 'history'); setOffset(0) }}><option value="current">Current only</option><option value="history">All version history</option></select></label></div>} />
          {transcripts.data ? <VersionSelectionSummary selection={transcripts.data.selection} kind="transcript" /> : null}
          {transcripts.isPending ? <PageLoading label="Loading audio transcripts" /> : transcripts.isError ? <ErrorState error={transcripts.error} onRetry={() => void transcripts.refetch()} /> : !transcripts.data?.items.length ? <EmptyState title="No matching audio transcripts" description="Start the local CPU transcription job or choose another status." action={<button className="button button--secondary" type="button" disabled={startAudio.isPending || activeKinds.has('audio_transcription')} onClick={() => startAudio.mutate()}><Play size={14} aria-hidden="true" />Start audio transcription</button>} /> : <div className="enrichment-record-list">{transcripts.data.items.map((item) => <article key={item.id} className="enrichment-record"><header><div><StatusBadge status={item.status} /><StatusBadge status={item.is_current_version ? 'current' : 'historical'} /><strong>{item.language ? item.language.toUpperCase() : 'Language unknown'}</strong><span>{formatSeconds((item.duration_ms ?? 0) / 1000)} · {formatCount(item.segment_count)} segments{item.language_probability != null ? ` · ${Math.round(item.language_probability * 100)}% language score` : ''}</span></div><code>{item.blob_sha256.slice(0, 16)}…</code></header>{item.error ? <p className="enrichment-error">{item.error}</p> : null}{item.text ? <p className="enrichment-record__text">{item.text}</p> : null}<details><summary>Timestamped segments</summary><ol className="transcript-segments">{item.segments.map((segment) => <li key={segment.id}><div><time>{formatSeconds(segment.start_ms / 1000)}</time><p>{segment.text || 'No speech retained'}</p>{segment.confidence != null ? <small>{Math.round(segment.confidence * 100)}% local decoder score · not calibrated</small> : null}</div><JumpLinks jumps={segment.jumps} /></li>)}</ol></details><RecordMetadata warnings={item.warnings} provenance={item.provenance} /><footer><span>{item.model_name ? `${item.model_name} · ${item.model_revision} · ${item.device ?? 'CPU'} ${item.compute_type ?? ''}` : 'Model provenance unavailable'} · pipeline <code>{item.pipeline_version}</code></span></footer></article>)}</div>}
          {transcripts.data ? <Pager offset={offset} limit={transcripts.data.page.limit} nextOffset={transcripts.data.page.next_offset} onChange={setOffset} /> : null}
        </section>
      ) : <section id="enrichment-panel-transcripts" role="tabpanel" aria-labelledby="enrichment-tab-transcripts" hidden />}

      {view === 'documents' ? (
        <section id="enrichment-panel-documents" role="tabpanel" aria-labelledby="enrichment-tab-documents" tabIndex={0} className="panel">
          <SectionHeader title="Extracted attachment text" description="Inert text extracted under local sandbox limits. Executable and credential-like formats are excluded from search." action={<div className="enrichment-section-filters"><label className="filter-field"><span>Status</span><select value={statusFilter} onChange={(event) => { setStatusFilter(event.target.value); setOffset(0) }}><option value="all">All</option><option value="completed">Completed</option><option value="failed">Failed</option><option value="excluded">Excluded</option><option value="unsupported">Unsupported</option><option value="limit_exceeded">Limit exceeded</option></select></label><label className="filter-field"><span>Pipeline results</span><select aria-label="Extraction version scope" value={extractionVersion} onChange={(event) => { setExtractionVersion(event.target.value as 'current' | 'history'); setOffset(0) }}><option value="current">Current only</option><option value="history">All version history</option></select></label></div>} />
          {documents.data ? <VersionSelectionSummary selection={documents.data.selection} kind="extraction" /> : null}
          {documents.isPending ? <PageLoading label="Loading extracted attachment text" /> : documents.isError ? <ErrorState error={documents.error} onRetry={() => void documents.refetch()} /> : !documents.data?.items.length ? <EmptyState title="No matching extraction records" description="Start the fail-closed local sandbox job or choose another status." action={<button className="button button--secondary" type="button" disabled={startDocuments.isPending || activeKinds.has('attachment_text')} onClick={() => startDocuments.mutate()}><Play size={14} aria-hidden="true" />Start attachment extraction</button>} /> : <div className="enrichment-record-list">{documents.data.items.map((item) => <article key={item.id} className="enrichment-record"><header><div><StatusBadge status={item.status} /><StatusBadge status={item.is_current_version ? 'current' : 'historical'} /><strong>{item.filename || 'Unnamed attachment'}</strong><span>{item.media_type || 'Unknown file type'} · {formatCount(item.character_count)} characters</span></div><span>{item.sandboxed ? 'Sandboxed' : 'Not executed'}</span></header>{item.error ? <p className="enrichment-error">{item.error}</p> : null}<ExtractedTextPreview textPreview={item.text_preview} characterCount={item.character_count} previewTruncated={item.preview_truncated} previewLimit={item.preview_limit} /><RecordMetadata warnings={item.warnings} provenance={{ ...item.provenance, sandbox_status: item.sandbox_status, limits: item.limits }} /><footer><span>{item.index_eligible ? 'Included in local search' : 'Excluded from local search'}{item.truncated ? ' · Extraction truncated at a recorded safety limit' : ''} · {item.extractor_name} · {item.extractor_revision} · pipeline <code>{item.pipeline_version}</code></span><JumpLinks jumps={item.jumps} /></footer></article>)}</div>}
          {documents.data ? <Pager offset={offset} limit={documents.data.page.limit} nextOffset={documents.data.page.next_offset} onChange={setOffset} /> : null}
        </section>
      ) : <section id="enrichment-panel-documents" role="tabpanel" aria-labelledby="enrichment-tab-documents" hidden />}

      {view === 'nlp' ? (
        <section id="enrichment-panel-nlp" role="tabpanel" aria-labelledby="enrichment-tab-nlp" tabIndex={0} className="panel">
          <SectionHeader title="Derived language signals" description="Full model probability vectors and revisions are retained. These labels are uncertain derived data and are never used to rank relationships." action={<label className="filter-field"><span>Task</span><select value={taskFilter} onChange={(event) => { setTaskFilter(event.target.value); setOffset(0) }}><option value="all">All tasks</option><option value="sentiment">Sentiment</option><option value="emotion">Emotion</option><option value="toxicity">Toxicity</option></select></label>} />
          {annotations.isPending ? <PageLoading label="Loading language annotations" /> : annotations.isError ? <ErrorState error={annotations.error} onRetry={() => void annotations.refetch()} /> : !annotations.data?.items.length ? <EmptyState title="No matching language annotations" description="Start the local model job or choose another task." action={<button className="button button--secondary" type="button" disabled={startNlp.isPending || activeKinds.has('nlp_annotations')} onClick={() => startNlp.mutate()}><Play size={14} aria-hidden="true" />Start local annotation</button>} /> : <div className="enrichment-annotation-grid">{annotations.data.items.map((item) => <article key={item.id}><header><StatusBadge status={item.task} /><strong>{sentenceCase(item.predicted_label)}</strong><span>{Math.round(item.confidence * 100)}% model estimate</span></header><dl>{Object.entries(item.probabilities).sort((a, b) => b[1] - a[1]).map(([label, probability]) => <div key={label}><dt>{sentenceCase(label)}</dt><dd>{(probability * 100).toFixed(1)}%</dd></div>)}</dl><RecordMetadata warnings={item.warnings} provenance={{ ...item.provenance, attribution: item.attribution, probability_activation: item.probability_activation }} title="Model attribution, warnings, and provenance" /><footer><code>{item.target_type}:{item.target_id}</code><span>{item.model_name} · {item.model_revision}</span></footer></article>)}</div>}
          {annotations.data ? <Pager offset={offset} limit={annotations.data.page.limit} nextOffset={annotations.data.page.next_offset} onChange={setOffset} /> : null}
        </section>
      ) : <section id="enrichment-panel-nlp" role="tabpanel" aria-labelledby="enrichment-tab-nlp" hidden />}

      {view === 'conflict' ? (
        <section id="enrichment-panel-conflict" role="tabpanel" aria-labelledby="enrichment-tab-conflict" tabIndex={0} className="panel conflict-workbench">
          <SectionHeader title="Conflict search and human review" description="Semantic retrieval is reranked locally with six ModernBERT NLI hypotheses. Scores are raw entailment-logit ranking features—not calibrated probabilities—and aggregate charts stay disabled until the explicit evaluation gate passes." action={<StatusBadge status={status.data.conflict.status} />} />
          {conflictReviewStatus.isPending ? <PageLoading label="Auditing the manual conflict-review set" /> : conflictReviewStatus.isError || !conflictReviewStatus.data ? <ErrorState title="Conflict review status is unavailable" error={conflictReviewStatus.error} onRetry={() => void conflictReviewStatus.refetch()} /> : (
            <section className="conflict-calibration-panel" aria-labelledby="conflict-calibration-heading">
              <div className="section-header"><div><h2 id="conflict-calibration-heading">Human calibration progress</h2><p>Conversation IDs are assigned once by {conflictReviewStatus.data.evaluation_split_algorithm.version}; every window from one conversation stays in the same evaluation split.</p></div><StatusBadge status={conflictReviewStatus.data.validation.eligible_for_calibration ? 'eligible' : 'collecting labels'} /></div>
              <div className="metric-grid metric-grid--four" role="region" aria-label="Conflict review totals">
                <MetricCard icon={Tags} label="Manual labels" value={formatCount(conflictReviewStatus.data.manual_label_count)} detail={`${formatCount(Math.max(0, conflictReviewStatus.data.minimum_label_count - conflictReviewStatus.data.manual_label_count))} still required`} />
                <MetricCard icon={MessageSquareText} label="Unlabelled windows" value={formatCount(conflictReviewStatus.data.unlabelled_window_count)} detail={`${formatCount(conflictReviewStatus.data.eligible_window_count)} eligible message windows`} />
                <MetricCard icon={ShieldCheck} label="Conversation leakage" value={formatCount(conflictReviewStatus.data.validation.leaked_conversation_count)} detail={conflictReviewStatus.data.validation.conversation_disjoint ? 'Conversation-disjoint' : 'Calibration blocked'} tone={conflictReviewStatus.data.validation.conversation_disjoint ? 'positive' : 'warning'} />
                <MetricCard icon={Languages} label="Missing label/split cells" value={formatCount(conflictReviewStatus.data.validation.missing_strata.length)} detail={conflictReviewStatus.data.validation.stratified ? 'All six labels represented in every split' : 'Continue stratified review'} tone={conflictReviewStatus.data.validation.stratified ? 'positive' : 'warning'} />
              </div>
              <ConflictReviewMatrix status={conflictReviewStatus.data} />
              {conflictReviewStatus.data.validation.blockers.length ? <ul className="conflict-calibration-blockers" aria-label="Calibration blockers">{conflictReviewStatus.data.validation.blockers.map((blocker) => <li key={blocker.code}><strong>{blocker.message}</strong>{blocker.count != null ? <span>{formatCount(blocker.count)}</span> : null}</li>)}</ul> : <p className="enrichment-success">The manual dataset is eligible for held-out calibration. Running it still cannot enable aggregate charts unless both model-quality thresholds pass.</p>}
              <div className="conflict-calibration-actions"><button className="button button--primary" type="button" disabled={!conflictReviewStatus.data.validation.eligible_for_calibration || createConflictCalibration.isPending} onClick={() => createConflictCalibration.mutate()}><ShieldCheck size={15} aria-hidden="true" />{createConflictCalibration.isPending ? 'Running held-out calibration…' : 'Run held-out calibration'}</button><span>{conflictReviewStatus.data.warning}</span></div>
              {createConflictCalibration.data ? <p className="enrichment-success" role="status">Calibration recorded: direct-conflict precision {((createConflictCalibration.data.direct_conflict_precision ?? 0) * 100).toFixed(1)}%, macro-F1 {((createConflictCalibration.data.macro_f1 ?? 0) * 100).toFixed(1)}%. Aggregate gate {createConflictCalibration.data.aggregation_enabled ? 'passed' : 'remains closed'}.</p> : null}
              {createConflictCalibration.isError ? <p className="enrichment-error" role="alert">{createConflictCalibration.error.message}</p> : null}
            </section>
          )}

          <section className="conflict-review-queue" aria-labelledby="conflict-review-queue-heading">
            <div className="section-header"><div><h2 id="conflict-review-queue-heading">Deterministic review queue</h2><p>Labels and notes are always manual. Evaluation splits are server-assigned and cannot be chosen to improve metrics.</p></div><button className="button button--secondary" type="button" disabled={nextConflictReview.isPending} onClick={() => nextConflictReview.mutate()}>{nextConflictReview.isPending ? 'Finding next…' : 'Next unlabelled window'}</button></div>
            <div className="conflict-review-filters">
              <label className="filter-field"><span>Manual label state</span><select value={reviewLabel} onChange={(event) => { setReviewLabel(event.target.value as ConflictReviewLabelFilter); setReviewAfter(undefined); setFocusedReviewItem(null) }}><option value="unlabelled">Unlabelled</option>{conflictLabels.map((label) => <option value={label} key={label}>{sentenceCase(label)}</option>)}</select></label>
              <label className="filter-field"><span>Deterministic split</span><select value={reviewSplit} onChange={(event) => { setReviewSplit(event.target.value as 'all' | ConflictEvaluationSplit); setReviewAfter(undefined); setFocusedReviewItem(null) }}><option value="all">All splits</option><option value="train">Train</option><option value="validation">Validation</option><option value="test">Test</option></select></label>
            </div>
            {nextConflictReview.isError ? <p className="enrichment-error" role="alert">{nextConflictReview.error.message}</p> : null}
            {focusedReviewItem ? <div className="conflict-focused-review"><h3>Next unlabelled window</h3><ConflictReviewCard item={focusedReviewItem} onSaved={refreshConflictReview} /></div> : null}
            {conflictReviewQueue.isPending ? <PageLoading label="Loading conflict review windows" /> : conflictReviewQueue.isError ? <ErrorState error={conflictReviewQueue.error} onRetry={() => void conflictReviewQueue.refetch()} /> : conflictReviewQueue.data?.items.length ? <><div className="conflict-review-list">{conflictReviewQueue.data.items.map((item) => <ConflictReviewCard item={item} key={item.window_id} onSaved={refreshConflictReview} />)}</div><div className="enrichment-pager"><span>{formatCount(conflictReviewQueue.data.page.returned)} of {formatCount(conflictReviewQueue.data.page.total)} matching windows</span><button className="button button--secondary" type="button" disabled={!conflictReviewQueue.data.page.next_after} onClick={() => setReviewAfter(conflictReviewQueue.data?.page.next_after ?? undefined)}>Next queue page</button></div></> : <EmptyState title="No windows match this review stratum" description="Choose another manual-label state or deterministic split. The server will never fabricate labels to fill a stratum." />}
          </section>

          <div className="section-header conflict-discovery-heading"><div><h2>Semantic discovery</h2><p>Find additional windows to add to the manual queue. The model’s suggested ranking label is never preselected as the human answer.</p></div></div>
          <form className="conflict-search-form" onSubmit={(event) => { event.preventDefault(); if (conflictDraft.q.trim()) setConflictApplied({ ...conflictDraft, q: conflictDraft.q.trim() }) }}>
            <label className="filter-field conflict-search-form__query"><span>Semantic description</span><span className="filter-input"><Search size={15} aria-hidden="true" /><input value={conflictDraft.q} onChange={(event) => setConflictDraft((current) => ({ ...current, q: event.target.value }))} placeholder="e.g. argument or tense disagreement" /></span></label>
            <label className="filter-field"><span>Rank for label</span><select value={conflictDraft.label} onChange={(event) => setConflictDraft((current) => ({ ...current, label: event.target.value as ConflictLabel }))}>{conflictLabels.map((label) => <option key={label} value={label}>{sentenceCase(label)}</option>)}</select></label>
            <label className="filter-field"><span>Participant contains</span><input value={conflictDraft.participant} onChange={(event) => setConflictDraft((current) => ({ ...current, participant: event.target.value }))} /></label>
            <label className="filter-field"><span>Conversation ID</span><input value={conflictDraft.conversationId} onChange={(event) => setConflictDraft((current) => ({ ...current, conversationId: event.target.value }))} /></label>
            <button className="button button--primary" type="submit" disabled={!conflictDraft.q.trim() || conflict.isFetching}><Search size={15} aria-hidden="true" />{conflict.isFetching ? 'Loading local models…' : 'Search conflict windows'}</button>
          </form>
          <div className="enrichment-safety" role="note"><ShieldCheck size={18} aria-hidden="true" /><div><strong>Calibration: {formatCount(conflictLabelCount)} / 500 labels</strong><span>{aggregate?.available ? `${formatCount(aggregate.run?.scored_count)} windows are available in time-only conflict analytics.` : aggregateReason}</span></div>{canStartAggregate ? <button className="button button--secondary" type="button" disabled={startConflictAggregate.isPending} onClick={() => startConflictAggregate.mutate()}><Play size={14} aria-hidden="true" />Build conflict analytics</button> : null}</div>
          {conflict.isError ? <ErrorState title="Local conflict scoring is unavailable" error={conflict.error} onRetry={() => void conflict.refetch()} /> : conflict.isPending && conflictApplied ? <PageLoading label="Retrieving and reranking message windows" /> : conflict.data ? <>{conflict.data.items.length ? <div className="conflict-result-list">{conflict.data.items.map((item) => <ConflictResultCard item={item} key={item.window_id ?? item.id} />)}</div> : <EmptyState title="No matching message windows" description="Try a broader semantic description or remove participant and conversation filters." />}<p className="conflict-workbench__footnote">{formatCount(conflict.data.page.returned)} of {formatCount(conflict.data.page.candidate_count)} locally scored candidates shown. Aggregate gate: {conflict.data.aggregation.aggregation_enabled ? 'enabled after calibration' : 'disabled'}.</p></> : <EmptyState title="Describe the interaction you want to review" description="The local semantic and NLI models will retrieve windows without sending messages to a cloud service." />}
        </section>
      ) : <section id="enrichment-panel-conflict" role="tabpanel" aria-labelledby="enrichment-tab-conflict" hidden />}
    </div>
  )
}
