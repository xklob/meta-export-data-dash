import { useInfiniteQuery, useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  ArrowRight,
  Bookmark,
  BrainCircuit,
  CalendarRange,
  Check,
  CircleHelp,
  Cloud,
  DatabaseZap,
  FileSearch,
  Filter,
  MessageSquareText,
  Search,
  Sparkles,
  ThumbsDown,
  ThumbsUp,
  Trash2,
} from 'lucide-react'
import { type FormEvent, useMemo, useState } from 'react'
import { CursorWindowControls } from '../components/CursorWindowControls'
import { EmptyState, ErrorState, PageHeading, PageLoading, ProgressBar, StatusBadge } from '../components/ui'
import { api, type SearchQuery } from '../lib/api'
import type {
  RagAuditItem,
  SavedSearch,
  SearchEntityType,
  SearchFeedbackResponse,
  SearchFilters,
  SearchMode,
  SearchResult,
  SemanticIndexAcceptanceEvaluation,
} from '../lib/contracts'
import { formatCount, formatTimestampMs, localDateToTimestamp, sentenceCase, stringifyValue } from '../lib/format'
import { useArchiveTimezone } from '../lib/useArchiveTimezone'
import { useCursorWindowQuery } from '../lib/useCursorWindowQuery'
import './SearchPage.css'

interface SearchDraft {
  q: string
  mode: SearchMode
  entityType: SearchEntityType
  conversationId: string
  participant: string
  sourceCategory: string
  from: string
  to: string
}

const initialDraft: SearchDraft = {
  q: '',
  mode: 'keyword',
  entityType: 'all',
  conversationId: '',
  participant: '',
  sourceCategory: '',
  from: '',
  to: '',
}

const modes: { mode: SearchMode; label: string; description: string }[] = [
  { mode: 'keyword', label: 'Keyword', description: 'FTS phrase and prefix matching' },
  { mode: 'exact', label: 'Exact literal', description: 'Case-sensitive text substring' },
  { mode: 'semantic', label: 'Semantic', description: 'Meaning-based conversation windows' },
  { mode: 'hybrid', label: 'Hybrid', description: 'Keyword and semantic ranks combined' },
]

function parseSavedSearch(saved: SavedSearch): SearchDraft {
  const query = saved.query
  const stringValue = (key: string) => (typeof query[key] === 'string' ? String(query[key]) : '')
  const rawMode = stringValue('mode')
  const rawEntity = stringValue('entityType')
  return {
    q: stringValue('q'),
    mode: modes.some(({ mode }) => mode === rawMode) ? (rawMode as SearchMode) : 'keyword',
    entityType: ['all', 'message', 'activity', 'audio_transcript', 'attachment_text'].includes(rawEntity)
      ? (rawEntity as SearchEntityType)
      : 'all',
    conversationId: stringValue('conversationId'),
    participant: stringValue('participant'),
    sourceCategory: stringValue('sourceCategory'),
    from: stringValue('from'),
    to: stringValue('to'),
  }
}

function toQuery(draft: SearchDraft, timezone: string): SearchQuery {
  return {
    q: draft.q.trim(),
    mode: draft.mode,
    entityType: draft.entityType,
    conversationId: draft.conversationId,
    participant: draft.participant,
    sourceCategory: draft.sourceCategory,
    fromMs: localDateToTimestamp(draft.from, timezone),
    toMs: localDateToTimestamp(draft.to, timezone, true),
    limit: 50,
  }
}

function toFilterScope(draft: SearchDraft, timezone: string): SearchFilters {
  return {
    entity_type: draft.entityType,
    conversation_id: draft.conversationId.trim() || null,
    participant: draft.participant.trim() || null,
    source_category: draft.sourceCategory.trim() || null,
    from_ms: localDateToTimestamp(draft.from, timezone) ?? null,
    to_ms: localDateToTimestamp(draft.to, timezone, true) ?? null,
  }
}

function filterScopeSummary(scope: Partial<SearchFilters> | undefined): string {
  if (!scope || !Object.keys(scope).length) return 'All archive records'
  const parts = [
    scope.entity_type && scope.entity_type !== 'all' ? sentenceCase(scope.entity_type) : null,
    scope.conversation_id ? `conversation ${scope.conversation_id}` : null,
    scope.participant ? `participant ${scope.participant}` : null,
    scope.source_category ? `source ${scope.source_category}` : null,
    scope.from_ms != null ? `from ${new Date(scope.from_ms).toLocaleDateString()}` : null,
    scope.to_ms != null ? `through ${new Date(scope.to_ms).toLocaleDateString()}` : null,
  ].filter(Boolean)
  return parts.length ? parts.join(' · ') : 'All archive records'
}

function scoreLabel(item: SearchResult): string {
  if (item.score_kind === 'exact') return 'Exact match'
  if (!Number.isFinite(item.score)) return sentenceCase(item.score_kind)
  return `${sentenceCase(item.score_kind)} ${item.score.toFixed(item.score_kind === 'rrf' ? 4 : 3)}`
}

function retrievalGateSummary(evaluation: SemanticIndexAcceptanceEvaluation): string {
  const metrics = evaluation.automatic_known_item.metrics
  const heldOut = evaluation.held_out_feedback
  const heldOutSummary = heldOut.used_for_activation
    ? ` · held-out ${heldOut.metrics.query_count} queries ${heldOut.status}`
    : heldOut.metrics.query_count
      ? ` · held-out ${heldOut.metrics.query_count} queries (sample too small to gate)`
      : ' · no held-out feedback yet'
  return `Retrieval gate ${evaluation.passed ? 'passed' : 'failed'} · known-item Recall@${metrics.k} ${metrics.recall_at_k.toFixed(3)} · MRR ${metrics.mrr.toFixed(3)} · nDCG@${metrics.k} ${metrics.ndcg_at_k.toFixed(3)}${heldOutSummary}`
}

function RagAuditDetails({ item }: { item: RagAuditItem }) {
  const usage = Object.entries(item.usage)
  return (
    <details className="rag-audit-detail">
      <summary>Source IDs and usage</summary>
      <dl>
        <div>
          <dt>Retrieved source IDs</dt>
          <dd className="rag-audit-detail__ids">
            {item.retrieved_ids.length
              ? item.retrieved_ids.map((id) => <code key={id}>{id}</code>)
              : <span>None</span>}
          </dd>
        </div>
        <div>
          <dt>Cited source IDs</dt>
          <dd className="rag-audit-detail__ids">
            {item.citation_ids.length
              ? item.citation_ids.map((id) => <code key={id}>{id}</code>)
              : <span>None</span>}
          </dd>
        </div>
        <div>
          <dt>Query hash</dt>
          <dd><code>{item.query_sha256}</code></dd>
        </div>
        {item.remote_response_id ? (
          <div>
            <dt>Remote response ID</dt>
            <dd><code>{item.remote_response_id}</code></dd>
          </div>
        ) : null}
      </dl>
      <h4>Usage</h4>
      {usage.length ? (
        <dl className="rag-audit-detail__usage">
          {usage.map(([key, value]) => (
            <div key={key}>
              <dt>{sentenceCase(key)}</dt>
              <dd>{stringifyValue(value)}</dd>
            </div>
          ))}
        </dl>
      ) : <p>No usage or token data was recorded.</p>}
    </details>
  )
}

function ResultCard({
  item,
  timezone,
  filterScope,
  feedback,
  onFeedback,
}: {
  item: SearchResult
  timezone: string
  filterScope: SearchFilters
  feedback?: string
  onFeedback: (
    label: 'relevant' | 'not_relevant' | 'uncertain',
    evaluationSplit?: 'train' | 'validation' | 'test',
    notes?: string,
  ) => void
}) {
  const [evaluationSplit, setEvaluationSplit] = useState<'' | 'train' | 'validation' | 'test'>('')
  const [notes, setNotes] = useState('')
  const [visibleOccurrenceCount, setVisibleOccurrenceCount] = useState(5)
  const [occurrences, setOccurrences] = useState(item.occurrences ?? [])
  const [nextOccurrenceOffset, setNextOccurrenceOffset] = useState(item.occurrence_page?.next_offset ?? null)
  const occurrenceTotal = item.occurrence_page?.total ?? occurrences.length
  const loadOccurrences = useMutation({
    mutationFn: () => api.searchDerivedOccurrences({
      entityType: item.entity_type as 'audio_transcript' | 'attachment_text',
      sourceId: String(item.source_id),
      filterScope,
      offset: nextOccurrenceOffset ?? occurrences.length,
      limit: 20,
    }),
    onSuccess: (page) => {
      setOccurrences((current) => {
        const byId = new Map(current.map((occurrence) => [occurrence.occurrence_id, occurrence]))
        for (const occurrence of page.items) byId.set(occurrence.occurrence_id, occurrence)
        return [...byId.values()]
      })
      setNextOccurrenceOffset(page.page.next_offset ?? null)
      setVisibleOccurrenceCount((count) => count + 5)
    },
  })
  const title =
    item.conversation?.title ??
    (item.entity_type === 'activity'
      ? sentenceCase(item.subtype ?? item.source_category)
      : item.entity_type === 'audio_transcript'
        ? 'Audio transcript'
        : item.entity_type === 'attachment_text'
          ? 'Extracted attachment text'
          : 'Message result')
  const recordFeedback = (label: 'relevant' | 'not_relevant' | 'uncertain') =>
    onFeedback(label, evaluationSplit || undefined, notes.trim() || undefined)
  return (
    <article className="search-result" id={`search-result-${item.id}`}>
      <header>
        <div className="search-result__identity">
          <span className="search-result__rank"><span className="sr-only">Result rank </span>{item.rank}</span>
          <div>
            <h3>{title || 'Untitled conversation'}</h3>
            <p>
              {sentenceCase(item.entity_type)}
              {item.sender ? ` · ${item.sender}` : ''}
              {item.source_category ? ` · ${sentenceCase(item.source_category)}` : ''}
            </p>
          </div>
        </div>
        <div className="search-result__meta">
          <span>{scoreLabel(item)}</span>
          <span>Coverage: {sentenceCase(item.coverage.status)}</span>
          <span>
            {item.confidence == null
              ? 'Record confidence: not applicable'
              : `Audio decoder score: ${(item.confidence * 100).toFixed(1)}% · uncalibrated`}
          </span>
          <time>{formatTimestampMs(item.timestamp_ms, timezone)}</time>
        </div>
      </header>
      <p className="search-result__excerpt">{item.excerpt || 'The source record has no searchable display text.'}</p>
      {item.window ? (
        <div className="search-result__window">
          <MessageSquareText size={15} aria-hidden="true" />
          <span>
            {formatCount(item.window.turn_count)} turns
            {item.participants?.length ? ` · ${item.participants.join(', ')}` : ''}
          </span>
          <span>
            {formatTimestampMs(item.window.first_timestamp_ms, timezone)} –{' '}
            {formatTimestampMs(item.window.last_timestamp_ms, timezone)}
          </span>
        </div>
      ) : null}
      {occurrences.length ? (
        <details className="search-result__occurrences">
          <summary>{formatCount(occurrenceTotal)} source {occurrenceTotal === 1 ? 'location' : 'locations'}</summary>
          <ol>
            {occurrences.slice(0, visibleOccurrenceCount).map((occurrence) => (
              <li key={occurrence.occurrence_id}>
                <span>
                  <strong>{occurrence.conversation_title || 'Media library'}</strong>
                  <small>
                    {formatTimestampMs(occurrence.timestamp_ms, timezone)}
                    {occurrence.source_category ? ` · ${sentenceCase(occurrence.source_category)}` : ''}
                    {occurrence.start_ms != null ? ` · audio at ${(occurrence.start_ms / 1000).toFixed(1)}s` : ''}
                  </small>
                </span>
                <a className="button button--secondary button--compact" href={occurrence.href}>Jump to this occurrence</a>
              </li>
            ))}
          </ol>
          {Math.min(visibleOccurrenceCount, occurrences.length) < occurrenceTotal ? (
            <button className="button button--secondary button--compact" type="button" disabled={loadOccurrences.isPending} onClick={() => {
              if (visibleOccurrenceCount < occurrences.length) setVisibleOccurrenceCount((count) => count + 5)
              else if (nextOccurrenceOffset != null) loadOccurrences.mutate()
            }}>
              {loadOccurrences.isPending ? 'Loading source locations…' : 'Show more source locations'}
            </button>
          ) : null}
          {loadOccurrences.isError ? <span className="search-inline-error">More source locations could not be loaded.</span> : null}
        </details>
      ) : null}
      <footer>
        <div className="search-result__feedback" role="group" aria-label={`Relevance feedback for result ${item.rank}`}>
          <span>{feedback ? `Marked ${sentenceCase(feedback)}` : 'Was this useful?'}</span>
          <button
            type="button"
            className={feedback === 'relevant' ? 'is-selected' : undefined}
            aria-label={`Mark result ${item.rank} relevant`}
            aria-pressed={feedback === 'relevant'}
            onClick={() => recordFeedback('relevant')}
          >
            <ThumbsUp size={14} aria-hidden="true" />
          </button>
          <button
            type="button"
            className={feedback === 'not_relevant' ? 'is-selected' : undefined}
            aria-label={`Mark result ${item.rank} not relevant`}
            aria-pressed={feedback === 'not_relevant'}
            onClick={() => recordFeedback('not_relevant')}
          >
            <ThumbsDown size={14} aria-hidden="true" />
          </button>
          <button
            type="button"
            className={feedback === 'uncertain' ? 'is-selected' : undefined}
            aria-label={`Mark result ${item.rank} uncertain`}
            aria-pressed={feedback === 'uncertain'}
            onClick={() => recordFeedback('uncertain')}
          >
            <CircleHelp size={14} aria-hidden="true" />
          </button>
          <details className="search-result__evaluation">
            <summary>Evaluation details</summary>
            <label>
              <span>Dataset split</span>
              <select value={evaluationSplit} onChange={(event) => setEvaluationSplit(event.target.value as typeof evaluationSplit)}>
                <option value="">Unassigned</option>
                <option value="train">Train</option>
                <option value="validation">Validation</option>
                <option value="test">Test</option>
              </select>
            </label>
            <label><span>Notes</span><input value={notes} onChange={(event) => setNotes(event.target.value)} maxLength={2_000} /></label>
          </details>
        </div>
        {item.jump?.href ? (
          <a className="button button--secondary button--compact" href={item.jump.href}>
            Jump to source <ArrowRight size={14} aria-hidden="true" />
          </a>
        ) : (
          <span className="search-result__no-jump">No source anchor available</span>
        )}
      </footer>
    </article>
  )
}

function EvaluationRow({
  item,
  saving,
  onSave,
  onDelete,
}: {
  item: SearchFeedbackResponse
  saving: boolean
  onSave: (item: SearchFeedbackResponse, update: { label: 'relevant' | 'not_relevant' | 'uncertain'; evaluation_split: 'train' | 'validation' | 'test' | null; notes: string | null }) => void
  onDelete: (id: string) => void
}) {
  const [label, setLabel] = useState(item.label)
  const [evaluationSplit, setEvaluationSplit] = useState<'' | 'train' | 'validation' | 'test'>(item.evaluation_split ?? '')
  const [notes, setNotes] = useState(item.notes ?? '')
  return (
    <li>
      <form onSubmit={(event) => { event.preventDefault(); onSave(item, { label, evaluation_split: evaluationSplit || null, notes: notes.trim() || null }) }}>
        <span className="search-evaluations__identity">
          <strong>{item.query || item.result_id}</strong>
          <small>{sentenceCase(item.search_mode ?? 'keyword')} · {item.result_id}</small>
          <small>Scope: {filterScopeSummary(item.filter_scope)}</small>
        </span>
        <label>
          <span>Judgment</span>
          <select aria-label={`Evaluation label for ${item.id}`} value={label} onChange={(event) => setLabel(event.target.value as typeof label)}>
            <option value="relevant">Relevant</option>
            <option value="not_relevant">Not relevant</option>
            <option value="uncertain">Uncertain</option>
          </select>
        </label>
        <label>
          <span>Dataset split</span>
          <select aria-label={`Evaluation split for ${item.id}`} value={evaluationSplit} onChange={(event) => setEvaluationSplit(event.target.value as typeof evaluationSplit)}>
            <option value="">Unassigned</option>
            <option value="train">Train</option>
            <option value="validation">Validation</option>
            <option value="test">Test</option>
          </select>
        </label>
        <label className="search-evaluations__notes">
          <span>Notes</span>
          <input aria-label={`Evaluation notes for ${item.id}`} value={notes} maxLength={2_000} onChange={(event) => setNotes(event.target.value)} />
        </label>
        <button className="button button--secondary button--compact" type="submit" disabled={saving}>{saving ? 'Saving…' : 'Save changes'}</button>
        <button className="icon-button" type="button" aria-label={`Delete evaluation for ${item.result_id}`} onClick={() => onDelete(item.id)}><Trash2 size={14} aria-hidden="true" /></button>
      </form>
    </li>
  )
}

export function SearchPage() {
  const timezone = useArchiveTimezone()
  const queryClient = useQueryClient()
  const [draft, setDraft] = useState<SearchDraft>(initialDraft)
  const [applied, setApplied] = useState<SearchDraft | null>(null)
  const [savedName, setSavedName] = useState('')
  const [feedback, setFeedback] = useState<Record<string, string>>({})
  const [ragQuestion, setRagQuestion] = useState('')
  const [evaluationsOpen, setEvaluationsOpen] = useState(false)
  const searchEnabled = Boolean(applied?.q.trim())

  const status = useQuery({
    queryKey: ['search-status'],
    queryFn: api.searchStatus,
    refetchInterval: (query) =>
      ['queued', 'running'].includes(query.state.data?.latest_job?.status ?? '') ? 1_500 : false,
  })
  const settings = useQuery({ queryKey: ['settings'], queryFn: api.settings, staleTime: 30_000 })
  const saved = useQuery({ queryKey: ['saved-searches'], queryFn: api.savedSearches })
  const feedbackHistory = useInfiniteQuery({
    queryKey: ['search-feedback-history'],
    enabled: evaluationsOpen,
    initialPageParam: undefined as string | undefined,
    queryFn: ({ pageParam }) => api.searchFeedbackHistory({ limit: 20, cursor: pageParam }),
    getNextPageParam: (lastPage) => lastPage.page.next_cursor ?? undefined,
  })
  const ragAudits = useCursorWindowQuery({
    queryKey: ['rag-audits', 'all'],
    queryFn: (cursor) => api.ragAudits('all', cursor, 20),
  })
  const query = useInfiniteQuery({
    queryKey: ['archive-search', applied, timezone],
    enabled: searchEnabled,
    initialPageParam: undefined as string | undefined,
    queryFn: ({ pageParam }) => api.search({ ...toQuery(applied!, timezone), cursor: pageParam }),
    getNextPageParam: (lastPage) => lastPage.page.next_cursor ?? undefined,
  })
  const buildIndex = useMutation({
    mutationFn: () => api.buildSearchIndex(),
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ['search-status'] })
    },
  })
  const saveSearch = useMutation({
    mutationFn: () => api.createSavedSearch({ name: savedName.trim(), query: { ...(applied ?? draft) } }),
    onSuccess: async () => {
      setSavedName('')
      await queryClient.invalidateQueries({ queryKey: ['saved-searches'] })
    },
  })
  const deleteSaved = useMutation({
    mutationFn: (id: string) => api.deleteSavedSearch(id),
    onSuccess: async () => queryClient.invalidateQueries({ queryKey: ['saved-searches'] }),
  })
  const submitFeedback = useMutation({
    mutationFn: ({ result, label, evaluationSplit, notes }: { result: SearchResult; label: 'relevant' | 'not_relevant' | 'uncertain'; evaluationSplit?: 'train' | 'validation' | 'test'; notes?: string }) =>
      api.searchFeedback({
        query: applied?.q ?? '',
        search_mode: applied?.mode ?? 'keyword',
        filter_scope: toFilterScope(applied ?? draft, timezone),
        result_id: result.id,
        label,
        evaluation_split: evaluationSplit,
        notes,
      }),
    onSuccess: async (response) => {
      setFeedback((current) => ({ ...current, [response.result_id]: response.label }))
      await queryClient.invalidateQueries({ queryKey: ['search-feedback-history'] })
    },
  })
  const deleteFeedback = useMutation({
    mutationFn: (id: string) => api.deleteSearchFeedback(id),
    onSuccess: async () => queryClient.invalidateQueries({ queryKey: ['search-feedback-history'] }),
  })
  const updateFeedback = useMutation({
    mutationFn: ({ id, update }: { id: string; update: { label: 'relevant' | 'not_relevant' | 'uncertain'; evaluation_split: 'train' | 'validation' | 'test' | null; notes: string | null } }) => api.updateSearchFeedback(id, update),
    onSuccess: async () => queryClient.invalidateQueries({ queryKey: ['search-feedback-history'] }),
  })
  const rag = useMutation({
    mutationFn: () => {
      const source = applied ?? draft
      return api.ragAnswer({
        question: ragQuestion.trim(),
        searchMode: source.mode === 'semantic' ? 'semantic' : 'hybrid',
        conversationId: source.conversationId,
        entityType: source.entityType,
        participant: source.participant,
        sourceCategory: source.sourceCategory,
        fromMs: localDateToTimestamp(source.from, timezone),
        toMs: localDateToTimestamp(source.to, timezone, true),
      })
    },
    onSuccess: async () => queryClient.invalidateQueries({ queryKey: ['rag-audits'] }),
  })

  const items = useMemo(() => query.data?.pages.flatMap((page) => page.items) ?? [], [query.data])
  const feedbackItems = useMemo(() => feedbackHistory.data?.pages.flatMap((page) => page.items) ?? [], [feedbackHistory.data])
  const firstPage = query.data?.pages[0]
  const semanticMode = (applied ?? draft).mode === 'semantic' || (applied ?? draft).mode === 'hybrid'
  const acceptedEvaluation = status.data?.acceptance
  const latestEvaluation = status.data?.latest_evaluation
  const latestCandidateRejected = latestEvaluation?.passed === false
  const semanticIndexStale = Boolean(status.data?.needs_rebuild)
  const displayedEvaluation = latestCandidateRejected
    ? latestEvaluation
    : acceptedEvaluation ?? latestEvaluation
  const submit = (event: FormEvent) => {
    event.preventDefault()
    if (!draft.q.trim()) return
    setApplied({ ...draft, q: draft.q.trim() })
    setRagQuestion(draft.q.trim())
    setFeedback({})
    rag.reset()
  }

  return (
    <div className="page-stack search-page">
      <PageHeading
        eyebrow="Stage 3 · Local retrieval"
        title="Search the archive"
        description="Search messages, conversation windows, audio transcripts, comments, and other indexed activity by literal text or meaning. Every result stays tied to a source jump."
      />

      <section className="panel search-index-card" aria-labelledby="search-index-title">
        <div className="search-index-card__main">
          <span className="search-index-card__icon"><DatabaseZap size={19} aria-hidden="true" /></span>
          <div>
            <div className="search-index-card__title">
              <h2 id="search-index-title">Semantic index</h2>
              <StatusBadge status={status.data?.ready ? 'ready' : semanticIndexStale ? 'stale' : status.data?.latest_job?.status ?? 'not ready'} />
            </div>
            <p>
              {status.data?.ready && status.data.active
                ? `${status.data.active.model_name} · ${formatCount(status.data.active.window_count)} searchable windows`
                : semanticIndexStale
                  ? 'New searchable transcript or attachment text is available. Rebuild the local index before using semantic or hybrid search.'
                  : 'Keyword and exact search work without embeddings. Build the local index to unlock semantic and hybrid modes.'}
            </p>
            {displayedEvaluation ? (
              <p className="search-index-card__quality">
                <span className="sr-only">Semantic retrieval quality gate: </span>{retrievalGateSummary(displayedEvaluation)}
              </p>
            ) : null}
            {latestCandidateRejected ? (
              <p className="search-index-card__warning" role="status">
                Latest candidate was rejected by the retrieval quality gate.
                {status.data?.ready ? ' The prior ready index remains active.' : ' Semantic modes remain disabled.'}
              </p>
            ) : null}
            {semanticIndexStale ? (
              <p className="search-index-card__warning" role="status">
                The prior semantic index is stale{status.data?.stale_reason?.reason ? `: ${status.data.stale_reason.reason}.` : '.'} Keyword and exact search remain current.
              </p>
            ) : null}
          </div>
        </div>
        {status.data?.latest_job && ['queued', 'running'].includes(status.data.latest_job.status) ? (
          <ProgressBar
            current={status.data.latest_job.progress_current}
            total={status.data.latest_job.progress_total}
            label="Semantic index progress"
          />
        ) : !status.data?.ready || latestCandidateRejected ? (
          <button className="button button--secondary" type="button" disabled={buildIndex.isPending} onClick={() => buildIndex.mutate()}>
            <BrainCircuit size={15} aria-hidden="true" />
            {buildIndex.isPending
              ? 'Queuing…'
              : latestCandidateRejected
                ? 'Retry semantic index build'
                : semanticIndexStale
                  ? 'Rebuild local semantic index'
                : 'Build local semantic index'}
          </button>
        ) : null}
        {status.isError ? <span className="search-inline-error">Index status unavailable.</span> : null}
        {buildIndex.isError ? <span className="search-inline-error">{buildIndex.error.message}</span> : null}
      </section>

      <form className="panel search-form" onSubmit={submit}>
        <fieldset className="search-modes">
          <legend>Search method</legend>
          {modes.map((option) => (
            <label key={option.mode} className={draft.mode === option.mode ? 'is-selected' : undefined}>
              <input
                type="radio"
                name="search-mode"
                value={option.mode}
                checked={draft.mode === option.mode}
                disabled={(option.mode === 'semantic' || option.mode === 'hybrid') && !status.data?.ready}
                onChange={() => setDraft((current) => ({ ...current, mode: option.mode }))}
              />
              <span><strong>{option.label}</strong><small>{option.description}</small></span>
            </label>
          ))}
        </fieldset>
        <label className="search-query-field">
          <span>What do you want to find?</span>
          <span className="search-query-field__control">
            <Search size={19} aria-hidden="true" />
            <input
              value={draft.q}
              onChange={(event) => setDraft((current) => ({ ...current, q: event.target.value }))}
              placeholder={draft.mode === 'semantic' ? 'e.g. a tense argument about plans' : 'Words or an exact phrase'}
              autoComplete="off"
            />
            <button className="button button--primary" type="submit" disabled={!draft.q.trim()}>
              Search archive
            </button>
          </span>
        </label>
        <details className="search-filters">
          <summary><Filter size={15} aria-hidden="true" /> Scope and date filters</summary>
          <div className="search-filter-grid">
            <label><span>Record type</span><select value={draft.entityType} onChange={(event) => setDraft((current) => ({ ...current, entityType: event.target.value as SearchEntityType }))}><option value="all">All searchable records</option><option value="message">Messages only</option><option value="activity">Posts/comments/activity</option><option value="audio_transcript">Audio transcripts</option><option value="attachment_text">Extracted attachment text</option></select></label>
            <label><span>Participant or actor</span><input value={draft.participant} onChange={(event) => setDraft((current) => ({ ...current, participant: event.target.value }))} placeholder="Name contains…" /></label>
            <label><span>Conversation ID</span><input value={draft.conversationId} onChange={(event) => setDraft((current) => ({ ...current, conversationId: event.target.value }))} placeholder="Optional exact ID" /></label>
            <label><span>Source category</span><input value={draft.sourceCategory} onChange={(event) => setDraft((current) => ({ ...current, sourceCategory: event.target.value }))} placeholder="inbox, groups…" /></label>
            <label><span>From</span><input type="date" value={draft.from} onChange={(event) => setDraft((current) => ({ ...current, from: event.target.value }))} /></label>
            <label><span>Through</span><input type="date" value={draft.to} onChange={(event) => setDraft((current) => ({ ...current, to: event.target.value }))} /></label>
          </div>
          <p><CalendarRange size={14} aria-hidden="true" /> Dates use {timezone}; unknown dates remain unknown.</p>
        </details>
      </form>

      <section className="panel saved-searches" aria-labelledby="saved-searches-title">
        <div className="saved-searches__header">
          <div><h2 id="saved-searches-title">Saved searches</h2><p>Keep a repeatable query and its complete filter scope.</p></div>
          <form onSubmit={(event) => { event.preventDefault(); if (savedName.trim()) saveSearch.mutate() }}>
            <input aria-label="Saved search name" value={savedName} onChange={(event) => setSavedName(event.target.value)} placeholder="Name this search" />
            <button className="button button--secondary" type="submit" disabled={!savedName.trim() || !(applied ?? draft).q.trim() || saveSearch.isPending}><Bookmark size={14} aria-hidden="true" /> Save</button>
          </form>
        </div>
        {saved.data?.items.length ? (
          <ul>{saved.data.items.map((item) => <li key={item.id}><button type="button" onClick={() => { const next = parseSavedSearch(item); setDraft(next); if (next.q) setApplied(next) }}><FileSearch size={14} aria-hidden="true" /><span><strong>{item.name}</strong><small>{String(item.query.q ?? 'Empty query')} · {sentenceCase(String(item.query.mode ?? 'keyword'))}</small></span></button><button type="button" aria-label={`Delete saved search ${item.name}`} onClick={() => deleteSaved.mutate(item.id)}><Trash2 size={14} aria-hidden="true" /></button></li>)}</ul>
        ) : <p className="saved-searches__empty">No saved searches yet.</p>}
        {saveSearch.isError ? <p className="search-inline-error">{saveSearch.error.message}</p> : null}
      </section>

      <details
        className="panel search-evaluations"
        aria-labelledby="search-evaluations-title"
        onToggle={(event) => setEvaluationsOpen(event.currentTarget.open)}
      >
        <summary>
          <span>
            <small>Advanced review</small>
            <strong id="search-evaluations-title">Retrieval evaluation set</strong>
            <span>Edit or delete saved relevance judgements without crowding the primary search workflow.</span>
          </span>
          <span className="search-evaluations__toggle">{evaluationsOpen ? 'Close ledger' : 'Open ledger'}</span>
        </summary>
        <div className="search-evaluations__body">
          <p>Reviewable local judgements used to compare search changes. Re-submitting feedback updates the same query, mode, and result record instead of creating contradictions.</p>
          {feedbackHistory.isPending ? <p className="saved-searches__empty" role="status">Loading retrieval evaluations…</p> : feedbackItems.length ? <ul>{feedbackItems.map((item) => <EvaluationRow key={item.id} item={item} saving={updateFeedback.isPending && updateFeedback.variables?.id === item.id} onSave={(feedbackItem, update) => updateFeedback.mutate({ id: feedbackItem.id, update })} onDelete={(id) => deleteFeedback.mutate(id)} />)}</ul> : <p className="saved-searches__empty">No relevance judgements recorded yet.</p>}
          {feedbackHistory.hasNextPage ? <button className="button button--secondary search-load-more" type="button" disabled={feedbackHistory.isFetchingNextPage} onClick={() => void feedbackHistory.fetchNextPage()}>{feedbackHistory.isFetchingNextPage ? 'Loading…' : 'Load more evaluations'}</button> : null}
          {feedbackHistory.isError ? <p className="search-inline-error">Evaluation history is unavailable.</p> : null}
          {updateFeedback.isError ? <p className="search-inline-error">Evaluation changes could not be saved.</p> : null}
        </div>
      </details>

      {semanticMode && !status.data?.ready ? (
        <section className="search-callout" role="status"><BrainCircuit size={18} aria-hidden="true" /><div><strong>Semantic retrieval is not ready</strong><p>Build the local index above, or switch to Keyword or Exact literal.</p></div></section>
      ) : null}
      {searchEnabled && query.isPending ? <PageLoading label="Searching the local archive" /> : null}
      {query.isError ? <ErrorState title="Search could not be completed" error={query.error} onRetry={() => void query.refetch()} /> : null}

      {firstPage ? (
        <section className="search-results" aria-labelledby="search-results-title">
          <div className="search-results__heading">
            <div><h2 id="search-results-title">Search results</h2><p>{formatCount(firstPage.page.candidate_count)} ranked candidates for “{firstPage.query}”</p></div>
            <span><Check size={14} aria-hidden="true" /> {sentenceCase(firstPage.mode)} · capped at {formatCount(firstPage.page.result_cap)}</span>
          </div>
          {items.length ? items.map((item) => <ResultCard key={`${firstPage.query}:${JSON.stringify(firstPage.filters)}:${item.id}`} item={item} timezone={timezone} filterScope={toFilterScope(applied ?? draft, timezone)} feedback={feedback[item.id]} onFeedback={(label, evaluationSplit, notes) => submitFeedback.mutate({ result: item, label, evaluationSplit, notes })} />) : <EmptyState title="No matching archive records" description="Try a broader query, another search method, or remove one of the scope filters." />}
          {query.hasNextPage ? <button className="button button--secondary search-load-more" type="button" disabled={query.isFetchingNextPage} onClick={() => void query.fetchNextPage()}>{query.isFetchingNextPage ? 'Loading…' : 'Load more results'}</button> : null}
        </section>
      ) : null}

      <section className="panel rag-panel" aria-labelledby="rag-title">
        <div className="rag-panel__heading">
          <span><Sparkles size={19} aria-hidden="true" /></span>
          <div><h2 id="rag-title">Ask across retrieved excerpts</h2><p>Optional cited RAG sends only the top local excerpts to OpenAI. Source text is treated as untrusted data and every supported answer must cite a returned result.</p></div>
          <StatusBadge status={settings.data?.cloud_enabled ? 'cloud enabled' : 'cloud disabled'} />
        </div>
        {settings.data?.cloud_enabled ? (
          <form onSubmit={(event) => { event.preventDefault(); if (ragQuestion.trim()) rag.mutate() }}>
            <label><span>Question</span><textarea rows={3} value={ragQuestion} onChange={(event) => setRagQuestion(event.target.value)} placeholder="Ask a question grounded in the archive…" /></label>
            <button className="button button--primary" type="submit" disabled={!ragQuestion.trim() || rag.isPending}><Cloud size={15} aria-hidden="true" />{rag.isPending ? 'Answering…' : 'Generate cited answer'}</button>
          </form>
        ) : (
          <p className="rag-panel__disabled">Cloud processing is off. Enable it deliberately in <a href="/settings">Settings</a> before any excerpt can leave this device.</p>
        )}
        {rag.isError ? <div className="search-inline-error" role="alert">{rag.error.message}</div> : null}
        {rag.data ? (
          <article className="rag-answer">
            <header><strong>{rag.data.insufficient_evidence ? 'Insufficient evidence' : 'Cited answer'}</strong><span>{rag.data.model} · audit {rag.data.audit_id}</span></header>
            <p>{rag.data.answer}</p>
            <h3>Sources</h3>
            <ol>{rag.data.sources.filter((source) => rag.data.citation_ids.includes(source.id)).map((source) => <li key={source.id}><a href={source.jump?.href ?? `#search-result-${source.id}`}>{source.conversation?.title ?? sentenceCase(source.entity_type)}</a><span>{source.excerpt}</span></li>)}</ol>
          </article>
        ) : null}
        <details className="rag-audits">
          <summary>Local RAG audit history ({formatCount(ragAudits.data?.items.length ?? 0)} on this page)</summary>
          {ragAudits.isError ? (
            <ErrorState error={ragAudits.error} onRetry={() => void ragAudits.refetch()} />
          ) : ragAudits.data?.items.length ? (
            <ol className="rag-audits__list">
              {ragAudits.data.items.map((item) => (
                <li key={item.id}>
                  <div><strong>{item.status}</strong><time>{new Date(item.created_at).toLocaleString()}</time></div>
                  <span>{item.model} · {sentenceCase(item.search_mode)} · {formatCount(item.retrieved_ids.length)} retrieved · {formatCount(item.citation_ids.length)} cited</span>
                  {item.error ? <em>{item.error}</em> : null}
                  <RagAuditDetails item={item} />
                </li>
              ))}
            </ol>
          ) : ragAudits.isPending ? (
            <p>Loading local RAG audit history…</p>
          ) : (
            <p>No cloud RAG calls have been recorded.</p>
          )}
          {ragAudits.data ? (
            <CursorWindowControls
              label="RAG audit history"
              pageNumber={ragAudits.pageNumber}
              targetPageNumber={ragAudits.targetPageNumber}
              itemCount={ragAudits.data.items.length}
              total={null}
              hasPreviousPage={ragAudits.hasPreviousPage}
              hasNextPage={ragAudits.hasNextPage}
              loading={ragAudits.transitionPending || ragAudits.isPending}
              onPreviousPage={ragAudits.previousPage}
              onNextPage={ragAudits.nextPage}
            />
          ) : null}
        </details>
      </section>
    </div>
  )
}
