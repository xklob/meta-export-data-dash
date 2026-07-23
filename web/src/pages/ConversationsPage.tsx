import { useInfiniteQuery } from '@tanstack/react-query'
import { Link } from '@tanstack/react-router'
import { useVirtualizer } from '@tanstack/react-virtual'
import {
  ArrowDown,
  ArrowUp,
  CalendarRange,
  Files,
  MessageSquareText,
  Save,
  Search,
  SlidersHorizontal,
  Users,
  X,
} from 'lucide-react'
import { type CSSProperties, type FormEvent, useMemo, useRef, useState } from 'react'
import { BoundedResultsHint } from '../components/BoundedResultsHint'
import { StarButton } from '../components/StarButton'
import { EmptyState, ErrorState, MetricCard, PageHeading, PageLoading, StatusBadge } from '../components/ui'
import { api, type ConversationQuery } from '../lib/api'
import type { ConversationSummary, SortDirection } from '../lib/contracts'
import {
  formatBytes,
  formatCompactCount,
  formatCount,
  formatDateMs,
  localDateToTimestamp,
  sentenceCase,
} from '../lib/format'
import { useArchiveTimezone } from '../lib/useArchiveTimezone'

type AttachmentFilter = 'all' | 'yes' | 'no'

interface ConversationView {
  q: string
  conversationType: string
  sourceCategory: string
  completeness: string
  minParticipants: string
  maxParticipants: string
  minMessages: string
  maxMessages: string
  activityFrom: string
  activityTo: string
  hasAttachments: AttachmentFilter
  minPhotos: string
  maxPhotos: string
  minVideos: string
  maxVideos: string
  minAudio: string
  maxAudio: string
  sort: string
  direction: SortDirection
}

interface SavedConversationView {
  id: string
  name: string
  view: ConversationView
}

const SAVED_VIEWS_KEY = 'meta-archive:conversation-views:v1'

const defaultView: ConversationView = {
  q: '',
  conversationType: '',
  sourceCategory: '',
  completeness: '',
  minParticipants: '',
  maxParticipants: '',
  minMessages: '',
  maxMessages: '',
  activityFrom: '',
  activityTo: '',
  hasAttachments: 'all',
  minPhotos: '',
  maxPhotos: '',
  minVideos: '',
  maxVideos: '',
  minAudio: '',
  maxAudio: '',
  sort: 'last_activity',
  direction: 'desc',
}

const sortOptions = [
  ['last_activity', 'Last activity'],
  ['first_activity', 'First activity'],
  ['title', 'Title'],
  ['type', 'Conversation type'],
  ['source_category', 'Source category'],
  ['participants', 'Participants'],
  ['messages', 'Total messages'],
  ['sent', 'Messages sent'],
  ['received', 'Messages received'],
  ['attachments', 'Attachment occurrences'],
  ['unique_attachments', 'Unique files'],
  ['photos', 'Photos'],
  ['videos', 'Videos'],
  ['audio', 'Audio files'],
  ['attachment_bytes', 'Attachment bytes'],
  ['active_days', 'Active days'],
  ['completeness', 'Completeness'],
] as const

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function isConversationView(value: unknown): value is ConversationView {
  if (!isRecord(value)) return false
  const requiredFields = Object.keys(defaultView) as (keyof ConversationView)[]
  if (!requiredFields.every((field) => typeof value[field] === 'string')) return false
  return ['all', 'yes', 'no'].includes(value.hasAttachments as string)
    && ['asc', 'desc'].includes(value.direction as string)
    && sortOptions.some(([sort]) => sort === value.sort)
}

function isSavedConversationView(value: unknown): value is SavedConversationView {
  return isRecord(value)
    && typeof value.id === 'string'
    && value.id.length > 0
    && typeof value.name === 'string'
    && value.name.trim().length > 0
    && isConversationView(value.view)
}

function loadSavedViews(): SavedConversationView[] {
  try {
    const value = localStorage.getItem(SAVED_VIEWS_KEY)
    if (!value) return []
    const parsed = JSON.parse(value) as unknown
    return Array.isArray(parsed) ? parsed.filter(isSavedConversationView) : []
  } catch {
    return []
  }
}

function parseOptionalNumber(value: string): number | undefined {
  if (!value.trim()) return undefined
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : undefined
}

function toQuery(view: ConversationView, timezone: string): ConversationQuery {
  return {
    q: view.q,
    conversationType: view.conversationType,
    sourceCategory: view.sourceCategory,
    completeness: view.completeness,
    minParticipants: parseOptionalNumber(view.minParticipants),
    maxParticipants: parseOptionalNumber(view.maxParticipants),
    minMessages: parseOptionalNumber(view.minMessages),
    maxMessages: parseOptionalNumber(view.maxMessages),
    activityFromMs: localDateToTimestamp(view.activityFrom, timezone),
    activityToMs: localDateToTimestamp(view.activityTo, timezone, true),
    hasAttachments:
      view.hasAttachments === 'all' ? undefined : view.hasAttachments === 'yes',
    minPhotos: parseOptionalNumber(view.minPhotos),
    maxPhotos: parseOptionalNumber(view.maxPhotos),
    minVideos: parseOptionalNumber(view.minVideos),
    maxVideos: parseOptionalNumber(view.maxVideos),
    minAudio: parseOptionalNumber(view.minAudio),
    maxAudio: parseOptionalNumber(view.maxAudio),
    sort: view.sort,
    direction: view.direction,
    limit: 75,
  }
}

function ConversationTable({
  items,
  total,
  view,
  timezone,
  onSort,
}: {
  items: ConversationSummary[]
  total: number
  view: ConversationView
  timezone: string
  onSort: (sort: string) => void
}) {
  const scrollRef = useRef<HTMLDivElement>(null)
  const virtualizer = useVirtualizer({
    count: items.length,
    getScrollElement: () => scrollRef.current,
    estimateSize: () => 76,
    overscan: 7,
    initialRect: { width: 1280, height: 610 },
    getItemKey: (index) => items[index]?.id ?? index,
  })
  const columns: { sort?: string; label: string }[] = [
    { sort: 'title', label: 'Conversation' },
    { sort: 'type', label: 'Type' },
    { sort: 'participants', label: 'People' },
    { sort: 'last_activity', label: 'Activity' },
    { sort: 'messages', label: 'Messages' },
    { label: 'Sent / received' },
    { sort: 'attachments', label: 'Files / media' },
    { sort: 'attachment_bytes', label: 'Size' },
    { sort: 'completeness', label: 'Coverage' },
  ]
  const virtualRows = virtualizer.getVirtualItems()
  const renderedRows = virtualRows.length > 0
    ? virtualRows
    : items.slice(0, 10).map((_item, index) => ({ index, key: items[index]?.id ?? index, start: index * 76 }))

  return (
    <>
      <BoundedResultsHint id="conversation-results-scroll-hint" subject="conversations" />
      <div
      className="conversation-table"
      role="table"
      aria-label="Conversations"
      aria-colcount={columns.length}
      aria-rowcount={Math.max(total, items.length) + 1}
      data-loaded-row-count={items.length}
      data-rendered-row-count={renderedRows.length}
      data-acceptance-row-projection={JSON.stringify(items.map((item) => ({
        id: item.id,
        href: item.href,
        message_total: item.message_counts.total,
        attachment_occurrences: item.attachment_counts.occurrences,
      })))}
    >
      <div className="conversation-table__header" role="row" aria-rowindex={1}>
        {columns.map(({ sort, label }) => (
          <div
            role="columnheader"
            key={label}
            aria-sort={sort && view.sort === sort ? (view.direction === 'asc' ? 'ascending' : 'descending') : undefined}
          >
            {sort ? (
              <button type="button" onClick={() => onSort(sort)} aria-label={`Sort by ${label}`}>
                {label}
                {view.sort === sort ? (
                  view.direction === 'asc' ? <ArrowUp size={12} /> : <ArrowDown size={12} />
                ) : null}
              </button>
            ) : label}
          </div>
        ))}
      </div>
      <div
        className="conversation-table__viewport bounded-results-viewport"
        ref={scrollRef}
        role="rowgroup"
        tabIndex={0}
        aria-describedby="conversation-results-scroll-hint"
        style={{ '--conversation-row-count': Math.min(items.length, 8) } as CSSProperties}
      >
        <div className="conversation-table__canvas" style={{ height: virtualizer.getTotalSize() }}>
          {renderedRows.map((virtualRow) => {
            const conversation = items[virtualRow.index]!
            return (
              <div
                className="conversation-table__row"
                role="row"
                aria-rowindex={virtualRow.index + 2}
                key={conversation.id}
                data-index={virtualRow.index}
                ref={virtualizer.measureElement}
                style={{ transform: `translateY(${virtualRow.start}px)` }}
              >
                <div className="conversation-table__identity" role="cell">
                  <div className="conversation-table__identity-heading">
                    <Link
                      to="/conversations/$conversationId"
                      params={{ conversationId: conversation.id }}
                      search={{ around: undefined }}
                    >
                      {conversation.title || 'Untitled conversation'}
                    </Link>
                    <StarButton
                      target={conversation.star_target ?? { target_type: 'conversation', target_id: conversation.id }}
                      star={conversation.star}
                      label={`conversation ${conversation.title || 'Untitled conversation'}`}
                    />
                  </div>
                  <span title={conversation.source_categories.join(', ')}>
                    {conversation.source_categories.map(sentenceCase).join(' · ') || 'Unknown source'}
                  </span>
                  {conversation.content_profile.kind === 'media_only' ? <strong>Media-only thread · no message records</strong> : null}
                  {conversation.content_profile.kind === 'metadata_only' ? <strong>Metadata-only thread</strong> : null}
                </div>
                <div role="cell"><span className="compact-label">Type</span>{sentenceCase(conversation.type)}</div>
                <div className="number-cell" role="cell"><span className="compact-label">People</span>{formatCount(conversation.participant_count)}</div>
                <div className="conversation-table__activity" role="cell">
                  <span className="compact-label">Activity</span>
                  <strong title={`Last activity: ${formatDateMs(conversation.last_activity_ms, timezone)}`}>{formatDateMs(conversation.last_activity_ms, timezone)}</strong>
                  <span title={`First activity: ${formatDateMs(conversation.first_activity_ms, timezone)} · ${formatCount(conversation.active_days)} active days`}>{formatDateMs(conversation.first_activity_ms, timezone)} first · {formatCount(conversation.active_days)} active days</span>
                </div>
                <div className="number-cell" role="cell"><span className="compact-label">Messages</span>{formatCount(conversation.message_counts.total)}</div>
                <div className="number-cell conversation-table__metric-stack" role="cell"><span className="compact-label">Sent / received</span><strong>{formatCount(conversation.message_counts.sent)} / {formatCount(conversation.message_counts.received)}</strong></div>
                <div className="number-cell conversation-table__metric-stack conversation-table__files" role="cell">
                  <span className="compact-label">Files / media</span>
                  <strong>{formatCount(conversation.attachment_counts.occurrences)} / {formatCount(conversation.attachment_counts.unique)}</strong>
                  <small>
                    <span className="sr-only">{formatCount(conversation.media_counts.photo)} photos, {formatCount(conversation.media_counts.video)} videos, {formatCount(conversation.media_counts.audio)} audio files</span>
                    <span aria-hidden="true">{formatCompactCount(conversation.media_counts.photo)} P · {formatCompactCount(conversation.media_counts.video)} V · {formatCompactCount(conversation.media_counts.audio)} A</span>
                  </small>
                </div>
                <div className="number-cell" role="cell"><span className="compact-label">Size</span>{formatBytes(conversation.attachment_bytes)}</div>
                <div role="cell"><span className="compact-label">Coverage</span><StatusBadge status={conversation.completeness} /></div>
              </div>
            )
          })}
        </div>
      </div>
      </div>
    </>
  )
}

export function ConversationsPage() {
  const timezone = useArchiveTimezone()
  const [draft, setDraft] = useState<ConversationView>(defaultView)
  const [applied, setApplied] = useState<ConversationView>(defaultView)
  const [savedViews, setSavedViews] = useState<SavedConversationView[]>(loadSavedViews)
  const [saveName, setSaveName] = useState('')
  const [savingView, setSavingView] = useState(false)
  const [filtersExpanded, setFiltersExpanded] = useState(false)
  const resultsHeadingRef = useRef<HTMLHeadingElement>(null)
  const query = useMemo(() => toQuery(applied, timezone), [applied, timezone])
  const conversations = useInfiniteQuery({
    queryKey: ['conversations', query],
    queryFn: ({ pageParam }) => api.conversations({ ...query, cursor: pageParam }),
    initialPageParam: undefined as string | undefined,
    getNextPageParam: (lastPage) => lastPage.page.next_cursor ?? undefined,
  })
  const items = useMemo(() => {
    const seen = new Set<string>()
    return (conversations.data?.pages.flatMap((page) => page.items) ?? []).filter((item) => {
      if (seen.has(item.id)) return false
      seen.add(item.id)
      return true
    })
  }, [conversations.data])
  const total = conversations.data?.pages[0]?.page.total ?? 0
  const totals = useMemo(
    () => ({
      messages: items.reduce((sum, item) => sum + item.message_counts.total, 0),
      attachments: items.reduce((sum, item) => sum + item.attachment_counts.occurrences, 0),
      people: items.reduce((sum, item) => sum + item.participant_count, 0),
    }),
    [items],
  )

  const applyFilters = (event?: FormEvent) => {
    event?.preventDefault()
    setApplied({ ...draft })
    if (window.matchMedia?.('(max-width: 520px)').matches) {
      setFiltersExpanded(false)
      requestAnimationFrame(() => resultsHeadingRef.current?.focus())
    }
  }
  const activeFilterCount = useMemo(
    () => (Object.keys(draft) as (keyof ConversationView)[]).filter(
      (key) => key !== 'sort' && key !== 'direction' && draft[key] !== defaultView[key],
    ).length,
    [draft],
  )
  const updateDraft = <Key extends keyof ConversationView>(key: Key, value: ConversationView[Key]) =>
    setDraft((current) => ({ ...current, [key]: value }))
  const applySort = (sort: string) => {
    const next = {
      ...draft,
      sort,
      direction: applied.sort === sort && applied.direction === 'desc' ? ('asc' as const) : ('desc' as const),
    }
    setDraft(next)
    setApplied(next)
  }
  const storeSavedViews = (next: SavedConversationView[]) => {
    setSavedViews(next)
    localStorage.setItem(SAVED_VIEWS_KEY, JSON.stringify(next))
  }
  const saveCurrentView = () => {
    const name = saveName.trim()
    if (!name) return
    storeSavedViews([
      ...savedViews,
      { id: `${Date.now()}-${Math.random().toString(16).slice(2)}`, name, view: { ...applied } },
    ])
    setSaveName('')
    setSavingView(false)
  }

  if (conversations.isLoading) return <PageLoading label="Loading conversations" />
  if (conversations.isError) {
    return <ErrorState error={conversations.error} onRetry={() => conversations.refetch()} />
  }

  return (
    <div className="page-stack conversations-page">
      <PageHeading
        eyebrow="Message archive"
        title="Conversations"
        description="Browse direct, group, and system histories without loading the full archive into the browser. Counts distinguish attachment occurrences from unique files."
        actions={
          <Link className="button button--secondary" to="/media" search={{ conversation: undefined }}>
            <Files size={15} aria-hidden="true" /> Browse all media
          </Link>
        }
      />

      <section className="metric-grid" aria-label="Loaded conversation totals">
        <MetricCard icon={MessageSquareText} label="Matching conversations" value={formatCount(total)} detail={`${formatCount(items.length)} loaded`} />
        <MetricCard icon={MessageSquareText} label="Messages in loaded rows" value={formatCompactCount(totals.messages)} detail="Sent and received combined" />
        <MetricCard icon={Files} label="File occurrences loaded" value={formatCompactCount(totals.attachments)} detail="Duplicates counted separately" />
        <MetricCard icon={Users} label="Participant slots loaded" value={formatCompactCount(totals.people)} detail="Includes you in each thread" />
      </section>

      <form className="panel filter-panel" onSubmit={applyFilters} aria-label="Conversation filters">
        <div className="filter-panel__heading">
          <div>
            <SlidersHorizontal size={17} aria-hidden="true" />
            <span><strong>Filter and sort</strong><small>All filtering runs against the local catalog.</small></span>
          </div>
          <div className="saved-view-controls">
            <label>
              <span className="sr-only">Open a saved view</span>
              <select
                value=""
                onChange={(event) => {
                  const saved = savedViews.find((view) => view.id === event.target.value)
                  if (saved) {
                    setDraft(saved.view)
                    setApplied(saved.view)
                  }
                }}
              >
                <option value="">Saved views ({savedViews.length})</option>
                {savedViews.map((view) => <option key={view.id} value={view.id}>{view.name}</option>)}
              </select>
            </label>
            <button className="button button--secondary" type="button" onClick={() => { setSavingView((value) => !value); setFiltersExpanded(true) }}>
              <Save size={14} aria-hidden="true" /> Save view
            </button>
          </div>
        </div>
        <button className="conversation-filter-toggle button button--secondary" type="button" aria-expanded={filtersExpanded} aria-controls="conversation-detailed-filters" onClick={() => setFiltersExpanded((value) => !value)}>{filtersExpanded ? 'Hide detailed filters' : `Show detailed filters${activeFilterCount ? ` (${activeFilterCount} active)` : ''}`}</button>
        <div id="conversation-detailed-filters" className={filtersExpanded ? 'conversation-detailed-filters conversation-detailed-filters--open' : 'conversation-detailed-filters'}>
          {savingView ? (
            <div className="save-view-row">
              <label>View name<input value={saveName} onChange={(event) => setSaveName(event.target.value)} maxLength={60} autoFocus /></label>
              <button className="button button--primary" type="button" disabled={!saveName.trim()} onClick={saveCurrentView}>Save current filters</button>
              <button className="icon-button" type="button" aria-label="Cancel saving view" onClick={() => setSavingView(false)}><X size={17} /></button>
            </div>
          ) : null}
          <div className="filter-grid filter-grid--conversations">
          <label className="filter-field filter-field--wide">
            <span>Conversation title</span>
            <span className="filter-input"><Search size={15} aria-hidden="true" /><input type="search" value={draft.q} onChange={(event) => updateDraft('q', event.target.value)} placeholder="Search names or group titles" /></span>
          </label>
          <label className="filter-field"><span>Type</span><select value={draft.conversationType} onChange={(event) => updateDraft('conversationType', event.target.value)}><option value="">All types</option><option value="direct">Direct</option><option value="group">Group</option><option value="system">System</option><option value="unknown">Unknown</option></select></label>
          <label className="filter-field"><span>Completeness</span><select value={draft.completeness} onChange={(event) => updateDraft('completeness', event.target.value)}><option value="">Any coverage</option><option value="complete">Complete</option><option value="incomplete">Incomplete</option></select></label>
          <label className="filter-field"><span>Source category</span><input value={draft.sourceCategory} onChange={(event) => updateDraft('sourceCategory', event.target.value)} placeholder="e.g. archived_threads" /></label>
          <label className="filter-field"><span>Minimum people</span><input type="number" min="0" inputMode="numeric" value={draft.minParticipants} onChange={(event) => updateDraft('minParticipants', event.target.value)} /></label>
          <label className="filter-field"><span>Maximum people</span><input type="number" min="0" inputMode="numeric" value={draft.maxParticipants} onChange={(event) => updateDraft('maxParticipants', event.target.value)} /></label>
          <label className="filter-field"><span>Minimum messages</span><input type="number" min="0" inputMode="numeric" value={draft.minMessages} onChange={(event) => updateDraft('minMessages', event.target.value)} /></label>
          <label className="filter-field"><span>Maximum messages</span><input type="number" min="0" inputMode="numeric" value={draft.maxMessages} onChange={(event) => updateDraft('maxMessages', event.target.value)} /></label>
          <label className="filter-field"><span>Active after</span><input type="date" value={draft.activityFrom} onChange={(event) => updateDraft('activityFrom', event.target.value)} /></label>
          <label className="filter-field"><span>Active before</span><input type="date" value={draft.activityTo} onChange={(event) => updateDraft('activityTo', event.target.value)} /></label>
          <label className="filter-field"><span>Attachments</span><select value={draft.hasAttachments} onChange={(event) => updateDraft('hasAttachments', event.target.value as AttachmentFilter)}><option value="all">With or without</option><option value="yes">Has attachments</option><option value="no">No attachments</option></select></label>
          <label className="filter-field"><span>Minimum photos</span><input type="number" min="0" inputMode="numeric" value={draft.minPhotos} onChange={(event) => updateDraft('minPhotos', event.target.value)} /></label>
          <label className="filter-field"><span>Maximum photos</span><input type="number" min="0" inputMode="numeric" value={draft.maxPhotos} onChange={(event) => updateDraft('maxPhotos', event.target.value)} /></label>
          <label className="filter-field"><span>Minimum videos</span><input type="number" min="0" inputMode="numeric" value={draft.minVideos} onChange={(event) => updateDraft('minVideos', event.target.value)} /></label>
          <label className="filter-field"><span>Maximum videos</span><input type="number" min="0" inputMode="numeric" value={draft.maxVideos} onChange={(event) => updateDraft('maxVideos', event.target.value)} /></label>
          <label className="filter-field"><span>Minimum audio files</span><input type="number" min="0" inputMode="numeric" value={draft.minAudio} onChange={(event) => updateDraft('minAudio', event.target.value)} /></label>
          <label className="filter-field"><span>Maximum audio files</span><input type="number" min="0" inputMode="numeric" value={draft.maxAudio} onChange={(event) => updateDraft('maxAudio', event.target.value)} /></label>
          <label className="filter-field"><span>Sort by</span><select value={draft.sort} onChange={(event) => updateDraft('sort', event.target.value)}>{sortOptions.map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>
          <label className="filter-field"><span>Direction</span><select value={draft.direction} onChange={(event) => updateDraft('direction', event.target.value as SortDirection)}><option value="desc">Descending</option><option value="asc">Ascending</option></select></label>
          </div>
          <div className="filter-actions">
            <span><CalendarRange size={14} aria-hidden="true" /> Dates use {timezone}</span>
            <button className="button button--secondary" type="button" onClick={() => { setDraft(defaultView); setApplied(defaultView) }}>Reset</button>
            <button className="button button--primary" type="submit">Apply filters</button>
          </div>
        </div>
      </form>

      <section className="panel table-panel conversation-results" aria-labelledby="conversation-results-heading">
        <div className="table-toolbar">
          <div><h2 ref={resultsHeadingRef} id="conversation-results-heading" tabIndex={-1}>Conversation ledger</h2><p>{formatCount(items.length)} loaded · {formatCount(total)} matching</p></div>
          {conversations.isFetching && !conversations.isFetchingNextPage ? <span className="loading-label" role="status">Updating…</span> : null}
        </div>
        {items.length === 0 ? (
          <EmptyState title="No conversations match" description="Clear a filter or broaden the participant and activity ranges." action={<button className="button button--secondary" type="button" onClick={() => { setDraft(defaultView); setApplied(defaultView) }}>Reset filters</button>} />
        ) : (
          <>
            <ConversationTable items={items} total={total} view={applied} timezone={timezone} onSort={applySort} />
            <div className="load-more-row">
              <span>{formatCount(items.length)} of {formatCount(total)} conversations loaded</span>
              {conversations.hasNextPage ? <button className="button button--secondary" type="button" disabled={conversations.isFetchingNextPage} onClick={() => conversations.fetchNextPage()}>{conversations.isFetchingNextPage ? 'Loading…' : 'Load more conversations'}</button> : <span>End of results</span>}
            </div>
          </>
        )}
      </section>
    </div>
  )
}
