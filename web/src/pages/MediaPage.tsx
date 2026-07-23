import { useInfiniteQuery } from '@tanstack/react-query'
import { Link, useSearch } from '@tanstack/react-router'
import { useVirtualizer } from '@tanstack/react-virtual'
import {
  ArrowDown,
  ArrowUp,
  CalendarRange,
  Check,
  Copy,
  FileCheck2,
  Files,
  GalleryHorizontal,
  Grid2X2,
  HardDrive,
  History,
  ImageOff,
  Info,
  List,
  MessageSquareText,
  Paperclip,
  Search,
  SlidersHorizontal,
  X,
} from 'lucide-react'
import { type CSSProperties, type FormEvent, useEffect, useMemo, useRef, useState } from 'react'
import { AttachmentPreview } from '../components/AttachmentPreview'
import { BoundedResultsHint } from '../components/BoundedResultsHint'
import { EmptyState, ErrorState, MetricCard, PageHeading, PageLoading, StatusBadge } from '../components/ui'
import { api, type MediaQuery } from '../lib/api'
import type { MediaItem, MediaView, SortDirection } from '../lib/contracts'
import {
  formatBytes,
  formatCount,
  formatDateMs,
  localDateToTimestamp,
  sentenceCase,
} from '../lib/format'
import { useArchiveTimezone } from '../lib/useArchiveTimezone'

type DisplayMode = 'grid' | 'table'
type MediaVisibilityMode = 'resolved-first' | 'all' | 'missing' | 'filtered'

interface MediaFilters {
  q: string
  type: string
  kind: string
  availability: string
  conversationId: string
  conversationQ: string
  conversationType: string
  sourceCategory: string
  fromDate: string
  toDate: string
  minSizeMb: string
  maxSizeMb: string
  sort: string
  direction: SortDirection
  view: MediaView
}

const baseFilters: MediaFilters = {
  q: '',
  type: '',
  kind: '',
  availability: '',
  conversationId: '',
  conversationQ: '',
  conversationType: '',
  sourceCategory: '',
  fromDate: '',
  toDate: '',
  minSizeMb: '',
  maxSizeMb: '',
  sort: 'date',
  direction: 'desc',
  view: 'occurrences',
}

const mediaTypes = ['photo', 'video', 'audio', 'gif', 'sticker', 'document', 'other']
const availabilityValues = ['resolved', 'missing', 'unlinked', 'external']

function mbToBytes(value: string): number | undefined {
  if (!value.trim()) return undefined
  const parsed = Number(value)
  return Number.isFinite(parsed) && parsed >= 0 ? Math.round(parsed * 1024 * 1024) : undefined
}

function toMediaQuery(filters: MediaFilters, timezone: string): MediaQuery {
  return {
    view: filters.view,
    q: filters.q,
    type: filters.type,
    kind: filters.kind,
    availability: filters.availability,
    conversationId: filters.conversationId,
    conversationQ: filters.conversationQ,
    conversationType: filters.conversationType,
    sourceCategory: filters.sourceCategory,
    fromMs: localDateToTimestamp(filters.fromDate, timezone),
    toMs: localDateToTimestamp(filters.toDate, timezone, true),
    minSize: mbToBytes(filters.minSizeMb),
    maxSize: mbToBytes(filters.maxSizeMb),
    sort: filters.sort,
    direction: filters.direction,
    limit: 60,
  }
}

function AvailabilityLabels({ item }: { item: MediaItem }) {
  return (
    <div className="media-badges">
      <StatusBadge status={item.availability} />
      {item.unknown_date ? <span className="status-badge status-badge--muted">Unknown date</span> : null}
      {item.provenance ? <span className="status-badge status-badge--warning">Unlinked</span> : null}
      {(item.occurrence_count ?? 0) > 1 ? <span className="status-badge">{formatCount(item.occurrence_count)} occurrences</span> : null}
    </div>
  )
}

function JumpLink({ item, compact = false }: { item: MediaItem; compact?: boolean }) {
  const className = compact ? 'media-jump media-jump--compact' : 'button button--secondary media-jump'
  if (item.jump?.conversation_id) {
    return (
      <a className={className} href={item.jump.href}>
        <MessageSquareText size={14} aria-hidden="true" />
        {item.jump.message_id ? 'Jump to message' : 'Open conversation'}
      </a>
    )
  }
  if (item.jump?.activity_id) {
    return (
      <a className={className} href={item.jump.href}>
        <History size={14} aria-hidden="true" />
        Jump to activity
      </a>
    )
  }
  return <span className="media-no-jump">No source link</span>
}

function ProvenanceAction({ item, compact = false }: { item: MediaItem; compact?: boolean }) {
  const [copyState, setCopyState] = useState<'idle' | 'copied' | 'failed'>('idle')
  const provenance = item.provenance
  const path = provenance?.library_path || provenance?.source_path
  if (!provenance || !path) return null

  const copyPath = async () => {
    try {
      if (!navigator.clipboard?.writeText) throw new Error('Clipboard access is unavailable')
      await navigator.clipboard.writeText(path)
      setCopyState('copied')
    } catch {
      setCopyState('failed')
    }
    window.setTimeout(() => setCopyState('idle'), 1800)
  }

  return (
    <div className={compact ? 'media-provenance media-provenance--compact' : 'media-provenance'}>
      {!compact ? <p>{provenance.explanation}</p> : null}
      {!compact ? <code title={path}>{path}</code> : null}
      <button className={compact ? '' : 'button button--secondary'} type="button" onClick={copyPath}>
        {copyState === 'copied' ? <Check size={13} aria-hidden="true" /> : <Copy size={13} aria-hidden="true" />}
        {copyState === 'copied' ? 'Path copied' : copyState === 'failed' ? 'Copy unavailable' : 'Copy library path'}
      </button>
      <span className="sr-only" aria-live="polite">{copyState === 'copied' ? `Copied ${path}` : copyState === 'failed' ? `Could not copy ${path}` : ''}</span>
    </div>
  )
}

function MediaCard({
  item,
  timezone,
  onOccurrences,
  focused,
  seekMs,
  position,
  setSize,
}: {
  item: MediaItem
  timezone: string
  onOccurrences: (item: MediaItem) => void
  focused: boolean
  seekMs?: number
  position: number
  setSize: number
}) {
  return (
    <article
      id={`media-occurrence-${item.id}`}
      className={`media-card ${focused ? 'media-card--focused' : ''}`}
      role="listitem"
      aria-posinset={position}
      aria-setsize={setSize}
      tabIndex={focused ? -1 : undefined}
    >
      <div className="media-card__preview"><AttachmentPreview item={item} seekMs={focused ? seekMs : undefined} /></div>
      <div className="media-card__body">
        <AvailabilityLabels item={item} />
        <h3 title={item.filename ?? undefined}>{item.filename || `${sentenceCase(item.type)} attachment`}</h3>
        <p>{item.conversation?.title || item.activity?.title || sentenceCase(item.activity?.category) || 'Not linked to a source record'}</p>
        <dl>
          <div><dt>Date</dt><dd>{formatDateMs(item.timestamp_ms, timezone)}</dd></div>
          <div><dt>Size</dt><dd>{formatBytes(item.size_bytes)}</dd></div>
          <div><dt>Type</dt><dd>{sentenceCase(item.type)}</dd></div>
          <div><dt>Source</dt><dd>{sentenceCase(item.source_category)}</dd></div>
        </dl>
        <ProvenanceAction item={item} />
        <div className="media-card__actions">
          <JumpLink item={item} />
          {item.occurrences_href && (item.occurrence_count ?? 0) > 1 ? (
            <button className="button button--secondary" type="button" onClick={() => onOccurrences(item)}>
              <GalleryHorizontal size={14} aria-hidden="true" /> Occurrences
            </button>
          ) : null}
        </div>
      </div>
    </article>
  )
}

function VirtualMediaGrid({
  items,
  total,
  timezone,
  onOccurrences,
  focusedOccurrenceId,
  seekMs,
}: {
  items: MediaItem[]
  total: number
  timezone: string
  onOccurrences: (item: MediaItem) => void
  focusedOccurrenceId?: string
  seekMs?: number
}) {
  const viewportRef = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState(1100)
  useEffect(() => {
    const element = viewportRef.current
    if (!element || typeof ResizeObserver === 'undefined') return
    const observer = new ResizeObserver(([entry]) => {
      if (entry) setWidth(entry.contentRect.width)
    })
    observer.observe(element)
    return () => observer.disconnect()
  }, [])
  const columns = width < 560 ? 1 : width < 850 ? 2 : width < 1180 ? 3 : 4
  const rows = Math.ceil(items.length / columns)
  const virtualizer = useVirtualizer({
    count: rows,
    getScrollElement: () => viewportRef.current,
    estimateSize: () => 460,
    overscan: 2,
    initialRect: { width: 1100, height: 820 },
  })
  const virtualRows = virtualizer.getVirtualItems()
  const renderedRows = virtualRows.length > 0
    ? virtualRows
    : Array.from({ length: Math.min(rows, 3) }, (_value, index) => ({ index, key: index, start: index * 460 }))
  return (
    <>
      <BoundedResultsHint id="media-results-scroll-hint" subject="media results" />
      <div
        className="media-grid-viewport bounded-results-viewport"
        ref={viewportRef}
        role="list"
        tabIndex={0}
        aria-label="Media gallery"
        aria-describedby="media-results-scroll-hint"
        data-loaded-item-count={items.length}
        data-rendered-item-count={renderedRows.reduce((count, row) => count + items.slice(row.index * columns, row.index * columns + columns).length, 0)}
      >
        <div className="media-grid-canvas" style={{ height: virtualizer.getTotalSize() }}>
          {renderedRows.map((row) => (
            <div className="media-grid-row" key={row.key} data-index={row.index} ref={virtualizer.measureElement} style={{ transform: `translateY(${row.start}px)`, gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))` }}>
              {items.slice(row.index * columns, row.index * columns + columns).map((item, index) => <MediaCard key={item.view_key} item={item} timezone={timezone} onOccurrences={onOccurrences} focused={item.id === focusedOccurrenceId} seekMs={seekMs} position={row.index * columns + index + 1} setSize={Math.max(total, items.length)} />)}
            </div>
          ))}
        </div>
      </div>
    </>
  )
}

function VirtualMediaTable({ items, total, timezone, onOccurrences, focusedOccurrenceId, seekMs }: { items: MediaItem[]; total: number; timezone: string; onOccurrences: (item: MediaItem) => void; focusedOccurrenceId?: string; seekMs?: number }) {
  const viewportRef = useRef<HTMLDivElement>(null)
  const virtualizer = useVirtualizer({
    count: items.length,
    getScrollElement: () => viewportRef.current,
    estimateSize: () => 76,
    overscan: 8,
    initialRect: { width: 1200, height: 700 },
    getItemKey: (index) => items[index]?.view_key ?? index,
  })
  const virtualRows = virtualizer.getVirtualItems()
  const renderedRows = virtualRows.length > 0
    ? virtualRows
    : items.slice(0, 12).map((_item, index) => ({ index, key: items[index]?.view_key ?? index, start: index * 76 }))
  return (
    <>
      <BoundedResultsHint id="media-results-scroll-hint" subject="media results" />
      <div className="media-table" role="table" aria-label="Media files" aria-colcount={8} aria-rowcount={Math.max(total, items.length) + 1} data-loaded-row-count={items.length} data-rendered-row-count={renderedRows.length}>
        <div className="media-table__header" role="row" aria-rowindex={1}><div role="columnheader">File</div><div role="columnheader">Type</div><div role="columnheader">Source context</div><div role="columnheader">Date</div><div role="columnheader">Size</div><div role="columnheader">Availability</div><div role="columnheader">Occurrences</div><div role="columnheader">Source / filesystem</div></div>
        <div
          className="media-table__viewport bounded-results-viewport"
          role="rowgroup"
          ref={viewportRef}
          tabIndex={0}
          aria-describedby="media-results-scroll-hint"
          style={{ '--media-row-count': Math.min(items.length, 9) } as CSSProperties}
        >
          <div className="media-table__canvas" style={{ height: virtualizer.getTotalSize() }}>
            {renderedRows.map((row) => {
              const item = items[row.index]!
              return (
                <div id={`media-occurrence-${item.id}`} className={`media-table__row ${item.id === focusedOccurrenceId ? 'media-table__row--focused' : ''}`} role="row" aria-rowindex={row.index + 2} key={item.view_key} data-index={row.index} ref={virtualizer.measureElement} tabIndex={item.id === focusedOccurrenceId ? -1 : undefined} style={{ transform: `translateY(${row.start}px)` }}>
                  <div className="media-table__file" role="cell"><div><AttachmentPreview item={item} compact seekMs={item.id === focusedOccurrenceId ? seekMs : undefined} /></div><span><strong>{item.filename || 'Unnamed attachment'}</strong><JumpLink item={item} compact /></span></div>
                  <div role="cell"><span className="compact-label">Type</span>{sentenceCase(item.type)}</div>
                  <div role="cell"><span className="compact-label">Source context</span>{item.conversation?.title || item.activity?.title || sentenceCase(item.activity?.category) || (item.provenance ? 'Unlinked' : 'No retained source')}</div>
                  <div role="cell"><span className="compact-label">Date</span>{formatDateMs(item.timestamp_ms, timezone)}</div>
                  <div role="cell"><span className="compact-label">Size</span>{formatBytes(item.size_bytes)}</div>
                  <div role="cell"><span className="compact-label">Availability</span><StatusBadge status={item.availability} /></div>
                  <div role="cell"><span className="compact-label">Occurrences</span>{item.occurrence_count ?? 1}{item.occurrences_href && (item.occurrence_count ?? 0) > 1 ? <button type="button" onClick={() => onOccurrences(item)}>View</button> : null}</div>
                  <div role="cell"><span className="compact-label">Source / filesystem</span>{item.provenance ? <ProvenanceAction item={item} compact /> : sentenceCase(item.source_category)}</div>
                </div>
              )
            })}
          </div>
        </div>
      </div>
    </>
  )
}

function OccurrenceDialog({ item, timezone, onClose }: { item: MediaItem; timezone: string; onClose: () => void }) {
  const closeRef = useRef<HTMLButtonElement>(null)
  const dialogRef = useRef<HTMLElement>(null)
  const onCloseRef = useRef(onClose)
  onCloseRef.current = onClose
  const occurrences = useInfiniteQuery({
    queryKey: ['blob-occurrences', item.blob_sha256],
    queryFn: ({ pageParam }) => api.blobOccurrences(item.blob_sha256 ?? '', pageParam),
    initialPageParam: undefined as string | undefined,
    getNextPageParam: (page) => page.page.next_cursor ?? undefined,
    enabled: Boolean(item.blob_sha256),
  })
  const entries = occurrences.data?.pages.flatMap((page) => page.items) ?? []
  useEffect(() => {
    const previousFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null
    const previousOverflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    requestAnimationFrame(() => closeRef.current?.focus())
    const keydown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault()
        onCloseRef.current()
        return
      }
      if (event.key !== 'Tab') return
      const focusable = dialogRef.current
        ? Array.from(
            dialogRef.current.querySelectorAll<HTMLElement>(
              'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), [tabindex]:not([tabindex="-1"])',
            ),
          )
        : []
      const first = focusable[0]
      const last = focusable.at(-1)
      if (!first || !last) return
      if (event.shiftKey && (document.activeElement === first || !dialogRef.current?.contains(document.activeElement))) {
        event.preventDefault()
        last.focus()
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault()
        first.focus()
      }
    }
    window.addEventListener('keydown', keydown)
    return () => {
      window.removeEventListener('keydown', keydown)
      document.body.style.overflow = previousOverflow
      requestAnimationFrame(() => previousFocus?.focus())
    }
  }, [])
  return (
    <div className="modal-backdrop" onMouseDown={(event) => { if (event.currentTarget === event.target) onCloseRef.current() }}>
      <section ref={dialogRef} className="occurrence-dialog" role="dialog" aria-modal="true" aria-labelledby="occurrence-dialog-title" aria-describedby="occurrence-dialog-description">
        <header><div><p className="eyebrow">Unique file</p><h2 id="occurrence-dialog-title">Every occurrence</h2><p id="occurrence-dialog-description">{item.filename || item.blob_sha256}</p></div><button ref={closeRef} className="icon-button" type="button" aria-label="Close occurrences" onClick={() => onCloseRef.current()}><X size={19} /></button></header>
        {occurrences.isLoading ? <div className="inline-loading" role="status">Loading occurrences…</div> : occurrences.isError ? <ErrorState error={occurrences.error} onRetry={() => occurrences.refetch()} /> : (
          <div className="occurrence-list"><p>{formatCount(occurrences.data?.pages[0]?.page.total)} retained occurrences</p><ul>{entries.map((entry) => <li key={entry.id}><div><strong>{entry.conversation?.title || entry.activity?.title || sentenceCase(entry.activity?.category) || (entry.provenance ? 'Unlinked file' : 'No retained source')}</strong><span>{formatDateMs(entry.timestamp_ms, timezone)} · {sentenceCase(entry.source_category)}</span></div><JumpLink item={entry} /></li>)}</ul>{occurrences.hasNextPage ? <button className="button button--secondary" type="button" disabled={occurrences.isFetchingNextPage} onClick={() => occurrences.fetchNextPage()}>Load more occurrences</button> : null}</div>
        )}
      </section>
    </div>
  )
}

export function MediaPage() {
  const { conversation, occurrence, t } = useSearch({ from: '/media' })
  const timezone = useArchiveTimezone()
  const initialFilters = useMemo(() => ({ ...baseFilters, conversationId: conversation ?? '' }), [conversation])
  const [draft, setDraft] = useState<MediaFilters>(initialFilters)
  const [applied, setApplied] = useState<MediaFilters>(initialFilters)
  const [display, setDisplay] = useState<DisplayMode>('grid')
  const [visibility, setVisibility] = useState<MediaVisibilityMode>(occurrence ? 'all' : 'resolved-first')
  const [filtersExpanded, setFiltersExpanded] = useState(false)
  const [occurrenceItem, setOccurrenceItem] = useState<MediaItem>()
  const query = useMemo(
    () => ({ ...toMediaQuery(applied, timezone), occurrenceId: occurrence }),
    [applied, occurrence, timezone],
  )
  const media = useInfiniteQuery({
    queryKey: ['media', query],
    queryFn: ({ pageParam }) => api.media({ ...query, cursor: pageParam }),
    initialPageParam: undefined as string | undefined,
    getNextPageParam: (page) => page.page.next_cursor ?? undefined,
  })
  const catalogItems = useMemo(() => {
    const seen = new Set<string>()
    return (media.data?.pages.flatMap((page) => page.items) ?? []).filter((item) => {
      if (seen.has(item.view_key)) return false
      seen.add(item.view_key)
      return true
    })
  }, [media.data])
  const items = useMemo(() => {
    if (visibility !== 'resolved-first' || applied.availability) return catalogItems
    return [...catalogItems].sort((left, right) => (
      Number(left.availability !== 'resolved') - Number(right.availability !== 'resolved')
    ))
  }, [applied.availability, catalogItems, visibility])
  useEffect(() => {
    if (!occurrence || !items.some((item) => item.id === occurrence)) return
    requestAnimationFrame(() => document.getElementById(`media-occurrence-${occurrence}`)?.focus())
  }, [items, occurrence])
  const firstPage = media.data?.pages[0]
  const updateDraft = <Key extends keyof MediaFilters>(key: Key, value: MediaFilters[Key]) => setDraft((current) => ({ ...current, [key]: value }))
  const applyFilters = (event?: FormEvent) => {
    event?.preventDefault()
    setApplied({ ...draft })
    setVisibility(draft.availability === 'missing' ? 'missing' : draft.availability ? 'filtered' : 'all')
  }
  const setQuickFilter = (key: 'type' | 'availability', value: string) => {
    const next = { ...draft, [key]: draft[key] === value ? '' : value }
    setDraft(next)
    setApplied(next)
    if (key === 'availability') setVisibility(next.availability === 'missing' ? 'missing' : next.availability ? 'filtered' : 'all')
  }
  const setVisibilityMode = (nextVisibility: Exclude<MediaVisibilityMode, 'filtered'>) => {
    const availability = nextVisibility === 'missing' ? 'missing' : ''
    setDraft((current) => ({ ...current, availability }))
    setApplied((current) => ({ ...current, availability }))
    setVisibility(nextVisibility)
  }

  if (media.isLoading) return <PageLoading label="Loading the media catalog" />
  if (media.isError) return <ErrorState title="Media could not be loaded" error={media.error} onRetry={() => media.refetch()} />
  const summary = firstPage?.summary
  const total = firstPage?.page.total ?? 0

  return (
    <div className="page-stack media-page">
      <PageHeading eyebrow="Attachment archive" title="Media" description="Browse photos, videos, audio, stickers, GIFs, documents, and other files independently of message text. Every linked item keeps a jump back to its original context." />
      {occurrence ? (
        <aside className="unlinked-media-note media-focus-note" aria-label="Focused attachment">
          <Info size={17} aria-hidden="true" />
          <p><strong>Showing the requested attachment</strong><span>The gallery is narrowed to its exact retained occurrence{t != null ? ` at ${formatCount(t)} ms` : ''}.</span></p>
          <Link className="button button--secondary" to="/media" search={{ conversation: undefined, occurrence: undefined, t: undefined }}>Browse all media</Link>
        </aside>
      ) : null}
      <section className="metric-grid" aria-label="Media totals">
        <MetricCard icon={Paperclip} label="Occurrences" value={formatCount(summary?.occurrences)} detail="Every appearance in the archive" />
        <MetricCard icon={FileCheck2} label="Unique files" value={formatCount(summary?.unique_files)} detail="Content-identical bytes collapsed" />
        <MetricCard icon={HardDrive} label="Occurrence bytes" value={formatBytes(summary?.occurrence_bytes)} detail="Duplicates counted separately" />
        <MetricCard icon={HardDrive} label="Unique bytes" value={formatBytes(summary?.unique_bytes)} detail="Physical content total" />
      </section>

      <section className="panel media-visibility" aria-labelledby="media-visibility-title">
        <div>
          <strong id="media-visibility-title">Gallery visibility</strong>
          <span>{visibility === 'resolved-first' ? 'Resolved files appear first within the loaded results; every missing, external, and unlinked record remains included.' : visibility === 'missing' ? 'Only missing-file placeholders are shown.' : visibility === 'filtered' ? 'A detailed availability filter controls this view.' : 'Every matching file follows the selected sort order.'}</span>
        </div>
        <div className="segmented-control" role="group" aria-label="Media visibility">
          <button type="button" className={visibility === 'resolved-first' ? 'segmented-control__active' : ''} aria-pressed={visibility === 'resolved-first'} onClick={() => setVisibilityMode('resolved-first')}>Resolved first</button>
          <button type="button" className={visibility === 'all' ? 'segmented-control__active' : ''} aria-pressed={visibility === 'all'} onClick={() => setVisibilityMode('all')}>Show all</button>
          <button type="button" className={visibility === 'missing' ? 'segmented-control__active' : ''} aria-pressed={visibility === 'missing'} onClick={() => setVisibilityMode('missing')}>Missing only</button>
        </div>
      </section>

      <form className="panel filter-panel media-filter-panel" onSubmit={applyFilters} aria-label="Media filters">
        <div className="filter-panel__heading"><div><SlidersHorizontal size={17} /><span><strong>Filter media</strong><small>Counts and facets update from the complete matching catalog.</small></span></div><div className="view-switchers"><div className="segmented-control" role="group" aria-label="Count mode"><button type="button" className={draft.view === 'occurrences' ? 'segmented-control__active' : ''} aria-pressed={draft.view === 'occurrences'} onClick={() => { updateDraft('view', 'occurrences'); setApplied({ ...draft, view: 'occurrences' }) }}>Occurrences</button><button type="button" className={draft.view === 'unique' ? 'segmented-control__active' : ''} aria-pressed={draft.view === 'unique'} onClick={() => { updateDraft('view', 'unique'); setApplied({ ...draft, view: 'unique' }) }}>Unique files</button></div><div className="segmented-control" role="group" aria-label="Display mode"><button type="button" className={display === 'grid' ? 'segmented-control__active' : ''} aria-pressed={display === 'grid'} onClick={() => setDisplay('grid')}><Grid2X2 size={13} /> Grid</button><button type="button" className={display === 'table' ? 'segmented-control__active' : ''} aria-pressed={display === 'table'} onClick={() => setDisplay('table')}><List size={13} /> Table</button></div></div></div>
        <button className="media-filter-toggle button button--secondary" type="button" aria-expanded={filtersExpanded} aria-controls="media-detailed-filters" onClick={() => setFiltersExpanded((value) => !value)}>{filtersExpanded ? 'Hide detailed filters' : 'Show detailed filters'}</button>
        <div id="media-detailed-filters" className={filtersExpanded ? 'media-detailed-filters media-detailed-filters--open' : 'media-detailed-filters'}>
        <div className="filter-grid filter-grid--media">
          <label className="filter-field filter-field--wide"><span>Filename</span><span className="filter-input"><Search size={15} /><input type="search" value={draft.q} onChange={(event) => updateDraft('q', event.target.value)} placeholder="Search attachment filenames" /></span></label>
          <label className="filter-field"><span>Media type</span><select value={draft.type} onChange={(event) => updateDraft('type', event.target.value)}><option value="">All types</option>{mediaTypes.map((value) => <option key={value} value={value}>{sentenceCase(value)}</option>)}</select></label>
          <label className="filter-field"><span>Availability</span><select value={draft.availability} onChange={(event) => updateDraft('availability', event.target.value)}><option value="">Any availability</option>{availabilityValues.map((value) => <option key={value} value={value}>{sentenceCase(value)}</option>)}</select></label>
          <label className="filter-field"><span>Attachment kind</span><input value={draft.kind} onChange={(event) => updateDraft('kind', event.target.value)} placeholder="photos, audio_files…" /></label>
          <label className="filter-field filter-field--wide"><span>Conversation title</span><input value={draft.conversationQ} onChange={(event) => updateDraft('conversationQ', event.target.value)} placeholder="Search conversation names" /></label>
          <label className="filter-field"><span>Conversation type</span><select value={draft.conversationType} onChange={(event) => updateDraft('conversationType', event.target.value)}><option value="">All conversations</option><option value="direct">Direct</option><option value="group">Group</option><option value="system">System</option></select></label>
          <label className="filter-field"><span>Source category</span><input value={draft.sourceCategory} onChange={(event) => updateDraft('sourceCategory', event.target.value)} placeholder="archived_threads" /></label>
          <label className="filter-field"><span>From date</span><input type="date" value={draft.fromDate} onChange={(event) => updateDraft('fromDate', event.target.value)} /></label>
          <label className="filter-field"><span>To date</span><input type="date" value={draft.toDate} onChange={(event) => updateDraft('toDate', event.target.value)} /></label>
          <label className="filter-field"><span>Minimum size (MB)</span><input type="number" min="0" step="0.01" value={draft.minSizeMb} onChange={(event) => updateDraft('minSizeMb', event.target.value)} /></label>
          <label className="filter-field"><span>Maximum size (MB)</span><input type="number" min="0" step="0.01" value={draft.maxSizeMb} onChange={(event) => updateDraft('maxSizeMb', event.target.value)} /></label>
          <label className="filter-field"><span>Sort by</span><select value={draft.sort} onChange={(event) => updateDraft('sort', event.target.value)}><option value="date">Date</option><option value="size">File size</option><option value="type">Media type</option><option value="filename">Filename</option><option value="conversation">Conversation</option><option value="availability">Availability</option>{draft.view === 'unique' ? <option value="occurrences">Occurrence count</option> : null}</select></label>
          <label className="filter-field"><span>Direction</span><select value={draft.direction} onChange={(event) => updateDraft('direction', event.target.value as SortDirection)}><option value="desc">Descending</option><option value="asc">Ascending</option></select></label>
        </div>
        {draft.conversationId ? <div className="active-filter"><MessageSquareText size={14} /><span>Exact conversation filter: <code>{draft.conversationId}</code></span><button type="button" aria-label="Clear exact conversation filter" onClick={() => updateDraft('conversationId', '')}><X size={14} /></button></div> : null}
        <div className="filter-actions"><span><CalendarRange size={14} /> Dates use {timezone}</span><button className="button button--secondary" type="button" onClick={() => { setDraft(baseFilters); setApplied(baseFilters); setVisibility('resolved-first') }}>Reset</button><button className="button button--primary" type="submit">Apply filters</button></div>
        </div>
      </form>

      <section className="media-facets panel" aria-label="Media facets">
        <div><strong>Types</strong>{mediaTypes.map((value) => <button type="button" key={value} className={applied.type === value ? 'facet-chip facet-chip--active' : 'facet-chip'} aria-pressed={applied.type === value} onClick={() => setQuickFilter('type', value)}>{sentenceCase(value)} <span>{formatCount(firstPage?.facets.type[value] ?? 0)}</span></button>)}</div>
        <div><strong>Availability</strong>{availabilityValues.map((value) => <button type="button" key={value} className={applied.availability === value ? 'facet-chip facet-chip--active' : 'facet-chip'} aria-pressed={applied.availability === value} onClick={() => setQuickFilter('availability', value)}>{sentenceCase(value)} <span>{formatCount(firstPage?.facets.availability[value] ?? 0)}</span></button>)}</div>
      </section>

      <section className="panel media-results" aria-labelledby="media-results-heading">
        <div className="table-toolbar"><div><h2 id="media-results-heading">{applied.view === 'unique' ? 'Unique file gallery' : 'Attachment occurrences'}</h2><p>{formatCount(items.length)} loaded · {formatCount(total)} matching · {display} view</p></div>{media.isFetching && !media.isFetchingNextPage ? <span className="loading-label" role="status">Updating…</span> : null}</div>
        {items.some((item) => item.provenance) ? (
          <aside className="unlinked-media-note" aria-label="About unlinked files">
            <Info size={17} aria-hidden="true" />
            <p><strong>Why these files have no source jump</strong><span>They were present in the export but no imported message or activity referenced them. Their cards include safe archive-relative source and library paths you can copy for filesystem browsing.</span></p>
          </aside>
        ) : null}
        {items.length === 0 ? <EmptyState title="No media matches" description="Try clearing type, availability, conversation, date, or size filters." action={<button className="button button--secondary" type="button" onClick={() => { setDraft(baseFilters); setApplied(baseFilters); setVisibility('resolved-first') }}>Reset filters</button>} /> : display === 'grid' ? <VirtualMediaGrid items={items} total={total} timezone={timezone} onOccurrences={setOccurrenceItem} focusedOccurrenceId={occurrence} seekMs={t} /> : <VirtualMediaTable items={items} total={total} timezone={timezone} onOccurrences={setOccurrenceItem} focusedOccurrenceId={occurrence} seekMs={t} />}
        {items.length > 0 ? <div className="load-more-row"><span>{formatCount(items.length)} of {formatCount(total)} {applied.view === 'unique' ? 'unique files' : 'occurrences'} loaded</span>{media.hasNextPage ? <button className="button button--secondary" type="button" disabled={media.isFetchingNextPage} onClick={() => media.fetchNextPage()}>{media.isFetchingNextPage ? 'Loading…' : 'Load more media'}</button> : <span>End of results</span>}</div> : null}
      </section>
      <p className="reader-privacy-note"><ImageOff size={14} /> Missing and external files stay visible as metadata placeholders. Documents are never embedded as active web content.</p>
      {occurrenceItem ? <OccurrenceDialog item={occurrenceItem} timezone={timezone} onClose={() => setOccurrenceItem(undefined)} /> : null}
    </div>
  )
}
