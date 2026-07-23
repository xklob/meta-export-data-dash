import { useSearch } from '@tanstack/react-router'
import { useVirtualizer } from '@tanstack/react-virtual'
import { CalendarDays, Clock3, History, MessageSquareText, Search } from 'lucide-react'
import { type CSSProperties, type FormEvent, useEffect, useMemo, useRef, useState } from 'react'
import { BoundedResultsHint } from '../components/BoundedResultsHint'
import { CoverageNotice } from '../components/CoverageNotice'
import { CursorWindowControls } from '../components/CursorWindowControls'
import { StarButton } from '../components/StarButton'
import { EmptyState, ErrorState, PageHeading, PageLoading, StatusBadge } from '../components/ui'
import { api, type TimelineQuery } from '../lib/api'
import type { TimelineItem } from '../lib/contracts'
import { formatCount, formatTimestampMs, localDateToTimestamp, sentenceCase } from '../lib/format'
import { useArchiveTimezone } from '../lib/useArchiveTimezone'
import { useCursorWindowQuery } from '../lib/useCursorWindowQuery'

type TimelineMode = 'timeline' | 'on-this-day'

function TimelineEntry({ item, timezone, position, setSize }: { item: TimelineItem; timezone: string; position: number; setSize?: number | null }) {
  return (
    <article className="timeline-entry" id={`${item.entity_type}-${item.entity_id}`} aria-posinset={position} aria-setsize={setSize ?? undefined} tabIndex={-1}>
      <span className={`timeline-entry__marker timeline-entry__marker--${item.entity_type}`} aria-hidden="true">
        {item.entity_type === 'message' ? <MessageSquareText size={16} /> : <History size={16} />}
      </span>
      <div className="timeline-entry__body">
        <header>
          <div>
            <StatusBadge status={item.entity_type} />
            <strong>{item.title || item.sender || sentenceCase(item.subtype)}</strong>
          </div>
          <time dateTime={item.timestamp_ms ? new Date(item.timestamp_ms).toISOString() : undefined}>
            {formatTimestampMs(item.timestamp_ms, timezone)}
          </time>
        </header>
        <p>{item.content || 'No exported body text is available for this record.'}</p>
        <footer>
          <span>{sentenceCase(item.category)} · {sentenceCase(item.subtype)}</span>
          <span className="timeline-entry__actions">
            <StarButton
              target={item.star_target ?? { target_type: item.entity_type, target_id: item.entity_id }}
              star={item.star}
              label={`${item.entity_type} ${item.title || item.sender || sentenceCase(item.subtype)}`}
            />
            <a className="button button--secondary button--compact" href={item.jump.href}>
              {item.entity_type === 'message' ? 'Jump to conversation' : 'Open source point'}
            </a>
          </span>
        </footer>
      </div>
    </article>
  )
}

function VirtualTimeline({
  items,
  total,
  timezone,
  focusId,
}: {
  items: TimelineItem[]
  total?: number | null
  timezone: string
  focusId: string
}) {
  const viewportRef = useRef<HTMLDivElement>(null)
  const virtualizer = useVirtualizer({
    count: items.length,
    getScrollElement: () => viewportRef.current,
    estimateSize: () => 170,
    overscan: 6,
    initialRect: { width: 960, height: 700 },
    getItemKey: (index) => items[index]?.id ?? index,
  })
  const virtualRows = virtualizer.getVirtualItems()
  const renderedRows = virtualRows.length
    ? virtualRows
    : items.slice(0, 10).map((_item, index) => ({ index, key: items[index]?.id ?? index, start: index * 170 }))

  useEffect(() => {
    if (!focusId || !items.length) return
    const index = items.findIndex((item) => `${item.entity_type}-${item.entity_id}` === focusId)
    if (index < 0) return
    virtualizer.scrollToIndex(index, { align: 'center' })
    requestAnimationFrame(() => document.getElementById(focusId)?.focus())
  }, [focusId, items, virtualizer])

  return (
    <>
      <BoundedResultsHint id="timeline-results-scroll-hint" subject="timeline entries" />
      <div
        className="timeline-list__viewport bounded-results-viewport"
        ref={viewportRef}
        role="feed"
        tabIndex={0}
        aria-label="Scrollable virtualized timeline"
        aria-describedby="timeline-results-scroll-hint"
        data-loaded-entry-count={items.length}
        data-rendered-entry-count={renderedRows.length}
      >
        <div className="timeline-list" style={{ height: virtualizer.getTotalSize() }}>
          <span className="timeline-list__rail" aria-hidden="true" />
          {renderedRows.map((virtualRow) => {
            const item = items[virtualRow.index]!
            return (
              <div
                className="timeline-list__row"
                key={item.id}
                data-index={virtualRow.index}
                ref={virtualizer.measureElement}
                style={{ transform: `translateY(${virtualRow.start}px)` } as CSSProperties}
              >
                <TimelineEntry item={item} timezone={timezone} position={virtualRow.index + 1} setSize={Math.max(total ?? 0, items.length)} />
              </div>
            )
          })}
        </div>
      </div>
    </>
  )
}

export function TimelinePage() {
  const timezone = useArchiveTimezone()
  const routeSearch = useSearch({ from: '/timeline' })
  const today = new Date()
  const [mode, setMode] = useState<TimelineMode>('timeline')
  const [draftQ, setDraftQ] = useState('')
  const [q, setQ] = useState('')
  const [entityType, setEntityType] = useState('')
  const [from, setFrom] = useState('')
  const [to, setTo] = useState('')
  const [month, setMonth] = useState(today.getMonth() + 1)
  const [day, setDay] = useState(today.getDate())
  const timelineOptions = useMemo<TimelineQuery>(
    () => ({
      q,
      entityType,
      fromMs: localDateToTimestamp(from, timezone),
      toMs: localDateToTimestamp(to, timezone, true),
      direction: 'desc',
      limit: 75,
      around: q || entityType || from || to ? undefined : routeSearch.around,
    }),
    [entityType, from, q, routeSearch.around, timezone, to],
  )
  const timeline = useCursorWindowQuery({
    queryKey: ['timeline', timelineOptions],
    queryFn: (cursor) => api.timeline({ ...timelineOptions, cursor }),
    enabled: mode === 'timeline',
  })
  const onThisDay = useCursorWindowQuery({
    queryKey: ['on-this-day', month, day, timezone, entityType],
    queryFn: (cursor) => api.onThisDay(month, day, timezone, entityType || undefined, cursor),
    enabled: mode === 'on-this-day',
  })
  const selectedQuery = mode === 'timeline' ? timeline : onThisDay
  const items =
    mode === 'timeline'
      ? timeline.data?.items ?? []
      : onThisDay.data?.items ?? []
  const coverage = timeline.data?.coverage
  const total = mode === 'timeline'
    ? timeline.data?.page.total
    : onThisDay.data?.page.total

  const focusId = window.location.hash.slice(1)

  function applySearch(event: FormEvent) {
    event.preventDefault()
    setQ(draftQ)
  }

  const pending = mode === 'timeline' ? timeline.isPending : onThisDay.isPending
  const error = mode === 'timeline' ? timeline.error : onThisDay.error

  return (
    <div className="page-stack stage4-page">
      <PageHeading
        eyebrow="Across the archive"
        title="Timeline"
        description="See messages and Facebook activity together, or revisit everything recorded on a calendar day."
      />
      <nav className="segmented-control" aria-label="Timeline mode">
        <button
          type="button"
          className={mode === 'timeline' ? 'segmented-control__active' : undefined}
          aria-pressed={mode === 'timeline'}
          onClick={() => setMode('timeline')}
        >
          Full timeline
        </button>
        <button
          type="button"
          className={mode === 'on-this-day' ? 'segmented-control__active' : undefined}
          aria-pressed={mode === 'on-this-day'}
          onClick={() => setMode('on-this-day')}
        >
          On this day
        </button>
      </nav>

      <form className="panel timeline-controls" onSubmit={applySearch}>
        {mode === 'timeline' ? (
          <label className="filter-field timeline-controls__search">
            <span>Search timeline</span>
            <span className="filter-input">
              <Search size={15} aria-hidden="true" />
              <input
                value={draftQ}
                onChange={(event) => setDraftQ(event.target.value)}
                placeholder="Search messages and activity"
              />
            </span>
          </label>
        ) : (
          <>
            <label className="filter-field">
              <span>Month</span>
              <select value={month} onChange={(event) => setMonth(Number(event.target.value))}>
                {Array.from({ length: 12 }, (_, index) => index + 1).map((value) => (
                  <option key={value} value={value}>
                    {new Intl.DateTimeFormat('en-US', { month: 'long', timeZone: 'UTC' }).format(
                      new Date(Date.UTC(2000, value - 1, 1)),
                    )}
                  </option>
                ))}
              </select>
            </label>
            <label className="filter-field">
              <span>Day</span>
              <input type="number" min={1} max={31} value={day} onChange={(event) => setDay(Number(event.target.value))} />
            </label>
          </>
        )}
        <label className="filter-field">
          <span>Record type</span>
          <select value={entityType} onChange={(event) => setEntityType(event.target.value)}>
            <option value="">Messages & activity</option>
            <option value="message">Messages only</option>
            <option value="activity">Activity only</option>
          </select>
        </label>
        {mode === 'timeline' ? (
          <>
            <label className="filter-field">
              <span>From</span>
              <input type="date" value={from} onChange={(event) => setFrom(event.target.value)} />
            </label>
            <label className="filter-field">
              <span>Through</span>
              <input type="date" value={to} onChange={(event) => setTo(event.target.value)} />
            </label>
            <button className="button button--primary" type="submit">Apply</button>
          </>
        ) : null}
      </form>

      {pending ? <PageLoading label="Building timeline" /> : null}
      {error ? (
        <ErrorState
          error={error}
          onRetry={() => void (mode === 'timeline' ? timeline.refetch() : onThisDay.refetch())}
        />
      ) : null}
      {mode === 'timeline' && coverage ? <CoverageNotice coverage={coverage} /> : null}

      {(!pending && !error) || selectedQuery.hasPreviousPage ? (
        <section
          key={`${mode}-results`}
          className="panel timeline-panel"
          aria-label={!pending && !error ? undefined : 'Timeline result navigation'}
          aria-labelledby={!pending && !error ? 'timeline-results-title' : undefined}
        >
          {!pending && !error ? (
            <>
              <div className="section-header">
                <div>
                  <h2 id="timeline-results-title">{mode === 'timeline' ? 'Archive chronology' : `On ${month}/${day}`}</h2>
                  <p>
                    {mode === 'timeline'
                      ? total == null ? 'Total unavailable' : `${formatCount(total)} matching records`
                      : `${formatCount(total)} matching records; ${formatCount(items.length)} shown from ${formatCount(new Set(items.map((item) => item.year)).size)} years on this page`}
                  </p>
                </div>
                <span className="timeline-timezone"><Clock3 size={14} /> {timezone}</span>
              </div>
              {items.length ? (
                <VirtualTimeline
                  key={`${mode}-${selectedQuery.pageNumber}`}
                  items={items}
                  total={total}
                  timezone={timezone}
                  focusId={focusId}
                />
              ) : (
                <EmptyState
                  title={mode === 'on-this-day' ? 'Nothing recorded on this day' : 'No matching timeline records'}
                  description="This only reflects dated records available in the export."
                  action={<CalendarDays size={20} aria-hidden="true" />}
                />
              )}
            </>
          ) : null}
          <CursorWindowControls
            key="timeline-results-pager"
            label={mode === 'timeline' ? 'Timeline results' : 'On this day results'}
            pageNumber={selectedQuery.pageNumber}
            targetPageNumber={selectedQuery.targetPageNumber}
            itemCount={!pending && !error ? items.length : 0}
            total={!pending && !error ? total : null}
            hasPreviousPage={selectedQuery.hasPreviousPage}
            hasNextPage={!error && selectedQuery.hasNextPage}
            loading={selectedQuery.transitionPending || pending}
            onPreviousPage={selectedQuery.previousPage}
            onNextPage={selectedQuery.nextPage}
          />
        </section>
      ) : null}
    </div>
  )
}
