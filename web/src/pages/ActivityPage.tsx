import { useVirtualizer } from '@tanstack/react-virtual'
import { ArrowDown, ArrowUp, CalendarClock, Files, Filter, Search } from 'lucide-react'
import { type CSSProperties, type FormEvent, useMemo, useRef, useState } from 'react'
import { BoundedResultsHint } from '../components/BoundedResultsHint'
import { CoverageNotice } from '../components/CoverageNotice'
import { CursorWindowControls } from '../components/CursorWindowControls'
import { StarButton } from '../components/StarButton'
import { EmptyState, ErrorState, PageHeading, PageLoading, StatusBadge } from '../components/ui'
import { api, type ActivityQuery } from '../lib/api'
import type { ActivityItem, SortDirection } from '../lib/contracts'
import { formatCount, formatTimestampMs, localDateToTimestamp, sentenceCase } from '../lib/format'
import { useArchiveTimezone } from '../lib/useArchiveTimezone'
import { useCursorWindowQuery } from '../lib/useCursorWindowQuery'

const views = [
  ['all', 'All activity'],
  ['posts', 'Posts'],
  ['comments', 'Comments'],
  ['groups', 'Groups'],
  ['marketplace', 'Marketplace'],
  ['pages', 'Pages'],
  ['check-ins', 'Check-ins'],
  ['support', 'Support'],
] as const

interface ActivityFilters {
  q: string
  from: string
  to: string
  availability: string
  attachments: 'all' | 'yes' | 'no'
  postContext: string
  sort: string
  direction: SortDirection
}

const initialFilters: ActivityFilters = {
  q: '',
  from: '',
  to: '',
  availability: '',
  attachments: 'all',
  postContext: '',
  sort: 'date',
  direction: 'desc',
}

function postContextBasisLabel(value?: string | null): string {
  switch (value?.replaceAll(' ', '_')) {
    case 'explicit_source_field':
      return 'Explicit source field'
    case 'explicit_source_subtype':
    case 'source_subtype':
      return 'Source subtype classification'
    case 'source_category_and_subtype':
      return 'Source category and subtype'
    case 'not_explicit_in_source':
      return 'No explicit source classification'
    default:
      return value ? sentenceCase(value) : 'Basis unavailable'
  }
}

function ActivityVirtualTable({ items, total, timezone }: { items: ActivityItem[]; total?: number | null; timezone: string }) {
  const viewportRef = useRef<HTMLDivElement>(null)
  const virtualizer = useVirtualizer({
    count: items.length,
    getScrollElement: () => viewportRef.current,
    estimateSize: () => 108,
    overscan: 6,
    initialRect: { width: 1180, height: 680 },
    getItemKey: (index) => items[index]?.id ?? index,
  })
  const virtualRows = virtualizer.getVirtualItems()
  const renderedRows = virtualRows.length
    ? virtualRows
    : items.slice(0, 10).map((_item, index) => ({ index, key: items[index]?.id ?? index, start: index * 108 }))

  return (
    <>
      <BoundedResultsHint id="activity-results-scroll-hint" subject="activity records" />
      <div
        className="data-table-wrap activity-table__viewport bounded-results-viewport"
        ref={viewportRef}
        role="region"
        tabIndex={0}
        data-loaded-row-count={items.length}
        data-rendered-row-count={renderedRows.length}
        aria-label="Scrollable virtualized activity results"
        aria-describedby="activity-results-scroll-hint"
      >
        <table className="data-table activity-table" aria-colcount={8} aria-rowcount={Math.max(total ?? 0, items.length) + 1}>
          <thead>
            <tr aria-rowindex={1}>
              <th>Activity</th><th>Category</th><th>Post context</th><th>Actor</th>
              <th>Date</th><th>Files</th><th>Source</th><th>Actions</th>
            </tr>
          </thead>
          <tbody style={{ height: virtualizer.getTotalSize() }}>
            {renderedRows.map((virtualRow) => {
              const item = items[virtualRow.index]!
              return (
                <tr
                  key={item.id}
                  id={`activity-${item.id}`}
                  aria-rowindex={virtualRow.index + 2}
                  data-index={virtualRow.index}
                  ref={virtualizer.measureElement}
                  style={{ transform: `translateY(${virtualRow.start}px)` } as CSSProperties}
                >
                  <td data-label="Activity"><div className="table-primary"><strong>{item.title || sentenceCase(item.subtype)}</strong><span>{item.content || 'No exported body text'}</span><StatusBadge status={item.availability} /></div></td>
                  <td data-label="Category"><strong>{sentenceCase(item.category)}</strong><div className="muted-cell">{sentenceCase(item.subtype)}</div></td>
                  <td data-label="Post context">{item.post_context ? <><strong>{sentenceCase(item.post_context)}</strong><div className="muted-cell" title={item.post_context_basis ?? undefined}>{postContextBasisLabel(item.post_context_basis)}</div></> : <span className="muted-cell">Not applicable</span>}</td>
                  <td data-label="Actor">{item.actor || <span className="muted-cell">Not recorded</span>}</td>
                  <td data-label="Date">{formatTimestampMs(item.timestamp_ms, timezone)}</td>
                  <td className="number-cell" data-label="Files">{item.attachment_count ? <span className="inline-icon-value"><Files size={14} aria-hidden="true" /> {formatCount(item.attachment_count)} {item.attachment_count === 1 ? 'file' : 'files'}</span> : 'None'}</td>
                  <td className="activity-table__source" data-label="Source"><span className="source-path" title={item.source.relative_path}>{item.source.relative_path}</span></td>
                  <td className="activity-table__jump" data-label="Actions"><StarButton target={item.star_target ?? { target_type: 'activity', target_id: item.id }} star={item.star} label={`${sentenceCase(item.subtype)} activity`} /><a aria-label="Open source record" className="button button--secondary button--compact" href={item.jump.href}>Open<span className="activity-table__mobile-action-label" aria-hidden="true"> source record</span></a></td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </>
  )
}

export function ActivityPage() {
  const timezone = useArchiveTimezone()
  const [view, setView] = useState<(typeof views)[number][0]>('all')
  const [draft, setDraft] = useState(initialFilters)
  const [filters, setFilters] = useState(initialFilters)
  const supportsPostContext = view === 'all' || view === 'posts' || view === 'comments'
  const queryOptions = useMemo<ActivityQuery>(
    () => ({
      view,
      q: filters.q,
      availability: filters.availability,
      fromMs: localDateToTimestamp(filters.from, timezone),
      toMs: localDateToTimestamp(filters.to, timezone, true),
      hasAttachments:
        filters.attachments === 'all' ? undefined : filters.attachments === 'yes',
      postContext: supportsPostContext ? filters.postContext : undefined,
      sort: filters.sort,
      direction: filters.direction,
      limit: 75,
    }),
    [filters, supportsPostContext, timezone, view],
  )
  const query = useCursorWindowQuery({
    queryKey: ['activity', queryOptions],
    queryFn: (cursor) => api.activity({ ...queryOptions, cursor }),
  })
  const items = query.data?.items ?? []
  const firstPage = query.data
  const resultTotal = firstPage?.page.total

  function apply(event: FormEvent) {
    event.preventDefault()
    setFilters(draft)
  }

  return (
    <div className="page-stack stage4-page">
      <PageHeading
        eyebrow="Activity archive"
        title="Posts, comments & Facebook activity"
        description="Browse every activity record that this export actually contains, with source coverage kept visible."
      />

      <div className="horizontal-tabs">
        <nav className="segmented-control horizontal-tabs__scroller" aria-label="Activity category">
          {views.map(([key, label]) => (
            <button
              key={key}
              type="button"
              className={view === key ? 'segmented-control__active' : undefined}
              aria-pressed={view === key}
              onClick={() => setView(key)}
            >
              {label}
            </button>
          ))}
        </nav>
        <span className="horizontal-tabs__hint" aria-hidden="true">Swipe to see more categories →</span>
      </div>

      <form className="panel filter-panel" onSubmit={apply}>
        <div className="filter-panel__heading">
          <div>
            <Filter size={18} aria-hidden="true" />
            <span>
              <strong>Filter this activity</strong>
              <small>Dates are interpreted in {timezone}.</small>
            </span>
          </div>
          <button
            className="button button--ghost"
            type="button"
            onClick={() => {
              setDraft(initialFilters)
              setFilters(initialFilters)
            }}
          >
            Reset
          </button>
        </div>
        <div className="filter-grid">
          <label className="filter-field filter-field--wide">
            <span>Search title, text, or actor</span>
            <span className="filter-input">
              <Search size={15} aria-hidden="true" />
              <input
                value={draft.q}
                onChange={(event) => setDraft((current) => ({ ...current, q: event.target.value }))}
                placeholder="Search exported activity"
              />
            </span>
          </label>
          <label className="filter-field">
            <span>From</span>
            <input
              type="date"
              value={draft.from}
              onChange={(event) => setDraft((current) => ({ ...current, from: event.target.value }))}
            />
          </label>
          {supportsPostContext ? (
            <label className="filter-field">
              <span>Post/comment context</span>
              <select value={draft.postContext} onChange={(event) => setDraft((current) => ({ ...current, postContext: event.target.value }))}>
                <option value="">Every explicit context</option>
                <option value="own_wall">Your own wall</option>
                <option value="others_wall">Someone else&apos;s wall</option>
                <option value="group">Group</option>
                <option value="reshared">Reshared</option>
                <option value="tagged">Tagged</option>
                <option value="others_on_your_wall">Others on your wall</option>
                <option value="unknown">Unknown / not explicit</option>
              </select>
            </label>
          ) : null}
          <label className="filter-field">
            <span>Through</span>
            <input
              type="date"
              value={draft.to}
              onChange={(event) => setDraft((current) => ({ ...current, to: event.target.value }))}
            />
          </label>
          <label className="filter-field">
            <span>Availability</span>
            <select
              value={draft.availability}
              onChange={(event) => setDraft((current) => ({ ...current, availability: event.target.value }))}
            >
              <option value="">Any status</option>
              <option value="available">Available</option>
              <option value="unavailable_body">Unavailable body</option>
            </select>
          </label>
          <label className="filter-field">
            <span>Attachments</span>
            <select
              value={draft.attachments}
              onChange={(event) =>
                setDraft((current) => ({
                  ...current,
                  attachments: event.target.value as ActivityFilters['attachments'],
                }))
              }
            >
              <option value="all">Either</option>
              <option value="yes">Has files</option>
              <option value="no">No files</option>
            </select>
          </label>
          <label className="filter-field">
            <span>Sort</span>
            <select
              value={draft.sort}
              onChange={(event) => setDraft((current) => ({ ...current, sort: event.target.value }))}
            >
              <option value="date">Date</option>
              <option value="title">Title</option>
              <option value="actor">Actor</option>
              <option value="category">Category</option>
              <option value="subtype">Subtype</option>
            </select>
          </label>
          <label className="filter-field">
            <span>Direction</span>
            <select
              value={draft.direction}
              onChange={(event) =>
                setDraft((current) => ({ ...current, direction: event.target.value as SortDirection }))
              }
            >
              <option value="desc">Newest / Z–A</option>
              <option value="asc">Oldest / A–Z</option>
            </select>
          </label>
        </div>
        <div className="filter-actions">
          <span>
            <CalendarClock size={14} aria-hidden="true" />
            Unknown dates remain unknown and sort after dated records.
          </span>
          <button className="button button--primary" type="submit">
            Apply filters
          </button>
        </div>
      </form>

      {query.isPending ? <PageLoading label="Loading activity records" /> : null}
      {query.isError ? <ErrorState error={query.error} onRetry={() => void query.refetch()} /> : null}
      {firstPage ? <CoverageNotice coverage={firstPage.coverage} /> : null}
      {firstPage?.coverage_components.map((coverage) => (
        <CoverageNotice key={coverage.key ?? coverage.label} coverage={coverage} />
      ))}
      {filters.postContext && firstPage?.post_context_coverage ? (
        <div className={`coverage-notice ${firstPage.post_context_coverage.status === 'unknown' ? 'coverage-notice--partial' : ''}`} role="note">
          <span className="coverage-notice__icon" aria-hidden="true"><Filter size={18} /></span>
          <div><div className="coverage-notice__title"><strong>Post context: {sentenceCase(filters.postContext)}</strong><StatusBadge status={firstPage.post_context_coverage.status} /></div><p>{firstPage.post_context_coverage.description}</p><p className="coverage-notice__caution">Classification uses retained source fields, categories, and subtypes only; missing evidence is never treated as zero history.</p></div>
        </div>
      ) : null}

      {query.isSuccess || query.hasPreviousPage ? (
        <section
          key="activity-results"
          className="panel table-panel"
          aria-label={query.isSuccess ? undefined : 'Activity result navigation'}
          aria-labelledby={query.isSuccess ? 'activity-results-title' : undefined}
        >
          {query.isSuccess ? (
            <>
              <div className="table-toolbar table-toolbar--wrap">
                <div>
                  <h2 id="activity-results-title">{views.find(([key]) => key === view)?.[1]}</h2>
                  <p>
                    {resultTotal == null
                      ? 'Count unavailable from this export'
                      : `${formatCount(resultTotal)} matching source records`}
                  </p>
                </div>
                <span className="status-badge status-badge--info">
                  {filters.direction === 'desc' ? <ArrowDown size={13} /> : <ArrowUp size={13} />}
                  {sentenceCase(filters.sort)}
                </span>
              </div>
              {items.length === 0 ? (
                <EmptyState
                  title={firstPage?.coverage.status === 'unavailable' ? 'History unavailable' : 'No matching activity'}
                  description={
                    firstPage?.coverage.description ?? 'Try broadening the filters for this source-backed view.'
                  }
                />
              ) : (
                <ActivityVirtualTable
                  key={query.pageNumber}
                  items={items}
                  total={resultTotal}
                  timezone={timezone}
                />
              )}
            </>
          ) : null}
          <CursorWindowControls
            key="activity-results-pager"
            label="Activity results"
            pageNumber={query.pageNumber}
            targetPageNumber={query.targetPageNumber}
            itemCount={query.isSuccess ? items.length : 0}
            total={query.isSuccess ? resultTotal : null}
            hasPreviousPage={query.hasPreviousPage}
            hasNextPage={query.isSuccess && query.hasNextPage}
            loading={query.transitionPending || query.isPending}
            onPreviousPage={query.previousPage}
            onNextPage={query.nextPage}
          />
        </section>
      ) : null}
    </div>
  )
}
