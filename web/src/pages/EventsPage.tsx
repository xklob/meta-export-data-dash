import { useSearch } from '@tanstack/react-router'
import { BadgeCheck, CalendarCheck2, CalendarDays, Search, UsersRound } from 'lucide-react'
import { type FormEvent, useEffect, useMemo, useState } from 'react'
import { CoverageNotice } from '../components/CoverageNotice'
import { CursorWindowControls } from '../components/CursorWindowControls'
import { StarButton } from '../components/StarButton'
import { EmptyState, ErrorState, MetricCard, PageHeading, PageLoading, StatusBadge } from '../components/ui'
import { api, type EventQuery } from '../lib/api'
import type { SortDirection } from '../lib/contracts'
import { formatCount, formatTimestampMs, localDateToTimestamp, sentenceCase } from '../lib/format'
import { useArchiveTimezone } from '../lib/useArchiveTimezone'
import { useCursorWindowQuery } from '../lib/useCursorWindowQuery'

interface EventFilters {
  q: string
  status: string
  hosted: 'all' | 'yes' | 'no'
  from: string
  to: string
  sort: string
  direction: SortDirection
}

const initialFilters: EventFilters = {
  q: '', status: '', hosted: 'all', from: '', to: '', sort: 'date', direction: 'desc',
}
export function EventsPage() {
  const timezone = useArchiveTimezone()
  const routeSearch = useSearch({ from: '/events' })
  const [draft, setDraft] = useState(initialFilters)
  const [filters, setFilters] = useState(initialFilters)
  const options = useMemo<EventQuery>(
    () => {
      const seekEligible = (Object.keys(initialFilters) as (keyof EventFilters)[])
        .every((key) => filters[key] === initialFilters[key])
      return {
        q: filters.q,
        status: filters.status,
        hosted: filters.hosted === 'all' ? undefined : filters.hosted === 'yes',
        fromMs: localDateToTimestamp(filters.from, timezone),
        toMs: localDateToTimestamp(filters.to, timezone, true),
        sort: filters.sort,
        direction: filters.direction,
        limit: 75,
        around: seekEligible ? routeSearch.around : undefined,
      }
    },
    [filters, routeSearch.around, timezone],
  )
  const query = useCursorWindowQuery({
    queryKey: ['events', options],
    queryFn: (cursor) => api.events({ ...options, cursor }),
  })
  const firstPage = query.data
  const items = query.data?.items ?? []

  useEffect(() => {
    const hash = window.location.hash.slice(1)
    if (!hash || !items.length) return
    requestAnimationFrame(() => document.getElementById(hash)?.scrollIntoView({ block: 'center' }))
  }, [items, query.pageNumber])

  function apply(event: FormEvent) {
    event.preventDefault()
    setFilters(draft)
  }

  return (
    <div className="page-stack stage4-page">
      <PageHeading
        eyebrow="Facebook events"
        title="Events"
        description="Browse hosted, attended, invited, interested, declined, and cancelled event records when present in the export."
      />
      <form className="panel event-filters" onSubmit={apply}>
        <label className="filter-field event-filters__search">
          <span>Search events</span>
          <span className="filter-input">
            <Search size={15} aria-hidden="true" />
            <input value={draft.q} onChange={(event) => setDraft((current) => ({ ...current, q: event.target.value }))} placeholder="Event title or description" />
          </span>
        </label>
        <label className="filter-field">
          <span>RSVP status</span>
          <select value={draft.status} onChange={(event) => setDraft((current) => ({ ...current, status: event.target.value }))}>
            <option value="">Every status</option>
            {['hosted', 'created', 'went', 'interested', 'invited', 'declined', 'cancelled', 'unknown'].map((status) => (
              <option key={status} value={status}>{sentenceCase(status)}</option>
            ))}
          </select>
        </label>
        <label className="filter-field">
          <span>Hosted by you</span>
          <select value={draft.hosted} onChange={(event) => setDraft((current) => ({ ...current, hosted: event.target.value as EventFilters['hosted'] }))}>
            <option value="all">Either</option>
            <option value="yes">Hosted</option>
            <option value="no">Not hosted</option>
          </select>
        </label>
        <label className="filter-field"><span>From</span><input type="date" value={draft.from} onChange={(event) => setDraft((current) => ({ ...current, from: event.target.value }))} /></label>
        <label className="filter-field"><span>Through</span><input type="date" value={draft.to} onChange={(event) => setDraft((current) => ({ ...current, to: event.target.value }))} /></label>
        <label className="filter-field">
          <span>Sort</span>
          <select value={draft.sort} onChange={(event) => setDraft((current) => ({ ...current, sort: event.target.value }))}>
            <option value="date">Date</option><option value="title">Title</option><option value="status">Status</option>
          </select>
        </label>
        <label className="filter-field">
          <span>Direction</span>
          <select value={draft.direction} onChange={(event) => setDraft((current) => ({ ...current, direction: event.target.value as SortDirection }))}>
            <option value="desc">Newest / Z–A</option><option value="asc">Oldest / A–Z</option>
          </select>
        </label>
        <button className="button button--primary" type="submit">Apply filters</button>
      </form>

      {query.isPending ? <PageLoading label="Loading events" /> : null}
      {query.isError ? <ErrorState error={query.error} onRetry={() => void query.refetch()} /> : null}
      {firstPage ? <CoverageNotice coverage={firstPage.coverage} /> : null}

      {firstPage?.summary ? (
        <section className="metric-grid metric-grid--four" aria-label="Event totals">
          <MetricCard icon={CalendarDays} label="Matching events" value={formatCount(firstPage.page.total)} detail="Source-backed records" />
          <MetricCard icon={BadgeCheck} label="Hosted / created" value={formatCount(firstPage.summary.hosted)} detail="Explicit source status" />
          <MetricCard icon={CalendarCheck2} label="Attended" value={formatCount(firstPage.summary.attended)} detail="Only explicit Went records" />
          <MetricCard icon={UsersRound} label="Other statuses" value={firstPage.summary.by_status ? formatCount(Object.values(firstPage.summary.by_status).reduce((sum, count) => sum + count, 0) - (firstPage.summary.attended ?? 0)) : '—'} detail="Invited, interested, declined, etc." />
        </section>
      ) : null}

      {query.isSuccess || query.hasPreviousPage ? (
        <section
          key="event-results"
          className="panel event-results"
          aria-label={query.isSuccess ? undefined : 'Event result navigation'}
          aria-labelledby={query.isSuccess ? 'event-results-title' : undefined}
        >
          {query.isSuccess ? (
            <>
              <div className="section-header"><div><h2 id="event-results-title">Event history</h2><p>Unknown dates and hosted status stay explicit.</p></div></div>
              {items.length ? (
                <div className="event-grid">
                  {items.map((event) => (
                    <article key={event.id} className="event-card" id={`event-${event.id}`}>
                      <div className="event-card__date"><CalendarDays size={18} /><span>{formatTimestampMs(event.timestamp_ms, timezone)}</span></div>
                      <div className="event-card__body">
                        <header><h3>{event.title || 'Untitled event'}</h3><span className="event-card__actions"><StatusBadge status={event.status} /><StarButton target={event.star_target ?? { target_type: 'activity', target_id: event.id }} star={event.star} label={`event ${event.title || 'Untitled event'}`} /></span></header>
                        <p>{event.description || 'No exported event description.'}</p>
                        <footer>
                          <span>{event.hosted == null ? 'Host unknown' : event.hosted ? 'Hosted by you' : 'Not hosted by you'}</span>
                          <span className="source-path" title={event.source.relative_path}>{event.source.relative_path}</span>
                        </footer>
                      </div>
                    </article>
                  ))}
                </div>
              ) : (
                <EmptyState
                  title={firstPage?.coverage.status === 'unavailable' ? 'Event history unavailable' : 'No matching events'}
                  description={firstPage?.coverage.description ?? 'Try broader filters.'}
                />
              )}
            </>
          ) : null}
          <CursorWindowControls
            key="event-results-pager"
            label="Event results"
            pageNumber={query.pageNumber}
            targetPageNumber={query.targetPageNumber}
            itemCount={query.isSuccess ? items.length : 0}
            total={query.isSuccess ? firstPage?.page.total : null}
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
