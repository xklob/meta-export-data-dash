import { useInfiniteQuery } from '@tanstack/react-query'
import {
  CalendarClock,
  FileHeart,
  Files,
  MessageSquareText,
  Star,
  UserRound,
} from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { AttachmentPreview } from '../components/AttachmentPreview'
import { StarButton } from '../components/StarButton'
import { EmptyState, ErrorState, MetricCard, PageHeading, PageLoading, StatusBadge } from '../components/ui'
import { api } from '../lib/api'
import type { StarCategory, StarredItem, SortDirection } from '../lib/contracts'
import { formatCompactCount, formatCount, formatTimestampMs, sentenceCase } from '../lib/format'
import { useArchiveTimezone } from '../lib/useArchiveTimezone'
import './StarredPage.css'

const categories: Array<{ value: StarCategory; label: string }> = [
  { value: 'all', label: 'All' },
  { value: 'conversations', label: 'Conversations' },
  { value: 'messages', label: 'Messages' },
  { value: 'media', label: 'Media' },
  { value: 'activity', label: 'Activity' },
  { value: 'people', label: 'People' },
]

function kindLabel(item: StarredItem): string {
  if (item.kind !== 'media') return sentenceCase(item.kind)
  return item.media_scope === 'unique' ? 'Unique file' : 'Exact occurrence'
}

function StarredCard({
  item,
  timezone,
  onRemoved,
}: {
  item: StarredItem
  timezone: string
  onRemoved: (element: HTMLElement | null) => void
}) {
  const cardRef = useRef<HTMLElement>(null)
  const sourceDate = formatTimestampMs(item.timestamp_ms, timezone)
  const starredDate = formatTimestampMs(Date.parse(item.star.starred_at), timezone)
  const media = item.kind === 'media' ? item.media : null
  const destination = item.jump?.href ?? item.href

  return (
    <article
      ref={cardRef}
      className={`starred-card starred-card--${item.kind}`}
      data-starred-card
      data-star-target={`${item.star_target.target_type}:${item.star_target.target_id}`}
    >
      {media ? (
        <div className="starred-card__preview">
          <AttachmentPreview item={media} />
        </div>
      ) : (
        <span className="starred-card__icon" aria-hidden="true">
          {item.kind === 'conversation' || item.kind === 'message'
            ? <MessageSquareText size={20} />
            : item.kind === 'person'
              ? <UserRound size={20} />
              : <FileHeart size={20} />}
        </span>
      )}
      <div className="starred-card__body">
        <header>
          <div>
            <StatusBadge status={kindLabel(item)} />
            {item.kind === 'person' && item.provisional ? (
              <span className="status-badge status-badge--warning">Messenger label</span>
            ) : null}
          </div>
          <StarButton
            target={item.star_target}
            star={item.star}
            label={item.title || kindLabel(item).toLowerCase()}
            onChanged={(starred) => {
              if (!starred) onRemoved(cardRef.current)
            }}
          />
        </header>
        <h2>{item.title || `Untitled ${kindLabel(item).toLowerCase()}`}</h2>
        {item.subtitle ? <p className="starred-card__subtitle">{item.subtitle}</p> : null}
        {item.excerpt ? <p className="starred-card__excerpt">{item.excerpt}</p> : null}
        {media ? (
          <dl className="starred-card__media-meta">
            <div><dt>Type</dt><dd>{sentenceCase(media.type || media.kind)}</dd></div>
            <div><dt>Availability</dt><dd>{sentenceCase(media.availability)}</dd></div>
            <div><dt>Occurrences</dt><dd>{formatCount(media.occurrence_count ?? 1)}</dd></div>
          </dl>
        ) : null}
        {item.kind === 'person' && item.counts ? (
          <p className="starred-card__counts">
            {formatCompactCount(item.counts.messages)} messages · {formatCompactCount(item.counts.photos)} photos ·{' '}
            {formatCompactCount(item.counts.videos)} videos · {formatCompactCount(item.counts.audio)} audio
          </p>
        ) : null}
        <footer>
          <span>
            <CalendarClock size={14} aria-hidden="true" />
            {item.timestamp_ms == null ? 'Source date unavailable' : sourceDate}
            <small>Starred {starredDate}</small>
          </span>
          {destination ? (
            <a className="button button--secondary button--compact" href={destination}>
              {item.kind === 'message' ? 'Jump to message' : item.kind === 'media' ? 'Jump to source' : 'Open source'}
            </a>
          ) : (
            <span className="starred-card__no-jump">No retained source jump</span>
          )}
        </footer>
      </div>
    </article>
  )
}

export function StarredPage() {
  const timezone = useArchiveTimezone()
  const [category, setCategory] = useState<StarCategory>('all')
  const [direction, setDirection] = useState<SortDirection>('desc')
  const [dismissed, setDismissed] = useState<Set<string>>(() => new Set())
  const headingRef = useRef<HTMLHeadingElement>(null)
  const query = useInfiniteQuery({
    queryKey: ['stars', category, direction],
    initialPageParam: undefined as string | undefined,
    queryFn: ({ pageParam }) => api.stars({ category, direction, limit: 50, cursor: pageParam }),
    getNextPageParam: (page) => page.page.next_cursor ?? undefined,
  })

  useEffect(() => setDismissed(new Set()), [category, direction])
  const items = useMemo(() => {
    const seen = new Set<string>()
    return (query.data?.pages.flatMap((page) => page.items) ?? []).filter((item) => {
      const key = `${item.star_target.target_type}:${item.star_target.target_id}`
      if (seen.has(key) || dismissed.has(key)) return false
      seen.add(key)
      return true
    })
  }, [dismissed, query.data])
  const firstPage = query.data?.pages[0]
  const total = firstPage?.page.total ?? 0
  const facets = firstPage?.facets

  const removeCard = (item: StarredItem, element: HTMLElement | null) => {
    const cards = Array.from(document.querySelectorAll<HTMLElement>('[data-starred-card]'))
    const index = element ? cards.indexOf(element) : -1
    const focusCandidate = cards[index + 1] ?? cards[index - 1]
    const focusTarget = focusCandidate?.querySelector<HTMLButtonElement>('.star-button')
    focusTarget?.focus()
    const key = `${item.star_target.target_type}:${item.star_target.target_id}`
    setDismissed((value) => new Set(value).add(key))
    requestAnimationFrame(() => {
      if (focusTarget?.isConnected) focusTarget.focus()
      else headingRef.current?.focus()
    })
  }

  if (query.isLoading) return <PageLoading label="Loading starred items" />
  if (query.isError && !query.data) {
    return <ErrorState title="Starred items could not be loaded" error={query.error} onRetry={() => query.refetch()} />
  }

  return (
    <div className="page-stack starred-page">
      <PageHeading
        eyebrow="Personal library"
        title="Starred"
        description="Keep conversations, individual messages, media, Facebook activity, and Messenger contacts together without changing the source archive."
      />

      <section className="metric-grid metric-grid--four starred-metrics" aria-label="Starred item totals">
        <MetricCard icon={Star} label="All starred items" value={formatCount(facets?.all ?? (category === 'all' ? total : undefined))} detail="Active archive targets" />
        <MetricCard icon={MessageSquareText} label="Conversations & messages" value={formatCount((facets?.conversations ?? 0) + (facets?.messages ?? 0))} detail="Whole histories and exact messages" />
        <MetricCard icon={Files} label="Media" value={formatCount(facets?.media)} detail="Occurrences and unique files" />
        <MetricCard icon={UserRound} label="People" value={formatCount(facets?.people)} detail="Provisional Messenger labels" />
      </section>

      <section className="panel starred-toolbar" aria-label="Starred filters">
        <div className="horizontal-tabs starred-toolbar__categories">
          <nav className="segmented-control horizontal-tabs__scroller" aria-label="Starred item type">
            {categories.map((option) => (
              <button
                key={option.value}
                type="button"
                className={category === option.value ? 'segmented-control__active' : undefined}
                aria-label={option.label}
                aria-pressed={category === option.value}
                onClick={() => setCategory(option.value)}
              >
                {option.label}
                {facets?.[option.value] != null ? <span>{formatCount(facets[option.value])}</span> : null}
              </button>
            ))}
          </nav>
          <span className="horizontal-tabs__hint" aria-hidden="true">Swipe to see Activity &amp; People →</span>
        </div>
        <label className="filter-field">
          <span>Starred order</span>
          <select value={direction} onChange={(event) => setDirection(event.target.value as SortDirection)}>
            <option value="desc">Newest starred first</option>
            <option value="asc">Oldest starred first</option>
          </select>
        </label>
      </section>

      <section className="starred-results" aria-labelledby="starred-results-title">
        <div className="section-header">
          <div>
            <h2 id="starred-results-title" ref={headingRef} tabIndex={-1}>
              {categories.find((option) => option.value === category)?.label} items
            </h2>
            <p>{formatCount(items.length)} loaded · {formatCount(total)} matching</p>
          </div>
        </div>
        {items.length ? (
          <div className="starred-list" role="list">
            {items.map((item) => (
              <div role="listitem" key={`${item.star_target.target_type}:${item.star_target.target_id}`}>
                <StarredCard item={item} timezone={timezone} onRemoved={(element) => removeCard(item, element)} />
              </div>
            ))}
          </div>
        ) : (
          <div className="panel">
            <EmptyState
              title={category === 'all' ? 'Nothing starred yet' : `No starred ${category}`}
              description={category === 'all'
                ? 'Use the star button beside a conversation, message, file, activity record, or person to keep it here.'
                : 'Choose another category or star an item from its archive page.'}
              action={<Star size={20} aria-hidden="true" />}
            />
          </div>
        )}
        {query.isError && query.data ? (
          <div className="starred-pagination-error" role="alert">
            <span>
              {query.isFetchNextPageError
                ? 'The next starred page could not be loaded. Already loaded items remain available.'
                : 'Starred items could not be refreshed. Already loaded items remain available.'}
            </span>
            <button
              className="button button--secondary button--compact"
              type="button"
              onClick={() => void (query.isFetchNextPageError ? query.fetchNextPage() : query.refetch())}
            >
              Try again
            </button>
          </div>
        ) : null}
        {query.hasNextPage && !query.isFetchNextPageError ? (
          <div className="load-more-row">
            <span>{formatCount(items.length)} of {formatCount(total)} loaded</span>
            <button
              className="button button--secondary"
              type="button"
              disabled={query.isFetchingNextPage}
              onClick={() => query.fetchNextPage()}
            >
              {query.isFetchingNextPage ? 'Loading…' : 'Load more starred items'}
            </button>
          </div>
        ) : null}
      </section>
    </div>
  )
}
