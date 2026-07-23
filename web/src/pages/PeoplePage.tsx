import { Link, useNavigate, useSearch } from '@tanstack/react-router'
import { Images, Info, MessageSquareText, Search, UserRoundCheck, Users, Video } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { CoverageNotice } from '../components/CoverageNotice'
import { CursorWindowControls } from '../components/CursorWindowControls'
import { EmptyState, ErrorState, MetricCard, PageHeading, PageLoading, StatusBadge } from '../components/ui'
import { api, type PeopleQuery } from '../lib/api'
import type { PersonItem, SortDirection } from '../lib/contracts'
import { formatCompactCount, formatCount, formatDateMs, sentenceCase } from '../lib/format'
import { useArchiveTimezone } from '../lib/useArchiveTimezone'
import { useCursorWindowQuery } from '../lib/useCursorWindowQuery'

const views = [
  ['contacts', 'Messenger contacts'],
  ['friends', 'Friends'],
  ['requests', 'Requests'],
  ['followers', 'Followers'],
  ['facebook-blocks', 'Facebook blocks'],
  ['messenger-blocks', 'Messenger blocks'],
] as const

const MOBILE_PEOPLE_QUERY = '(max-width: 560px)'
const DESKTOP_PEOPLE_PAGE_SIZE = 24
const MOBILE_PEOPLE_PAGE_SIZE = 8

function useCompactPeopleWindow() {
  const [compact, setCompact] = useState(() => (
    Boolean(typeof window !== 'undefined' && window.matchMedia?.(MOBILE_PEOPLE_QUERY).matches)
  ))

  useEffect(() => {
    const media = window.matchMedia?.(MOBILE_PEOPLE_QUERY)
    if (!media) return undefined
    const update = () => setCompact(media.matches)
    update()
    media.addEventListener('change', update)
    return () => media.removeEventListener('change', update)
  }, [])

  return compact
}

function ContactCard({ person, timezone }: { person: PersonItem; timezone: string }) {
  const counts = person.direct_counts ?? person.counts
  return (
    <article className="person-card" id={`relationship-${person.id}`}>
      <header className="person-card__header">
        <span className="person-avatar" aria-hidden="true">
          {(person.name || '?').slice(0, 1).toLocaleUpperCase()}
        </span>
        <div>
          <h2>{person.name || 'Unnamed record'}</h2>
          <p>
            {person.contact_basis === 'messenger_participant_label'
              ? 'Messenger participant label'
              : sentenceCase(person.status)}
          </p>
        </div>
        <StatusBadge status={person.friendship_status ?? person.status ?? 'unknown'} />
      </header>

      {counts ? (
        <dl className="person-card__metrics" aria-label="Direct one-to-one exchanges">
          <div><dt>Messages</dt><dd>{formatCompactCount(counts.messages)}</dd></div>
          <div><dt>Photos</dt><dd>{formatCompactCount(counts.photos)}</dd></div>
          <div><dt>Videos</dt><dd>{formatCompactCount(counts.videos)}</dd></div>
          <div><dt>Audio</dt><dd>{formatCompactCount(counts.audio)}</dd></div>
        </dl>
      ) : null}

      <div className="person-card__dates">
        <span>First seen <strong>{formatDateMs(person.first_activity_ms ?? person.timestamp_ms, timezone)}</strong></span>
        <span>Last seen <strong>{formatDateMs(person.last_activity_ms ?? person.timestamp_ms, timezone)}</strong></span>
      </div>

      {person.conversations?.length ? (
        <details className="person-conversations">
          <summary>
            {formatCount(person.conversation_count)} linked conversations
            <small>{formatCount(person.direct_conversation_count)} direct · {formatCount(person.group_conversation_count)} group</small>
          </summary>
          <div>
            {person.conversations.map((conversation) => (
              <Link
                key={conversation.id}
                to="/conversations/$conversationId"
                params={{ conversationId: conversation.id }}
                search={{ around: undefined }}
              >
                <span>
                  <strong>{conversation.title}</strong>
                  <small>{sentenceCase(conversation.type)} · {formatCount(conversation.message_count)} whole-thread messages</small>
                </span>
                Open
              </Link>
            ))}
          </div>
        </details>
      ) : null}

      {person.shared_group_counts && person.group_conversation_count ? (
        <div className="person-card__semantics">
          <strong>Shared group threads (whole-thread totals)</strong>
          <span>{formatCompactCount(person.shared_group_counts.messages)} messages · {formatCompactCount(person.shared_group_counts.photos)} photos · {formatCompactCount(person.shared_group_counts.videos)} videos · {formatCompactCount(person.shared_group_counts.audio)} audio</span>
          <p>These totals cover full group conversations and are not attributed to this person.</p>
        </div>
      ) : null}
    </article>
  )
}

export function PeoplePage() {
  const timezone = useArchiveTimezone()
  const compactWindow = useCompactPeopleWindow()
  const navigate = useNavigate({ from: '/people' })
  const routeSearch = useSearch({ from: '/people' })
  const view = views.some(([key]) => key === routeSearch.view)
    ? routeSearch.view as (typeof views)[number][0]
    : 'contacts'
  const [q, setQ] = useState('')
  const [conversationType, setConversationType] = useState('')
  const [sort, setSort] = useState('messages')
  const [direction, setDirection] = useState<SortDirection>('desc')
  const options = useMemo<PeopleQuery>(
    () => {
      const contactSort = ['messages', 'photos', 'videos', 'audio', 'conversations', 'last_activity', 'name'].includes(sort) ? sort : 'messages'
      const relationshipSort = ['name', 'date', 'status'].includes(sort) ? sort : 'date'
      return {
        view,
        q,
        conversationType: view === 'contacts' ? conversationType : undefined,
        sort: view === 'contacts' ? contactSort : relationshipSort,
        direction,
        limit: compactWindow ? MOBILE_PEOPLE_PAGE_SIZE : DESKTOP_PEOPLE_PAGE_SIZE,
        around: q ? undefined : routeSearch.around,
      }
    },
    [compactWindow, conversationType, direction, q, routeSearch.around, sort, view],
  )
  const query = useCursorWindowQuery({
    queryKey: ['people', options],
    queryFn: (cursor) => api.people({ ...options, cursor }),
  })
  const firstPage = query.data
  const items = query.data?.items ?? []

  useEffect(() => {
    const hash = window.location.hash.slice(1)
    if (!hash || !items.length) return
    requestAnimationFrame(() => document.getElementById(hash)?.scrollIntoView({ block: 'center' }))
  }, [items, query.pageNumber])
  const totals = useMemo(
    () => items.reduce(
      (current, person) => ({
        messages: current.messages + ((person.direct_counts ?? person.counts)?.messages ?? 0),
        photos: current.photos + ((person.direct_counts ?? person.counts)?.photos ?? 0),
        videos: current.videos + ((person.direct_counts ?? person.counts)?.videos ?? 0),
      }),
      { messages: 0, photos: 0, videos: 0 },
    ),
    [items],
  )

  return (
    <div className="page-stack stage4-page people-page">
      <PageHeading
        eyebrow="People & relationship records"
        title="People"
        description="Explore conversation contacts and any authoritative friend, request, follower, or block records available in this export."
      />
      <div className="horizontal-tabs">
        <nav className="segmented-control horizontal-tabs__scroller" aria-label="People view">
          {views.map(([key, label]) => (
            <button
              key={key}
              type="button"
              className={view === key ? 'segmented-control__active' : undefined}
              aria-pressed={view === key}
              onClick={() => void navigate({ search: { view: key === 'contacts' ? undefined : key, around: undefined }, replace: true })}
            >
              {label}
            </button>
          ))}
        </nav>
        <span className="horizontal-tabs__hint" aria-hidden="true">Swipe to see more relationships →</span>
      </div>

      <section className="panel people-toolbar" aria-label="People filters">
        <label className="filter-field people-toolbar__search">
          <span>Search names</span>
          <span className="filter-input">
            <Search size={15} aria-hidden="true" />
            <input value={q} onChange={(event) => setQ(event.target.value)} placeholder="Find a person" />
          </span>
        </label>
        {view === 'contacts' ? (
            <label className="filter-field">
              <span>Conversation type</span>
              <select value={conversationType} onChange={(event) => setConversationType(event.target.value)}>
                <option value="">Direct & groups</option>
                <option value="direct">Has direct chat</option>
                <option value="group">Has shared group</option>
              </select>
            </label>
        ) : null}
        <label className="filter-field">
          <span>Sort by</span>
          {view === 'contacts' ? (
            <select value={['messages', 'photos', 'videos', 'audio', 'conversations', 'last_activity', 'name'].includes(sort) ? sort : 'messages'} onChange={(event) => setSort(event.target.value)}>
                <option value="messages">Messages</option>
                <option value="photos">Photos</option>
                <option value="videos">Videos</option>
                <option value="audio">Audio</option>
                <option value="conversations">Conversations</option>
                <option value="last_activity">Last activity</option>
                <option value="name">Name</option>
            </select>
          ) : (
            <select value={['name', 'date', 'status'].includes(sort) ? sort : 'date'} onChange={(event) => setSort(event.target.value)}>
              <option value="date">Date</option><option value="name">Name</option><option value="status">Status</option>
            </select>
          )}
        </label>
        <label className="filter-field">
          <span>Direction</span>
          <select value={direction} onChange={(event) => setDirection(event.target.value as SortDirection)}>
            <option value="desc">Highest / newest</option>
            <option value="asc">Lowest / oldest</option>
          </select>
        </label>
      </section>

      {query.isPending ? <PageLoading label="Loading people records" /> : null}
      {query.isError ? <ErrorState error={query.error} onRetry={() => void query.refetch()} /> : null}
      {firstPage ? <CoverageNotice coverage={firstPage.coverage} /> : null}

      {view === 'contacts' && firstPage ? (
        <section className="metric-grid metric-grid--four" aria-label="Visible contact exchange totals">
          <MetricCard icon={Users} label="Contacts" value={formatCount(firstPage.page.total)} detail="Participant labels, not friends" />
          <MetricCard icon={MessageSquareText} label="Messages" value={formatCompactCount(totals.messages)} detail="Visible direct chats" />
          <MetricCard icon={Images} label="Photos" value={formatCompactCount(totals.photos)} detail="Direct-chat occurrences" />
          <MetricCard icon={Video} label="Videos" value={formatCompactCount(totals.videos)} detail="Direct-chat occurrences" />
        </section>
      ) : null}

      {view === 'contacts' && items.length > 0 && (
        firstPage?.messenger_contact_matches_are_provisional
        || items.some((person) => person.messenger_contact_match)
      ) ? (
        <p className="panel people-identity-note" role="note">
          <Info size={16} aria-hidden="true" />
          Conversation links use provisional exact display-label matches, not Facebook identity matches.
        </p>
      ) : null}

      {query.isSuccess || query.hasPreviousPage ? (
        <section key="people-results" aria-label="People results">
          {query.isSuccess ? (
            items.length ? (
              <div className="people-grid">
                {items.map((person) => <ContactCard key={person.id} person={person} timezone={timezone} />)}
              </div>
            ) : (
              <div className="panel">
                <EmptyState
                  title={firstPage?.coverage.status === 'unavailable' ? 'Relationship history unavailable' : 'No matching people'}
                  description={firstPage?.coverage.description ?? 'Try a broader name search.'}
                  action={<UserRoundCheck size={20} aria-hidden="true" />}
                />
              </div>
            )
          ) : null}
          <div className="panel cursor-window-panel">
            <CursorWindowControls
              key="people-results-pager"
              label="People results"
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
          </div>
        </section>
      ) : null}
    </div>
  )
}
