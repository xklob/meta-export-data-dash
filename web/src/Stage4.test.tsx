import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { createMemoryHistory, createRouter, RouterProvider } from '@tanstack/react-router'
import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { routeTree } from './router'
import { fixtures } from './test/fixtures'

vi.mock('./components/ArchiveChart', () => ({
  ArchiveChart: ({ label, option }: { label: string; option?: { legend?: unknown; series?: unknown } }) => {
    const legend = Array.isArray(option?.legend) ? option.legend[0] : option?.legend
    const series = Array.isArray(option?.series) ? option.series : []
    const labelsHidden = series.length > 0 && series.every((item) => item?.label?.show === false)
    return <div role="img" aria-label={label} data-legend-visibility={legend?.show === false ? 'hidden' : 'visible'} data-slice-labels={labelsHidden ? 'hidden' : 'visible'} />
  },
}))

const partialCoverage = {
  key: 'posts', label: 'Posts and albums', status: 'partial', raw_count: 2, logical_count: 2,
  description: 'Only exported post-related records are shown.', source_paths: {}, absence_is_not_zero_history: false,
}
const unavailableCoverage = {
  key: 'events', label: 'Events', status: 'unavailable', raw_count: null, logical_count: null,
  description: 'Event history is unavailable in this export.', source_paths: {}, absence_is_not_zero_history: true,
}
const activityItem = {
  id: 'activity-one', parent_id: null, category: 'posts', subtype: 'album_photo', timestamp_ms: 1_700_000_000_000,
  unknown_date: false, title: 'Fixture album photo', content: 'A retained caption', actor: 'Archive User',
  availability: 'available', target_uri: null, attachment_count: 1,
  post_context: 'group', post_context_basis: 'explicit source subtype',
  source: { source_file_id: 1, source_record_id: 'source-record', chunk: 'package-a', relative_path: 'your_facebook_activity/posts/album.json', json_pointer: '/photos/0' },
  jump: { href: '/timeline?around=activity:activity-one#activity-activity-one', fragment: 'activity-activity-one' },
}

function stageFourFetchMock() {
  const mock = vi.fn(async (input: RequestInfo | URL) => {
    const url = new URL(String(input), 'http://localhost')
    if (url.pathname === '/api/v1/health') return Response.json(fixtures.health)
    if (url.pathname === '/api/v1/settings') return Response.json(fixtures.settings)
    if (url.pathname === '/api/v1/activity' || url.pathname.startsWith('/api/v1/activity/views/')) {
      return Response.json({ coverage: partialCoverage, coverage_components: [], post_context_coverage: { status: url.searchParams.has('post_context') ? 'source_backed' : 'summary', requested: url.searchParams.get('post_context')?.split(',') ?? [], counts: { group: 2 }, classification: 'source_fields_categories_and_subtypes', description: 'Retained source fields, categories, and subtypes classify context.', absence_is_not_zero_history: false }, items: [activityItem], page: { total: 2, limit: 75, next_cursor: null, count_is_unknown: false } })
    }
    if (url.pathname === '/api/v1/timeline/on-this-day') {
      const second = url.searchParams.has('cursor')
      return Response.json({ month: 7, day: 20, timezone: 'America/New_York', items: [{ ...activityItem, id: second ? 'activity:activity-two' : 'activity:activity-one', entity_type: 'activity', entity_id: second ? 'activity-two' : 'activity-one', local_date: second ? '2022-07-20' : '2023-07-20', year: second ? 2022 : 2023 }], limit: 100, page: { total: 2, limit: 100, next_cursor: second ? null : 'on-this-day-page-two', count_is_unknown: false } })
    }
    if (url.pathname === '/api/v1/timeline') {
      return Response.json({ coverage: { status: 'source_backed', absence_is_not_zero_history: false }, items: [{ ...activityItem, id: 'activity:activity-one', entity_type: 'activity', entity_id: 'activity-one' }], page: { total: 1, limit: 75, next_cursor: null, count_is_unknown: false } })
    }
    if (url.pathname === '/api/v1/events') {
      if (url.searchParams.has('around')) return Response.json({ coverage: { ...unavailableCoverage, status: 'available', raw_count: 1, logical_count: 1, absence_is_not_zero_history: false }, items: [{ id: 'event-one', title: 'Hosted fixture', description: 'A retained event', timestamp_ms: 1_700_000_000_000, unknown_date: false, status: 'hosted', hosted: true, source: activityItem.source, jump: { href: '/events?around=event-one#event-event-one', fragment: 'event-event-one' } }], summary: { by_status: { hosted: 1 }, hosted: 1, attended: 0 }, page: { total: 1, limit: 75, next_cursor: null, count_is_unknown: false }, around: { requested: 'event-one', event_id: 'event-one', resolved: true } })
      return Response.json({ coverage: unavailableCoverage, items: [], summary: { by_status: null, hosted: null, attended: null }, page: { total: null, limit: 75, next_cursor: null, count_is_unknown: true } })
    }
    if (url.pathname === '/api/v1/people') {
      const view = url.searchParams.get('view')
      if (view !== 'contacts') {
        if (url.searchParams.has('around')) return Response.json({ coverage: { ...unavailableCoverage, key: view, label: 'Friends', status: 'available', raw_count: 1, logical_count: 1, absence_is_not_zero_history: false }, authority: 'source_records', contacts_are_not_included: true, items: [{ id: 'friend-one', name: 'Former Friend', status: 'removed', friendship_status: 'removed', timestamp_ms: 1_700_000_000_000, source: activityItem.source, jump: { href: '/people?view=friends&around=friend-one#relationship-friend-one', fragment: 'relationship-friend-one' } }], page: { total: 1, limit: 100, next_cursor: null, count_is_unknown: false }, around: { requested: 'friend-one', relationship_id: 'friend-one', resolved: true } })
        return Response.json({ coverage: { ...unavailableCoverage, key: view, label: 'Friends', description: 'Accepted-friend history is unavailable in this export.' }, authority: 'source_records_required', contacts_are_not_included: true, items: [], page: { total: null, limit: 100, next_cursor: null, count_is_unknown: true } })
      }
      return Response.json({
        coverage: { key: 'messenger_contacts', label: 'Messenger contacts', status: 'derived', raw_count: 1, logical_count: 1, description: 'Contacts are participant labels and not authoritative Facebook friends.', absence_is_not_zero_history: false },
        authority: 'messenger_contact_only',
        messenger_contact_matches_are_provisional: true,
        items: [{
          id: 'person-alex', name: 'Alex Example', contact_basis: 'messenger_participant_label', is_friend: null, friendship_status: 'unknown',
          counts: { messages: 120, photos: 2, videos: 1, audio: 1 }, direct_counts: { messages: 120, photos: 2, videos: 1, audio: 1 }, shared_group_counts: { messages: 20, photos: 1, videos: 0, audio: 0 }, combined_thread_counts: { messages: 140, photos: 3, videos: 1, audio: 1 },
          conversation_count: 2, direct_conversation_count: 1, group_conversation_count: 1, first_activity_ms: 1_600_000_000_000, last_activity_ms: 1_700_000_000_000,
          conversations: [{ id: 'conv-alex', title: 'Alex Example', type: 'direct', message_count: 120, first_activity_ms: 1_600_000_000_000, last_activity_ms: 1_700_000_000_000, href: '/conversations/conv-alex' }],
          messenger_contact_match: { contact_id: 'person-alex', match_basis: 'exact_normalized_display_label', authoritative_identity: false },
          count_semantics: 'Shared group totals are not attributed to this person.',
        }],
        page: { total: 1, limit: 100, next_cursor: null, count_is_unknown: false },
      })
    }
    if (url.pathname === '/api/v1/support') {
      const secondPage = url.searchParams.get('cursor') === 'support-page-two'
      return Response.json({
        coverage: { key: 'support_messages', label: 'Support messages', status: 'available', raw_count: 14, logical_count: 12, description: 'Exported support-message conversations are available.', source_paths: {}, absence_is_not_zero_history: false },
        items: secondPage
          ? [{ id: 'support-two', title: 'Billing appeal case', message_count: 2, first_activity_ms: 1_500_000_000_000, last_activity_ms: 1_550_000_000_000, href: '/conversations/support-two' }]
          : [{ id: 'support-one', title: 'Account access case', message_count: 12, first_activity_ms: 1_600_000_000_000, last_activity_ms: 1_700_000_000_000, href: '/conversations/support-one' }],
        page: { total: 2, limit: 50, next_cursor: secondPage ? null : 'support-page-two', count_is_unknown: false },
      })
    }
    const nlp = { sentiment: { status: 'unavailable', reason: 'Not generated.' }, emotion: { status: 'unavailable', reason: 'Not generated.' }, toxicity: { status: 'unavailable', reason: 'Not generated.', relationship_ranking_enabled: false }, conflict: { status: 'unavailable', reason: 'Not calibrated.', relationship_ranking_enabled: false } }
    if (url.pathname === '/api/v1/analytics/overview') return Response.json({ coverage: { ...partialCoverage, key: 'messages', label: 'Messages', status: 'available' }, timezone: url.searchParams.get('timezone') ?? 'America/New_York', messages: { total: 240, sent: 100, received: 140, first_timestamp_ms: 1, last_timestamp_ms: 2, calendar_days: 60, average_per_calendar_day: 4, active_conversations: 4 }, interactions: { message_reactions: 24, reaction_types: { heart: 1 }, calls: 3, missed_calls: 1, call_duration_seconds: 360, audio_attachments: 4 }, attachments: { occurrences: 18, occurrence_bytes: 1_000_000 }, activity_records: 12, link_domains: [{ domain: 'example.com', occurrences: 6 }], derived_nlp: nlp, provenance: 'SQLite rows' })
    if (url.pathname === '/api/v1/analytics/messages') {
      const secondPage = url.searchParams.get('cursor') === 'messages-page-two'
      return Response.json({
        coverage: partialCoverage,
        bucket: url.searchParams.get('bucket') ?? 'month',
        timezone: url.searchParams.get('timezone'),
        series: [{ period: secondPage ? '2023-02' : '2023-01', messages: 30, sent: 10, received: 20, active_conversations: 3 }],
        page: { total: 2, limit: 120, returned: 1, next_cursor: secondPage ? null : 'messages-page-two' },
        summary: { messages: 60, observed_periods: 2, average_per_observed_period: 30 },
        derived_nlp: nlp,
      })
    }
    if (url.pathname === '/api/v1/analytics/activity-patterns') return Response.json({ timezone: url.searchParams.get('timezone') ?? 'America/New_York', heatmap: [{ date: '2022-12-31', messages: 8, activities: 1, total: 9 }, { date: '2023-01-01', messages: 30, activities: 2, total: 32 }], page: { total: 2, limit: 120, returned: 2, next_cursor: null }, hours: Array.from({ length: 24 }, (_, hour) => ({ hour, messages: hour, sent: 0, received: hour, activities: 0 })), weekdays: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'].map((label, weekday) => ({ weekday, label, messages: weekday, activities: 0 })), streaks: { active_days: 2, longest_days: 2, latest_days: 2, first_active_date: '2022-12-31', last_active_date: '2023-01-01' }, peak_days: [{ date: '2023-01-01', messages: 30, activities: 2, total: 32 }], summary: { messages_in_scope: 38, dated_activity_records_in_scope: 3, active_activity_records_archive_wide: 8, unknown_date_activity_records: 5, unknown_date_scope: 'archive_wide', unknown_date_included_in_time_buckets: false, date_filter_active: false }, range_provenance: 'Timestamp bounds are inclusive at millisecond precision. Unknown-date Facebook activity is reported archive-wide but cannot be tested against a selected range and is never assigned to a time bucket.', derived_nlp: nlp })
    if (url.pathname === '/api/v1/analytics/interaction-patterns') return Response.json({ timezone: 'America/New_York', bucket: url.searchParams.get('bucket') ?? 'month', responses: { owner_replied: { count: 10, mean_ms: 1000, median_ms: 900, p90_ms: 2000, buckets: { under_1_minute: 10, '1_to_5_minutes': 0, '5_to_30_minutes': 0, '30_minutes_to_2_hours': 0, '2_to_24_hours': 0, '1_to_7_days': 0, over_7_days: 0 } }, other_replied: { count: 12, mean_ms: 1200, median_ms: 1000, p90_ms: 2200, buckets: { under_1_minute: 12, '1_to_5_minutes': 0, '5_to_30_minutes': 0, '30_minutes_to_2_hours': 0, '2_to_24_hours': 0, '1_to_7_days': 0, over_7_days: 0 } }, definition: 'Adjacent direction changes.' }, initiations: { owner: 4, other: 6, definition: 'First retained message direction.' }, active_participants: [{ period: '2023-01', participants: 8, conversations: 3 }], page: { total: 1, limit: 120, returned: 1, next_cursor: null }, ranking_policy: 'Timing is not used to rank people.' })
    if (url.pathname === '/api/v1/analytics/interaction-series') return Response.json({ bucket: url.searchParams.get('bucket') ?? 'month', timezone: url.searchParams.get('timezone') ?? 'America/New_York', series: [{ period: '2023-01', calls: 3, call_duration_seconds: 360, reactions: 24, reaction_explicit_timestamp: 20, reaction_message_timestamp_fallback: 4, audio_attachments: 4, link_occurrences: 6 }], page: { total: 1, limit: 120, returned: 1, next_cursor: null }, domains: [{ domain: 'example.com', occurrences: 6, periods: [{ period: '2023-01', occurrences: 6 }] }], summary: { calls: 3, call_duration_seconds: 360, reactions: 24, reaction_explicit_timestamp: 20, reaction_message_timestamp_fallback: 4, audio_attachments: 4, audio_unknown_date: 1, link_occurrences: 6, domain_count: 1 }, provenance: { calls: 'Calls use the source message timestamp.', reactions: 'Reactions use their own timestamp, falling back to the retained source-message timestamp.', audio: 'Dated audio uses the attachment occurrence timestamp; unknown-date audio is excluded from the series.', links: 'Links use the source message timestamp where the URL appeared.' } })
    if (url.pathname === '/api/v1/analytics/nlp-trends') {
      const emptyRange = url.searchParams.has('from_ms') || url.searchParams.has('to_ms')
      return Response.json({
      timezone: 'America/New_York', bucket: url.searchParams.get('bucket') ?? 'month', direction: url.searchParams.get('direction') ?? 'all', source_sample_count: emptyRange ? 0 : 12, page_sample_count: emptyRange ? 0 : 12,
      models: { sentiment: { model_name: 'fixture/sentiment', model_revision: 'revision-one', configuration_sha256: 'a'.repeat(64) } },
      series: {
        sentiment: emptyRange ? [] : [{ period: '2023-01', direction: 'all', sample_count: 12, mean_confidence: .82, score: .35, score_definition: 'Mean positive minus negative probability.', dominant_label: 'positive', predicted_label_counts: { positive: 9, neutral: 2, negative: 1 }, mean_probabilities: { positive: .6, neutral: .15, negative: .25 } }],
        emotion: emptyRange ? [] : [{ period: '2023-01', direction: 'all', sample_count: 12, mean_confidence: .71, score: null, score_definition: 'Mean label vector.', dominant_label: 'joy', predicted_label_counts: { joy: 7 }, mean_probabilities: { joy: .58, neutral: .22 } }],
        toxicity: emptyRange ? [] : [{ period: '2023-01', direction: 'all', sample_count: 12, mean_confidence: .78, score: .12, score_definition: 'Highest dimension.', dominant_label: 'non-toxic', predicted_label_counts: { 'non-toxic': 11 }, mean_probabilities: { toxic: .12, 'non-toxic': .88 } }],
      },
      hour_of_day: { sentiment: emptyRange ? [] : [{ hour: '09', direction: 'all', sample_count: 4, mean_confidence: .8, score: .2, score_definition: 'Mean positive minus negative probability.', dominant_label: 'positive', predicted_label_counts: { positive: 3 }, mean_probabilities: { positive: .55, negative: .35 } }], emotion: [], toxicity: [] },
      hour_of_day_scope: 'current_page', page: { total: null, limit: 24, returned: emptyRange ? 0 : 1, next_cursor: null, count_is_unknown: true },
      availability: nlp, rollup_provenance: { source_resolution: 'UTC hour', range_filter_semantics: 'whole UTC hours overlapping the requested interval', local_bucket_label: 'local time at the UTC rollup hour start', partial_boundary_hours_included: url.searchParams.has('from_ms') || url.searchParams.has('to_ms'), non_whole_hour_timezone_offset_observed: false }, warnings: ['No people are ranked.'],
    })
    }
    if (url.pathname === '/api/v1/analytics/conflict-rate') return Response.json({
      available: true, status: 'ready', reason: null, timezone: 'America/New_York', bucket: url.searchParams.get('bucket') ?? 'month',
      series: [{ period: '2023-01', sample_count: 120, direct_conflict_window_count: 18, direct_conflict_window_rate: .15, low_sample_warning: false }],
      page: { total: null, limit: 24, returned: 1, next_cursor: null, count_is_unknown: true },
      summary: { sample_count: 120, direct_conflict_window_count: 18, direct_conflict_window_rate: .15 },
      warnings: ['Model estimate; no people, conversations, or relationships are ranked.'],
      provenance: { run_id: 'fixture-conflict-run', model_revision: 'fixture-revision', range_filter_semantics: 'whole UTC hours overlapping the requested interval', local_bucket_label: 'local time at the UTC rollup hour start', partial_boundary_hours_included: url.searchParams.has('from_ms') || url.searchParams.has('to_ms') }, relationship_ranking_enabled: false, relationship_dimensions: [],
    })
    if (url.pathname === '/api/v1/analytics/network') return Response.json({ nodes: [{ id: 'conversation:one', kind: 'conversation', label: 'Alex Example', value: 100, href: '/conversations/one' }, { id: 'participant:one', kind: 'participant_label', label: 'Alex Example', value: 100, is_owner: false }], edges: [{ id: 'edge-one', source: 'conversation:one', target: 'participant:one', value: 100 }], conversation_count: 1, participant_label_count: 1, scope: 'Top retained conversation.', caveat: 'Participant labels are provisional.' })
    if (url.pathname === '/api/v1/analytics/topics') return Response.json({ status: 'unavailable', ready: false, latest_job: null, semantic_index: null, result: null, reason: 'Build the semantic text index before generating topic clusters.' })
    if (url.pathname === '/api/v1/analytics/attachments') return Response.json({ bucket: url.searchParams.get('bucket') ?? 'month', timezone: url.searchParams.get('timezone') ?? 'America/New_York', types: [{ type: 'photo', occurrences: 10, unique_files: 8, occurrence_bytes: 1000 }], availability: [{ availability: 'resolved', occurrences: 10, bytes: 1000, rate: 1 }], trends: [{ period: '2023-01', occurrences: 10, unique_files: 8, occurrence_bytes: 1000, by_type: { photo: { occurrences: 10, unique_files: 8, occurrence_bytes: 1000 } } }], page: { total: 1, limit: 120, returned: 1, next_cursor: null }, summary: { occurrences: 10, dated_occurrences: 10, unknown_date_occurrences: 0, unknown_date_scope: 'archive_wide', unknown_date_included_in_totals: true, missing_occurrences: 0, missing_rate: 0, unlinked_occurrences: 0, unlinked_rate: 0 }, count_semantics: 'Occurrence bytes count each send.', provenance: 'Fixture attachment timestamps.' })
    if (url.pathname === '/api/v1/analytics/conversations') return Response.json({ conversation_count: 4, by_type: { direct: 2, group: 1, system: 1 }, longevity_days: { mean: 10, median: 2, maximum: 100, histogram: { same_day: 2 } }, active_days: { mean: 20, maximum: 300 }, ranking_policy: 'No people or relationships are ranked.', derived_nlp: nlp })
    if (url.pathname === '/api/v1/analytics/storage') return Response.json({ source_occurrences: { files: 128, bytes: 125_829_120 }, unique_blobs: { files: 96, bytes: 94_371_840 }, deduplication: { redundant_occurrences: 32, duplicate_bytes: 31_457_280 }, by_media_class: [{ class: 'image', files: 10, bytes: 1000 }] })
    if (url.pathname === '/api/v1/analytics/coverage') return Response.json({ summary: { available: 1, unavailable: 1 }, items: [partialCoverage, unavailableCoverage] })
    return new Response(JSON.stringify({ detail: 'Not found' }), { status: 404 })
  })
  vi.stubGlobal('fetch', mock)
  return mock
}

async function renderAt(path: string) {
  const history = createMemoryHistory({ initialEntries: [path] })
  const router = createRouter({ routeTree, history })
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } })
  await router.load()
  return render(<QueryClientProvider client={queryClient}><RouterProvider router={router} /></QueryClientProvider>)
}

describe('Stage 4 archive views', () => {
  beforeEach(() => stageFourFetchMock())

  it('keeps Messenger contacts separate from authoritative friend history', async () => {
    const user = userEvent.setup()
    await renderAt('/people')
    expect(await screen.findByRole('heading', { name: 'People' })).toBeVisible()
    expect(await screen.findByRole('heading', { name: 'Alex Example' })).toBeVisible()
    expect(screen.getByText('Messenger participant label')).toBeVisible()
    expect(screen.getByText(/not attributed to this person/i)).toBeVisible()
    expect(screen.getAllByText(/not Facebook identity matches/i)).toHaveLength(1)
    await user.click(screen.getByRole('button', { name: 'Friends' }))
    expect((await screen.findAllByText('Accepted-friend history is unavailable in this export.')).length).toBeGreaterThan(0)
    expect(screen.getByText(/cannot establish that your history was empty/i)).toBeVisible()
  })

  it('renders unavailable event history as unknown instead of zero', async () => {
    await renderAt('/events')
    expect(await screen.findByRole('heading', { name: 'Events' })).toBeVisible()
    expect((await screen.findAllByText('Event history is unavailable in this export.')).length).toBeGreaterThan(0)
    expect(screen.getByText('Event history unavailable')).toBeVisible()
    expect(screen.getAllByText('—').length).toBeGreaterThanOrEqual(3)
  })

  it('resolves event and relationship deep links into the rendered page', async () => {
    const fetchMock = stageFourFetchMock()
    await renderAt('/events?around=event-one#event-event-one')
    expect(await screen.findByRole('heading', { name: 'Hosted fixture' })).toBeVisible()
    expect(fetchMock.mock.calls.some(([input]) => String(input).includes('/events?') && String(input).includes('around=event-one'))).toBe(true)

    await renderAt('/people?view=friends&around=friend-one#relationship-friend-one')
    expect(await screen.findByRole('heading', { name: 'Former Friend' })).toBeVisible()
    expect(screen.getByRole('button', { name: 'Friends' })).toHaveAttribute('aria-pressed', 'true')
    expect(fetchMock.mock.calls.some(([input]) => String(input).includes('/people?') && String(input).includes('around=friend-one'))).toBe(true)
  })

  it('keeps mobile people pages compact without dropping a deep-link target', async () => {
    const matchMedia = vi.spyOn(window, 'matchMedia').mockImplementation((query) => ({
      matches: query === '(max-width: 560px)',
      media: query,
      onchange: null,
      addListener: () => undefined,
      removeListener: () => undefined,
      addEventListener: () => undefined,
      removeEventListener: () => undefined,
      dispatchEvent: () => false,
    }))
    const fetchMock = stageFourFetchMock()
    await renderAt('/people?around=person-alex#relationship-person-alex')

    expect(await screen.findByRole('heading', { name: 'Alex Example' })).toBeVisible()
    const peopleViews = screen.getByRole('navigation', { name: 'People view' })
    expect(peopleViews).toHaveClass('horizontal-tabs__scroller')
    expect(peopleViews.closest('.horizontal-tabs')).not.toBeNull()
    expect(screen.getByText('Swipe to see more relationships →')).toHaveAttribute(
      'aria-hidden',
      'true',
    )
    await waitFor(() => expect(fetchMock.mock.calls.some(([input]) => {
      const url = new URL(String(input), 'http://localhost')
      return url.pathname === '/api/v1/people'
        && url.searchParams.get('limit') === '8'
        && url.searchParams.get('around') === 'person-alex'
    })).toBe(true))
    expect(document.getElementById('relationship-person-alex')).not.toBeNull()
    matchMedia.mockRestore()
  })

  it('browses actual support-message threads separately from support activity', async () => {
    const user = userEvent.setup()
    await renderAt('/support')
    expect(await screen.findByRole('heading', { name: 'Support conversations' })).toBeVisible()
    expect(screen.getByRole('heading', { name: 'Account access case' })).toBeVisible()
    expect(screen.getByText('12 messages')).toBeVisible()
    expect(screen.getByRole('link', { name: 'Open conversation' })).toHaveAttribute('href', '/conversations/support-one')
    const pages = screen.getByRole('navigation', { name: 'Support conversations pages' })
    await user.click(within(pages).getByRole('button', { name: 'Next page' }))
    expect(await screen.findByRole('heading', { name: 'Billing appeal case' })).toBeVisible()
    expect(screen.getByRole('link', { name: 'Open conversation' })).toHaveAttribute('href', '/conversations/support-two')
    await user.click(within(pages).getByRole('button', { name: 'Previous page' }))
    expect(await screen.findByRole('heading', { name: 'Account access case' })).toBeVisible()
  })

  it('filters activity and preserves source-backed jump links', async () => {
    const fetchMock = stageFourFetchMock()
    const user = userEvent.setup()
    await renderAt('/activity')
    expect(await screen.findByText('Fixture album photo')).toBeVisible()
    expect(screen.getByText('your_facebook_activity/posts/album.json')).toBeVisible()
    const sourceJump = screen.getByRole('link', { name: 'Open source record' })
    expect(sourceJump).toHaveAttribute('href', expect.stringContaining('/timeline?around='))
    expect(sourceJump.closest('td')).toHaveClass('activity-table__jump')
    const activityViewport = screen.getByRole('region', { name: 'Scrollable virtualized activity results' })
    expect(activityViewport).toHaveAttribute('aria-describedby', 'activity-results-scroll-hint')
    expect(document.getElementById('activity-results-scroll-hint')).toHaveTextContent('Scroll inside the framed pane to browse all loaded activity records.')
    expect(screen.getByText('Source subtype classification')).toBeVisible()
    await user.click(screen.getByRole('button', { name: 'Posts' }))
    await waitFor(() => expect(fetchMock.mock.calls.some(([input]) => String(input).includes('/activity/views/posts'))).toBe(true))
    await user.selectOptions(screen.getByLabelText('Post/comment context'), 'group')
    await user.click(screen.getByRole('button', { name: 'Apply filters' }))
    await waitFor(() => expect(fetchMock.mock.calls.some(([input]) => String(input).includes('post_context=group'))).toBe(true))
    expect(await screen.findByText('Classification uses retained source fields, categories, and subtypes only; missing evidence is never treated as zero history.')).toBeVisible()
  })

  it('switches between the combined timeline and On This Day', async () => {
    const user = userEvent.setup()
    await renderAt('/timeline')
    expect(await screen.findByText('Archive chronology')).toBeVisible()
    const timelineViewport = screen.getByRole('feed', { name: 'Scrollable virtualized timeline' })
    expect(timelineViewport).toHaveAttribute('aria-describedby', 'timeline-results-scroll-hint')
    expect(document.getElementById('timeline-results-scroll-hint')).toHaveTextContent('Scroll inside the framed pane to browse all loaded timeline entries.')
    expect(screen.getByText('A retained caption')).toBeVisible()
    await user.click(screen.getByRole('button', { name: 'On this day' }))
    expect(await screen.findByText(/2 matching records; 1 shown from 1 years on this page/)).toBeVisible()
    await user.click(screen.getByRole('button', { name: 'Next page' }))
    expect(await screen.findByText('Page 2')).toBeVisible()
    await waitFor(() => expect(document.getElementById('activity-activity-two')).not.toBeNull())
    await user.click(screen.getByRole('button', { name: 'Previous page' }))
    expect(await screen.findByText('Page 1')).toBeVisible()
    await waitFor(() => expect(document.getElementById('activity-activity-one')).not.toBeNull())
  })

  it('passes an activity seek target to the timeline and renders it first', async () => {
    const fetchMock = stageFourFetchMock()
    await renderAt('/timeline?around=activity:activity-one#activity-activity-one')
    expect(await screen.findByText('Archive chronology')).toBeVisible()
    expect(fetchMock.mock.calls.some(([input]) => String(input).includes('/timeline?') && String(input).includes('around=activity%3Aactivity-one'))).toBe(true)
    expect(document.getElementById('activity-activity-one')).not.toBeNull()
  })

  it('shows source-backed charts and marks derived NLP as unavailable', async () => {
    const user = userEvent.setup()
    await renderAt('/analytics')
    expect(await screen.findByRole('heading', { name: 'Archive analytics' })).toBeVisible()
    expect(await screen.findByText('240')).toBeVisible()
    expect(screen.getByRole('img', { name: 'Sent, received, and active conversations by month' })).toBeVisible()
    expect(screen.getByRole('img', { name: 'Calendar heatmap of retained archive activity in 2023' })).toBeVisible()
    const calendarYear = screen.getByRole('combobox', { name: 'Calendar year' })
    expect(calendarYear).toHaveValue('2023')
    await user.selectOptions(calendarYear, '2022')
    expect(screen.getByRole('img', { name: 'Calendar heatmap of retained archive activity in 2022' })).toBeVisible()
    expect(screen.getByRole('img', { name: 'Response time distributions by speaker direction' })).toBeVisible()
    expect(screen.getByRole('img', { name: 'Bipartite graph of conversations and participant labels' })).toBeVisible()
    expect(screen.getByRole('img', { name: 'Unique archive bytes by media class' })).toHaveAttribute('data-legend-visibility', 'hidden')
    expect(screen.getByRole('img', { name: 'Unique archive bytes by media class' })).toHaveAttribute('data-slice-labels', 'hidden')
    const storageKey = screen.getByRole('list', { name: 'Storage composition key' })
    expect(within(storageKey).getAllByRole('listitem')).toHaveLength(1)
    expect(storageKey).toHaveTextContent('Image')
    expect(screen.getByRole('img', { name: 'Conversation count by type' })).toHaveAttribute('data-legend-visibility', 'hidden')
    expect(screen.getByRole('img', { name: 'Conversation count by type' })).toHaveAttribute('data-slice-labels', 'hidden')
    const conversationKey = screen.getByRole('list', { name: 'Conversation types key' })
    expect(within(conversationKey).getAllByRole('listitem')).toHaveLength(3)
    expect(conversationKey).toHaveTextContent('Direct')
    expect(conversationKey).toHaveTextContent('Group')
    expect(conversationKey).toHaveTextContent('System')
    expect(screen.getByRole('heading', { name: 'Topic clusters' })).toBeVisible()
    expect(screen.getByRole('button', { name: 'Build topics' })).toBeDisabled()
    expect(screen.getByRole('img', { name: 'Model-estimated sentiment by month and speaker direction' })).toBeVisible()
    expect(screen.getByRole('img', { name: 'Model-estimated direct-conflict-window rate by month' })).toBeVisible()
    expect(screen.getAllByText('15.0%')).toHaveLength(2)
    expect(screen.getByLabelText('Time bucket')).toHaveValue('month')
    await user.click(screen.getByText('Inspect sentiment sample sizes and confidence'))
    expect(screen.getByText('82.0%')).toBeVisible()
    const languageSignals = screen.getByRole('heading', { name: 'Derived language signals' }).closest('section')
    expect(languageSignals).not.toBeNull()
    expect(within(languageSignals!).getAllByText('Unavailable')).toHaveLength(4)
    expect(screen.getAllByText('Relationship ranking is disabled.')).toHaveLength(2)
  })

  it('keeps mobile analytics compact and expands every drill-down on demand', async () => {
    const matchMedia = vi.spyOn(window, 'matchMedia').mockImplementation((query) => ({
      matches: query === '(max-width: 680px)',
      media: query,
      onchange: null,
      addListener: () => undefined,
      removeListener: () => undefined,
      addEventListener: () => undefined,
      removeEventListener: () => undefined,
      dispatchEvent: () => false,
    }))
    const user = userEvent.setup()
    await renderAt('/analytics')

    expect(await screen.findByRole('heading', { name: 'Archive analytics' })).toBeVisible()
    const drilldowns = document.querySelectorAll('details.analytics-drilldown')
    expect(drilldowns).toHaveLength(7)
    expect(await screen.findByLabelText('Analytics overview')).toBeVisible()
    expect(screen.queryByLabelText('Language-model trend filters')).not.toBeInTheDocument()
    await user.click(screen.getByText('Language-model signals'))
    expect(await screen.findByLabelText('Language-model trend filters')).toBeVisible()
    expect(screen.getByRole('img', { name: 'Model-estimated sentiment by month and speaker direction' })).toBeVisible()
    matchMedia.mockRestore()
  })

  it('navigates bounded analytics periods with the returned cursor', async () => {
    const user = userEvent.setup()
    const fetchMock = stageFourFetchMock()
    await renderAt('/analytics')
    const pager = await screen.findByRole('navigation', { name: 'Message volume periods pages' })
    expect(within(pager).getByText('Page 1 · 1 periods of 2')).toBeVisible()
    await user.click(within(pager).getByRole('button', { name: 'Next page' }))
    expect(await within(pager).findByText('Page 2 · 1 periods of 2')).toBeVisible()
    expect(fetchMock.mock.calls.some(([input]) => {
      const url = new URL(String(input), 'http://localhost')
      return url.pathname === '/api/v1/analytics/messages'
        && url.searchParams.get('cursor') === 'messages-page-two'
        && url.searchParams.get('limit') === '120'
    })).toBe(true)
    await user.click(within(pager).getByRole('button', { name: 'Previous page' }))
    expect(await within(pager).findByText('Page 1 · 1 periods of 2')).toBeVisible()
  })

  it('keeps language-model filters usable when a date range has zero annotated results', async () => {
    const user = userEvent.setup()
    await renderAt('/analytics')
    const heading = await screen.findByRole('heading', { name: 'Local language-model trends' })
    const section = heading.closest('section')
    expect(section).not.toBeNull()
    const filters = within(section!).getByLabelText('Language-model trend filters')
    const fromInput = within(filters).getByLabelText('From')
    await user.type(fromInput, '2035-01-01')
    expect(await screen.findByText('No versioned language-model annotations match these filters. Clear or widen the date and source filters to continue.')).toBeVisible()
    expect(fromInput).toHaveValue('2035-01-01')
    expect(screen.getByLabelText('Language-model trend filters')).toBeVisible()
  })
})
