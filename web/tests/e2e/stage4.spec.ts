import { expect, type Page, test } from '@playwright/test'
import AxeBuilder from '@axe-core/playwright'
import { fixtures } from '../../src/test/fixtures'

const partialCoverage = { key: 'posts', label: 'Posts and albums', status: 'partial', raw_count: 2, logical_count: 2, description: 'Only exported post-related records are shown.', source_paths: {}, absence_is_not_zero_history: false }
const unavailableCoverage = { key: 'events', label: 'Events', status: 'unavailable', raw_count: null, logical_count: null, description: 'Event history is unavailable in this export.', source_paths: {}, absence_is_not_zero_history: true }
const activityItem = { id: 'activity-one', parent_id: null, category: 'posts', subtype: 'album_photo', timestamp_ms: 1_700_000_000_000, unknown_date: false, title: 'Fixture album photo', content: 'A retained caption', actor: 'Archive User', availability: 'available', target_uri: null, attachment_count: 1, post_context: 'group', post_context_basis: 'explicit source subtype', source: { source_file_id: 1, source_record_id: 'source-record', chunk: 'package-a', relative_path: 'your_facebook_activity/posts/album.json', json_pointer: '/photos/0' }, jump: { href: '/timeline?around=activity:activity-one#activity-activity-one', fragment: 'activity-activity-one' } }
const eventItem = { id: 'event-one', title: 'Hosted fixture', description: 'A retained event', timestamp_ms: 1_700_000_000_000, unknown_date: false, status: 'hosted', hosted: true, source: activityItem.source, jump: { href: '/events?around=event-one#event-event-one', fragment: 'event-event-one' } }
const contactItem = { id: 'person-alex', name: 'Alex Example', contact_basis: 'messenger_participant_label', is_friend: null, friendship_status: 'unknown', counts: { messages: 120, photos: 2, videos: 1, audio: 1 }, direct_counts: { messages: 120, photos: 2, videos: 1, audio: 1 }, shared_group_counts: { messages: 20, photos: 1, videos: 0, audio: 0 }, combined_thread_counts: { messages: 140, photos: 3, videos: 1, audio: 1 }, conversation_count: 2, direct_conversation_count: 1, group_conversation_count: 1, first_activity_ms: 1_600_000_000_000, last_activity_ms: 1_700_000_000_000, conversations: [{ id: 'conv-alex', title: 'Alex Example', type: 'direct', message_count: 120, first_activity_ms: 1_600_000_000_000, last_activity_ms: 1_700_000_000_000, href: '/conversations/conv-alex' }, { id: 'group-alex', title: 'Fixture group', type: 'group', message_count: 20, first_activity_ms: 1_610_000_000_000, last_activity_ms: 1_690_000_000_000, href: '/conversations/group-alex' }], count_semantics: 'Counts and direct_counts describe one-to-one exchanges. Shared-group counts cover whole shared conversations and are not attributed to this person; combined_thread_counts is the explicit sum of those two scopes.' }
const nlp = { sentiment: { status: 'unavailable', reason: 'Not generated.' }, emotion: { status: 'unavailable', reason: 'Not generated.' }, toxicity: { status: 'unavailable', reason: 'Not generated.', relationship_ranking_enabled: false }, conflict: { status: 'unavailable', reason: 'Not calibrated.', relationship_ranking_enabled: false } }

function activityPage(nextCursor: string | null = null, items = [activityItem], total = items.length ? 2 : 0) {
  return { coverage: partialCoverage, coverage_components: [], post_context_coverage: { status: 'summary', requested: [], counts: { group: total }, classification: 'source_fields_categories_and_subtypes', description: 'Retained source fields, categories, and subtypes classify context.', absence_is_not_zero_history: false }, items, page: { total, limit: 75, next_cursor: nextCursor, count_is_unknown: false } }
}

function eventsPage(nextCursor: string | null = null, items = [eventItem], total = items.length ? 2 : 0) {
  const populated = items.length > 0
  return { coverage: { ...unavailableCoverage, status: 'available', raw_count: total, logical_count: total, absence_is_not_zero_history: false }, items, summary: { by_status: populated ? { hosted: 1, went: Math.max(0, total - 1) } : {}, hosted: populated ? 1 : 0, attended: populated ? Math.max(0, total - 1) : 0 }, page: { total, limit: 75, next_cursor: nextCursor, count_is_unknown: false } }
}

function contactsPage(nextCursor: string | null = null, items = [contactItem], total = items.length ? 2 : 0) {
  return { coverage: { key: 'messenger_contacts', label: 'Messenger contacts', status: 'derived', raw_count: total, logical_count: total, description: 'Contacts are participant labels and not authoritative Facebook friends.', absence_is_not_zero_history: false }, authority: 'messenger_contact_only', items, page: { total, limit: 100, next_cursor: nextCursor, count_is_unknown: false } }
}

async function mockStageFourApi(page: Page) {
  await page.route('**/api/v1/**', async (route) => {
    const url = new URL(route.request().url())
    const path = url.pathname
    if (path === '/api/v1/health') return route.fulfill({ json: fixtures.health })
    if (path === '/api/v1/settings') return route.fulfill({ json: fixtures.settings })
    if (path === '/api/v1/activity' || path.startsWith('/api/v1/activity/views/')) return route.fulfill({ json: { coverage: partialCoverage, coverage_components: [], post_context_coverage: { status: url.searchParams.has('post_context') ? 'source_backed' : 'summary', requested: url.searchParams.get('post_context')?.split(',') ?? [], counts: { group: 2 }, classification: 'source_fields_categories_and_subtypes', description: 'Retained source fields, categories, and subtypes classify context.', absence_is_not_zero_history: false }, items: [activityItem], page: { total: 2, limit: 75, next_cursor: null, count_is_unknown: false } } })
    if (path === '/api/v1/timeline/on-this-day') { const second = url.searchParams.has('cursor'); return route.fulfill({ json: { month: 7, day: 20, timezone: 'America/New_York', items: [{ ...activityItem, id: second ? 'activity:activity-two' : 'activity:activity-one', entity_type: 'activity', entity_id: second ? 'activity-two' : 'activity-one', local_date: second ? '2022-07-20' : '2023-07-20', year: second ? 2022 : 2023 }], limit: 100, page: { total: 2, limit: 100, next_cursor: second ? null : 'on-this-day-page-two', count_is_unknown: false } } }) }
    if (path === '/api/v1/timeline') return route.fulfill({ json: { coverage: { status: 'source_backed', absence_is_not_zero_history: false }, items: [{ ...activityItem, id: 'activity:activity-one', entity_type: 'activity', entity_id: 'activity-one' }], page: { total: 1, limit: 75, next_cursor: null, count_is_unknown: false } } })
    if (path === '/api/v1/events') {
      if (url.searchParams.has('around')) return route.fulfill({ json: { coverage: { ...unavailableCoverage, status: 'available', raw_count: 1, logical_count: 1, absence_is_not_zero_history: false }, items: [{ id: 'event-one', title: 'Hosted fixture', description: 'A retained event', timestamp_ms: 1_700_000_000_000, unknown_date: false, status: 'hosted', hosted: true, source: activityItem.source, jump: { href: '/events?around=event-one#event-event-one', fragment: 'event-event-one' } }], summary: { by_status: { hosted: 1 }, hosted: 1, attended: 0 }, page: { total: 1, limit: 75, next_cursor: null, count_is_unknown: false }, around: { requested: 'event-one', event_id: 'event-one', resolved: true } } })
      return route.fulfill({ json: { coverage: unavailableCoverage, items: [], summary: { by_status: null, hosted: null, attended: null }, page: { total: null, limit: 75, next_cursor: null, count_is_unknown: true } } })
    }
    if (path === '/api/v1/people') {
      const view = url.searchParams.get('view')
      if (view !== 'contacts') {
        if (url.searchParams.has('around')) return route.fulfill({ json: { coverage: { ...unavailableCoverage, key: view, label: 'Friends', status: 'available', raw_count: 1, logical_count: 1, absence_is_not_zero_history: false }, authority: 'source_records', contacts_are_not_included: true, items: [{ id: 'friend-one', name: 'Former Friend', status: 'removed', friendship_status: 'removed', timestamp_ms: 1_700_000_000_000, source: activityItem.source, jump: { href: '/people?view=friends&around=friend-one#relationship-friend-one', fragment: 'relationship-friend-one' } }], page: { total: 1, limit: 100, next_cursor: null, count_is_unknown: false }, around: { requested: 'friend-one', relationship_id: 'friend-one', resolved: true } } })
        return route.fulfill({ json: { coverage: { ...unavailableCoverage, key: view, label: 'Friends', description: 'Accepted-friend history is unavailable in this export.' }, authority: 'source_records_required', contacts_are_not_included: true, items: [], page: { total: null, limit: 100, next_cursor: null, count_is_unknown: true } } })
      }
      return route.fulfill({ json: { coverage: { key: 'messenger_contacts', label: 'Messenger contacts', status: 'derived', raw_count: 1, logical_count: 1, description: 'Contacts are participant labels and not authoritative Facebook friends.', absence_is_not_zero_history: false }, authority: 'messenger_contact_only', items: [contactItem], page: { total: 1, limit: 100, next_cursor: null, count_is_unknown: false } } })
    }
    if (path === '/api/v1/support') {
      const secondPage = url.searchParams.get('cursor') === 'support-page-two'
      return route.fulfill({ json: {
        coverage: { key: 'support_messages', label: 'Support messages', status: 'available', raw_count: 14, logical_count: 12, description: 'Exported support-message conversations are available.', source_paths: {}, absence_is_not_zero_history: false },
        items: secondPage
          ? [{ id: 'support-two', title: 'Billing appeal case', message_count: 2, first_activity_ms: 1_500_000_000_000, last_activity_ms: 1_550_000_000_000, href: '/conversations/support-two' }]
          : [{ id: 'support-one', title: 'Account access case', message_count: 12, first_activity_ms: 1_600_000_000_000, last_activity_ms: 1_700_000_000_000, href: '/conversations/support-one' }],
        page: { total: 2, limit: 50, next_cursor: secondPage ? null : 'support-page-two', count_is_unknown: false },
      } })
    }
    if (path === '/api/v1/analytics/overview') return route.fulfill({ json: { coverage: { ...partialCoverage, key: 'messages', label: 'Messages', status: 'available' }, timezone: url.searchParams.get('timezone') ?? 'America/New_York', messages: { total: 240, sent: 100, received: 140, first_timestamp_ms: 1, last_timestamp_ms: 2, calendar_days: 60, average_per_calendar_day: 4, active_conversations: 4 }, interactions: { message_reactions: 24, reaction_types: { heart: 1 }, calls: 3, missed_calls: 1, call_duration_seconds: 360, audio_attachments: 4 }, attachments: { occurrences: 18, occurrence_bytes: 1_000_000 }, activity_records: 12, link_domains: [{ domain: 'example.com', occurrences: 6 }], derived_nlp: nlp, provenance: 'SQLite rows' } })
    if (path === '/api/v1/analytics/messages') return route.fulfill({ json: { coverage: partialCoverage, bucket: url.searchParams.get('bucket') ?? 'month', timezone: url.searchParams.get('timezone'), series: [{ period: '2023-01', messages: 30, sent: 10, received: 20, active_conversations: 3 }], page: { total: 1, limit: 120, returned: 1, next_cursor: null }, summary: { messages: 30, observed_periods: 1, average_per_observed_period: 30 }, derived_nlp: nlp } })
    if (path === '/api/v1/analytics/activity-patterns') return route.fulfill({ json: { timezone: url.searchParams.get('timezone') ?? 'America/New_York', heatmap: [{ date: '2023-01-01', messages: 30, activities: 2, total: 32 }], page: { total: 1, limit: 120, returned: 1, next_cursor: null }, hours: Array.from({ length: 24 }, (_, hour) => ({ hour, messages: hour, sent: 0, received: hour, activities: 0 })), weekdays: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'].map((label, weekday) => ({ weekday, label, messages: weekday, activities: 0 })), streaks: { active_days: 1, longest_days: 1, latest_days: 1, first_active_date: '2023-01-01', last_active_date: '2023-01-01' }, peak_days: [{ date: '2023-01-01', messages: 30, activities: 2, total: 32 }], summary: { messages_in_scope: 30, dated_activity_records_in_scope: 2, active_activity_records_archive_wide: 7, unknown_date_activity_records: 5, unknown_date_scope: 'archive_wide', unknown_date_included_in_time_buckets: false, date_filter_active: false }, range_provenance: 'Timestamp bounds are inclusive at millisecond precision. Unknown-date Facebook activity is reported archive-wide but cannot be tested against a selected range and is never assigned to a time bucket.', derived_nlp: nlp } })
    if (path === '/api/v1/analytics/interaction-patterns') return route.fulfill({ json: { timezone: 'America/New_York', bucket: url.searchParams.get('bucket') ?? 'month', responses: { owner_replied: { count: 10, mean_ms: 1000, median_ms: 900, p90_ms: 2000, buckets: { under_1_minute: 10, '1_to_5_minutes': 0, '5_to_30_minutes': 0, '30_minutes_to_2_hours': 0, '2_to_24_hours': 0, '1_to_7_days': 0, over_7_days: 0 } }, other_replied: { count: 12, mean_ms: 1200, median_ms: 1000, p90_ms: 2200, buckets: { under_1_minute: 12, '1_to_5_minutes': 0, '5_to_30_minutes': 0, '30_minutes_to_2_hours': 0, '2_to_24_hours': 0, '1_to_7_days': 0, over_7_days: 0 } }, definition: 'Adjacent direction changes.' }, initiations: { owner: 4, other: 6, definition: 'First retained message direction.' }, active_participants: [{ period: '2023-01', participants: 8, conversations: 3 }], page: { total: 1, limit: 120, returned: 1, next_cursor: null }, ranking_policy: 'Timing is not used to rank people.' } })
    if (path === '/api/v1/analytics/interaction-series') return route.fulfill({ json: { bucket: url.searchParams.get('bucket') ?? 'month', timezone: url.searchParams.get('timezone') ?? 'America/New_York', series: [{ period: '2023-01', calls: 3, call_duration_seconds: 360, reactions: 24, reaction_explicit_timestamp: 20, reaction_message_timestamp_fallback: 4, audio_attachments: 4, link_occurrences: 6 }], page: { total: 1, limit: 120, returned: 1, next_cursor: null }, domains: [{ domain: 'example.com', occurrences: 6, periods: [{ period: '2023-01', occurrences: 6 }] }], summary: { calls: 3, call_duration_seconds: 360, reactions: 24, reaction_explicit_timestamp: 20, reaction_message_timestamp_fallback: 4, audio_attachments: 4, audio_unknown_date: 1, link_occurrences: 6, domain_count: 1 }, provenance: { calls: 'Calls use the source message timestamp.', reactions: 'Reactions use their own timestamp, falling back to the retained source-message timestamp.', audio: 'Dated audio uses the attachment occurrence timestamp; unknown-date audio is excluded from the series.', links: 'Links use the source message timestamp where the URL appeared.' } } })
    if (path === '/api/v1/analytics/nlp-trends') return route.fulfill({ json: { timezone: url.searchParams.get('timezone') ?? 'America/New_York', bucket: url.searchParams.get('bucket') ?? 'month', direction: url.searchParams.get('direction') ?? 'all', source_sample_count: 12, page_sample_count: 12, models: { sentiment: { model_name: 'fixture/sentiment', model_revision: 'revision-one', configuration_sha256: 'a'.repeat(64) } }, series: { sentiment: [{ period: '2023-01', direction: url.searchParams.get('direction') ?? 'all', sample_count: 12, mean_confidence: .82, score: .35, score_definition: 'Mean positive minus negative probability.', dominant_label: 'positive', predicted_label_counts: { positive: 9 }, mean_probabilities: { positive: .6, negative: .25 } }], emotion: [{ period: '2023-01', direction: url.searchParams.get('direction') ?? 'all', sample_count: 12, mean_confidence: .71, score: null, score_definition: 'Mean vector.', dominant_label: 'joy', predicted_label_counts: { joy: 7 }, mean_probabilities: { joy: .58 } }], toxicity: [{ period: '2023-01', direction: url.searchParams.get('direction') ?? 'all', sample_count: 12, mean_confidence: .78, score: .12, score_definition: 'Highest dimension.', dominant_label: 'non-toxic', predicted_label_counts: { 'non-toxic': 11 }, mean_probabilities: { toxic: .12 } }] }, hour_of_day: { sentiment: [{ hour: '09', direction: url.searchParams.get('direction') ?? 'all', sample_count: 4, mean_confidence: .8, score: .2, score_definition: 'Mean score.', dominant_label: 'positive', predicted_label_counts: { positive: 3 }, mean_probabilities: { positive: .55, negative: .35 } }], emotion: [], toxicity: [] }, hour_of_day_scope: 'current_page', page: { total: null, limit: 24, returned: 1, next_cursor: null, count_is_unknown: true }, availability: nlp, rollup_provenance: { source_resolution: 'UTC hour', range_filter_semantics: 'whole UTC hours overlapping the requested interval', local_bucket_label: 'local time at the UTC rollup hour start', partial_boundary_hours_included: url.searchParams.has('from_ms') || url.searchParams.has('to_ms'), non_whole_hour_timezone_offset_observed: false }, warnings: ['No people are ranked.'] } })
    if (path === '/api/v1/analytics/conflict-rate') return route.fulfill({ json: { available: true, status: 'ready', reason: null, timezone: url.searchParams.get('timezone') ?? 'America/New_York', bucket: url.searchParams.get('bucket') ?? 'month', series: [{ period: '2023-01', sample_count: 120, direct_conflict_window_count: 18, direct_conflict_window_rate: .15, low_sample_warning: false }], page: { total: null, limit: 24, returned: 1, next_cursor: null, count_is_unknown: true }, summary: { sample_count: 120, direct_conflict_window_count: 18, direct_conflict_window_rate: .15 }, warnings: ['Model estimate; no people, conversations, or relationships are ranked.'], provenance: { run_id: 'fixture-conflict-run', model_revision: 'fixture-revision', rollup_resolution: 'UTC hour', range_filter_semantics: 'whole UTC hours overlapping the requested interval', local_bucket_label: 'local time at the UTC rollup hour start', partial_boundary_hours_included: url.searchParams.has('from_ms') || url.searchParams.has('to_ms'), non_whole_hour_timezone_offset_observed: false }, relationship_ranking_enabled: false, relationship_dimensions: [] } })
    if (path === '/api/v1/analytics/network') return route.fulfill({ json: { nodes: [{ id: 'conversation:one', kind: 'conversation', label: 'Alex Example', value: 100, href: '/conversations/one' }, { id: 'participant:one', kind: 'participant_label', label: 'Alex Example', value: 100, is_owner: false }], edges: [{ id: 'edge-one', source: 'conversation:one', target: 'participant:one', value: 100 }], conversation_count: 1, participant_label_count: 1, scope: 'Top retained conversation.', caveat: 'Participant labels are provisional.' } })
    if (path === '/api/v1/analytics/topics') return route.fulfill({ json: { status: 'unavailable', ready: false, latest_job: null, semantic_index: null, result: null, reason: 'Build the semantic text index before generating topic clusters.' } })
    if (path === '/api/v1/analytics/attachments') return route.fulfill({ json: { bucket: url.searchParams.get('bucket') ?? 'month', timezone: url.searchParams.get('timezone') ?? 'America/New_York', types: [{ type: 'photo', occurrences: 10, unique_files: 8, occurrence_bytes: 1000 }], availability: [{ availability: 'resolved', occurrences: 10, bytes: 1000, rate: 1 }], trends: [{ period: '2023-01', occurrences: 10, unique_files: 8, occurrence_bytes: 1000, by_type: { photo: { occurrences: 10, unique_files: 8, occurrence_bytes: 1000 } } }], page: { total: 1, limit: 120, returned: 1, next_cursor: null }, summary: { occurrences: 10, dated_occurrences: 10, unknown_date_occurrences: 0, unknown_date_scope: 'archive_wide', unknown_date_included_in_totals: true, missing_occurrences: 0, missing_rate: 0, unlinked_occurrences: 0, unlinked_rate: 0 }, count_semantics: 'Occurrence bytes count each send.', provenance: 'Fixture attachment timestamps.' } })
    if (path === '/api/v1/analytics/conversations') return route.fulfill({ json: { conversation_count: 4, by_type: { direct: 2, group: 1, system: 1 }, longevity_days: { mean: 10, median: 2, maximum: 100, histogram: { same_day: 2 } }, active_days: { mean: 20, maximum: 300 }, ranking_policy: 'No people or relationships are ranked.', derived_nlp: nlp } })
    if (path === '/api/v1/analytics/storage') return route.fulfill({ json: { source_occurrences: { files: 128, bytes: 125_829_120 }, unique_blobs: { files: 96, bytes: 94_371_840 }, deduplication: { redundant_occurrences: 32, duplicate_bytes: 31_457_280 }, by_media_class: [{ class: 'image', files: 10, bytes: 1000 }] } })
    if (path === '/api/v1/analytics/coverage') return route.fulfill({ json: { summary: { available: 1, unavailable: 1 }, items: [partialCoverage, unavailableCoverage], history: { mode: 'single_observed_export_snapshot', historical_comparison_available: false, description: 'Coverage reflects the current export inventory; no earlier export snapshots were retained.', observed_at: '2026-07-20T12:00:00Z', observed_at_basis: 'Latest coverage-area catalog update time, not the export creation time.', snapshots: [{ id: 'coverage-current', observed_at: '2026-07-20T12:00:00Z', summary: { available: 1, unavailable: 1 }, items: [partialCoverage, unavailableCoverage] }] } } })
    return route.fulfill({ status: 404, json: { detail: `Unhandled ${path}` } })
  })
}

test.beforeEach(async ({ page }) => {
  await mockStageFourApi(page)
})

test('activity, timeline, people, and unavailable event coverage remain source-traceable', async ({ page }) => {
  await page.goto('/activity')
  await expect(page.locator('.topbar__title > span')).toHaveText('Stage 4')
  await expect(page.getByText('Fixture album photo')).toBeVisible()
  await expect(page.getByText('your_facebook_activity/posts/album.json')).toBeVisible()
  await expect(page.getByRole('link', { name: 'Open' })).toHaveAttribute('href', /timeline/)
  const contextRequest = page.waitForRequest((request) => request.url().includes('post_context=group'))
  await page.getByLabel('Post/comment context').selectOption('group')
  await page.getByRole('button', { name: 'Apply filters' }).click()
  await contextRequest
  await expect(page.getByText('Classification uses retained source fields, categories, and subtypes only; missing evidence is never treated as zero history.')).toBeVisible()
  await page.screenshot({ path: 'test-results/stage4-activity-desktop.png', fullPage: true })

  await page.getByRole('link', { name: 'Open' }).click()
  await expect(page).toHaveURL(/timeline\?around=activity(?::|%3A)activity-one#activity-activity-one/)
  await expect(page.getByText('Archive chronology')).toBeVisible()
  await expect(page.locator('#activity-activity-one')).toBeVisible()
  await page.getByRole('button', { name: 'On this day' }).click()
  await expect(page.getByText(/2 matching records; 1 shown from 1 years on this page/)).toBeVisible()
  await page.getByRole('button', { name: 'Next page' }).click()
  await expect(page.getByText('Page 2')).toBeVisible()
  await expect(page.locator('#activity-activity-two')).toBeVisible()

  await page.getByRole('link', { name: 'People' }).click()
  await expect(page.getByRole('heading', { name: 'Alex Example' })).toBeVisible()
  await page.getByRole('button', { name: 'Friends' }).click()
  await expect(page.getByText(/cannot establish that your history was empty/i)).toBeVisible()

  await page.getByRole('link', { name: 'Events' }).click()
  await expect(page.getByText('Event history unavailable')).toBeVisible()
  await expect(page.getByText('Event history is unavailable in this export.').first()).toBeVisible()
})

test('event and relationship deep links resolve the requested source record', async ({ page }) => {
  await page.goto('/events?around=event-one#event-event-one')
  await expect(page.getByRole('heading', { name: 'Hosted fixture' })).toBeVisible()
  await expect(page.locator('#event-event-one')).toBeVisible()
  await page.goto('/people?view=friends&around=friend-one#relationship-friend-one')
  await expect(page.getByRole('heading', { name: 'Former Friend' })).toBeVisible()
  await expect(page.locator('#relationship-friend-one')).toBeVisible()
  await expect(page.getByRole('button', { name: 'Friends' })).toHaveAttribute('aria-pressed', 'true')
})

test('people selectors, direct-versus-group semantics, links, and cursor paging stay honest', async ({ page }) => {
  const relationshipFixtures = {
    friends: { name: 'Former Friend', status: 'removed' },
    requests: { name: 'Pending Friend Request', status: 'received' },
    followers: { name: 'Retained Follower', status: 'following' },
    'facebook-blocks': { name: 'Blocked Facebook Profile', status: 'blocked' },
    'messenger-blocks': { name: 'Blocked Messenger Profile', status: 'blocked' },
  } as const
  await page.route(/\/api\/v1\/people(?:\?.*)?$/, (route) => {
    const url = new URL(route.request().url())
    const view = url.searchParams.get('view') ?? 'contacts'
    if (view !== 'contacts') {
      const fixture = relationshipFixtures[view as keyof typeof relationshipFixtures]
      const relationshipItem = { id: `relationship-${view}`, name: fixture.name, status: fixture.status, friendship_status: fixture.status, timestamp_ms: 1_650_000_000_000, source: activityItem.source, jump: { href: `/people?view=${view}&around=relationship-${view}#relationship-relationship-${view}`, fragment: `relationship-relationship-${view}` } }
      const items = url.searchParams.get('q') === 'Nobody' ? [] : [relationshipItem]
      return route.fulfill({ json: { coverage: { ...partialCoverage, key: view, label: 'Relationship records', status: 'available' }, authority: 'source_records', contacts_are_not_included: true, items, page: { total: items.length, limit: 100, next_cursor: null, count_is_unknown: false } } })
    }
    if (url.searchParams.get('cursor') === 'people-page-two') {
      return route.fulfill({ json: contactsPage(null, [{ ...contactItem, id: 'person-jordan', name: 'Jordan Example', counts: { messages: 2, photos: 1, videos: 0, audio: 0 }, direct_counts: { messages: 2, photos: 1, videos: 0, audio: 0 }, shared_group_counts: undefined, combined_thread_counts: { messages: 2, photos: 1, videos: 0, audio: 0 }, conversation_count: 1, direct_conversation_count: 1, group_conversation_count: 0, conversations: [contactItem.conversations[0]] }]) })
    }
    return route.fulfill({ json: contactsPage('people-page-two') })
  })

  await page.goto('/people')
  const directMetrics = page.getByLabel('Direct one-to-one exchanges')
  await expect(directMetrics).toContainText('Messages120Photos2Videos1Audio1')
  await expect(page.getByLabel('Visible contact exchange totals')).toContainText('Visible direct chats')
  await expect(directMetrics.getByRole('definition')).toHaveCount(4)

  const groupSemantics = page.locator('.person-card__semantics').filter({ hasText: 'Shared group threads (whole-thread totals)' })
  await expect(groupSemantics).toContainText('20 messages · 1 photos · 0 videos · 0 audio')
  await expect(groupSemantics).toContainText('not attributed to this person')
  const semanticsLayout = await groupSemantics.evaluate((element) => {
    const style = getComputedStyle(element)
    const heading = element.querySelector('strong')!.getBoundingClientRect()
    const totals = element.querySelector('span')!.getBoundingClientRect()
    const caveat = element.querySelector('p')!.getBoundingClientRect()
    return { display: style.display, fontSize: Number.parseFloat(style.fontSize), lineHeight: Number.parseFloat(style.lineHeight), headingBottom: heading.bottom, totalsTop: totals.top, totalsBottom: totals.bottom, caveatTop: caveat.top }
  })
  expect(semanticsLayout.display).toBe('grid')
  expect(semanticsLayout.fontSize).toBeGreaterThanOrEqual(11)
  expect(semanticsLayout.lineHeight).toBeGreaterThanOrEqual(15)
  expect(semanticsLayout.totalsTop).toBeGreaterThanOrEqual(semanticsLayout.headingBottom)
  expect(semanticsLayout.caveatTop).toBeGreaterThanOrEqual(semanticsLayout.totalsBottom)

  await page.getByText('2 linked conversations').click()
  await expect(page.getByRole('link', { name: /Alex Example.*Direct.*120 whole-thread messages/ })).toHaveAttribute('href', '/conversations/conv-alex')
  await expect(page.getByRole('link', { name: /Fixture group.*Group.*20 whole-thread messages/ })).toHaveAttribute('href', '/conversations/group-alex')

  const nextPeople = page.waitForRequest((request) => new URL(request.url()).searchParams.get('cursor') === 'people-page-two')
  await page.getByRole('button', { name: 'Next page' }).click()
  await nextPeople
  await expect(page.getByRole('heading', { name: 'Jordan Example' })).toBeVisible()

  const searchRequest = page.waitForRequest((request) => new URL(request.url()).pathname === '/api/v1/people' && new URL(request.url()).searchParams.get('q') === 'Alex')
  await page.getByLabel('Search names').fill('Alex')
  await searchRequest
  for (const value of ['photos', 'videos', 'audio', 'conversations', 'last_activity', 'name']) {
    const requested = page.waitForRequest((request) => {
      const url = new URL(request.url())
      return url.pathname === '/api/v1/people' && url.searchParams.get('sort') === value
    })
    await page.getByLabel('Sort by').selectOption(value)
    await requested
  }
  await page.getByLabel('Sort by').selectOption('messages')
  for (const value of ['direct', 'group']) {
    const requested = page.waitForRequest((request) => new URL(request.url()).searchParams.get('conversation_type') === value)
    await page.getByLabel('Conversation type').selectOption(value)
    await requested
  }
  const ascending = page.waitForRequest((request) => new URL(request.url()).searchParams.get('direction') === 'asc')
  await page.getByLabel('Direction').selectOption('asc')
  await ascending

  for (const [name, value, recordName] of [['Friends', 'friends', 'Former Friend'], ['Requests', 'requests', 'Pending Friend Request'], ['Followers', 'followers', 'Retained Follower'], ['Facebook blocks', 'facebook-blocks', 'Blocked Facebook Profile'], ['Messenger blocks', 'messenger-blocks', 'Blocked Messenger Profile']] as const) {
    const requested = page.waitForRequest((request) => new URL(request.url()).searchParams.get('view') === value)
    await page.getByRole('button', { name }).click()
    await requested
    await expect(page.getByRole('heading', { name: recordName })).toBeVisible()
  }
  for (const value of ['status', 'name']) {
    const requested = page.waitForRequest((request) => {
      const url = new URL(request.url())
      return url.searchParams.get('view') === 'messenger-blocks' && url.searchParams.get('sort') === value
    })
    await page.getByLabel('Sort by').selectOption(value)
    await requested
  }
  const descending = page.waitForRequest((request) => {
    const url = new URL(request.url())
    return url.searchParams.get('view') === 'messenger-blocks' && url.searchParams.get('direction') === 'desc'
  })
  await page.getByLabel('Direction').selectOption('desc')
  await descending

  const empty = page.waitForRequest((request) => new URL(request.url()).searchParams.get('q') === 'Nobody')
  await page.getByLabel('Search names').fill('Nobody')
  await empty
  await expect(page.getByText('No matching people', { exact: true })).toBeVisible()
})

test('people recovers from failure and traverses beyond its former cap with a bounded reversible window', async ({ page }) => {
  let attempts = 0
  await page.route(/\/api\/v1\/people(?:\?.*)?$/, (route) => {
    if (attempts++ < 2) return route.fulfill({ status: 503, json: { detail: 'Fixture people service unavailable' } })
    const url = new URL(route.request().url())
    const cursor = url.searchParams.get('cursor')
    const pageNumber = cursor ? Number(cursor.replace('people-window-', '')) : 1
    const item = { ...contactItem, id: `person-window-${pageNumber}`, name: `Person window ${pageNumber}`, conversation_count: 0, direct_conversation_count: 0, group_conversation_count: 0, conversations: [], shared_group_counts: undefined }
    return route.fulfill({ json: contactsPage(pageNumber < 6 ? `people-window-${pageNumber + 1}` : null, [item], 600) })
  })
  await page.goto('/people')
  await expect(page.getByRole('alert')).toContainText('Fixture people service unavailable')
  await page.getByRole('button', { name: 'Try again' }).click()
  await expect(page.getByRole('heading', { name: 'Person window 1' })).toBeVisible()
  for (let pageNumber = 2; pageNumber <= 5; pageNumber += 1) {
    const requested = page.waitForRequest((request) => new URL(request.url()).searchParams.get('cursor') === `people-window-${pageNumber}`)
    await page.getByRole('button', { name: 'Next page' }).click()
    await requested
  }
  await expect(page.getByRole('heading', { name: 'Person window 5' })).toBeVisible()
  await expect(page.getByRole('heading', { name: 'Person window 1' })).toHaveCount(0)
  await expect(page.getByLabel('People results pages')).toHaveAttribute('data-page-number', '5')
  await expect(page.getByLabel('People results pages')).toHaveAttribute('data-loaded-item-count', '1')
  const previous = page.waitForRequest((request) => new URL(request.url()).searchParams.get('cursor') === 'people-window-4')
  await page.getByRole('button', { name: 'Previous page' }).click()
  await previous
  await expect(page.getByRole('heading', { name: 'Person window 4' })).toBeVisible()
  await expect(page.getByRole('heading', { name: /^Person window/ })).toHaveCount(1)
})

test('event filters, every status/sort choice, hosted scope, and cursor paging reach the API', async ({ page }) => {
  test.setTimeout(60_000)
  const observed: URL[] = []
  const secondEvent = { ...eventItem, id: 'event-two', title: 'Attended fixture', status: 'went', hosted: false }
  await page.route(/\/api\/v1\/events(?:\?.*)?$/, (route) => {
    const url = new URL(route.request().url())
    observed.push(url)
    return route.fulfill({ json: url.searchParams.get('cursor') === 'events-page-two' ? eventsPage(null, [secondEvent]) : eventsPage('events-page-two') })
  })

  await page.goto('/events')
  await expect(page.getByRole('heading', { name: 'Hosted fixture' })).toBeVisible()
  const nextEvents = page.waitForRequest((request) => new URL(request.url()).searchParams.get('cursor') === 'events-page-two')
  await page.getByRole('button', { name: 'Next page' }).click()
  await nextEvents
  await expect(page.getByRole('heading', { name: 'Attended fixture' })).toBeVisible()

  for (const status of ['hosted', 'created', 'went', 'interested', 'invited', 'declined', 'cancelled', 'unknown']) {
    await page.getByLabel('RSVP status').selectOption(status)
    const requested = page.waitForRequest((request) => new URL(request.url()).searchParams.get('event_status') === status)
    await page.getByRole('button', { name: 'Apply filters' }).click()
    await requested
  }
  for (const [value, expected] of [['yes', 'true'], ['no', 'false']] as const) {
    await page.getByLabel('Hosted by you').selectOption(value)
    const requested = page.waitForRequest((request) => new URL(request.url()).searchParams.get('hosted') === expected)
    await page.getByRole('button', { name: 'Apply filters' }).click()
    await requested
  }
  await page.getByLabel('Hosted by you').selectOption('all')
  await page.getByLabel('Search events').fill('all hosts')
  const everyHost = page.waitForRequest((request) => {
    const url = new URL(request.url())
    return url.pathname === '/api/v1/events' && !url.searchParams.has('hosted') && url.searchParams.get('event_status') === 'unknown' && url.searchParams.get('q') === 'all hosts'
  })
  await page.getByRole('button', { name: 'Apply filters' }).click()
  await everyHost

  for (const sort of ['title', 'status']) {
    await page.getByLabel('Sort').selectOption(sort)
    const requested = page.waitForRequest((request) => {
      const url = new URL(request.url())
      return url.searchParams.get('sort') === sort && url.searchParams.get('event_status') === 'unknown'
    })
    await page.getByRole('button', { name: 'Apply filters' }).click()
    await requested
  }
  await page.getByLabel('Sort').selectOption('date')

  await page.getByLabel('Search events').fill('fixture reunion')
  await page.getByLabel('RSVP status').selectOption('went')
  await page.getByLabel('Hosted by you').selectOption('no')
  await page.getByLabel('From').fill('2023-01-01')
  await page.getByLabel('Through').fill('2023-01-31')
  await page.getByLabel('Sort').selectOption('title')
  await page.getByLabel('Direction').selectOption('asc')
  const composite = page.waitForRequest((request) => {
    const url = new URL(request.url())
    return url.pathname === '/api/v1/events' && url.searchParams.get('q') === 'fixture reunion' && url.searchParams.get('event_status') === 'went' && url.searchParams.get('hosted') === 'false' && url.searchParams.has('from_ms') && url.searchParams.has('to_ms') && url.searchParams.get('sort') === 'title' && url.searchParams.get('direction') === 'asc'
  })
  await page.getByRole('button', { name: 'Apply filters' }).click()
  await composite
  expect(new Set(observed.map((url) => url.searchParams.get('event_status')).filter(Boolean))).toEqual(new Set(['hosted', 'created', 'went', 'interested', 'invited', 'declined', 'cancelled', 'unknown']))
  expect(new Set(observed.map((url) => url.searchParams.get('sort')))).toEqual(new Set(['date', 'title', 'status']))
})

test('events recover from failure and traverse beyond their former cap with a bounded reversible window', async ({ page }) => {
  test.setTimeout(60_000)
  let attempts = 0
  await page.route(/\/api\/v1\/events(?:\?.*)?$/, (route) => {
    if (attempts++ < 2) return route.fulfill({ status: 503, json: { detail: 'Fixture events service unavailable' } })
    const url = new URL(route.request().url())
    if (url.searchParams.get('q') === 'empty') return route.fulfill({ json: eventsPage(null, []) })
    const cursor = url.searchParams.get('cursor')
    const pageNumber = cursor ? Number(cursor.replace('events-window-', '')) : 1
    return route.fulfill({ json: eventsPage(pageNumber < 10 ? `events-window-${pageNumber + 1}` : null, [{ ...eventItem, id: `event-window-${pageNumber}`, title: `Event window ${pageNumber}` }], 750) })
  })
  await page.goto('/events')
  await expect(page.getByRole('alert')).toContainText('Fixture events service unavailable')
  await page.getByRole('button', { name: 'Try again' }).click()
  await expect(page.getByRole('heading', { name: 'Event window 1' })).toBeVisible()
  for (let pageNumber = 2; pageNumber <= 9; pageNumber += 1) {
    const requested = page.waitForRequest((request) => new URL(request.url()).searchParams.get('cursor') === `events-window-${pageNumber}`)
    await page.getByRole('button', { name: 'Next page' }).click()
    await requested
  }
  await expect(page.getByRole('heading', { name: 'Event window 9' })).toBeVisible()
  await expect(page.getByRole('heading', { name: 'Event window 1' })).toHaveCount(0)
  await expect(page.getByLabel('Event results pages')).toHaveAttribute('data-page-number', '9')
  await expect(page.getByLabel('Event results pages')).toHaveAttribute('data-loaded-item-count', '1')
  const previous = page.waitForRequest((request) => new URL(request.url()).searchParams.get('cursor') === 'events-window-8')
  await page.getByRole('button', { name: 'Previous page' }).click()
  await previous
  await expect(page.getByRole('heading', { name: 'Event window 8' })).toBeVisible()
  await expect(page.locator('.event-card')).toHaveCount(1)
  await page.getByLabel('Search events').fill('empty')
  const empty = page.waitForRequest((request) => new URL(request.url()).searchParams.get('q') === 'empty')
  await page.getByRole('button', { name: 'Apply filters' }).click()
  await empty
  await expect(page.getByText('No matching events', { exact: true })).toBeVisible()
})

test('activity views, filters, every context/sort choice, reset, and cursor paging reach the API', async ({ page }) => {
  test.setTimeout(60_000)
  const observed: URL[] = []
  const secondActivity = { ...activityItem, id: 'activity-two', title: 'Second fixture activity', jump: { href: '/timeline?around=activity:activity-two#activity-activity-two', fragment: 'activity-activity-two' } }
  await page.route(/\/api\/v1\/activity(?:\/views\/[^?]+)?(?:\?.*)?$/, (route) => {
    const url = new URL(route.request().url())
    observed.push(url)
    return route.fulfill({ json: url.searchParams.get('cursor') === 'activity-page-two' ? activityPage(null, [secondActivity]) : activityPage('activity-page-two') })
  })

  await page.goto('/activity')
  const nextActivity = page.waitForRequest((request) => new URL(request.url()).searchParams.get('cursor') === 'activity-page-two')
  await page.getByRole('button', { name: 'Next page' }).click()
  await nextActivity
  await expect(page.getByText('Second fixture activity')).toBeVisible()

  for (const [name, path] of [['Posts', '/api/v1/activity/views/posts'], ['Comments', '/api/v1/activity/views/comments'], ['Groups', '/api/v1/activity/views/groups'], ['Marketplace', '/api/v1/activity/views/marketplace'], ['Pages', '/api/v1/activity/views/pages'], ['Check-ins', '/api/v1/activity/views/check-ins'], ['Support', '/api/v1/activity/views/support']] as const) {
    const requested = page.waitForRequest((request) => new URL(request.url()).pathname === path)
    await page.getByRole('button', { name, exact: true }).click()
    await requested
  }
  await page.getByRole('button', { name: 'All activity' }).click()
  await expect(page.getByLabel('Post/comment context')).toBeVisible()

  for (const context of ['own_wall', 'others_wall', 'group', 'reshared', 'tagged', 'others_on_your_wall', 'unknown']) {
    await page.getByLabel('Post/comment context').selectOption(context)
    const requested = page.waitForRequest((request) => new URL(request.url()).searchParams.get('post_context') === context)
    await page.getByRole('button', { name: 'Apply filters' }).click()
    await requested
  }
  for (const sort of ['title', 'actor', 'category', 'subtype']) {
    await page.getByLabel('Sort').selectOption(sort)
    const requested = page.waitForRequest((request) => {
      const url = new URL(request.url())
      return url.pathname === '/api/v1/activity' && url.searchParams.get('sort') === sort && url.searchParams.get('post_context') === 'unknown'
    })
    await page.getByRole('button', { name: 'Apply filters' }).click()
    await requested
  }
  await page.getByLabel('Sort').selectOption('date')
  for (const availability of ['available', 'unavailable_body']) {
    await page.getByLabel('Availability').selectOption(availability)
    const requested = page.waitForRequest((request) => new URL(request.url()).searchParams.get('availability') === availability)
    await page.getByRole('button', { name: 'Apply filters' }).click()
    await requested
  }
  for (const [value, expected] of [['yes', 'true'], ['no', 'false']] as const) {
    await page.getByLabel('Attachments').selectOption(value)
    const requested = page.waitForRequest((request) => new URL(request.url()).searchParams.get('has_attachments') === expected)
    await page.getByRole('button', { name: 'Apply filters' }).click()
    await requested
  }

  await page.getByLabel('Search title, text, or actor').fill('caption')
  await page.getByLabel('From').fill('2023-02-01')
  await page.getByLabel('Through').fill('2023-02-28')
  await page.getByLabel('Direction').selectOption('asc')
  const composite = page.waitForRequest((request) => {
    const url = new URL(request.url())
    return url.pathname === '/api/v1/activity' && url.searchParams.get('q') === 'caption' && url.searchParams.has('from_ms') && url.searchParams.has('to_ms') && url.searchParams.get('direction') === 'asc'
  })
  await page.getByRole('button', { name: 'Apply filters' }).click()
  await composite
  await page.getByRole('button', { name: 'Reset' }).click()
  await expect(page.getByLabel('Search title, text, or actor')).toHaveValue('')
  await expect(page.getByLabel('Direction')).toHaveValue('desc')

  expect(new Set(observed.map((url) => url.pathname))).toEqual(new Set(['/api/v1/activity', '/api/v1/activity/views/posts', '/api/v1/activity/views/comments', '/api/v1/activity/views/groups', '/api/v1/activity/views/marketplace', '/api/v1/activity/views/pages', '/api/v1/activity/views/check-ins', '/api/v1/activity/views/support']))
  expect(new Set(observed.map((url) => url.searchParams.get('sort')))).toEqual(new Set(['date', 'title', 'actor', 'category', 'subtype']))
})

test('activity recovers from failure and traverses beyond its former cap with a bounded reversible window', async ({ page }) => {
  test.setTimeout(60_000)
  let attempts = 0
  await page.route(/\/api\/v1\/activity(?:\?.*)?$/, (route) => {
    if (attempts++ < 2) return route.fulfill({ status: 503, json: { detail: 'Fixture activity service unavailable' } })
    const url = new URL(route.request().url())
    if (url.searchParams.get('q') === 'empty') return route.fulfill({ json: activityPage(null, []) })
    const cursor = url.searchParams.get('cursor')
    const pageNumber = cursor ? Number(cursor.replace('activity-window-', '')) : 1
    return route.fulfill({ json: activityPage(pageNumber < 18 ? `activity-window-${pageNumber + 1}` : null, [{ ...activityItem, id: `activity-window-${pageNumber}`, title: `Activity window ${pageNumber}` }], 1_350) })
  })
  await page.goto('/activity')
  await expect(page.getByRole('alert')).toContainText('Fixture activity service unavailable')
  await page.getByRole('button', { name: 'Try again' }).click()
  await expect(page.getByText('Activity window 1')).toBeVisible()
  for (let pageNumber = 2; pageNumber <= 17; pageNumber += 1) {
    const requested = page.waitForRequest((request) => new URL(request.url()).searchParams.get('cursor') === `activity-window-${pageNumber}`)
    await page.getByRole('button', { name: 'Next page' }).click()
    await requested
  }
  await expect(page.getByText('Activity window 17')).toBeVisible()
  await expect(page.getByText('Activity window 1', { exact: true })).toHaveCount(0)
  await expect(page.getByLabel('Activity results pages')).toHaveAttribute('data-page-number', '17')
  await expect(page.getByLabel('Scrollable virtualized activity results')).toHaveAttribute('data-loaded-row-count', '1')
  const previous = page.waitForRequest((request) => new URL(request.url()).searchParams.get('cursor') === 'activity-window-16')
  await page.getByRole('button', { name: 'Previous page' }).click()
  await previous
  await expect(page.getByText('Activity window 16')).toBeVisible()
  await expect(page.locator('.activity-table tbody tr')).toHaveCount(1)
  await page.getByLabel('Search title, text, or actor').fill('empty')
  const empty = page.waitForRequest((request) => new URL(request.url()).searchParams.get('q') === 'empty')
  await page.getByRole('button', { name: 'Apply filters' }).click()
  await empty
  await expect(page.getByText('No matching activity', { exact: true })).toBeVisible()
})

test('timeline modes, record types, date/text filters, source jumps, and both pagers stay wired', async ({ page }) => {
  const timelineSecond = { ...activityItem, id: 'activity:activity-two', entity_type: 'activity', entity_id: 'activity-two', title: 'Earlier timeline fixture', jump: { href: '/activity#activity-activity-two', fragment: 'activity-activity-two' } }
  await page.route(/\/api\/v1\/timeline(?:\?.*)?$/, (route) => {
    const url = new URL(route.request().url())
    const second = url.searchParams.get('cursor') === 'timeline-page-two'
    return route.fulfill({ json: { coverage: { status: 'source_backed', absence_is_not_zero_history: false }, items: [second ? timelineSecond : { ...activityItem, id: 'activity:activity-one', entity_type: 'activity', entity_id: 'activity-one' }], page: { total: 2, limit: 75, next_cursor: second ? null : 'timeline-page-two', count_is_unknown: false } } })
  })

  await page.goto('/timeline')
  await expect(page.getByRole('link', { name: 'Open source point' })).toHaveAttribute('href', '/timeline?around=activity:activity-one#activity-activity-one')
  const nextTimeline = page.waitForRequest((request) => new URL(request.url()).searchParams.get('cursor') === 'timeline-page-two')
  await page.getByRole('button', { name: 'Next page' }).click()
  await nextTimeline
  await expect(page.getByText('Earlier timeline fixture')).toBeVisible()

  await page.getByLabel('Search timeline').fill('retained caption')
  await page.getByLabel('Record type').selectOption('message')
  await page.getByLabel('From').fill('2023-03-01')
  await page.getByLabel('Through').fill('2023-03-31')
  const filtered = page.waitForRequest((request) => {
    const url = new URL(request.url())
    return url.pathname === '/api/v1/timeline' && url.searchParams.get('q') === 'retained caption' && url.searchParams.get('entity_type') === 'message' && url.searchParams.has('from_ms') && url.searchParams.has('to_ms') && !url.searchParams.has('around')
  })
  await page.getByRole('button', { name: 'Apply' }).click()
  await filtered
  const activityType = page.waitForRequest((request) => new URL(request.url()).searchParams.get('entity_type') === 'activity')
  await page.getByLabel('Record type').selectOption('activity')
  await activityType

  await page.getByRole('button', { name: 'On this day' }).click()
  await expect(page.getByLabel('Month')).toBeVisible()
  const onThisDay = page.waitForRequest((request) => {
    const url = new URL(request.url())
    return url.pathname === '/api/v1/timeline/on-this-day' && url.searchParams.get('month') === '3' && url.searchParams.get('day') === '15' && url.searchParams.get('entity_type') === 'activity'
  })
  await page.getByLabel('Month').selectOption('3')
  await page.getByLabel('Day', { exact: true }).fill('15')
  await onThisDay
  await expect(page.getByRole('heading', { name: 'On 3/15' })).toBeVisible()
  const moreYears = page.waitForRequest((request) => new URL(request.url()).searchParams.get('cursor') === 'on-this-day-page-two')
  await page.getByRole('button', { name: 'Next page' }).click()
  await moreYears
  await expect(page.getByText('Page 2')).toBeVisible()
  await expect(page.locator('#activity-activity-two')).toBeVisible()
})

test('timeline traverses beyond its former cap with a bounded reversible window and keeps error states usable', async ({ page }) => {
  let errorAttempts = 0
  await page.route(/\/api\/v1\/timeline(?:\?.*)?$/, (route) => {
    const url = new URL(route.request().url())
    const q = url.searchParams.get('q')
    if (q === 'force error' && errorAttempts++ < 2) return route.fulfill({ status: 503, json: { detail: 'Fixture timeline unavailable' } })
    if (q === 'empty' || q === 'force error') return route.fulfill({ json: { coverage: { status: 'source_backed', absence_is_not_zero_history: false }, items: [], page: { total: 0, limit: 75, next_cursor: null, count_is_unknown: false } } })
    const cursor = url.searchParams.get('cursor')
    if (cursor === 'timeline-window-18') return route.fulfill({ status: 503, json: { detail: 'Fixture later timeline page unavailable' } })
    const pageNumber = cursor ? Number(cursor.replace('timeline-window-', '')) : 1
    const item = { ...activityItem, id: `activity:window-${pageNumber}`, entity_type: 'activity', entity_id: `window-${pageNumber}`, title: `Timeline window ${pageNumber}`, jump: { href: `/activity#activity-window-${pageNumber}`, fragment: `activity-window-${pageNumber}` } }
    return route.fulfill({ json: { coverage: { status: 'source_backed', absence_is_not_zero_history: false }, items: [item], page: { total: 1_350, limit: 75, next_cursor: pageNumber < 18 ? `timeline-window-${pageNumber + 1}` : null, count_is_unknown: false } } })
  })

  await page.goto('/timeline')
  for (let pageNumber = 2; pageNumber <= 17; pageNumber += 1) {
    const requested = page.waitForRequest((request) => new URL(request.url()).searchParams.get('cursor') === `timeline-window-${pageNumber}`)
    await page.getByRole('button', { name: 'Next page' }).click()
    await requested
  }
  await expect(page.getByText('Timeline window 17')).toBeVisible()
  await expect(page.getByText('Timeline window 1', { exact: true })).toHaveCount(0)
  await expect(page.getByLabel('Timeline results pages')).toHaveAttribute('data-page-number', '17')
  await expect(page.getByLabel('Scrollable virtualized timeline')).toHaveAttribute('data-loaded-entry-count', '1')
  const failedLaterPage = page.waitForResponse((response) => new URL(response.url()).searchParams.get('cursor') === 'timeline-window-18' && response.status() === 503)
  await page.getByRole('button', { name: 'Next page' }).click()
  await failedLaterPage
  await expect(page.getByRole('alert')).toContainText('Fixture later timeline page unavailable')
  const recoverPrevious = page.waitForRequest((request) => new URL(request.url()).searchParams.get('cursor') === 'timeline-window-17')
  await page.getByRole('button', { name: 'Previous page' }).click()
  await recoverPrevious
  await expect(page.getByText('Timeline window 17')).toBeVisible()
  const previous = page.waitForRequest((request) => new URL(request.url()).searchParams.get('cursor') === 'timeline-window-16')
  await page.getByRole('button', { name: 'Previous page' }).click()
  await previous
  await expect(page.getByText('Timeline window 16')).toBeVisible()
  await expect(page.locator('.timeline-entry')).toHaveCount(1)

  await page.getByLabel('Search timeline').fill('empty')
  const empty = page.waitForRequest((request) => new URL(request.url()).searchParams.get('q') === 'empty')
  await page.getByRole('button', { name: 'Apply' }).click()
  await empty
  await expect(page.getByText('No matching timeline records', { exact: true })).toBeVisible()

  await page.getByLabel('Search timeline').fill('force error')
  const failed = page.waitForResponse((response) => new URL(response.url()).pathname === '/api/v1/timeline' && response.status() === 503)
  await page.getByRole('button', { name: 'Apply' }).click()
  await failed
  await expect(page.getByRole('alert')).toContainText('Fixture timeline unavailable')
  await page.getByRole('button', { name: 'Try again' }).click()
  await expect(page.getByText('No matching timeline records', { exact: true })).toBeVisible()
})

test('large activity and timeline result sets keep their rendered DOM bounded', async ({ page }) => {
  const activityItems = Array.from({ length: 240 }, (_, index) => ({
    ...activityItem,
    id: `activity-volume-${index}`,
    title: `Fixture activity ${index}`,
    timestamp_ms: 1_700_000_000_000 - index * 60_000,
    jump: { href: `/timeline?around=activity:activity-volume-${index}#activity-activity-volume-${index}`, fragment: `activity-activity-volume-${index}` },
  }))
  await page.route(/\/api\/v1\/activity(?:\?.*)?$/, (route) => route.fulfill({ json: {
    coverage: partialCoverage,
    coverage_components: [],
    post_context_coverage: { status: 'summary', requested: [], counts: { group: 240 }, classification: 'source_fields_categories_and_subtypes', description: 'Retained source fields, categories, and subtypes classify context.', absence_is_not_zero_history: false },
    items: activityItems,
    page: { total: 240, limit: 240, next_cursor: null, count_is_unknown: false },
  } }))
  await page.goto('/activity')
  const activityViewport = page.getByLabel('Scrollable virtualized activity results')
  await expect(activityViewport).toHaveAttribute('data-loaded-row-count', '240')
  expect(Number(await activityViewport.getAttribute('data-rendered-row-count'))).toBeLessThan(40)
  await expect(activityViewport.getByRole('table')).toHaveAttribute('aria-rowcount', '241')
  await expect(activityViewport.getByRole('row').nth(1)).toHaveAttribute('aria-rowindex', '2')

  const timelineItems = activityItems.map((item, index) => ({
    ...item,
    id: `activity:activity-volume-${index}`,
    entity_type: 'activity',
    entity_id: `activity-volume-${index}`,
  }))
  await page.route(/\/api\/v1\/timeline(?:\?.*)?$/, (route) => route.fulfill({ json: {
    coverage: { status: 'source_backed', absence_is_not_zero_history: false },
    items: timelineItems,
    page: { total: 240, limit: 240, next_cursor: null, count_is_unknown: false },
  } }))
  await page.goto('/timeline')
  const timelineViewport = page.getByLabel('Scrollable virtualized timeline')
  await expect(timelineViewport).toHaveAttribute('role', 'feed')
  await expect(timelineViewport).toHaveAttribute('data-loaded-entry-count', '240')
  expect(Number(await timelineViewport.getAttribute('data-rendered-entry-count'))).toBeLessThan(40)
  await expect(timelineViewport.getByRole('article').first()).toHaveAttribute('aria-posinset', '1')
  await expect(timelineViewport.getByRole('article').first()).toHaveAttribute('aria-setsize', '240')
})

test('desktop virtualized activity rows remain unobscured and axe-complete', async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 1000 })
  const activityItems = Array.from({ length: 75 }, (_, index) => ({
    ...activityItem,
    id: `activity-accessibility-${index}`,
    title: `Activity accessibility fixture ${index}`,
    content: index % 2 === 0
      ? 'A short retained caption'
      : 'A longer retained caption that exercises dynamic virtual-row measurement without allowing adjacent rows or the containing panel to paint over its text.',
    timestamp_ms: 1_700_000_000_000 - index * 60_000,
    jump: {
      href: `/timeline?around=activity:activity-accessibility-${index}#activity-activity-accessibility-${index}`,
      fragment: `activity-activity-accessibility-${index}`,
    },
  }))
  await page.route(/\/api\/v1\/activity(?:\?.*)?$/, (route) => route.fulfill({ json: {
    coverage: partialCoverage,
    coverage_components: [],
    post_context_coverage: { status: 'summary', requested: [], counts: { group: activityItems.length }, classification: 'source_fields_categories_and_subtypes', description: 'Retained source fields, categories, and subtypes classify context.', absence_is_not_zero_history: false },
    items: activityItems,
    page: { total: activityItems.length, limit: 75, next_cursor: null, count_is_unknown: false },
  } }))

  await page.goto('/activity')
  const viewport = page.getByLabel('Scrollable virtualized activity results')
  await expect(viewport).toHaveAttribute('data-loaded-row-count', '75')
  await page.evaluate(() => new Promise<void>((resolve) => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))))

  const rowGeometry = await viewport.locator('tbody tr[data-index]').evaluateAll((rows) => rows.map((row) => {
    const bounds = row.getBoundingClientRect()
    return {
      top: bounds.top,
      bottom: bounds.bottom,
      background: getComputedStyle(row).backgroundColor,
    }
  }))
  expect(rowGeometry.length).toBeGreaterThan(1)
  expect(rowGeometry.every((row) => row.background !== 'rgba(0, 0, 0, 0)')).toBe(true)
  for (let index = 1; index < rowGeometry.length; index += 1) {
    expect(rowGeometry[index]!.top).toBeGreaterThanOrEqual(rowGeometry[index - 1]!.bottom - 0.5)
  }

  const result = await new AxeBuilder({ page })
    .withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'wcag22a', 'wcag22aa'])
    .analyze()
  expect(result.violations, result.violations.map((issue) => `${issue.id}:${issue.nodes.length}`).join(', ')).toEqual([])
  expect(result.incomplete, result.incomplete.map((issue) => `${issue.id}:${issue.nodes.length}`).join(', ')).toEqual([])
})

test('analytics stays busy through queued placeholder reads and settles before navigation', async ({ page }) => {
  test.setTimeout(60_000)
  const started: string[] = []
  const active = new Set<number>()
  const pending: Array<{ release: () => void }> = []
  const failures: string[] = []
  let sequence = 0
  let settledObserved = false
  let startsAfterSettled = 0

  page.on('requestfailed', (request) => {
    const url = new URL(request.url())
    if (url.pathname.startsWith('/api/v1/analytics/')) failures.push(url.pathname)
  })
  await page.route('**/api/v1/analytics/**', async (route) => {
    const url = new URL(route.request().url())
    if (url.searchParams.get('bucket') !== 'day') return route.fallback()
    if (settledObserved) startsAfterSettled += 1
    const id = sequence++
    started.push(url.pathname)
    active.add(id)
    await new Promise<void>((resolve) => pending.push({ release: resolve }))
    try {
      await route.fallback()
    } finally {
      active.delete(id)
    }
  })
  await page.route(/\/api\/v1\/deletions(?:\?.*)?$/, (route) => route.fulfill({
    json: { items: [], page: { total: 0, limit: 50, next_cursor: null, count_is_unknown: false } },
  }))

  await page.goto('/analytics')
  const analytics = page.locator('.analytics-page')
  await expect(analytics).toHaveAttribute('data-analytics-route-state', 'settled')
  await expect(page.getByLabel('Analytics overview')).toBeVisible()

  await page.locator('.analytics-controls').getByLabel('Bucket').selectOption('day')
  await expect(analytics).toHaveAttribute('aria-busy', 'true')
  await expect(page.getByText('Computing archive analytics')).toBeVisible()
  await expect(page.getByLabel('Analytics overview')).toBeVisible()
  await expect.poll(() => started.length).toBe(2)
  expect(active.size).toBe(2)

  while (started.length < 4) {
    const previousStarted = started.length
    await expect.poll(() => pending.length).toBeGreaterThan(0)
    pending.shift()?.release()
    await expect.poll(() => started.length).toBe(previousStarted + 1)
    await expect(analytics).toHaveAttribute('data-analytics-route-state', 'loading')
  }
  while (pending.length) pending.shift()?.release()

  await expect(analytics).toHaveAttribute('data-analytics-route-state', 'settled')
  await expect(analytics).toHaveAttribute('aria-busy', 'false')
  await expect(page.getByText('Computing archive analytics')).toHaveCount(0)
  await expect.poll(() => active.size).toBe(0)
  expect(new Set(started)).toEqual(new Set([
    '/api/v1/analytics/messages',
    '/api/v1/analytics/interaction-patterns',
    '/api/v1/analytics/interaction-series',
    '/api/v1/analytics/attachments',
  ]))

  settledObserved = true
  await page.waitForTimeout(750)
  expect(startsAfterSettled).toBe(0)
  expect(active.size).toBe(0)
  expect(started).toHaveLength(4)

  await page.goto('/deletion')
  await expect(page.getByRole('heading', { name: 'Deletion and quarantine' })).toBeVisible()
  expect(failures).toEqual([])
})

test('analytics renders reconciled charts and explicit NLP availability', async ({ page }) => {
  await page.goto('/analytics')
  await expect(page.getByRole('heading', { name: 'Archive analytics' })).toBeVisible()
  await expect(page.getByText('240')).toBeVisible()
  await expect(page.getByRole('img', { name: 'Sent, received, and active conversations by month' })).toBeVisible()
  await expect(page.getByRole('img', { name: 'Calendar heatmap of retained archive activity' })).toBeVisible()
  await expect(page.getByRole('img', { name: 'Response time distributions by speaker direction' })).toBeVisible()
  await expect(page.getByRole('img', { name: 'Calls, reactions, audio attachments, and linked URLs by month' })).toBeVisible()
  await expect(page.getByRole('img', { name: 'Top linked-domain occurrences by month' })).toBeVisible()
  await expect(page.getByRole('img', { name: 'Attachment occurrences and bytes by month' })).toBeVisible()
  await expect(page.getByRole('img', { name: 'Bipartite graph of conversations and participant labels' })).toBeVisible()
  const storageKey = page.getByRole('list', { name: 'Storage composition key' })
  const conversationKey = page.getByRole('list', { name: 'Conversation types key' })
  await expect(storageKey).toBeVisible()
  await expect(storageKey.getByRole('listitem')).toHaveCount(1)
  await expect(conversationKey).toBeVisible()
  await expect(conversationKey.getByRole('listitem')).toHaveCount(3)
  await expect(page.getByText('5 archive-wide Facebook activity records have unknown dates and are excluded from time buckets.')).toBeVisible()
  await expect(page.getByText('Timestamp bounds are inclusive at millisecond precision. Unknown-date Facebook activity is reported archive-wide but cannot be tested against a selected range and is never assigned to a time bucket.')).toBeVisible()
  await expect(page.getByText('Rollup provenance: whole UTC hours overlapping the requested interval; local time at the UTC rollup hour start.').first()).toBeVisible()
  await expect(page.getByRole('heading', { name: 'Topic clusters' })).toBeVisible()
  await expect(page.getByRole('button', { name: 'Build topics' })).toBeDisabled()
  await expect(page.getByRole('heading', { name: 'Derived language signals' })).toBeVisible()
  await expect(page.getByRole('img', { name: 'Model-estimated sentiment by month and speaker direction' })).toBeVisible()
  await page.getByText('Inspect sentiment sample sizes and confidence').click()
  await expect(page.getByText('82.0%')).toBeVisible()
  await expect(page.getByText('Relationship ranking is disabled.')).toHaveCount(2)
  await expect(page.getByText('This is one observed export snapshot. Prior coverage snapshots were not captured, so no historical trend is inferred. Observed 2026-07-20.')).toBeVisible()
  await expect(page.getByText('Latest coverage-area catalog update time, not the export creation time.')).toBeVisible()
  await page.screenshot({ path: 'test-results/stage4-analytics-desktop.png', fullPage: true })
})

test('analytics controls requery every bucket/direction/date scope and topic build recovers from failure', async ({ page }) => {
  const observed: URL[] = []
  page.on('request', (request) => {
    const url = new URL(request.url())
    if (url.pathname.startsWith('/api/v1/analytics/')) observed.push(url)
  })
  let topicBuildAttempts = 0
  let topicsReady = false
  await page.route(/\/api\/v1\/analytics\/topics\/build(?:\?.*)?$/, (route) => {
    topicBuildAttempts += 1
    if (topicBuildAttempts === 1) return route.fulfill({ status: 503, json: { detail: 'Fixture topic build failed' } })
    topicsReady = true
    return route.fulfill({ json: { job_id: 'topic-job-one', status: 'queued', status_href: '/api/v1/jobs/topic-job-one' } })
  })
  await page.route(/\/api\/v1\/analytics\/topics(?:\?.*)?$/, (route) => route.fulfill({ json: topicsReady ? {
    status: 'ready', ready: true, reason: null, latest_job: { id: 'topic-job-one', status: 'completed', stage: 'Complete', progress_current: 1, progress_total: 1 }, semantic_index: { id: 'semantic-one', model_name: 'fixture/embedding', model_revision: 'revision-one', window_count: 12 }, result: { pipeline_version: 'fixture-topics-v1', generated_at: '2026-07-20T12:00:00Z', corpus_window_count: 12, sample_count: 12, sampling: 'Deterministic bounded fixture sample.', warning: 'Topic labels are model estimates.', clusters: [{ id: 'topic-one', ordinal: 1, sample_windows: 7, sample_share: .583, terms: ['travel', 'plans', 'weekend'], representatives: [{ id: 'window:topic-one', excerpt: 'Let us make weekend travel plans.', jump: { href: '/conversations/conv-alex?around=message-002#message-message-002' } }] }] },
  } : { status: 'unavailable', ready: false, latest_job: null, semantic_index: { id: 'semantic-one', model_name: 'fixture/embedding', model_revision: 'revision-one', window_count: 12 }, result: null, reason: 'Topic clusters have not been built.' } }))

  await page.goto('/analytics')
  await expect(page.getByRole('heading', { name: 'Archive analytics' })).toBeVisible()
  const primaryControls = page.locator('.analytics-controls')
  for (const bucket of ['day', 'week', 'year']) {
    const requested = page.waitForRequest((request) => {
      const url = new URL(request.url())
      return url.pathname === '/api/v1/analytics/messages' && url.searchParams.get('bucket') === bucket
    })
    await primaryControls.getByLabel('Bucket').selectOption(bucket)
    await requested
  }
  const timezoneRequest = page.waitForRequest((request) => {
    const url = new URL(request.url())
    return url.pathname === '/api/v1/analytics/overview' && url.searchParams.get('timezone') === 'UTC'
  })
  await primaryControls.getByLabel('Timezone').fill('UTC')
  await timezoneRequest
  await expect(page.getByLabel('Analytics overview')).toContainText('local calendar days in UTC')

  const nlpFilters = page.getByLabel('Language-model trend filters')
  for (const bucket of ['hour', 'day', 'week', 'year']) {
    const requested = page.waitForRequest((request) => {
      const url = new URL(request.url())
      return url.pathname === '/api/v1/analytics/nlp-trends' && url.searchParams.get('bucket') === bucket
    })
    await nlpFilters.getByLabel('Time bucket').selectOption(bucket)
    await requested
  }
  for (const direction of ['owner', 'other', 'activity', 'audio', 'attachment']) {
    const requested = page.waitForRequest((request) => {
      const url = new URL(request.url())
      return url.pathname === '/api/v1/analytics/nlp-trends' && url.searchParams.get('direction') === direction
    })
    await nlpFilters.getByLabel('Speaker/source').selectOption(direction)
    await requested
  }
  await nlpFilters.getByLabel('From').fill('2023-04-01')
  const nlpDates = page.waitForRequest((request) => {
    const url = new URL(request.url())
    return url.pathname === '/api/v1/analytics/nlp-trends' && url.searchParams.has('from_ms') && url.searchParams.has('to_ms')
  })
  await nlpFilters.getByLabel('Through').fill('2023-04-30')
  await nlpDates
  await expect(page.getByText(/Selected dates include overlapping whole UTC hours, so partial boundary hours are approximate/)).toBeVisible()

  const conflictFilters = page.getByLabel('Conflict rate filters')
  for (const bucket of ['hour', 'day', 'week', 'year']) {
    const requested = page.waitForRequest((request) => {
      const url = new URL(request.url())
      return url.pathname === '/api/v1/analytics/conflict-rate' && url.searchParams.get('bucket') === bucket
    })
    await conflictFilters.getByLabel('Conflict time bucket').selectOption(bucket)
    await requested
  }
  await conflictFilters.getByLabel('From').fill('2023-05-01')
  const conflictDates = page.waitForRequest((request) => {
    const url = new URL(request.url())
    return url.pathname === '/api/v1/analytics/conflict-rate' && url.searchParams.has('from_ms') && url.searchParams.has('to_ms')
  })
  await conflictFilters.getByLabel('Through').fill('2023-05-31')
  await conflictDates
  await expect(page.getByText(/Selected dates can include partial boundary hours/)).toBeVisible()

  const charts = page.locator('.archive-chart')
  expect(await charts.count()).toBeGreaterThan(10)
  for (let index = 0; index < await charts.count(); index += 1) {
    const chart = charts.nth(index)
    await expect(chart).toHaveAttribute('role', 'img')
    await expect(chart).not.toHaveAccessibleName('')
    await expect(chart).not.toHaveAccessibleDescription('')
  }
  await page.getByText('Inspect sentiment sample sizes and confidence').click()
  await expect(page.getByRole('table').filter({ hasText: 'Mean confidence' })).toContainText('82.0%')
  await page.getByText('Inspect conflict-rate samples').click()
  await expect(page.getByRole('table').filter({ hasText: 'Estimated rate' })).toContainText('15.0%')

  await expect(page.getByText('Topic clusters have not been built.')).toBeVisible()
  await page.getByRole('button', { name: 'Build topics' }).click()
  await expect(page.getByRole('alert')).toContainText('Fixture topic build failed')
  const successfulBuild = page.waitForRequest((request) => request.method() === 'POST' && new URL(request.url()).pathname === '/api/v1/analytics/topics/build')
  await page.getByRole('button', { name: 'Build topics' }).click()
  await successfulBuild
  await expect(page.getByRole('button', { name: 'Rebuild topics' })).toBeVisible()
  await expect(page.getByText('travel · plans · weekend')).toBeVisible()
  await expect(page.getByRole('link', { name: 'Jump to source' })).toHaveAttribute('href', '/conversations/conv-alex?around=message-002#message-message-002')

  expect(new Set(observed.filter((url) => url.pathname === '/api/v1/analytics/messages').map((url) => url.searchParams.get('bucket')))).toEqual(new Set(['month', 'day', 'week', 'year']))
  expect(new Set(observed.filter((url) => url.pathname === '/api/v1/analytics/nlp-trends').map((url) => url.searchParams.get('direction')))).toEqual(new Set(['all', 'owner', 'other', 'activity', 'audio', 'attachment']))
  await page.getByRole('link', { name: 'Jump to source' }).click()
  await expect(page).toHaveURL(/\/conversations\/conv-alex\?around=message-002#message-message-002$/)
})

test('analytics exposes an aggregate service error and recovers through its retry action', async ({ page }) => {
  let attempts = 0
  await page.route(/\/api\/v1\/analytics\/overview(?:\?.*)?$/, (route) => {
    if (attempts++ < 2) return route.fulfill({ status: 503, json: { detail: 'Fixture analytics overview unavailable' } })
    const url = new URL(route.request().url())
    return route.fulfill({ json: { coverage: { ...partialCoverage, key: 'messages', label: 'Messages', status: 'available' }, timezone: url.searchParams.get('timezone') ?? 'America/New_York', messages: { total: 240, sent: 100, received: 140, first_timestamp_ms: 1, last_timestamp_ms: 2, calendar_days: 60, average_per_calendar_day: 4, active_conversations: 4 }, interactions: { message_reactions: 24, reaction_types: { heart: 1 }, calls: 3, missed_calls: 1, call_duration_seconds: 360, audio_attachments: 4 }, attachments: { occurrences: 18, occurrence_bytes: 1_000_000 }, activity_records: 12, link_domains: [{ domain: 'example.com', occurrences: 6 }], derived_nlp: nlp, provenance: 'SQLite rows' } })
  })
  await page.goto('/analytics')
  await expect(page.getByRole('alert')).toContainText('Fixture analytics overview unavailable')
  await page.getByRole('button', { name: 'Try again' }).click()
  await expect(page.getByLabel('Analytics overview')).toContainText('240')
})

test('zero-result language-model date ranges keep their controls available for recovery', async ({ page }) => {
  await page.route(/\/api\/v1\/analytics\/nlp-trends(?:\?.*)?$/, (route) => {
    const url = new URL(route.request().url())
    return route.fulfill({ json: { timezone: url.searchParams.get('timezone') ?? 'America/New_York', bucket: url.searchParams.get('bucket') ?? 'month', direction: url.searchParams.get('direction') ?? 'all', source_sample_count: 0, page_sample_count: 0, models: {}, series: { sentiment: [], emotion: [], toxicity: [] }, hour_of_day: { sentiment: [], emotion: [], toxicity: [] }, hour_of_day_scope: 'current_page', page: { total: null, limit: 24, returned: 0, next_cursor: null, count_is_unknown: true }, availability: nlp, rollup_provenance: { source_resolution: 'UTC hour', range_filter_semantics: 'whole UTC hours overlapping the requested interval', local_bucket_label: 'local time at the UTC rollup hour start', partial_boundary_hours_included: url.searchParams.has('from_ms') || url.searchParams.has('to_ms'), non_whole_hour_timezone_offset_observed: false }, warnings: ['No people are ranked.'] } })
  })
  await page.goto('/analytics')
  const filters = page.getByLabel('Language-model trend filters')
  await expect(filters).toBeVisible()
  await filters.getByLabel('From').fill('2035-01-01')
  const ranged = page.waitForRequest((request) => {
    const url = new URL(request.url())
    return url.pathname === '/api/v1/analytics/nlp-trends' && url.searchParams.has('from_ms') && url.searchParams.has('to_ms')
  })
  await filters.getByLabel('Through').fill('2035-01-31')
  await ranged
  await expect(page.getByText('No versioned language-model annotations match these filters. Clear or widen the date and source filters to continue.')).toBeVisible()
  await expect(filters.getByLabel('From')).toHaveValue('2035-01-01')
  await expect(filters.getByLabel('Through')).toHaveValue('2035-01-31')
  await filters.getByLabel('From').fill('')
  await filters.getByLabel('Through').fill('')
  await expect(filters.getByLabel('From')).toHaveValue('')
  await expect(filters.getByLabel('Through')).toHaveValue('')
  const recovered = page.waitForRequest((request) => {
    const url = new URL(request.url())
    return url.pathname === '/api/v1/analytics/nlp-trends' && url.searchParams.get('direction') === 'owner' && !url.searchParams.has('from_ms') && !url.searchParams.has('to_ms')
  })
  await filters.getByLabel('Speaker/source').selectOption('owner')
  await recovered
  await expect(filters).toBeVisible()
})

test('support browser opens retained support-message conversations', async ({ page }) => {
  await page.goto('/support')
  await expect(page.locator('.topbar__title > span')).toHaveText('Stage 4')
  await expect(page.getByRole('heading', { name: 'Support conversations' })).toBeVisible()
  await expect(page.getByRole('heading', { name: 'Account access case' })).toBeVisible()
  await expect(page.getByRole('link', { name: 'Open conversation' })).toHaveAttribute('href', '/conversations/support-one')
  const pages = page.getByRole('navigation', { name: 'Support conversations pages' })
  await pages.getByRole('button', { name: 'Next page' }).click()
  await expect(page.getByRole('heading', { name: 'Billing appeal case' })).toBeVisible()
  await pages.getByRole('button', { name: 'Previous page' }).click()
  await expect(page.getByRole('heading', { name: 'Account access case' })).toBeVisible()
  await page.screenshot({ path: 'test-results/stage4-support-desktop.png', fullPage: true })
  await page.getByRole('link', { name: 'Open conversation' }).click()
  await expect(page).toHaveURL(/\/conversations\/support-one$/)
})

test('support distinguishes unavailable history from a retryable service error', async ({ page }) => {
  let attempts = 0
  await page.route(/\/api\/v1\/support(?:\?.*)?$/, (route) => {
    if (attempts++ < 2) return route.fulfill({ status: 503, json: { detail: 'Fixture support service unavailable' } })
    return route.fulfill({ json: { coverage: { key: 'support_messages', label: 'Support messages', status: 'unavailable', raw_count: null, logical_count: null, description: 'No support-message source was retained; this does not prove zero history.', source_paths: {}, absence_is_not_zero_history: true }, items: [], page: { total: null, limit: 50, next_cursor: null, count_is_unknown: true } } })
  })
  await page.goto('/support')
  await expect(page.getByRole('alert')).toContainText('Fixture support service unavailable')
  await page.getByRole('button', { name: 'Try again' }).click()
  await expect(page.getByRole('heading', { name: 'Support conversations' })).toBeVisible()
  await expect(page.getByText('Support-message history unavailable', { exact: true })).toBeVisible()
  await expect(page.getByText('No support-message source was retained; this does not prove zero history.').first()).toBeVisible()
})

test('cursor windows retain keyboard focus and truthful busy state through success, 503, and back recovery', async ({ page }) => {
  test.setTimeout(120_000)
  const scenarios = [
    {
      key: 'activity',
      path: '/activity',
      endpoint: '/api/v1/activity',
      pagesLabel: 'Activity results pages',
      itemLabel: (pageNumber: number) => `Activity focus page ${pageNumber}`,
      payload: (pageNumber: number) => activityPage(
        pageNumber < 3 ? `activity-focus-${pageNumber + 1}` : null,
        [{ ...activityItem, id: `activity-focus-${pageNumber}`, title: `Activity focus page ${pageNumber}` }],
        225,
      ),
    },
    {
      key: 'timeline',
      path: '/timeline',
      endpoint: '/api/v1/timeline',
      pagesLabel: 'Timeline results pages',
      itemLabel: (pageNumber: number) => `Timeline focus page ${pageNumber}`,
      payload: (pageNumber: number) => ({
        coverage: { status: 'source_backed', absence_is_not_zero_history: false },
        items: [{
          ...activityItem,
          id: `activity:timeline-focus-${pageNumber}`,
          entity_type: 'activity',
          entity_id: `timeline-focus-${pageNumber}`,
          title: `Timeline focus page ${pageNumber}`,
        }],
        page: {
          total: 225,
          limit: 75,
          next_cursor: pageNumber < 3 ? `timeline-focus-${pageNumber + 1}` : null,
          count_is_unknown: false,
        },
      }),
    },
    {
      key: 'people',
      path: '/people',
      endpoint: '/api/v1/people',
      pagesLabel: 'People results pages',
      itemLabel: (pageNumber: number) => `Person focus page ${pageNumber}`,
      payload: (pageNumber: number) => contactsPage(
        pageNumber < 3 ? `people-focus-${pageNumber + 1}` : null,
        [{ ...contactItem, id: `person-focus-${pageNumber}`, name: `Person focus page ${pageNumber}` }],
        300,
      ),
    },
    {
      key: 'events',
      path: '/events',
      endpoint: '/api/v1/events',
      pagesLabel: 'Event results pages',
      itemLabel: (pageNumber: number) => `Event focus page ${pageNumber}`,
      payload: (pageNumber: number) => eventsPage(
        pageNumber < 3 ? `events-focus-${pageNumber + 1}` : null,
        [{ ...eventItem, id: `event-focus-${pageNumber}`, title: `Event focus page ${pageNumber}` }],
        225,
      ),
    },
  ] as const

  for (const viewport of [{ width: 1440, height: 1000 }, { width: 390, height: 844 }]) {
    await page.setViewportSize(viewport)
    for (const scenario of scenarios) {
      let releaseSecondPage: (() => void) | undefined
      let holdSecondPage = true
      const secondPageGate = new Promise<void>((resolve) => {
        releaseSecondPage = resolve
      })
      const endpointPattern = new RegExp(`${scenario.endpoint.replaceAll('/', '\\/')}(?:\\?.*)?$`)
      await page.route(endpointPattern, async (route) => {
        const url = new URL(route.request().url())
        const cursor = url.searchParams.get('cursor')
        const pageNumber = cursor ? Number(cursor.replace(`${scenario.key}-focus-`, '')) : 1
        if (pageNumber === 2 && holdSecondPage) await secondPageGate
        if (pageNumber === 3) {
          return route.fulfill({
            status: 503,
            json: { detail: `Fixture ${scenario.key} focus page unavailable` },
          })
        }
        return route.fulfill({ json: scenario.payload(pageNumber) })
      })

      await page.goto(scenario.path)
      await expect(page.getByText(scenario.itemLabel(1), { exact: true })).toBeVisible()
      const pager = page.getByLabel(scenario.pagesLabel)
      const previous = pager.getByRole('button', { name: 'Previous page' })
      const next = pager.getByRole('button', { name: 'Next page' })
      await expect(previous).toHaveAttribute('aria-disabled', 'true')
      expect(await previous.getAttribute('disabled')).toBeNull()

      await next.focus()
      const secondRequest = page.waitForRequest((request) => (
        new URL(request.url()).searchParams.get('cursor') === `${scenario.key}-focus-2`
      ))
      await next.click()
      await secondRequest
      await expect(pager).toHaveAttribute('aria-busy', 'true')
      await expect(pager).toContainText('Loading page 2')
      await expect(pager).toContainText('1 current results remain shown')
      await expect(next).toHaveAttribute('aria-disabled', 'true')
      await expect(next).toBeFocused()
      holdSecondPage = false
      releaseSecondPage?.()

      await expect(page.getByText(scenario.itemLabel(2), { exact: true })).toBeVisible()
      await expect(pager).toHaveAttribute('aria-busy', 'false')
      await expect(pager).toContainText('Page 2')
      await expect(next).toBeFocused()

      const failedResponse = page.waitForResponse((response) => (
        new URL(response.url()).searchParams.get('cursor') === `${scenario.key}-focus-3`
        && response.status() === 503
      ))
      await next.click()
      await failedResponse
      await expect(page.getByRole('alert')).toContainText(`Fixture ${scenario.key} focus page unavailable`)
      await expect(pager).toContainText('Page 3')
      await expect(next).toHaveAttribute('aria-disabled', 'true')
      await expect(next).toBeFocused()

      await page.keyboard.press('Shift+Tab')
      await expect(previous).toBeFocused()
      await previous.click()
      await expect(page.getByText(scenario.itemLabel(2), { exact: true })).toBeVisible()
      await expect(pager).toContainText('Page 2')
      await expect(previous).toBeFocused()

      const widths = await page.locator('body').evaluate((element) => ({
        scroll: element.scrollWidth,
        client: element.clientWidth,
      }))
      expect(widths.scroll, `${scenario.path} recovery at ${viewport.width}px`).toBeLessThanOrEqual(widths.client + 1)
      const result = await new AxeBuilder({ page })
        .withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'wcag22a', 'wcag22aa'])
        .analyze()
      expect(result.violations, `${scenario.path} recovery at ${viewport.width}px`).toEqual([])
      expect(result.incomplete, `${scenario.path} recovery at ${viewport.width}px`).toEqual([])
      await page.unroute(endpointPattern)
    }
  }
})

test('Stage 4 pages remain overflow-free and WCAG 2.2 axe-clean at desktop and mobile sizes', async ({ page }) => {
  test.setTimeout(60_000)
  for (const viewport of [{ width: 1440, height: 1000 }, { width: 390, height: 844 }]) {
    await page.setViewportSize(viewport)
    for (const path of ['/people', '/events', '/activity', '/timeline', '/support', '/analytics']) {
      await page.goto(path)
      await expect(page.locator('main h1')).toBeVisible()
      if (path === '/activity' && viewport.width === 390) {
        await expect(page.getByText(/Swipe to see more categories/)).toBeVisible()
        const activityViewport = page.getByLabel('Scrollable virtualized activity results')
        await expect(page.locator('#activity-results-scroll-hint')).toBeVisible()
        await expect(activityViewport).toHaveAttribute('aria-describedby', 'activity-results-scroll-hint')
        expect(await activityViewport.evaluate((element) => getComputedStyle(element).scrollbarGutter)).toContain('stable')
        const actionOrder = await page.locator('.activity-table tbody tr').first().evaluate((row) => {
          const primary = row.querySelector<HTMLElement>('td:first-child')
          const jump = row.querySelector<HTMLElement>('.activity-table__jump')
          const source = row.querySelector<HTMLElement>('.activity-table__source')
          if (!primary || !jump || !source) throw new Error('Expected activity primary, jump, and source cells')
          return {
            primaryBottom: primary.getBoundingClientRect().bottom,
            jumpTop: jump.getBoundingClientRect().top,
            jumpBottom: jump.getBoundingClientRect().bottom,
            sourceTop: source.getBoundingClientRect().top,
          }
        })
        expect(actionOrder.jumpTop).toBeGreaterThanOrEqual(actionOrder.primaryBottom - 1)
        expect(actionOrder.sourceTop).toBeGreaterThanOrEqual(actionOrder.jumpBottom - 1)
      }
      if (path === '/timeline' && viewport.width === 390) {
        const timelineViewport = page.getByRole('feed', { name: 'Scrollable virtualized timeline' })
        await expect(page.locator('#timeline-results-scroll-hint')).toBeVisible()
        await expect(timelineViewport).toHaveAttribute('aria-describedby', 'timeline-results-scroll-hint')
        expect(await timelineViewport.evaluate((element) => getComputedStyle(element).scrollbarGutter)).toContain('stable')
        const actionOrder = await page.locator('.timeline-entry__body').first().evaluate((entry) => {
          const footer = entry.querySelector<HTMLElement>('footer')
          const body = entry.querySelector<HTMLElement>(':scope > p')
          if (!footer || !body) throw new Error('Expected timeline action footer and body')
          return { footerTop: footer.getBoundingClientRect().top, bodyTop: body.getBoundingClientRect().top }
        })
        expect(actionOrder.footerTop).toBeLessThan(actionOrder.bodyTop)
      }
      if (path === '/people' && viewport.width === 390) {
        await expect(page.getByText(/Swipe to see more relationships/)).toBeVisible()
        const peopleTabs = page.getByRole('navigation', { name: 'People view' })
        const tabMetrics = await peopleTabs.evaluate((element) => {
          const style = getComputedStyle(element)
          return {
            client: element.clientWidth,
            scroll: element.scrollWidth,
            overflowX: style.overflowX,
            rightEdge: Number.parseFloat(style.borderRightWidth),
          }
        })
        expect(tabMetrics.scroll).toBeGreaterThan(tabMetrics.client)
        expect(tabMetrics.overflowX).toBe('auto')
        expect(tabMetrics.rightEdge).toBeGreaterThanOrEqual(3)
        const firstTabHeight = await peopleTabs.getByRole('button').first().evaluate(
          (button) => button.getBoundingClientRect().height,
        )
        expect(firstTabHeight).toBeGreaterThanOrEqual(44)
      }
      const widths = await page.locator('body').evaluate((element) => ({ scroll: element.scrollWidth, client: element.clientWidth }))
      expect(widths.scroll, `${path} at ${viewport.width}px`).toBeLessThanOrEqual(widths.client + 1)
      const result = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'wcag22a', 'wcag22aa']).analyze()
      expect(result.violations, `${path} at ${viewport.width}px: ${result.violations.map((violation) => `${violation.id}:${violation.nodes.length}`).join(', ')}`).toEqual([])
      expect(result.incomplete, `${path} at ${viewport.width}px: incomplete ${result.incomplete.map((issue) => `${issue.id}:${issue.nodes.length}`).join(', ')}`).toEqual([])
    }
  }
  for (const route of [
    { path: '/people', slug: 'people', pages: 'People results pages' },
    { path: '/events', slug: 'events', pages: 'Event results pages' },
    { path: '/activity', slug: 'activity', pages: 'Activity results pages' },
    { path: '/timeline', slug: 'timeline', pages: 'Timeline results pages' },
  ]) {
    await page.goto(route.path)
    await expect(page.getByLabel(route.pages)).toBeVisible()
    await page.screenshot({ path: `test-results/stage4-${route.slug}-pagination-mobile.png`, fullPage: true })
  }
  await page.goto('/analytics')
  await expect(page.getByRole('heading', { name: 'Archive analytics' })).toBeVisible()
  await expect(page.getByLabel('Analytics overview')).toBeVisible()
  await expect(page.locator('details.analytics-drilldown')).toHaveCount(7)
  await expect(page.locator('details.analytics-drilldown[open]')).toHaveCount(1)
  await expect(page.getByLabel('Language-model trend filters')).toHaveCount(0)
  await page.getByText('Language-model signals').click()
  await expect(page.getByLabel('Language-model trend filters')).toBeVisible()
  await expect(page.locator('details.analytics-drilldown[open]')).toHaveCount(2)
  await page.screenshot({ path: 'test-results/stage4-analytics-mobile.png', fullPage: true })
  await page.goto('/support')
  await expect(page.getByRole('heading', { name: 'Support conversations' })).toBeVisible()
  await page.screenshot({ path: 'test-results/stage4-support-mobile.png', fullPage: true })
})

test('Stage 4 analytics has deterministic visual and accessibility baselines', async ({ page }) => {
  await page.emulateMedia({ reducedMotion: 'reduce' })
  await page.goto('/analytics')
  const main = page.getByRole('main')
  await expect(main.getByRole('heading', { name: 'Archive analytics' })).toBeVisible()
  await page.addStyleTag({ content: 'canvas { visibility: hidden !important; }' })
  await expect(page).toHaveScreenshot('stage4-analytics-viewport.png', { animations: 'disabled', timeout: 15_000 })
  await expect(main.locator('.analytics-conflict-rate')).toMatchAriaSnapshot({ name: 'stage4-conflict-rate.aria.yml' })
  await expect(main).toMatchAriaSnapshot({ name: 'stage4-analytics-main.aria.yml' })
})

test('every remaining Stage 4 route has deterministic visual and ARIA baselines', async ({ page }) => {
  test.setTimeout(60_000)
  await page.emulateMedia({ reducedMotion: 'reduce' })
  for (const route of [
    { path: '/people', slug: 'people' },
    { path: '/events', slug: 'events' },
    { path: '/activity', slug: 'activity' },
    { path: '/timeline', slug: 'timeline' },
    { path: '/support', slug: 'support' },
  ]) {
    await page.goto(route.path)
    await page.addStyleTag({ content: '.topbar { visibility: hidden !important; }' })
    const main = page.getByRole('main')
    await expect(main.getByRole('heading', { level: 1 })).toBeVisible()
    await expect(main).toHaveScreenshot(`stage4-${route.slug}-main.png`, { animations: 'disabled', timeout: 15_000 })
    await expect(main).toMatchAriaSnapshot({ name: `stage4-${route.slug}-main.aria.yml` })
  }
})
