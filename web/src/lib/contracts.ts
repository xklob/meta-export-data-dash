export type HealthStatus = 'ok' | 'degraded' | 'error' | string

export interface HealthResponse {
  status: HealthStatus
  version?: string
  catalog_ready?: boolean
  catalog_state?: string
}

export interface Job {
  id: string
  kind: string
  status: string
  progress_current?: number | null
  progress_total?: number | null
  stage?: string | null
  counters?: Record<string, number> | null
  started_at?: string | null
  finished_at?: string | null
  error?: string | null
}

export interface OverviewResponse {
  archive?: {
    root_name?: string
    chunk_count?: number
    source_file_count?: number
    total_bytes?: number
    json_file_count?: number
    json_valid_count?: number
    json_invalid_count?: number
    json_pending_count?: number
    json_unsupported_count?: number
  }
  content?: {
    conversation_count?: number
    message_count?: number
    attachment_count?: number
    activity_count?: number
  }
  integrity?: {
    unresolved_anomalies?: number
    critical_anomalies?: number
    missing_attachments?: number
    unlinked_media?: number
  }
  latest_job?: Job | null
  lifecycle?: {
    state?: string
    ready?: boolean
    active_job_count?: number
    pending_json_count?: number
    invalid_json_count?: number
    verification_check_count?: number
    verification_pending_count?: number
    verification_failed_count?: number
  }
}

export type CoverageStatus = 'available' | 'partial' | 'unavailable' | 'unknown' | string

export interface CoverageItem {
  key: string
  label: string
  status: CoverageStatus
  raw_count?: number | null
  logical_count?: number | null
  description?: string | null
  updated_at?: string | null
}

export interface CoverageResponse {
  items?: CoverageItem[]
}

export type IntegritySeverity = 'critical' | 'warning' | 'info' | string

export interface IntegrityIssue {
  id: string
  code: string
  severity: IntegritySeverity
  summary: string
  detail?: unknown
  resolved?: boolean
  created_at?: string | null
}

export interface IntegrityCheck {
  key: string
  status: string
  expected?: unknown
  actual?: unknown
  detail?: string | null
}

export interface IntegrityResponse {
  summary?: {
    total?: number
    unresolved?: number
    critical?: number
    warning?: number
    info?: number
  }
  items?: IntegrityIssue[]
  checks?: IntegrityCheck[]
  page?: PageMetadata
}

export interface JobsResponse {
  items?: Job[]
  page?: PageMetadata & { total?: number }
  summary?: {
    running?: number
    completed?: number
    failed?: number
    total?: number
  }
}

export interface PageMetadata {
  limit?: number
  offset?: number
  filtered_total?: number
}

export interface SettingsResponse {
  timezone?: string
  owner_identity?: string
  cloud_enabled?: boolean
  image_semantic_stage_enabled?: false | boolean
  archive_root_display?: string
  access_mode?: 'localhost-only' | string
}

export interface SettingsPatch {
  timezone: string
  cloud_enabled: boolean
}

export interface ApiProblemDetail {
  code: string
  message: string
  [key: string]: unknown
}

export interface ApiProblem {
  detail?: string | ApiProblemDetail
  message?: string
  title?: string
}

export type SortDirection = 'asc' | 'desc'

export type StarTargetType =
  | 'conversation'
  | 'message'
  | 'activity'
  | 'attachment'
  | 'blob'
  | 'messenger_contact'

export type StarCategory = 'all' | 'conversations' | 'messages' | 'media' | 'activity' | 'people'

export interface StarTarget {
  target_type: StarTargetType
  target_id: string
}

export interface StarState {
  id: string
  starred_at: string
}

/**
 * Star metadata is optional while an older catalog response is still cached.
 * New API responses include both fields for every canonical star target.
 */
export interface Starrable {
  star_target?: StarTarget
  star?: StarState | null
}

export interface CursorPageMetadata {
  limit: number
  total: number
  next_cursor?: string | null
}

export interface MessageCounts {
  total: number
  sent: number
  received: number
}

export interface AttachmentCounts {
  occurrences: number
  unique: number
}

export interface MediaCounts {
  photo: number
  video: number
  audio: number
  gif: number
  sticker: number
  document: number
  other: number
}

export interface ConversationContentProfile {
  kind: 'messages_and_media' | 'messages_only' | 'media_only' | 'metadata_only'
  message_records: 'available' | 'not_present'
  attachment_records: 'available' | 'not_present'
}

export interface ConversationSummary extends Starrable {
  id: string
  title: string
  type: string
  primary_category?: string | null
  source_categories: string[]
  participant_count: number
  first_activity_ms?: number | null
  last_activity_ms?: number | null
  message_counts: MessageCounts
  content_profile: ConversationContentProfile
  attachment_counts: AttachmentCounts
  media_counts: MediaCounts
  attachment_bytes: number
  active_days: number
  completeness: string
  segment_count: number
  identity?: {
    status: 'provisional' | 'corrected' | 'source' | string
    basis: string
    corrected_segment_count: number
    can_correct: boolean
  }
  href: string
}

export interface ConversationsResponse {
  items: ConversationSummary[]
  page: CursorPageMetadata
  sort: { field: string; direction: SortDirection }
}

export interface ConversationParticipant {
  id: string
  name: string
  is_owner: boolean
  ordinal: number
}

export interface ConversationSegment {
  id: string
  title: string
  thread_path: string
  source_category: string
  sequence: number
  source_chunk_count: number
  is_still_participant?: boolean | null
  first_activity_ms?: number | null
  last_activity_ms?: number | null
  completeness: string
  assignment?: {
    status: 'provisional' | 'corrected' | string
    correction_id?: string | null
    base_conversation_id: string
  }
  participants: ConversationParticipant[]
}

export interface ConversationDetailResponse {
  conversation: ConversationSummary
  segments: ConversationSegment[]
}

export interface ConversationCorrectionSegment {
  id: string
  thread_path: string
  before_conversation_id: string
  after_conversation_id: string
  previous_correction_id?: string | null
}

export interface ConversationCorrection {
  id: string
  action: 'merge' | 'split' | string
  status: 'active' | 'reverted' | string
  reason?: string | null
  request: Record<string, unknown>
  before: { conversations?: Array<Record<string, unknown>> }
  after: { conversations?: Array<Record<string, unknown>> }
  effect: Record<string, unknown>
  created_at: string
  reverted_at?: string | null
  can_revert: boolean
  segments: ConversationCorrectionSegment[]
  audits: Array<{ id: string; action: string; detail: Record<string, unknown>; created_at: string }>
}

export interface ConversationCorrectionsResponse {
  items: ConversationCorrection[]
  identity_revision: string
  page: { limit: number; offset: number; total: number; next_offset?: number | null }
  notice: string
}

export interface ConversationCorrectionResult {
  correction_id: string
  action: string
  status: string
  identity_revision: string
  redirect_conversation_id: string
  effect: Record<string, unknown>
}

export interface ConversationJump {
  href: string
  conversation_id?: string
  message_id?: string
  activity_id?: string
  fragment?: string
}

export interface MediaDerivative {
  id: string
  kind: 'image_thumbnail' | 'video_poster' | 'audio_waveform'
  status: string
  media_type?: string | null
  size_bytes?: number | null
  duration_ms?: number | null
  width?: number | null
  height?: number | null
  href?: string | null
}

export interface ArchiveAttachment extends Starrable {
  id: string
  kind: string
  filename?: string | null
  media_type?: string | null
  size_bytes?: number | null
  timestamp_ms?: number | null
  availability: string
  blob_sha256?: string | null
  blob_href?: string | null
  is_external_reference?: boolean
  jump?: ConversationJump | null
  derivatives?: MediaDerivative[]
}

export interface MessageReaction {
  id: string
  actor: string
  reaction: string
  timestamp_ms?: number | null
}

export interface ConversationMessage extends Starrable {
  id: string
  conversation_id: string
  segment?: {
    id: string
    title: string
    source_category: string
    completeness: string
  } | null
  timestamp_ms: number
  sender: string
  sender_is_owner?: boolean | null
  content?: string | null
  transcription?: string | null
  kind: string
  call?: { duration_seconds?: number | null; missed?: boolean | null } | null
  has_payment_details: boolean
  has_sensitive_ip: boolean
  is_unsent: boolean
  is_unavailable_body: boolean
  is_duplicate: boolean
  duplicate_of_id?: string | null
  reactions: MessageReaction[]
  attachments: ArchiveAttachment[]
  anchor: ConversationJump
  matched_fields?: string[]
}

export interface ConversationMessagesResponse {
  items: ConversationMessage[]
  page: {
    limit: number
    total: number
    before_cursor?: string | null
    after_cursor?: string | null
  }
  focus?: ConversationJump | null
}

export interface ConversationSearchResponse {
  items: ConversationMessage[]
  page: CursorPageMetadata
  query: string
}

export interface ConversationDateJumpResponse {
  requested_timestamp_ms: number
  target_timestamp_ms?: number | null
  jump?: ConversationJump | null
}

export interface MessageAdvancedResponse {
  message_id: string
  has_sensitive_details: boolean
  sensitive: Record<string, unknown>
  notice: string
}

export type MediaView = 'occurrences' | 'unique'

export interface MediaProvenance {
  source_file_id?: number | null
  source_chunk?: string | null
  source_relative_path?: string | null
  source_path?: string | null
  source_copy_count: number
  library_path?: string | null
  explanation: string
}

export interface MediaItem extends ArchiveAttachment {
  view_key: string
  type: string
  unknown_date: boolean
  first_timestamp_ms?: number | null
  last_timestamp_ms?: number | null
  occurrence_count?: number
  conversation?: { id: string; title?: string | null } | null
  activity?: {
    id: string
    title?: string | null
    category: string
    subtype: string
  } | null
  source_category?: string | null
  message_id?: string | null
  occurrences_href?: string | null
  provenance?: MediaProvenance | null
}

export interface MediaResponse {
  items: MediaItem[]
  page: CursorPageMetadata
  view: MediaView
  sort: { field: string; direction: SortDirection }
  summary: {
    occurrences: number
    unique_files: number
    occurrence_bytes: number
    unique_bytes: number
  }
  facets: {
    availability: Record<string, number>
    type: Record<string, number>
  }
}

export interface BlobOccurrencesResponse {
  blob_sha256: string
  items: MediaItem[]
  page: CursorPageMetadata
}

export interface CoverageSummary {
  key?: string
  label?: string
  status: string
  raw_count?: number | null
  logical_count?: number | null
  description?: string
  source_paths?: Record<string, unknown>
  absence_is_not_zero_history: boolean
}

export interface NullableCursorPageMetadata {
  limit: number
  total: number | null
  next_cursor?: string | null
  count_is_unknown?: boolean
}

export interface SourceReference {
  source_file_id: number
  source_record_id?: string | null
  chunk: string
  relative_path: string
  json_pointer?: string | null
}

export interface ActivityItem extends Starrable {
  id: string
  parent_id?: string | null
  category: string
  subtype: string
  timestamp_ms?: number | null
  unknown_date: boolean
  title?: string | null
  content?: string | null
  actor?: string | null
  availability: string
  target_uri?: string | null
  attachment_count: number
  post_context?: string | null
  post_context_basis?: string | null
  source: SourceReference
  jump: ConversationJump
}

export interface ActivityResponse {
  coverage: CoverageSummary
  coverage_components: CoverageSummary[]
  items: ActivityItem[]
  page: NullableCursorPageMetadata
  post_context_coverage?: {
    status: string
    requested: string[]
    counts: Record<string, number>
    classification: string
    description: string
    absence_is_not_zero_history: boolean
  }
}

export interface AroundResolution {
  requested: string
  resolved: boolean
  entity_type?: string
  entity_id?: string
  event_id?: string
  relationship_id?: string
}

export interface TimelineItem extends Starrable {
  id: string
  entity_type: 'message' | 'activity'
  entity_id: string
  timestamp_ms?: number | null
  unknown_date: boolean
  title?: string | null
  content?: string | null
  category: string
  subtype: string
  sender?: string | null
  local_date?: string
  year?: number
  source: SourceReference
  jump: ConversationJump
}

export interface TimelineResponse {
  coverage: CoverageSummary
  items: TimelineItem[]
  page: NullableCursorPageMetadata
  around?: AroundResolution | null
}

export interface OnThisDayResponse {
  month: number
  day: number
  timezone: string
  items: TimelineItem[]
  limit: number
  page: NullableCursorPageMetadata
}

export interface EventItem extends Starrable {
  id: string
  title?: string | null
  description?: string | null
  timestamp_ms?: number | null
  unknown_date: boolean
  status: string
  hosted?: boolean | null
  source: SourceReference
  jump: ConversationJump
}

export interface EventsResponse {
  coverage: CoverageSummary
  items: EventItem[]
  summary: {
    by_status: Record<string, number> | null
    hosted: number | null
    attended: number | null
  }
  page: NullableCursorPageMetadata
  around?: AroundResolution | null
}

export interface ContactCounts {
  messages: number
  photos: number
  videos: number
  audio: number
}

export interface ContactConversation {
  id: string
  title: string
  type: string
  message_count: number
  first_activity_ms?: number | null
  last_activity_ms?: number | null
  href: string
}

export interface PersonItem extends Starrable {
  id: string
  name?: string | null
  contact_basis?: string
  is_friend?: boolean | null
  friendship_status?: string
  status?: string
  timestamp_ms?: number | null
  counts?: ContactCounts
  direct_counts?: ContactCounts
  shared_group_counts?: ContactCounts
  combined_thread_counts?: ContactCounts
  conversation_count?: number
  direct_conversation_count?: number
  group_conversation_count?: number
  first_activity_ms?: number | null
  last_activity_ms?: number | null
  conversations?: ContactConversation[]
  count_semantics?: string
  messenger_contact_match?: {
    contact_id: string
    match_basis: 'exact_normalized_display_label' | string
    authoritative_identity: false
  }
  source?: SourceReference
  jump?: ConversationJump
}

export interface PeopleResponse {
  coverage: CoverageSummary
  authority: string
  contacts_are_not_included?: boolean
  messenger_contact_matches_are_provisional?: boolean
  items: PersonItem[]
  page: NullableCursorPageMetadata
  around?: AroundResolution | null
  filters?: Record<string, unknown>
}

export interface SupportThread extends Starrable {
  id: string
  title: string
  message_count: number
  first_activity_ms?: number | null
  last_activity_ms?: number | null
  href: string
}

export interface SupportResponse {
  coverage: CoverageSummary
  items: SupportThread[]
  page: NullableCursorPageMetadata
}

export interface StarredItem {
  kind: 'conversation' | 'message' | 'media' | 'activity' | 'person'
  star_target: StarTarget
  star: StarState
  title: string
  subtitle?: string | null
  excerpt?: string | null
  timestamp_ms?: number | null
  jump?: ConversationJump | null
  href?: string | null
  media_scope?: 'occurrence' | 'unique'
  media?: MediaItem | null
  sender?: string | null
  conversation?: (ConversationSummary & { title: string }) | { id: string; title?: string | null } | null
  counts?: ContactCounts | null
  conversations?: ContactConversation[]
  provisional?: boolean
}

export interface StarsResponse {
  items: StarredItem[]
  page: CursorPageMetadata
  category: StarCategory
  direction: SortDirection
  facets?: Partial<Record<StarCategory, number>>
}

export interface NlpAvailability {
  status: string
  reason: string
  relationship_ranking_enabled?: boolean
}

export type DerivedNlpAvailability = Record<string, NlpAvailability>

export interface AnalyticsOverviewResponse {
  coverage: CoverageSummary
  timezone: string
  messages: {
    total: number
    sent: number
    received: number
    first_timestamp_ms?: number | null
    last_timestamp_ms?: number | null
    calendar_days: number | null
    average_per_calendar_day: number | null
    active_conversations: number
  }
  interactions: {
    message_reactions: number
    reaction_types: Record<string, number>
    calls: number
    missed_calls: number
    call_duration_seconds: number
    audio_attachments: number
  }
  attachments: { occurrences: number; occurrence_bytes: number }
  activity_records: number
  link_domains: { domain: string; occurrences: number }[]
  derived_nlp: DerivedNlpAvailability
  provenance: string
}

export interface AnalyticsSeriesPage {
  limit: number
  returned: number
  total: number | null
  next_cursor?: string | null
  count_is_unknown?: boolean
}

export interface AnalyticsMessagesResponse {
  coverage: CoverageSummary
  bucket: 'day' | 'week' | 'month' | 'year'
  timezone: string
  series: {
    period: string
    messages: number
    sent: number
    received: number
    active_conversations: number
  }[]
  page: AnalyticsSeriesPage
  summary: {
    messages: number
    observed_periods: number
    average_per_observed_period: number
  }
  derived_nlp: DerivedNlpAvailability
}

export interface AnalyticsPatternsResponse {
  timezone: string
  heatmap: { date: string; messages: number; activities: number; total: number }[]
  hours: { hour: number; messages: number; sent: number; received: number; activities: number }[]
  weekdays: { weekday: number; label: string; messages: number; activities: number }[]
  streaks: {
    active_days: number
    longest_days: number
    latest_days: number
    first_active_date?: string | null
    last_active_date?: string | null
  }
  peak_days: { date: string; messages: number; activities: number; total: number }[]
  page: AnalyticsSeriesPage
  summary: {
    messages_in_scope: number
    dated_activity_records_in_scope: number
    active_activity_records_archive_wide: number
    unknown_date_activity_records: number
    unknown_date_scope: 'archive_wide'
    unknown_date_included_in_time_buckets: false
    date_filter_active: boolean
  }
  range_provenance: string
  derived_nlp: DerivedNlpAvailability
}

export interface ResponseTimeSummary {
  count: number
  mean_ms?: number | null
  median_ms?: number | null
  p90_ms?: number | null
  buckets: Record<string, number>
}

export interface AnalyticsInteractionPatternsResponse {
  timezone: string
  bucket: 'day' | 'week' | 'month' | 'year'
  responses: {
    owner_replied: ResponseTimeSummary
    other_replied: ResponseTimeSummary
    definition: string
  }
  initiations: { owner: number; other: number; definition: string }
  active_participants: { period: string; participants: number; conversations: number }[]
  page: AnalyticsSeriesPage
  ranking_policy: string
}

export interface AnalyticsInteractionSeriesResponse {
  bucket: 'day' | 'week' | 'month' | 'year'
  timezone: string
  series: {
    period: string
    calls: number
    call_duration_seconds: number
    reactions: number
    reaction_explicit_timestamp: number
    reaction_message_timestamp_fallback: number
    audio_attachments: number
    link_occurrences: number
  }[]
  page: AnalyticsSeriesPage
  domains: {
    domain: string
    occurrences: number
    periods: { period: string; occurrences: number }[]
  }[]
  summary: {
    calls: number
    call_duration_seconds: number
    reactions: number
    reaction_explicit_timestamp: number
    reaction_message_timestamp_fallback: number
    audio_attachments: number
    audio_unknown_date: number
    link_occurrences: number
    domain_count: number
  }
  provenance: {
    calls: string
    reactions: string
    audio: string
    links: string
  }
}

export interface AnalyticsNetworkNode {
  id: string
  kind: 'conversation' | 'participant_label'
  label: string
  value: number
  conversation_type?: string
  is_owner?: boolean
  href?: string
}

export interface AnalyticsNetworkResponse {
  nodes: AnalyticsNetworkNode[]
  edges: { id: string; source: string; target: string; value: number }[]
  conversation_count: number
  participant_label_count: number
  scope: string
  caveat: string
}

export interface AnalyticsTopicsResponse {
  status: 'unavailable' | 'building' | 'ready' | 'stale' | 'failed' | string
  ready: boolean
  reason?: string | null
  latest_job?: Pick<Job, 'id' | 'status' | 'stage' | 'progress_current' | 'progress_total'> & {
    error?: string | null
    created_at?: string | null
    finished_at?: string | null
  } | null
  semantic_index?: {
    id: string
    model_name: string
    model_revision: string
    window_count: number
  } | null
  result?: {
    pipeline_version: string
    generated_at: string
    corpus_window_count: number
    sample_count: number
    sampling: string
    warning: string
    clusters: {
      id: string
      ordinal: number
      sample_windows: number
      sample_share: number
      terms: string[]
      representatives: SearchResult[]
    }[]
  } | null
}

export interface NlpTrendPoint {
  period?: string
  hour?: string
  direction: 'all' | 'owner' | 'other' | 'activity' | 'audio' | 'attachment' | string
  sample_count: number
  mean_confidence: number
  score?: number | null
  score_definition: string
  dominant_label?: string | null
  predicted_label_counts: Record<string, number>
  mean_probabilities: Record<string, number>
}

export interface AnalyticsNlpTrendsResponse {
  timezone: string
  bucket: 'hour' | 'day' | 'week' | 'month' | 'year'
  direction: string
  source_sample_count: number
  page_sample_count: number
  models: Record<string, { model_name: string; model_revision: string; configuration_sha256: string }>
  series: Record<'sentiment' | 'emotion' | 'toxicity', NlpTrendPoint[]>
  hour_of_day: Record<'sentiment' | 'emotion' | 'toxicity', NlpTrendPoint[]>
  hour_of_day_scope: 'current_page'
  page: AnalyticsSeriesPage
  availability: DerivedNlpAvailability
  rollup_provenance: {
    source_resolution: string
    range_filter_semantics: string
    local_bucket_label: string
    partial_boundary_hours_included: boolean
    non_whole_hour_timezone_offset_observed: boolean
  }
  warnings: string[]
}

export interface AnalyticsConflictRatePoint {
  period: string
  sample_count: number
  direct_conflict_window_count: number
  direct_conflict_window_rate?: number | null
  low_sample_warning: boolean
}

export interface AnalyticsConflictRateResponse {
  available: boolean
  status: 'ready' | 'building' | 'stale' | 'failed' | 'unavailable' | string
  reason?: string | null
  timezone: string
  bucket: 'hour' | 'day' | 'week' | 'month' | 'year'
  series: AnalyticsConflictRatePoint[]
  page: AnalyticsSeriesPage
  summary: {
    sample_count: number
    direct_conflict_window_count: number
    direct_conflict_window_rate?: number | null
  }
  warnings: string[]
  provenance?: Record<string, unknown> | null
  relationship_ranking_enabled: false
  relationship_dimensions: []
}

export interface EnrichmentJump {
  occurrence_id: string
  conversation_id?: string | null
  message_id?: string | null
  activity_id?: string | null
  start_ms?: number
  href: string
}

export interface AudioTranscriptSegment {
  id: string
  ordinal: number
  start_ms: number
  end_ms: number
  text?: string | null
  confidence?: number | null
  avg_log_probability?: number | null
  no_speech_probability?: number | null
  jumps: EnrichmentJump[]
}

export interface AudioTranscriptionItem {
  id: string
  blob_sha256: string
  pipeline_version: string
  is_current_version: boolean
  status: string
  model_name?: string | null
  model_revision?: string | null
  device?: string | null
  compute_type?: string | null
  language?: string | null
  language_probability?: number | null
  duration_ms?: number | null
  text?: string | null
  segment_count: number
  warnings: string[]
  provenance: Record<string, unknown>
  error?: string | null
  segments: AudioTranscriptSegment[]
}

export interface EnrichmentOffsetPage {
  limit: number
  offset: number
  next_offset?: number | null
}

export interface AudioTranscriptionsResponse {
  items: AudioTranscriptionItem[]
  page: EnrichmentOffsetPage
  selection: EnrichmentVersionSelection
}

export interface EnrichmentVersionSelection {
  scope: 'current' | 'history'
  pipeline_version?: string | null
  model_name?: string | null
  model_revision?: string | null
  extractor_name?: string | null
  extractor_revision?: string | null
  current: {
    pipeline_version: string
    model_name?: string | null
    model_revision?: string | null
    extractor_name?: string | null
    extractor_revision?: string | null
  }
}

export interface AttachmentTextExtractionItem {
  id: string
  blob_sha256: string
  pipeline_version: string
  extractor_name: string
  extractor_revision: string
  is_current_version: boolean
  status: string
  filename?: string | null
  media_type?: string | null
  text_preview?: string | null
  preview_character_count: number
  preview_truncated: boolean
  preview_limit: number
  character_count?: number | null
  truncated: boolean
  index_eligible: boolean
  sandboxed: boolean
  sandbox_status?: string | null
  limits: Record<string, unknown>
  warnings: string[]
  provenance: Record<string, unknown>
  error?: string | null
  jumps: EnrichmentJump[]
}

export interface AttachmentTextExtractionsResponse {
  items: AttachmentTextExtractionItem[]
  page: EnrichmentOffsetPage
  selection: EnrichmentVersionSelection
}

export interface NlpAnnotationItem {
  id: string
  target_type: string
  target_id: string
  task: string
  model_name: string
  model_revision: string
  predicted_label: string
  confidence: number
  probabilities: Record<string, number>
  probability_activation?: string | null
  attribution: Record<string, unknown>
  warnings: string[]
  provenance: Record<string, unknown>
}

export interface NlpAnnotationsResponse {
  items: NlpAnnotationItem[]
  page: EnrichmentOffsetPage
}

export interface EnrichmentStatusResponse {
  audio_transcriptions: Record<string, number>
  attachment_text_extractions: Record<string, number>
  active_nlp_annotations: Record<string, number>
  active_nlp_hourly_rollups: Record<string, number>
  nlp_models: Record<string, {
    model_name: string
    model_revision: string
    language_scope: string
    warnings: string[]
  }>
  attachment_sandbox: Record<string, unknown>
  attachment_limits: Record<string, unknown>
  audio_transcription_selection?: {
    current: EnrichmentVersionSelection['current']
    current_record_count: number
    historical_record_count: number
  }
  attachment_text_selection?: {
    current: EnrichmentVersionSelection['current']
    current_record_count: number
    historical_record_count: number
  }
  conflict: {
    status: string
    reason?: string
    warning?: string
    sample_count?: number
    label_count?: number
    aggregation_enabled?: boolean
    direct_conflict_precision?: number | null
    macro_f1?: number | null
    relationship_ranking_enabled?: boolean
    [key: string]: unknown
  }
  conflict_aggregates?: {
    available: boolean
    status: 'ready' | 'building' | 'stale' | 'failed' | 'unavailable' | string
    reason?: string | null
    gate: EnrichmentStatusResponse['conflict']
    run?: {
      id: string
      status: string
      active: boolean
      pipeline_version: string
      model_name: string
      model_revision: string
      calibration_id?: string | null
      semantic_config_id?: string | null
      corpus_revision: string
      source_revision: string
      configuration_sha256: string
      window_count: number
      scored_count: number
      direct_conflict_count: number
      warnings: string[]
      completed_at?: string | null
    } | null
    job?: Pick<Job, 'id' | 'status' | 'stage' | 'progress_current' | 'progress_total'> | null
    warnings: string[]
    relationship_ranking_enabled: false
    relationship_dimensions: []
  }
  jobs: Pick<Job, 'id' | 'kind' | 'status' | 'stage' | 'progress_current' | 'progress_total'>[]
}

export type ConflictLabel = 'direct conflict' | 'calm disagreement' | 'banter' | 'third-party conflict' | 'no conflict' | 'uncertain'

export interface ConflictSearchResult extends SearchResult {
  conflict_scoring: {
    target_label: ConflictLabel
    predicted_label: ConflictLabel
    scores: Record<ConflictLabel, number>
    score_kind: string
    model_name: string
    model_revision: string
    quantization: string
    ranking_feature_only: boolean
    aggregation_gate_bypassed: boolean
    warnings: string[]
    provenance: Record<string, unknown>
  }
}

export interface ConflictSearchResponse {
  items: ConflictSearchResult[]
  page: { limit: number; returned: number; candidate_count: number }
  query: string
  target_label: ConflictLabel
  retrieval_config?: Record<string, unknown> | null
  conflict_model: Record<string, unknown>
  aggregation: EnrichmentStatusResponse['conflict']
}

export interface ConflictLabelResponse {
  id: string
  window_id: string
  label: ConflictLabel
  evaluation_split: 'train' | 'validation' | 'test'
  notes?: string | null
  split_assignment: { version: string; source: 'conversation_id' | string }
  updated: boolean
  calibration_invalidated: boolean
}

export type ConflictEvaluationSplit = 'train' | 'validation' | 'test'
export type ConflictReviewLabelFilter = 'unlabelled' | ConflictLabel

export interface ConflictReviewValidation {
  eligible_for_calibration: boolean
  conversation_disjoint: boolean
  stratified: boolean
  exact_split_assignment: boolean
  missing_conversation_count: number
  split_assignment_mismatch_count: number
  leaked_conversation_count: number
  missing_strata: Array<{ evaluation_split: ConflictEvaluationSplit; label: ConflictLabel }>
  blockers: Array<{ code: string; message: string; count?: number }>
}

export interface ConflictReviewStatus {
  labels: ConflictLabel[]
  minimum_label_count: number
  eligible_window_count: number
  manual_label_count: number
  unlabelled_window_count: number
  evaluation_split_algorithm: {
    version: string
    ratios: Record<ConflictEvaluationSplit, number>
  }
  by_label: Record<ConflictLabel, number>
  by_split: Record<ConflictEvaluationSplit, {
    eligible_window_count: number
    manual_label_count: number
    unlabelled_window_count: number
    by_label: Record<ConflictLabel, number>
  }>
  validation: ConflictReviewValidation
  relationship_ranking_enabled: false
  warning: string
}

export interface ConflictReviewItem {
  window_id: string
  conversation_id: string
  evaluation_split: ConflictEvaluationSplit
  first_message_id?: string | null
  last_message_id?: string | null
  first_timestamp_ms?: number | null
  last_timestamp_ms?: number | null
  turn_count: number
  text_display: string
  current_manual_label?: {
    label: ConflictLabel
    notes?: string | null
    created_at: string
  } | null
  jump: { href: string; label: string }
}

export interface ConflictReviewQueueResponse {
  items: ConflictReviewItem[]
  page: { limit: number; returned: number; total: number; next_after?: string | null }
  filters: { label: ConflictReviewLabelFilter; evaluation_split: 'all' | ConflictEvaluationSplit }
  navigation: { next_unlabelled_href: string }
}

export interface ConflictReviewNextResponse {
  item?: ConflictReviewItem | null
  wrapped: boolean
  navigation: { next_unlabelled_href: string }
}

export interface ConflictCalibrationResponse {
  id: string
  status: string
  sample_count: number
  aggregation_enabled: boolean
  direct_conflict_precision?: number | null
  macro_f1?: number | null
  evaluation_split_algorithm: ConflictReviewStatus['evaluation_split_algorithm']
  relationship_ranking_enabled: false
}

export type DeletionEntityType = 'message' | 'activity' | 'conversation' | 'attachment'

export interface DeletionPreviewRequest {
  entity_types: DeletionEntityType[]
  text?: string
  text_mode: 'exact' | 'keyword'
  case_sensitive: boolean
  whole_field: boolean
  text_variant: 'raw' | 'display'
  from_date?: string
  to_date?: string
  timezone: string
  conversation_ids: string[]
  participant?: string
  media_types: string[]
  availability: string[]
  entity_ids: Partial<Record<DeletionEntityType, string[]>>
  allow_all: boolean
}

export interface DeletionPreviewResponse {
  operation_id: string
  commit_token: string
  catalog_revision: string
  selection_sha256: string
  impact: Record<string, unknown>
}

export interface DeletionSelectionItem {
  entity_type: DeletionEntityType
  entity_id: string
  reason: string
  label: string
  label_truncated: boolean
  snippet?: string | null
  snippet_truncated: boolean
  preview_limit: number
  timestamp_ms?: number | null
  resource_kind?: string | null
  availability?: string | null
  media_type?: string | null
  size_bytes?: number | null
  quarantined: boolean
  jump?: { href: string; label: string } | null
}

export interface DeletionSelectionResponse {
  operation_id: string
  revision: string
  status: 'previewed' | 'committed'
  items: DeletionSelectionItem[]
  page: CursorPageMetadata
  privacy: {
    snippet_limit: number
    label_limit: number
    raw_fields_exposed: false
  }
}

export interface DeletionMutationResult {
  operation_id: string
  status: string
  affected: Record<string, number>
  audit_id: string
}

export type DeletionPurgeCount =
  | 'normalized_rows'
  | 'search_windows'
  | 'candidate_blobs'
  | 'logically_hidden_blobs'
  | 'candidate_blob_bytes'
  | 'shared_blobs_preserved'
  | 'media_derivatives'
  | 'media_derivative_bytes'
  | 'audio_transcriptions'
  | 'audio_transcript_segments'
  | 'attachment_text_extractions'
  | 'nlp_annotations'
  | 'raw_source_files'
  | 'raw_source_files_rewritten'
  | 'raw_source_obsolete_blobs'
  | 'workspace_files'
  | 'workspace_bytes'
  | 'filesystem_library_files'
  | 'analytics_cache_files'
  | 'vector_store_files'
  | 'affected_messages'
  | 'affected_activities'
  | 'affected_conversations'
  | 'affected_attachments'
  | 'affected_blobs'
  | 'affected_source_records'
  | 'affected_search_windows'
  | 'staged_workspace_files'
  | 'deleted_vector_rows'
  | 'raw_source_rewrites'

export interface DeletionPurgeSummary {
  scope?: 'normalized' | 'workspace' | null
  status: {
    operation: string
    completed?: boolean
    publication_action?: 'commit' | 'restore'
    publication_phase?: 'preview' | 'prepared' | 'published'
    blockers?: string[]
  }
  counts: Partial<Record<DeletionPurgeCount, number>>
  digests: {
    purge_revision?: string
    logical_restore_state_sha256?: string
  }
  timestamps: {
    planned_at?: string
    completed_at?: string
  }
}

export interface DeletionRestoreRebuildSummary {
  required: true
  reason: 'intervening_hard_purge' | 'later_quarantine'
  product_kinds: ('search_windows' | 'semantic_index' | 'nlp_hourly_rollups' | 'conflict_aggregates' | 'topic_clusters')[]
  invalidated_by_count: number
}

export interface DeletionAuditDetail {
  counts: Partial<Record<DeletionPurgeCount, number>>
  digests: {
    catalog_revision?: string
    purge_revision?: string
  }
  status: {
    blockers?: string[]
    recovered?: boolean
  }
  restore_rebuild_required?: DeletionRestoreRebuildSummary
}

export interface DeletionOperation {
  id: string
  status: string
  item_count: number
  impact: Record<string, unknown>
  created_at: string
  updated_at: string
  committed_at?: string | null
  restored_at?: string | null
  purged_at?: string | null
  selector?: DeletionPreviewRequest
  catalog_revision?: string
  selection_sha256?: string
  purge_summary?: DeletionPurgeSummary | null
  restore_rebuild_required?: DeletionRestoreRebuildSummary | null
  error_text?: string | null
  audits?: {
    id: string
    action: 'commit' | 'restore' | 'purge_preview' | 'purge_precommit_recovery' | 'purge' | 'purge_finalize' | 'unknown'
    scope?: 'logical_quarantine' | 'normalized' | 'workspace' | null
    outcome: string
    detail: DeletionAuditDetail
    created_at: string
  }[]
}

export interface DeletionOperationsResponse {
  items: DeletionOperation[]
  page: CursorPageMetadata
}

export interface HardPurgePlan {
  operation_id: string
  scope: 'normalized' | 'workspace'
  executable: boolean
  blockers: string[]
  purge_token: string
  purge_revision: string
  impact: Partial<Record<DeletionPurgeCount, number>>
}

export interface AnalyticsAttachmentsResponse {
  bucket: 'day' | 'week' | 'month' | 'year'
  timezone: string
  types: { type: string; occurrences: number; unique_files: number; occurrence_bytes: number }[]
  availability: { availability: string; occurrences: number; bytes: number; rate?: number | null }[]
  trends: {
    period: string
    occurrences: number
    unique_files: number
    occurrence_bytes: number
    by_type: Record<string, { occurrences: number; unique_files: number; occurrence_bytes: number }>
  }[]
  page: AnalyticsSeriesPage
  summary: {
    occurrences: number
    dated_occurrences: number
    unknown_date_occurrences: number
    unknown_date_scope: 'archive_wide' | string
    unknown_date_included_in_totals: boolean
    missing_occurrences: number
    missing_rate?: number | null
    unlinked_occurrences: number
    unlinked_rate?: number | null
  }
  count_semantics: string
  provenance: string
}

export interface AnalyticsConversationsResponse {
  conversation_count: number
  by_type: Record<string, number>
  longevity_days: {
    mean: number
    median: number
    maximum: number
    histogram: Record<string, number>
  }
  active_days: { mean: number; maximum: number }
  ranking_policy: string
  derived_nlp: DerivedNlpAvailability
}

export interface AnalyticsStorageResponse {
  source_occurrences: { files: number; bytes: number }
  unique_blobs: { files: number; bytes: number }
  deduplication: { redundant_occurrences: number; duplicate_bytes: number }
  by_media_class: { class: string; files: number; bytes: number }[]
}

export interface AnalyticsCoverageResponse {
  summary: Record<string, number>
  items: CoverageSummary[]
  history?: {
    mode: string
    historical_comparison_available: boolean
    description: string
    observed_at?: string | null
    observed_at_basis?: string | null
    snapshots: {
      id: string
      observed_at?: string | null
      summary: Record<string, number>
      items: CoverageSummary[]
    }[]
  }
}

export type SearchMode = 'keyword' | 'exact' | 'semantic' | 'hybrid'
export type SearchEntityType = 'all' | 'message' | 'activity' | 'audio_transcript' | 'attachment_text'

export interface SearchFilters {
  entity_type: SearchEntityType
  conversation_id?: string | null
  participant?: string | null
  source_category?: string | null
  from_ms?: number | null
  to_ms?: number | null
}

export interface SearchWindow {
  start_message_id?: string | null
  end_message_id?: string | null
  message_ids: string[]
  turn_count: number
  first_timestamp_ms?: number | null
  last_timestamp_ms?: number | null
}

export interface SearchOccurrence extends Starrable {
  occurrence_id: string
  conversation_id?: string | null
  conversation_title?: string | null
  message_id?: string | null
  activity_id?: string | null
  activity?: {
    id: string
    title?: string | null
    category: string
    subtype: string
  } | null
  timestamp_ms?: number | null
  source_category?: string | null
  start_ms?: number | null
  href: string
}

export interface SearchOccurrencePage {
  limit: number
  offset: number
  returned: number
  total: number
  next_offset?: number | null
}

export interface DerivedOccurrencesResponse {
  items: SearchOccurrence[]
  page: SearchOccurrencePage
}

export interface SearchResult extends Starrable {
  id: string
  window_id?: string | null
  entity_type: 'message' | 'message_window' | 'activity' | 'audio_transcript' | 'attachment_text'
  source_id?: string | null
  rank: number
  score: number
  score_kind: string
  timestamp_ms?: number | null
  excerpt: string
  sender?: string | null
  conversation?: { id: string; title?: string | null } | null
  participants?: string[]
  source_category?: string | null
  subtype?: string | null
  coverage: CoverageSummary
  confidence: number | null
  confidence_kind: 'not_applicable' | 'audio_decoder_score' | string
  confidence_calibrated: boolean
  availability: string
  deletion_state: 'retained' | string
  provenance: {
    entity_type?: string | null
    source_id?: string | null
    window_id?: string | null
    source_category?: string | null
    representation: 'source_record' | 'semantic_window' | 'derived_text' | string
  }
  window?: SearchWindow | null
  occurrences?: SearchOccurrence[]
  occurrence_page?: SearchOccurrencePage
  jump?: ConversationJump | null
}

export interface SemanticIndexConfig {
  id: string
  status: string
  model_name: string
  model_revision: string
  dimensions: number
  window_version: string
  window_count: number
  audit?: SemanticIndexAudit
}

export interface SemanticManualActivationOverride {
  version: string
  activated_at: string
  candidate_config_id: string
  previous_active_config_id: string | null
  acknowledged_failed_quality_gate: true
  reason: string
}

export interface SemanticIndexAudit extends Record<string, unknown> {
  manual_activation_override?: SemanticManualActivationOverride | null
}

export interface SemanticRetrievalMetrics {
  query_count: number
  k: number
  recall_at_k: number
  mrr: number
  ndcg_at_k: number
}

export interface SemanticRetrievalBenchmark {
  benchmark_kind: 'automatic_known_item' | 'held_out_user_feedback' | string
  status: string
  passed: boolean | null
  used_for_activation: boolean
  metrics: SemanticRetrievalMetrics
}

export interface SemanticUpgradeMetricComparison {
  candidate: number
  incumbent: number
  delta: number
  regression_exceeds_tolerance: boolean
  demonstrated_improvement: boolean
}

export interface SemanticIncumbentComparison {
  status: string
  required_for_activation: boolean
  passed: boolean | null
  metrics?: Partial<
    Record<'recall_at_k' | 'mrr' | 'ndcg_at_k', SemanticUpgradeMetricComparison>
  >
}

export interface SemanticIndexAcceptanceEvaluation {
  evaluator_version: string
  evaluated_at: string
  passed: boolean
  decision: string
  automatic_known_item: SemanticRetrievalBenchmark
  held_out_feedback: SemanticRetrievalBenchmark
  active_incumbent_comparison?: SemanticIncumbentComparison
}

export interface SemanticIndexCandidateStatus {
  id: string
  status: string
  active: boolean
  window_count: number
  evaluation?: SemanticIndexAcceptanceEvaluation | null
}

export interface SearchResponse {
  items: SearchResult[]
  page: {
    limit: number
    returned: number
    candidate_count: number
    result_cap: number
    next_cursor?: string | null
  }
  query: string
  mode: SearchMode
  filters: SearchFilters
  index_config?: Partial<SemanticIndexConfig> | null
}

export interface SearchStatusResponse {
  ready: boolean
  needs_rebuild?: boolean
  stale_reason?: {
    reason?: string
    source_kind?: string
    source_id?: string
    stale_at?: string
  } | null
  stale_config?: Pick<
    SemanticIndexConfig,
    'id' | 'status' | 'model_name' | 'model_revision' | 'window_count'
  > | null
  active?: SemanticIndexConfig | null
  acceptance?: SemanticIndexAcceptanceEvaluation | null
  latest_evaluation?: SemanticIndexAcceptanceEvaluation | null
  latest_candidate?: SemanticIndexCandidateStatus | null
  latest_job?: Job | null
}

export interface SavedSearch {
  id: string
  name: string
  query: Record<string, unknown>
}

export interface SavedSearchesResponse {
  items: SavedSearch[]
}

export interface SearchFeedbackResponse {
  id: string
  query_sha256: string
  query?: string
  search_mode?: SearchMode
  filter_scope?: Partial<SearchFilters>
  result_id: string
  label: 'relevant' | 'not_relevant' | 'uncertain'
  evaluation_split?: 'train' | 'validation' | 'test' | null
  notes?: string | null
  created_at?: string
  updated?: boolean
}

export interface SearchFeedbackListResponse {
  items: SearchFeedbackResponse[]
  page: { limit: number; returned: number; next_cursor?: string | null }
}

export interface RagAnswerResponse {
  answer: string
  citation_ids: string[]
  insufficient_evidence: boolean
  sources: SearchResult[]
  audit_id: string
  model: string
}

export interface RagAuditItem {
  id: string
  status: string
  model: string
  query_sha256: string
  search_mode: 'semantic' | 'hybrid'
  retrieved_ids: string[]
  citation_ids: string[]
  remote_response_id?: string | null
  usage: Record<string, unknown>
  error?: string | null
  created_at: string
}

export interface RagAuditsResponse {
  items: RagAuditItem[]
  page: { limit: number; returned: number; next_cursor?: string | null }
}
