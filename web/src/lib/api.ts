import type {
  ActivityResponse,
  AnalyticsAttachmentsResponse,
  AnalyticsConversationsResponse,
  AnalyticsConflictRateResponse,
  AnalyticsCoverageResponse,
  AnalyticsMessagesResponse,
  AnalyticsInteractionPatternsResponse,
  AnalyticsInteractionSeriesResponse,
  AnalyticsNetworkResponse,
  AnalyticsNlpTrendsResponse,
  AnalyticsOverviewResponse,
  AnalyticsPatternsResponse,
  AnalyticsStorageResponse,
  AnalyticsTopicsResponse,
  AttachmentTextExtractionsResponse,
  ApiProblem,
  AudioTranscriptionsResponse,
  BlobOccurrencesResponse,
  ConversationDateJumpResponse,
  ConversationDetailResponse,
  ConversationCorrectionResult,
  ConversationCorrectionsResponse,
  ConversationMessagesResponse,
  ConversationSearchResponse,
  ConversationsResponse,
  ConflictLabel,
  ConflictLabelResponse,
  ConflictCalibrationResponse,
  ConflictEvaluationSplit,
  ConflictReviewLabelFilter,
  ConflictReviewNextResponse,
  ConflictReviewQueueResponse,
  ConflictReviewStatus,
  ConflictSearchResponse,
  CoverageResponse,
  DeletionMutationResult,
  DeletionOperation,
  DeletionOperationsResponse,
  DeletionPreviewRequest,
  DeletionPreviewResponse,
  DeletionSelectionResponse,
  DerivedOccurrencesResponse,
  EnrichmentStatusResponse,
  HardPurgePlan,
  HealthResponse,
  IntegrityResponse,
  JobsResponse,
  MediaResponse,
  MediaView,
  MessageAdvancedResponse,
  EventsResponse,
  NlpAnnotationsResponse,
  OnThisDayResponse,
  OverviewResponse,
  PeopleResponse,
  SortDirection,
  SettingsPatch,
  SettingsResponse,
  RagAnswerResponse,
  RagAuditsResponse,
  SavedSearch,
  SavedSearchesResponse,
  SearchEntityType,
  SearchFilters,
  SearchFeedbackResponse,
  SearchFeedbackListResponse,
  SearchMode,
  SearchResponse,
  SearchStatusResponse,
  SupportResponse,
  TimelineResponse,
} from './contracts'

const API_ROOT = '/api/v1'
const CSRF_COOKIE = 'meta_archive_csrf'
const ANALYTICS_READ_CONCURRENCY = 2

type AnalyticsReadRelease = () => void

let activeAnalyticsReads = 0
const pendingAnalyticsReads: Array<(release: AnalyticsReadRelease) => void> = []

function releaseAnalyticsRead() {
  activeAnalyticsReads -= 1
  const next = pendingAnalyticsReads.shift()
  if (!next) return
  activeAnalyticsReads += 1
  next(releaseAnalyticsRead)
}

function acquireAnalyticsRead(): Promise<AnalyticsReadRelease> {
  if (activeAnalyticsReads < ANALYTICS_READ_CONCURRENCY) {
    activeAnalyticsReads += 1
    return Promise.resolve(releaseAnalyticsRead)
  }
  return new Promise((resolve) => pendingAnalyticsReads.push(resolve))
}

export class ApiError extends Error {
  readonly status: number
  readonly code?: string

  constructor(message: string, status: number, code?: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.code = code
  }
}

function unwrapEnvelope<T>(payload: unknown): T {
  if (payload && typeof payload === 'object' && 'data' in payload) {
    return (payload as { data: T }).data
  }
  return payload as T
}

function readCookie(name: string): string | undefined {
  if (typeof document === 'undefined') return undefined
  const prefix = `${encodeURIComponent(name)}=`
  const value = document.cookie
    .split(';')
    .map((part) => part.trim())
    .find((part) => part.startsWith(prefix))
    ?.slice(prefix.length)
  return value ? decodeURIComponent(value) : undefined
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const method = (init?.method ?? 'GET').toUpperCase()
  const limitAnalyticsFanout = method === 'GET' && path.startsWith('/analytics/')
  const releaseAnalyticsRead = limitAnalyticsFanout ? await acquireAnalyticsRead() : undefined
  const csrfToken = method === 'GET' || method === 'HEAD' ? undefined : readCookie(CSRF_COOKIE)
  try {
    const response = await fetch(`${API_ROOT}${path}`, {
      credentials: 'same-origin',
      headers: {
        Accept: 'application/json',
        ...(init?.body ? { 'Content-Type': 'application/json' } : {}),
        ...(csrfToken ? { 'X-CSRF-Token': csrfToken } : {}),
        ...init?.headers,
      },
      ...init,
    })

    if (!response.ok) {
      let problem: ApiProblem | undefined
      try {
        problem = (await response.json()) as ApiProblem
      } catch {
        // The status text is still more useful than a JSON parse error.
      }
      const structuredDetail = problem?.detail && typeof problem.detail === 'object' ? problem.detail : undefined
      const message =
        (typeof problem?.detail === 'string' ? problem.detail : structuredDetail?.message) ??
        problem?.message ??
        problem?.title ??
        response.statusText ??
        'The archive service could not complete this request.'
      throw new ApiError(message, response.status, structuredDetail?.code)
    }

    if (response.status === 204) {
      return undefined as T
    }

    return unwrapEnvelope<T>(await response.json())
  } finally {
    // A read occupies its slot until the response body has settled. Releasing
    // at headers would let queued Analytics reads outlive the query state that
    // the route uses as its loading contract.
    releaseAnalyticsRead?.()
  }
}

function addOptional(query: URLSearchParams, name: string, value: string | number | boolean | null | undefined) {
  if (value == null || value === '') return
  query.set(name, String(value))
}

export interface ConversationQuery {
  q?: string
  conversationType?: string
  sourceCategory?: string
  completeness?: string
  minParticipants?: number
  maxParticipants?: number
  minMessages?: number
  maxMessages?: number
  activityFromMs?: number
  activityToMs?: number
  hasAttachments?: boolean
  minPhotos?: number
  maxPhotos?: number
  minVideos?: number
  maxVideos?: number
  minAudio?: number
  maxAudio?: number
  sort?: string
  direction?: SortDirection
  limit?: number
  cursor?: string
}

export interface MessagePageQuery {
  limit?: number
  cursor?: string
  before?: string
  after?: string
  around?: string
  atMs?: number
  start?: 'latest' | 'earliest'
}

export interface ThreadSearchQuery {
  q: string
  kind?: string
  sender?: string
  fromMs?: number
  toMs?: number
  direction?: SortDirection
  limit?: number
  cursor?: string
}

export interface MediaQuery {
  view?: MediaView
  occurrenceId?: string
  q?: string
  type?: string
  kind?: string
  availability?: string
  conversationId?: string
  conversationQ?: string
  conversationType?: string
  sourceCategory?: string
  fromMs?: number
  toMs?: number
  minSize?: number
  maxSize?: number
  sort?: string
  direction?: SortDirection
  limit?: number
  cursor?: string
}

export interface ActivityQuery {
  view?: string
  q?: string
  category?: string
  subtype?: string
  availability?: string
  fromMs?: number
  toMs?: number
  hasAttachments?: boolean
  logicalOnly?: boolean
  postContext?: string
  sort?: string
  direction?: SortDirection
  limit?: number
  cursor?: string
}

export interface TimelineQuery {
  q?: string
  entityType?: string
  fromMs?: number
  toMs?: number
  direction?: SortDirection
  limit?: number
  cursor?: string
  around?: string
}

export interface EventQuery {
  q?: string
  status?: string
  hosted?: boolean
  fromMs?: number
  toMs?: number
  sort?: string
  direction?: SortDirection
  limit?: number
  cursor?: string
  around?: string
}

export interface PeopleQuery {
  view?: string
  q?: string
  conversationType?: string
  sort?: string
  direction?: SortDirection
  limit?: number
  cursor?: string
  around?: string
}

export interface AnalyticsQuery {
  bucket?: 'day' | 'week' | 'month' | 'year'
  timezone?: string
  fromMs?: number
  toMs?: number
  conversationId?: string
  limit?: number
  cursor?: string
}

export interface SearchQuery {
  q: string
  mode?: SearchMode
  entityType?: SearchEntityType
  conversationId?: string
  participant?: string
  sourceCategory?: string
  fromMs?: number
  toMs?: number
  limit?: number
  cursor?: string
}

export interface RagQuestion {
  question: string
  searchMode?: 'semantic' | 'hybrid'
  topK?: number
  entityType?: SearchEntityType
  conversationId?: string
  participant?: string
  sourceCategory?: string
  fromMs?: number
  toMs?: number
}

export const api = {
  health: () => request<HealthResponse>('/health'),
  overview: () => request<OverviewResponse>('/overview'),
  coverage: () => request<CoverageResponse>('/coverage'),
  integrity: (options: {
    includeResolved?: boolean
    severity?: string
    q?: string
    limit?: number
    offset?: number
  } = {}) => {
    const query = new URLSearchParams({
      unresolved_only: options.includeResolved ? 'false' : 'true',
      limit: String(options.limit ?? 100),
      offset: String(options.offset ?? 0),
    })
    if (options.severity && options.severity !== 'all') query.set('severity', options.severity)
    if (options.q?.trim()) query.set('q', options.q.trim())
    return request<IntegrityResponse>(`/integrity?${query}`)
  },
  jobs: (options: { status?: string; q?: string; limit?: number; offset?: number } = {}) => {
    const query = new URLSearchParams({
      limit: String(options.limit ?? 50),
      offset: String(options.offset ?? 0),
    })
    if (options.status && options.status !== 'all') query.set('status', options.status)
    if (options.q?.trim()) query.set('q', options.q.trim())
    return request<JobsResponse>(`/jobs?${query}`)
  },
  settings: () => request<SettingsResponse>('/settings'),
  updateSettings: (settings: SettingsPatch) =>
    request<SettingsResponse>('/settings', {
      method: 'PATCH',
      body: JSON.stringify(settings),
    }),
  search: (options: SearchQuery) => {
    const query = new URLSearchParams({ q: options.q, mode: options.mode ?? 'keyword' })
    addOptional(query, 'entity_type', options.entityType ?? 'all')
    addOptional(query, 'conversation_id', options.conversationId?.trim())
    addOptional(query, 'participant', options.participant?.trim())
    addOptional(query, 'source_category', options.sourceCategory?.trim())
    addOptional(query, 'from_ms', options.fromMs)
    addOptional(query, 'to_ms', options.toMs)
    addOptional(query, 'limit', options.limit ?? 50)
    addOptional(query, 'cursor', options.cursor)
    return request<SearchResponse>(`/search?${query}`)
  },
  searchStatus: () => request<SearchStatusResponse>('/search/status'),
  searchDerivedOccurrences: (options: {
    entityType: 'audio_transcript' | 'attachment_text'
    sourceId: string
    filterScope?: Partial<SearchFilters>
    offset?: number
    limit?: number
  }) => {
    const query = new URLSearchParams({
      entity_type: options.entityType,
      source_id: options.sourceId,
      offset: String(options.offset ?? 0),
      limit: String(options.limit ?? 20),
    })
    addOptional(query, 'conversation_id', options.filterScope?.conversation_id)
    addOptional(query, 'participant', options.filterScope?.participant)
    addOptional(query, 'source_category', options.filterScope?.source_category)
    addOptional(query, 'from_ms', options.filterScope?.from_ms)
    addOptional(query, 'to_ms', options.filterScope?.to_ms)
    return request<DerivedOccurrencesResponse>(`/search/derived-occurrences?${query}`)
  },
  buildSearchIndex: (payload: { model?: string; rebuild_windows?: boolean } = {}) =>
    request<{ job_id: string; status: string; status_href: string }>('/search/index', {
      method: 'POST',
      body: JSON.stringify({ rebuild_windows: true, ...payload }),
    }),
  savedSearches: () => request<SavedSearchesResponse>('/saved-searches'),
  createSavedSearch: (payload: { name: string; query: Record<string, unknown> }) =>
    request<SavedSearch>('/saved-searches', { method: 'POST', body: JSON.stringify(payload) }),
  updateSavedSearch: (id: string, payload: { name?: string; query?: Record<string, unknown> }) =>
    request<SavedSearch>(`/saved-searches/${encodeURIComponent(id)}`, {
      method: 'PATCH',
      body: JSON.stringify(payload),
    }),
  deleteSavedSearch: (id: string) =>
    request<void>(`/saved-searches/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  searchFeedback: (payload: {
    query: string
    search_mode: SearchMode
    filter_scope?: Partial<SearchFilters>
    result_id: string
    label: 'relevant' | 'not_relevant' | 'uncertain'
    evaluation_split?: 'train' | 'validation' | 'test'
    notes?: string
  }) =>
    request<SearchFeedbackResponse>('/search/feedback', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  searchFeedbackHistory: (options: { query?: string; searchMode?: SearchMode; label?: string; evaluationSplit?: string; cursor?: string; limit?: number } = {}) => {
    const query = new URLSearchParams({ limit: String(options.limit ?? 50) })
    addOptional(query, 'query', options.query?.trim())
    addOptional(query, 'search_mode', options.searchMode)
    addOptional(query, 'label', options.label)
    addOptional(query, 'evaluation_split', options.evaluationSplit)
    addOptional(query, 'cursor', options.cursor)
    return request<SearchFeedbackListResponse>(`/search/feedback?${query}`)
  },
  updateSearchFeedback: (id: string, payload: { label?: 'relevant' | 'not_relevant' | 'uncertain'; evaluation_split?: 'train' | 'validation' | 'test' | null; notes?: string | null }) =>
    request<SearchFeedbackResponse>(`/search/feedback/${encodeURIComponent(id)}`, { method: 'PATCH', body: JSON.stringify(payload) }),
  deleteSearchFeedback: (id: string) =>
    request<void>(`/search/feedback/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  ragAnswer: (options: RagQuestion) =>
    request<RagAnswerResponse>('/rag/answer', {
      method: 'POST',
      body: JSON.stringify({
        question: options.question,
        search_mode: options.searchMode ?? 'hybrid',
        top_k: options.topK ?? 8,
        entity_type: options.entityType ?? 'all',
        conversation_id: options.conversationId,
        participant: options.participant,
        source_category: options.sourceCategory,
        from_ms: options.fromMs,
        to_ms: options.toMs,
      }),
    }),
  ragAudits: (status = 'all', cursor?: string, limit = 20) => {
    const query = new URLSearchParams({ status, limit: String(limit) })
    addOptional(query, 'cursor', cursor)
    return request<RagAuditsResponse>(`/rag/audits?${query}`)
  },
  conversations: (options: ConversationQuery = {}) => {
    const query = new URLSearchParams()
    addOptional(query, 'q', options.q?.trim())
    addOptional(query, 'conversation_type', options.conversationType)
    addOptional(query, 'source_category', options.sourceCategory)
    addOptional(query, 'completeness', options.completeness)
    addOptional(query, 'min_participants', options.minParticipants)
    addOptional(query, 'max_participants', options.maxParticipants)
    addOptional(query, 'min_messages', options.minMessages)
    addOptional(query, 'max_messages', options.maxMessages)
    addOptional(query, 'activity_from_ms', options.activityFromMs)
    addOptional(query, 'activity_to_ms', options.activityToMs)
    addOptional(query, 'has_attachments', options.hasAttachments)
    addOptional(query, 'min_photos', options.minPhotos)
    addOptional(query, 'max_photos', options.maxPhotos)
    addOptional(query, 'min_videos', options.minVideos)
    addOptional(query, 'max_videos', options.maxVideos)
    addOptional(query, 'min_audio', options.minAudio)
    addOptional(query, 'max_audio', options.maxAudio)
    addOptional(query, 'sort', options.sort ?? 'last_activity')
    addOptional(query, 'direction', options.direction ?? 'desc')
    addOptional(query, 'limit', options.limit ?? 75)
    addOptional(query, 'cursor', options.cursor)
    return request<ConversationsResponse>(`/conversations?${query}`)
  },
  conversation: (conversationId: string) =>
    request<ConversationDetailResponse>(`/conversations/${encodeURIComponent(conversationId)}`),
  conversationCorrections: (conversationId?: string) => {
    const query = new URLSearchParams({ limit: '100', offset: '0' })
    addOptional(query, 'conversation_id', conversationId)
    return request<ConversationCorrectionsResponse>(`/conversation-corrections?${query}`)
  },
  mergeConversationHistories: (payload: {
    conversation_ids: string[]
    target_conversation_id: string
    identity_revision: string
    reason?: string
  }) => request<ConversationCorrectionResult>('/conversation-corrections/merge', {
    method: 'POST',
    body: JSON.stringify(payload),
  }),
  splitConversationHistory: (payload: {
    conversation_id: string
    segment_ids: string[]
    title: string
    identity_revision: string
    reason?: string
  }) => request<ConversationCorrectionResult>('/conversation-corrections/split', {
    method: 'POST',
    body: JSON.stringify(payload),
  }),
  revertConversationCorrection: (correctionId: string, identityRevision: string) =>
    request<ConversationCorrectionResult>(
      `/conversation-corrections/${encodeURIComponent(correctionId)}/revert`,
      { method: 'POST', body: JSON.stringify({ identity_revision: identityRevision }) },
    ),
  conversationMessages: (conversationId: string, options: MessagePageQuery = {}) => {
    const query = new URLSearchParams()
    addOptional(query, 'limit', options.limit ?? 100)
    addOptional(query, 'cursor', options.cursor)
    addOptional(query, 'before', options.before)
    addOptional(query, 'after', options.after)
    addOptional(query, 'around', options.around)
    addOptional(query, 'at_ms', options.atMs)
    addOptional(query, 'start', options.start)
    return request<ConversationMessagesResponse>(
      `/conversations/${encodeURIComponent(conversationId)}/messages?${query}`,
    )
  },
  conversationDateJump: (conversationId: string, atMs: number) =>
    request<ConversationDateJumpResponse>(
      `/conversations/${encodeURIComponent(conversationId)}/messages/date-jump?at_ms=${atMs}`,
    ),
  messageAdvanced: (conversationId: string, messageId: string) =>
    request<MessageAdvancedResponse>(
      `/conversations/${encodeURIComponent(conversationId)}/messages/${encodeURIComponent(messageId)}/advanced`,
    ),
  conversationSearch: (conversationId: string, options: ThreadSearchQuery) => {
    const query = new URLSearchParams({ q: options.q })
    addOptional(query, 'kind', options.kind)
    addOptional(query, 'sender', options.sender?.trim())
    addOptional(query, 'from_ms', options.fromMs)
    addOptional(query, 'to_ms', options.toMs)
    addOptional(query, 'direction', options.direction ?? 'desc')
    addOptional(query, 'limit', options.limit ?? 50)
    addOptional(query, 'cursor', options.cursor)
    return request<ConversationSearchResponse>(
      `/conversations/${encodeURIComponent(conversationId)}/message-search?${query}`,
    )
  },
  media: (options: MediaQuery = {}) => {
    const query = new URLSearchParams()
    addOptional(query, 'view', options.view ?? 'occurrences')
    addOptional(query, 'occurrence', options.occurrenceId)
    addOptional(query, 'q', options.q?.trim())
    addOptional(query, 'type', options.type)
    addOptional(query, 'kind', options.kind)
    addOptional(query, 'availability', options.availability)
    addOptional(query, 'conversation_id', options.conversationId?.trim())
    addOptional(query, 'conversation_q', options.conversationQ?.trim())
    addOptional(query, 'conversation_type', options.conversationType)
    addOptional(query, 'source_category', options.sourceCategory?.trim())
    addOptional(query, 'from_ms', options.fromMs)
    addOptional(query, 'to_ms', options.toMs)
    addOptional(query, 'min_size', options.minSize)
    addOptional(query, 'max_size', options.maxSize)
    addOptional(query, 'sort', options.sort ?? 'date')
    addOptional(query, 'direction', options.direction ?? 'desc')
    addOptional(query, 'limit', options.limit ?? 60)
    addOptional(query, 'cursor', options.cursor)
    return request<MediaResponse>(`/media?${query}`)
  },
  blobOccurrences: (sha256: string, cursor?: string) => {
    const query = new URLSearchParams({ limit: '100' })
    addOptional(query, 'cursor', cursor)
    return request<BlobOccurrencesResponse>(
      `/media/blobs/${encodeURIComponent(sha256)}/occurrences?${query}`,
    )
  },
  activity: (options: ActivityQuery = {}) => {
    const query = new URLSearchParams()
    addOptional(query, 'q', options.q?.trim())
    addOptional(query, 'category', options.category)
    addOptional(query, 'subtype', options.subtype)
    addOptional(query, 'availability', options.availability)
    addOptional(query, 'from_ms', options.fromMs)
    addOptional(query, 'to_ms', options.toMs)
    addOptional(query, 'has_attachments', options.hasAttachments)
    addOptional(query, 'logical_only', options.logicalOnly)
    addOptional(query, 'post_context', options.postContext)
    addOptional(query, 'sort', options.sort ?? 'date')
    addOptional(query, 'direction', options.direction ?? 'desc')
    addOptional(query, 'limit', options.limit ?? 75)
    addOptional(query, 'cursor', options.cursor)
    const path = options.view && options.view !== 'all' ? `/activity/views/${encodeURIComponent(options.view)}` : '/activity'
    return request<ActivityResponse>(`${path}?${query}`)
  },
  timeline: (options: TimelineQuery = {}) => {
    const query = new URLSearchParams()
    addOptional(query, 'q', options.q?.trim())
    addOptional(query, 'entity_type', options.entityType)
    addOptional(query, 'from_ms', options.fromMs)
    addOptional(query, 'to_ms', options.toMs)
    addOptional(query, 'direction', options.direction ?? 'desc')
    addOptional(query, 'limit', options.limit ?? 75)
    addOptional(query, 'cursor', options.cursor)
    addOptional(query, 'around', options.around)
    return request<TimelineResponse>(`/timeline?${query}`)
  },
  onThisDay: (month: number, day: number, timezone: string, entityType?: string, cursor?: string, limit = 100) => {
    const query = new URLSearchParams({ month: String(month), day: String(day), timezone })
    addOptional(query, 'entity_type', entityType)
    addOptional(query, 'limit', limit)
    addOptional(query, 'cursor', cursor)
    return request<OnThisDayResponse>(`/timeline/on-this-day?${query}`)
  },
  events: (options: EventQuery = {}) => {
    const query = new URLSearchParams()
    addOptional(query, 'q', options.q?.trim())
    addOptional(query, 'event_status', options.status)
    addOptional(query, 'hosted', options.hosted)
    addOptional(query, 'from_ms', options.fromMs)
    addOptional(query, 'to_ms', options.toMs)
    addOptional(query, 'sort', options.sort ?? 'date')
    addOptional(query, 'direction', options.direction ?? 'desc')
    addOptional(query, 'limit', options.limit ?? 75)
    addOptional(query, 'cursor', options.cursor)
    addOptional(query, 'around', options.around)
    return request<EventsResponse>(`/events?${query}`)
  },
  people: (options: PeopleQuery = {}) => {
    const query = new URLSearchParams()
    addOptional(query, 'view', options.view ?? 'contacts')
    addOptional(query, 'q', options.q?.trim())
    addOptional(query, 'conversation_type', options.conversationType)
    addOptional(query, 'sort', options.sort ?? 'name')
    addOptional(query, 'direction', options.direction ?? 'asc')
    addOptional(query, 'limit', options.limit ?? 100)
    addOptional(query, 'cursor', options.cursor)
    addOptional(query, 'around', options.around)
    return request<PeopleResponse>(`/people?${query}`)
  },
  support: (options: { cursor?: string; limit?: number } = {}) => {
    const query = new URLSearchParams({ limit: String(options.limit ?? 50) })
    addOptional(query, 'cursor', options.cursor)
    return request<SupportResponse>(`/support?${query}`)
  },
  analyticsOverview: (timezone = 'America/New_York') => {
    const query = new URLSearchParams({ timezone })
    return request<AnalyticsOverviewResponse>(`/analytics/overview?${query}`)
  },
  analyticsMessages: (options: AnalyticsQuery = {}) => {
    const query = new URLSearchParams()
    addOptional(query, 'bucket', options.bucket ?? 'month')
    addOptional(query, 'timezone', options.timezone ?? 'America/New_York')
    addOptional(query, 'from_ms', options.fromMs)
    addOptional(query, 'to_ms', options.toMs)
    addOptional(query, 'conversation_id', options.conversationId)
    addOptional(query, 'limit', options.limit)
    addOptional(query, 'cursor', options.cursor)
    return request<AnalyticsMessagesResponse>(`/analytics/messages?${query}`)
  },
  analyticsPatterns: (
    timezone = 'America/New_York',
    options: Pick<AnalyticsQuery, 'fromMs' | 'toMs' | 'limit' | 'cursor'> = {},
  ) => {
    const query = new URLSearchParams({ timezone })
    addOptional(query, 'from_ms', options.fromMs)
    addOptional(query, 'to_ms', options.toMs)
    addOptional(query, 'limit', options.limit)
    addOptional(query, 'cursor', options.cursor)
    return request<AnalyticsPatternsResponse>(`/analytics/activity-patterns?${query}`)
  },
  analyticsInteractionPatterns: (
    bucket: AnalyticsQuery['bucket'] = 'month',
    timezone = 'America/New_York',
    options: Pick<AnalyticsQuery, 'fromMs' | 'toMs' | 'limit' | 'cursor'> = {},
  ) => {
    const query = new URLSearchParams({ bucket: bucket ?? 'month', timezone })
    addOptional(query, 'from_ms', options.fromMs)
    addOptional(query, 'to_ms', options.toMs)
    addOptional(query, 'limit', options.limit)
    addOptional(query, 'cursor', options.cursor)
    return request<AnalyticsInteractionPatternsResponse>(`/analytics/interaction-patterns?${query}`)
  },
  analyticsInteractionSeries: (
    bucket: AnalyticsQuery['bucket'] = 'month',
    timezone = 'America/New_York',
    options: Pick<AnalyticsQuery, 'fromMs' | 'toMs' | 'limit' | 'cursor'> & { domainLimit?: number } = {},
  ) => {
    const query = new URLSearchParams({ bucket: bucket ?? 'month', timezone })
    addOptional(query, 'from_ms', options.fromMs)
    addOptional(query, 'to_ms', options.toMs)
    addOptional(query, 'domain_limit', options.domainLimit)
    addOptional(query, 'limit', options.limit)
    addOptional(query, 'cursor', options.cursor)
    return request<AnalyticsInteractionSeriesResponse>(`/analytics/interaction-series?${query}`)
  },
  analyticsNetwork: (limit = 40) =>
    request<AnalyticsNetworkResponse>(`/analytics/network?limit=${limit}`),
  analyticsNlpTrends: (
    bucket: 'hour' | 'day' | 'week' | 'month' | 'year',
    timezone: string,
    options: { direction?: string; fromMs?: number; toMs?: number; limit?: number; cursor?: string } = {},
  ) => {
    const query = new URLSearchParams({ bucket, timezone })
    addOptional(query, 'direction', options.direction)
    addOptional(query, 'from_ms', options.fromMs)
    addOptional(query, 'to_ms', options.toMs)
    addOptional(query, 'limit', options.limit)
    addOptional(query, 'cursor', options.cursor)
    return request<AnalyticsNlpTrendsResponse>(`/analytics/nlp-trends?${query}`)
  },
  analyticsConflictRate: (
    bucket: 'hour' | 'day' | 'week' | 'month' | 'year',
    timezone: string,
    options: { fromMs?: number; toMs?: number; limit?: number; cursor?: string } = {},
  ) => {
    const query = new URLSearchParams({ bucket, timezone })
    addOptional(query, 'from_ms', options.fromMs)
    addOptional(query, 'to_ms', options.toMs)
    addOptional(query, 'limit', options.limit)
    addOptional(query, 'cursor', options.cursor)
    return request<AnalyticsConflictRateResponse>(`/analytics/conflict-rate?${query}`)
  },
  analyticsTopics: () => request<AnalyticsTopicsResponse>('/analytics/topics'),
  buildAnalyticsTopics: (options: { cluster_count?: number; sample_limit?: number } = {}) =>
    request<{ job_id: string; status: string; status_href: string }>('/analytics/topics/build', {
      method: 'POST',
      body: JSON.stringify(options),
    }),
  analyticsAttachments: (
    bucket: AnalyticsQuery['bucket'] = 'month',
    timezone = 'America/New_York',
    options: Pick<AnalyticsQuery, 'fromMs' | 'toMs' | 'limit' | 'cursor'> = {},
  ) => {
    const query = new URLSearchParams({ bucket: bucket ?? 'month', timezone })
    addOptional(query, 'from_ms', options.fromMs)
    addOptional(query, 'to_ms', options.toMs)
    addOptional(query, 'limit', options.limit)
    addOptional(query, 'cursor', options.cursor)
    return request<AnalyticsAttachmentsResponse>(`/analytics/attachments?${query}`)
  },
  analyticsConversations: () => request<AnalyticsConversationsResponse>('/analytics/conversations'),
  analyticsStorage: () => request<AnalyticsStorageResponse>('/analytics/storage'),
  analyticsCoverage: () => request<AnalyticsCoverageResponse>('/analytics/coverage'),
  enrichmentStatus: () => request<EnrichmentStatusResponse>('/enrichment/status'),
  startAudioTranscriptionJob: (retryFailed = true) =>
    request<{ job_id: string; status: string; status_href: string }>('/enrichment/transcriptions/jobs', { method: 'POST', body: JSON.stringify({ retry_failed: retryFailed }) }),
  startAttachmentTextJob: (retryFailed = true) =>
    request<{ job_id: string; status: string; status_href: string }>('/enrichment/attachment-text/jobs', { method: 'POST', body: JSON.stringify({ retry_failed: retryFailed }) }),
  startNlpAnnotationJob: (tasks: Array<'sentiment' | 'emotion' | 'toxicity'> = ['sentiment', 'emotion', 'toxicity'], batchSize = 16) =>
    request<{ job_id: string; status: string; status_href: string }>('/enrichment/nlp/jobs', { method: 'POST', body: JSON.stringify({ tasks, batch_size: batchSize }) }),
  startConflictAggregateJob: (batchSize = 32) =>
    request<{ job_id?: string | null; status: string; status_href: string; reused?: boolean }>('/enrichment/conflict/aggregate-jobs', { method: 'POST', body: JSON.stringify({ batch_size: batchSize }) }),
  audioTranscriptions: (status = 'all', offset = 0, limit = 25, version: 'current' | 'history' = 'current') => {
    const query = new URLSearchParams({ status, offset: String(offset), limit: String(limit), version })
    return request<AudioTranscriptionsResponse>(`/enrichment/transcriptions?${query}`)
  },
  attachmentTextExtractions: (status = 'all', offset = 0, limit = 25, version: 'current' | 'history' = 'current') => {
    const query = new URLSearchParams({ status, offset: String(offset), limit: String(limit), version })
    return request<AttachmentTextExtractionsResponse>(`/enrichment/attachment-text?${query}`)
  },
  nlpAnnotations: (task = 'all', offset = 0, limit = 50) => {
    const query = new URLSearchParams({ task, offset: String(offset), limit: String(limit) })
    return request<NlpAnnotationsResponse>(`/enrichment/nlp?${query}`)
  },
  conflictSearch: (options: { q: string; label: ConflictLabel; participant?: string; conversationId?: string; limit?: number }) => {
    const query = new URLSearchParams({ q: options.q, label: options.label, limit: String(options.limit ?? 25) })
    addOptional(query, 'participant', options.participant?.trim())
    addOptional(query, 'conversation_id', options.conversationId?.trim())
    return request<ConflictSearchResponse>(`/enrichment/conflict/search?${query}`)
  },
  conflictReviewStatus: () => request<ConflictReviewStatus>('/enrichment/conflict/review/status'),
  conflictReviewQueue: (options: { label?: ConflictReviewLabelFilter; split?: 'all' | ConflictEvaluationSplit; after?: string; limit?: number } = {}) => {
    const query = new URLSearchParams({
      label: options.label ?? 'unlabelled',
      split: options.split ?? 'all',
      limit: String(options.limit ?? 12),
    })
    addOptional(query, 'after', options.after)
    return request<ConflictReviewQueueResponse>(`/enrichment/conflict/review/queue?${query}`)
  },
  nextConflictReviewItem: (options: { split?: 'all' | ConflictEvaluationSplit; after?: string } = {}) => {
    const query = new URLSearchParams({ split: options.split ?? 'all' })
    addOptional(query, 'after', options.after)
    return request<ConflictReviewNextResponse>(`/enrichment/conflict/review/next?${query}`)
  },
  createConflictLabel: (payload: { window_id: string; label: ConflictLabel; notes?: string }) =>
    request<ConflictLabelResponse>('/enrichment/conflict/labels', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  createConflictCalibration: () => request<ConflictCalibrationResponse>('/enrichment/conflict/calibrations', {
    method: 'POST',
    body: JSON.stringify({}),
  }),
  deletionOperations: (options: { status?: string; cursor?: string; limit?: number } = {}) => {
    const query = new URLSearchParams({ limit: String(options.limit ?? 50) })
    addOptional(query, 'status', options.status)
    addOptional(query, 'cursor', options.cursor)
    return request<DeletionOperationsResponse>(`/deletions?${query}`)
  },
  deletionOperation: (operationId: string) =>
    request<DeletionOperation>(`/deletions/${encodeURIComponent(operationId)}`),
  deletionSelection: (
    operationId: string,
    revision: string,
    options: { cursor?: string; limit?: number } = {},
  ) => {
    const query = new URLSearchParams({ revision, limit: String(options.limit ?? 25) })
    addOptional(query, 'cursor', options.cursor)
    return request<DeletionSelectionResponse>(
      `/deletions/${encodeURIComponent(operationId)}/selection?${query}`,
    )
  },
  previewDeletion: (payload: DeletionPreviewRequest) =>
    request<DeletionPreviewResponse>('/deletions/preview', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  commitDeletion: (operationId: string, commitToken: string, catalogRevision: string) =>
    request<DeletionMutationResult>(`/deletions/${encodeURIComponent(operationId)}/commit`, {
      method: 'POST',
      body: JSON.stringify({ commit_token: commitToken, catalog_revision: catalogRevision }),
    }),
  restoreDeletion: (operationId: string) =>
    request<DeletionMutationResult>(`/deletions/${encodeURIComponent(operationId)}/restore`, {
      method: 'POST',
      body: JSON.stringify({ confirmation: operationId }),
    }),
  previewHardPurge: (operationId: string, scope: 'normalized' | 'workspace' = 'workspace') =>
    request<HardPurgePlan>(`/deletions/${encodeURIComponent(operationId)}/purge/preview`, {
      method: 'POST',
      body: JSON.stringify({ scope }),
    }),
  commitHardPurge: (plan: HardPurgePlan) =>
    request<DeletionMutationResult>(`/deletions/${encodeURIComponent(plan.operation_id)}/purge`, {
      method: 'POST',
      body: JSON.stringify({
        scope: plan.scope,
        purge_token: plan.purge_token,
        purge_revision: plan.purge_revision,
        confirmation: plan.operation_id,
      }),
    }),
}
