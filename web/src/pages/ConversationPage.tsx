import { useInfiniteQuery, useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Link, useNavigate, useParams, useSearch } from '@tanstack/react-router'
import { useVirtualizer } from '@tanstack/react-virtual'
import {
  ArrowDown,
  ArrowLeft,
  ArrowUp,
  CalendarSearch,
  CheckCircle2,
  Clock3,
  FileQuestion,
  Files,
  Flag,
  GitMerge,
  Info,
  LockKeyhole,
  MessageSquareText,
  Phone,
  RotateCcw,
  Search,
  Scissors,
  Users,
  X,
} from 'lucide-react'
import { type FormEvent, type UIEvent, useEffect, useMemo, useRef, useState } from 'react'
import { AttachmentPreview } from '../components/AttachmentPreview'
import { EmptyState, ErrorState, PageLoading, StatusBadge } from '../components/ui'
import { api } from '../lib/api'
import type { ConversationMessage, ConversationSegment, ConversationSummary } from '../lib/contracts'
import {
  formatBytes,
  formatCount,
  formatDateMs,
  formatSeconds,
  formatTimestampMs,
  localDateToTimestamp,
  sentenceCase,
  stringifyValue,
} from '../lib/format'
import { useArchiveTimezone } from '../lib/useArchiveTimezone'

function AdvancedDetails({ conversationId, message }: { conversationId: string; message: ConversationMessage }) {
  const [revealed, setRevealed] = useState(false)
  const advanced = useQuery({
    queryKey: ['message-advanced', conversationId, message.id],
    queryFn: () => api.messageAdvanced(conversationId, message.id),
    enabled: revealed,
    staleTime: Infinity,
  })

  if (!message.has_payment_details && !message.has_sensitive_ip) return null
  if (!revealed) {
    return (
      <button className="advanced-reveal" type="button" onClick={() => setRevealed(true)}>
        <LockKeyhole size={13} aria-hidden="true" />
        Show advanced details
      </button>
    )
  }
  if (advanced.isLoading) return <span className="message-note" role="status">Loading private details…</span>
  if (advanced.isError) {
    return <span className="message-note message-note--error">Private details could not be loaded.</span>
  }
  return (
    <div className="advanced-details">
      <div><LockKeyhole size={14} aria-hidden="true" /><strong>Private advanced details</strong></div>
      <p>{advanced.data?.notice}</p>
      <dl>
        {Object.entries(advanced.data?.sensitive ?? {}).map(([key, value]) => (
          <div key={key}><dt>{sentenceCase(key)}</dt><dd>{stringifyValue(value)}</dd></div>
        ))}
      </dl>
      <button type="button" onClick={() => setRevealed(false)}>Hide details</button>
    </div>
  )
}

function MessageCard({
  message,
  conversationId,
  timezone,
  segmentBoundary,
  focused,
  focusedAttachmentId,
  mediaStartMs,
  position,
  setSize,
  leadingMetadataHidden,
}: {
  message: ConversationMessage
  conversationId: string
  timezone: string
  segmentBoundary: boolean
  focused: boolean
  focusedAttachmentId?: string
  mediaStartMs?: number
  position?: number
  setSize: number
  leadingMetadataHidden?: boolean
}) {
  const emptyBody = !message.content && !message.transcription
  return (
    <article
      id={`message-${message.id}`}
      className={`message-row ${message.sender_is_owner ? 'message-row--owner' : ''} ${focused ? 'message-row--focused' : ''}`}
      aria-label={`Message from ${message.sender} on ${formatTimestampMs(message.timestamp_ms, timezone)}`}
      aria-posinset={position}
      aria-setsize={setSize}
      tabIndex={focused ? -1 : undefined}
    >
      {segmentBoundary ? (
        <div
          className="segment-boundary"
          role="separator"
          aria-hidden={leadingMetadataHidden || undefined}
          style={leadingMetadataHidden ? { visibility: 'hidden' } : undefined}
        >
          <span />
          <strong>{message.segment?.title || 'Conversation segment'}</strong>
          <small>{sentenceCase(message.segment?.source_category)}</small>
          <span />
        </div>
      ) : null}
      <div
        className="message-row__meta"
        aria-hidden={leadingMetadataHidden || undefined}
        style={leadingMetadataHidden ? { visibility: 'hidden' } : undefined}
      >
        <strong>{message.sender || 'Unknown sender'}</strong>
        {message.sender_is_owner ? <span>You</span> : null}
        <time dateTime={new Date(message.timestamp_ms).toISOString()}>{formatTimestampMs(message.timestamp_ms, timezone)}</time>
      </div>
      <div className="message-bubble">
        <div className="message-kind"><span>{sentenceCase(message.kind)}</span>{message.is_duplicate ? <span>Duplicate record</span> : null}</div>
        {message.is_unsent ? (
          <p className="message-placeholder"><FileQuestion size={15} aria-hidden="true" /> This message was unsent.</p>
        ) : message.is_unavailable_body ? (
          <p className="message-placeholder"><FileQuestion size={15} aria-hidden="true" /> Message body is unavailable in the export.</p>
        ) : null}
        {message.content ? <p className="message-content">{message.content}</p> : null}
        {message.transcription ? <p className="message-transcription"><strong>Transcript</strong>{message.transcription}</p> : null}
        {message.kind === 'call' && message.call ? (
          <p className="call-detail"><Phone size={15} aria-hidden="true" /> {message.call.missed ? 'Missed call' : `Call · ${formatSeconds(message.call.duration_seconds)}`}</p>
        ) : null}
        {message.kind === 'payment' && emptyBody ? <p className="message-placeholder"><LockKeyhole size={15} aria-hidden="true" /> Payment activity — details hidden.</p> : null}
        {emptyBody && !message.is_unsent && !message.is_unavailable_body && message.kind !== 'call' && message.kind !== 'payment' && message.attachments.length === 0 ? (
          <p className="message-placeholder"><Info size={15} aria-hidden="true" /> No text was preserved for this {sentenceCase(message.kind).toLowerCase()} record.</p>
        ) : null}
        {message.attachments.length > 0 ? (
          <div className="message-attachments" role="group" aria-label={`${message.attachments.length} attachments`}>
            {message.attachments.map((attachment) => (
              <div
                id={`attachment-${attachment.id}`}
                className={`message-attachment ${focusedAttachmentId === attachment.id ? 'message-attachment--focused' : ''}`}
                key={attachment.id}
                tabIndex={focusedAttachmentId === attachment.id ? -1 : undefined}
              >
                <AttachmentPreview
                  item={attachment}
                  compact
                  seekMs={focusedAttachmentId === attachment.id ? mediaStartMs : undefined}
                />
                <div>
                  <strong>{attachment.filename || sentenceCase(attachment.kind)}</strong>
                  <span>{formatBytes(attachment.size_bytes)} · {sentenceCase(attachment.availability)}</span>
                </div>
              </div>
            ))}
          </div>
        ) : null}
        {message.reactions.length > 0 ? (
          <ul className="reaction-list" aria-label="Reactions">
            {message.reactions.map((reaction) => <li key={reaction.id}><span>{reaction.reaction}</span>{reaction.actor}</li>)}
          </ul>
        ) : null}
        <AdvancedDetails conversationId={conversationId} message={message} />
      </div>
    </article>
  )
}

function MessageTimeline({
  messages,
  total,
  conversationId,
  timezone,
  focusId,
  focusAttachmentId,
  mediaStartMs,
  canLoadOlder,
  canLoadNewer,
  loadingOlder,
  loadingNewer,
  onLoadOlder,
  onLoadNewer,
  initialEdge,
}: {
  messages: ConversationMessage[]
  total: number
  conversationId: string
  timezone: string
  focusId?: string
  focusAttachmentId?: string
  mediaStartMs?: number
  canLoadOlder: boolean
  canLoadNewer: boolean
  loadingOlder: boolean
  loadingNewer: boolean
  onLoadOlder: () => Promise<unknown>
  onLoadNewer: () => Promise<unknown>
  initialEdge: 'latest' | 'earliest'
}) {
  const viewportRef = useRef<HTMLDivElement>(null)
  const restoreIdRef = useRef<string | undefined>(undefined)
  const loadLockRef = useRef<'older' | 'newer' | undefined>(undefined)
  const initializedRef = useRef(false)
  const virtualizer = useVirtualizer({
    count: messages.length,
    getScrollElement: () => viewportRef.current,
    estimateSize: () => 150,
    overscan: 6,
    initialRect: { width: 840, height: 670 },
    getItemKey: (index) => messages[index]?.id ?? index,
  })
  const initialFocusRef = useRef<string | undefined>(undefined)
  const virtualRows = virtualizer.getVirtualItems()
  const renderedRows = virtualRows.length > 0
    ? virtualRows
    : messages.slice(0, 12).map((_message, index) => ({
      index,
      key: messages[index]?.id ?? index,
      start: index * 150,
      end: (index + 1) * 150,
      size: 150,
      lane: 0,
    }))

  useEffect(() => {
    const restoreId = restoreIdRef.current
    const targetId = restoreId ?? focusId
    const focusKey = `${targetId ?? ''}:${focusAttachmentId ?? ''}:${mediaStartMs ?? ''}`
    if (!targetId || (!restoreId && initialFocusRef.current === focusKey)) return
    const index = messages.findIndex((message) => message.id === targetId)
    if (index < 0) return
    virtualizer.scrollToIndex(index, { align: restoreId ? 'start' : 'center' })
    if (restoreId) {
      restoreIdRef.current = undefined
      return
    }

    let cancelled = false
    let frame = 0
    let attempt = 0
    const revealAndFocus = () => {
      if (cancelled) return
      const attachment = focusAttachmentId
        ? document.getElementById(`attachment-${focusAttachmentId}`)
        : null
      const target = attachment ?? document.getElementById(`message-${targetId}`)
      if (target && !target.closest('[aria-hidden="true"]')) {
        initialFocusRef.current = focusKey
        target.focus()
        if (typeof target.scrollIntoView === 'function') target.scrollIntoView({ block: 'center' })
        return
      }

      // The virtualizer may not have observed its scroll element on the first effect pass.
      // Seed the viewport with the same conservative row estimate, then allow it to measure
      // and correct before focusing the stable message or attachment anchor.
      if (attempt === 0 && viewportRef.current) {
        viewportRef.current.scrollTop = Math.max(0, index * 150 - viewportRef.current.clientHeight / 2)
      }
      attempt += 1
      if (attempt < 8) frame = requestAnimationFrame(revealAndFocus)
    }
    frame = requestAnimationFrame(revealAndFocus)
    return () => {
      cancelled = true
      cancelAnimationFrame(frame)
    }
  }, [focusAttachmentId, focusId, mediaStartMs, messages, virtualizer])

  useEffect(() => {
    if (initializedRef.current || focusId || messages.length === 0) return
    initializedRef.current = true
    requestAnimationFrame(() => {
      virtualizer.scrollToIndex(initialEdge === 'latest' ? messages.length - 1 : 0, {
        align: initialEdge === 'latest' ? 'end' : 'start',
      })
    })
  }, [focusId, initialEdge, messages.length, virtualizer])

  const rememberVisibleMessage = () => {
    const firstVisible = virtualizer.getVirtualItems()[0]
    restoreIdRef.current = firstVisible ? messages[firstVisible.index]?.id : undefined
  }

  const loadOlder = async () => {
    if (!canLoadOlder || loadingOlder || loadLockRef.current) return
    loadLockRef.current = 'older'
    rememberVisibleMessage()
    try {
      await onLoadOlder()
    } finally {
      loadLockRef.current = undefined
    }
  }

  const loadNewer = async () => {
    if (!canLoadNewer || loadingNewer || loadLockRef.current) return
    loadLockRef.current = 'newer'
    rememberVisibleMessage()
    try {
      await onLoadNewer()
    } finally {
      loadLockRef.current = undefined
    }
  }

  const autoLoadAtEdge = (event: UIEvent<HTMLDivElement>) => {
    const viewport = event.currentTarget
    const edgeThreshold = Math.max(220, viewport.clientHeight * 0.35)
    if (viewport.scrollTop <= edgeThreshold) {
      void loadOlder()
    } else if (viewport.scrollHeight - viewport.clientHeight - viewport.scrollTop <= edgeThreshold) {
      void loadNewer()
    }
  }
  const knownPositionBase = !canLoadOlder
    ? 0
    : !canLoadNewer
      ? Math.max(0, total - messages.length)
      : undefined
  const viewportStart = virtualizer.scrollOffset ?? 0
  const viewportHeight = virtualizer.scrollRect?.height ?? viewportRef.current?.clientHeight ?? 0
  const viewportEnd = viewportStart + viewportHeight

  return (
    <div className="timeline-shell">
      <div className="timeline-load-row">
        {canLoadOlder ? <button className="button button--secondary" type="button" disabled={loadingOlder} onClick={() => void loadOlder()}><ArrowUp size={14} />{loadingOlder ? 'Loading older…' : 'Load older now'}</button> : <span>Beginning of conversation</span>}
        <span>{formatCount(messages.length)} cached of {formatCount(total)} · scroll to either edge to continue</span>
      </div>
      <div
        className="message-timeline"
        ref={viewportRef}
        role="feed"
        aria-label="Conversation messages"
        aria-busy={loadingOlder || loadingNewer}
        data-rendered-message-count={renderedRows.length}
        data-loaded-message-count={messages.length}
        data-acceptance-message-projection={JSON.stringify(messages.map((message) => ({
          id: message.id,
          timestamp_ms: message.timestamp_ms,
          kind: message.kind,
          attachment_count: message.attachments.length,
          sender_is_owner: Boolean(message.sender_is_owner),
        })))}
        tabIndex={0}
        onScroll={autoLoadAtEdge}
      >
        <div className="message-timeline__canvas" style={{ height: virtualizer.getTotalSize() }}>
          {renderedRows.map((virtualRow) => {
            const message = messages[virtualRow.index]!
            const prior = messages[virtualRow.index - 1]
            const outsideViewport = viewportHeight > 0 && (
              virtualRow.end <= viewportStart || virtualRow.start >= viewportEnd
            )
            return (
              <div
                className="message-timeline__item"
                key={message.id}
                data-index={virtualRow.index}
                aria-hidden={outsideViewport || undefined}
                ref={virtualizer.measureElement}
                style={{
                  transform: `translateY(${virtualRow.start}px)`,
                  visibility: outsideViewport ? 'hidden' : undefined,
                }}
              >
                <MessageCard
                  message={message}
                  conversationId={conversationId}
                  timezone={timezone}
                  segmentBoundary={!prior || prior.segment?.id !== message.segment?.id}
                  focused={focusId === message.id}
                  focusedAttachmentId={focusId === message.id ? focusAttachmentId : undefined}
                  mediaStartMs={mediaStartMs}
                  position={knownPositionBase == null ? undefined : knownPositionBase + virtualRow.index + 1}
                  setSize={total}
                  leadingMetadataHidden={viewportHeight > 0 && virtualRow.start < viewportStart}
                />
              </div>
            )
          })}
        </div>
      </div>
      <div className="timeline-load-row timeline-load-row--bottom">
        {canLoadNewer ? <button className="button button--secondary" type="button" disabled={loadingNewer} onClick={() => void loadNewer()}><ArrowDown size={14} />{loadingNewer ? 'Loading newer…' : 'Load newer now'}</button> : <span>End of conversation</span>}
      </div>
    </div>
  )
}

function SegmentList({ segments, timezone }: { segments: ConversationSegment[]; timezone: string }) {
  return (
    <details className="conversation-segments panel">
      <summary><span><Flag size={15} aria-hidden="true" /> Source segments</span><small>{formatCount(segments.length)}</small></summary>
      <div className="segment-list">
        {segments.map((segment) => (
          <article key={segment.id}>
            <div><strong>{segment.title || 'Untitled segment'}</strong><StatusBadge status={segment.completeness} /></div>
            <p>{sentenceCase(segment.source_category)} · {formatDateMs(segment.first_activity_ms, timezone)} to {formatDateMs(segment.last_activity_ms, timezone)}</p>
            <span>{segment.participants.map((participant) => participant.is_owner ? `${participant.name} (you)` : participant.name).join(', ') || 'No participant labels'}</span>
            {segment.assignment ? <small>{segment.assignment.status === 'corrected' ? 'Manually assigned · source provenance unchanged' : 'Provisional exact-label assignment'}</small> : null}
          </article>
        ))}
      </div>
    </details>
  )
}

function IdentityCorrectionPanel({
  conversation,
  segments,
  onRedirect,
}: {
  conversation: ConversationSummary
  segments: ConversationSegment[]
  onRedirect: (conversationId: string) => Promise<void>
}) {
  const queryClient = useQueryClient()
  const [mode, setMode] = useState<'merge' | 'split'>('split')
  const [selectedSegments, setSelectedSegments] = useState<string[]>([])
  const [splitTitle, setSplitTitle] = useState('')
  const [reason, setReason] = useState('')
  const [mergeDraft, setMergeDraft] = useState('')
  const [mergeQuery, setMergeQuery] = useState('')
  const [mergeTarget, setMergeTarget] = useState('')
  const [correctionsRequested, setCorrectionsRequested] = useState(false)
  const history = useQuery({
    queryKey: ['conversation-corrections', conversation.id],
    queryFn: () => api.conversationCorrections(conversation.id),
    enabled: correctionsRequested,
  })
  const candidates = useQuery({
    queryKey: ['conversation-correction-candidates', mergeQuery],
    queryFn: () => api.conversations({ q: mergeQuery, conversationType: 'direct', sort: 'title', direction: 'asc', limit: 75 }),
    enabled: mergeQuery.length > 0,
  })
  const refreshCorrectionViews = async () => {
    await Promise.all([
      queryClient.invalidateQueries({ queryKey: ['conversations'] }),
      queryClient.invalidateQueries({ queryKey: ['conversation'] }),
      queryClient.invalidateQueries({ queryKey: ['conversation-messages'] }),
      queryClient.invalidateQueries({ queryKey: ['conversation-corrections'] }),
      queryClient.invalidateQueries({ queryKey: ['media'] }),
      queryClient.invalidateQueries({ queryKey: ['search'] }),
    ])
  }
  const split = useMutation({
    mutationFn: () => api.splitConversationHistory({
      conversation_id: conversation.id,
      segment_ids: selectedSegments,
      title: splitTitle.trim(),
      identity_revision: history.data!.identity_revision,
      reason: reason.trim() || undefined,
    }),
    onSuccess: async (result) => {
      await refreshCorrectionViews()
      await onRedirect(result.redirect_conversation_id)
    },
  })
  const merge = useMutation({
    mutationFn: () => api.mergeConversationHistories({
      conversation_ids: [conversation.id, mergeTarget],
      target_conversation_id: mergeTarget,
      identity_revision: history.data!.identity_revision,
      reason: reason.trim() || undefined,
    }),
    onSuccess: async (result) => {
      await refreshCorrectionViews()
      await onRedirect(result.redirect_conversation_id)
    },
  })
  const revert = useMutation({
    mutationFn: (correctionId: string) => api.revertConversationCorrection(correctionId, history.data!.identity_revision),
    onSuccess: async (result) => {
      await refreshCorrectionViews()
      await onRedirect(result.redirect_conversation_id)
    },
  })
  const error = split.error ?? merge.error ?? revert.error
  const candidateItems = (candidates.data?.items ?? []).filter((item) => item.id !== conversation.id && item.type === 'direct')
  const splitInvalid = selectedSegments.length === 0 || selectedSegments.length >= segments.length || !splitTitle.trim()

  return (
    <details
      className="conversation-corrections panel"
      onToggle={(event) => {
        if (event.currentTarget.open) setCorrectionsRequested(true)
      }}
    >
      <summary>
        <span><GitMerge size={15} aria-hidden="true" /> Direct-history identity</span>
        <StatusBadge status={conversation.identity?.status ?? 'provisional'} />
      </summary>
      <div className="conversation-corrections__body">
        <div className="conversation-corrections__notice" role="note">
          <Info size={17} aria-hidden="true" />
          <p><strong>{conversation.identity?.status === 'corrected' ? 'This history includes a manual correction.' : 'This history is provisional.'}</strong><span>Facebook supplied names, not stable person IDs. Merge and split only change which source segments are browsed together; thread paths, messages, and source coordinates stay unchanged.</span></p>
        </div>
        <div className="segmented-control" role="group" aria-label="Identity correction type">
          <button type="button" className={mode === 'split' ? 'segmented-control__active' : ''} aria-pressed={mode === 'split'} onClick={() => setMode('split')}><Scissors size={13} /> Split segments</button>
          <button type="button" className={mode === 'merge' ? 'segmented-control__active' : ''} aria-pressed={mode === 'merge'} onClick={() => setMode('merge')}><GitMerge size={13} /> Merge history</button>
        </div>

        {mode === 'split' ? (
          <form className="conversation-correction-form" onSubmit={(event) => { event.preventDefault(); if (!splitInvalid && history.data) split.mutate() }}>
            <fieldset disabled={segments.length < 2 || split.isPending}>
              <legend>Move selected source segments into a new direct history</legend>
              <p>Select at least one segment and leave at least one here. The source data itself is not edited.</p>
              <div className="conversation-correction-segments">
                {segments.map((segment) => <label key={segment.id}><input type="checkbox" checked={selectedSegments.includes(segment.id)} onChange={(event) => setSelectedSegments((current) => event.target.checked ? [...current, segment.id] : current.filter((value) => value !== segment.id))} /><span><strong>{segment.title || 'Untitled segment'}</strong><small>{sentenceCase(segment.source_category)} · {segment.thread_path}</small></span></label>)}
              </div>
              <label className="filter-field"><span>New history title</span><input value={splitTitle} maxLength={240} onChange={(event) => setSplitTitle(event.target.value)} placeholder="Display name for the separated history" /></label>
              <label className="filter-field"><span>Reason (optional, saved to audit)</span><textarea value={reason} maxLength={1_000} rows={2} onChange={(event) => setReason(event.target.value)} /></label>
              <button className="button button--primary" type="submit" disabled={splitInvalid || !history.data || split.isPending}>{split.isPending ? 'Applying correction…' : 'Split selected segments'}</button>
            </fieldset>
          </form>
        ) : (
          <form className="conversation-correction-form" onSubmit={(event) => { event.preventDefault(); setMergeTarget(''); setMergeQuery(mergeDraft.trim()) }}>
            <fieldset disabled={merge.isPending}>
              <legend>Merge this entire history into another direct history</legend>
              <p>The selected target keeps its title. This history’s source segments, messages, and linked media move together.</p>
              <label className="filter-field"><span>Find target history</span><span className="filter-input"><Search size={14} /><input value={mergeDraft} onChange={(event) => setMergeDraft(event.target.value)} placeholder="Search counterpart display names" /></span></label>
              <button className="button button--secondary" type="submit" disabled={!mergeDraft.trim()}>Find direct histories</button>
              {candidates.isFetching ? <span role="status">Finding candidate histories…</span> : null}
              {mergeQuery && !candidates.isFetching ? <div className="conversation-correction-targets" role="radiogroup" aria-label="Merge target history">{candidateItems.length ? candidateItems.map((item) => <label key={item.id}><input type="radio" name="merge-target" value={item.id} checked={mergeTarget === item.id} onChange={() => setMergeTarget(item.id)} /><span><strong>{item.title}</strong><small>{formatCount(item.message_counts.total)} messages · {formatCount(item.segment_count)} source segments</small></span></label>) : <span>No other direct histories match that label.</span>}</div> : null}
              <label className="filter-field"><span>Reason (optional, saved to audit)</span><textarea value={reason} maxLength={1_000} rows={2} onChange={(event) => setReason(event.target.value)} /></label>
              <button className="button button--primary" type="button" disabled={!mergeTarget || !history.data || merge.isPending} onClick={() => merge.mutate()}>{merge.isPending ? 'Applying correction…' : 'Merge into selected history'}</button>
            </fieldset>
          </form>
        )}
        {error ? <p className="conversation-corrections__error" role="alert">{error instanceof Error ? error.message : 'The correction could not be applied.'}</p> : null}

        <section className="conversation-correction-history" aria-labelledby="correction-history-heading">
          <div><h3 id="correction-history-heading">Correction history</h3><p>Every apply and revert remains in the local audit ledger.</p></div>
          {history.isLoading ? <span role="status">Loading correction history…</span> : history.data?.items.length ? <ol>{history.data.items.map((item) => <li key={item.id}><div><StatusBadge status={item.status} /><strong>{sentenceCase(item.action)}</strong><time>{new Date(item.created_at).toLocaleString()}</time></div><p>{item.reason || `${formatCount(item.segments.length)} source segments reassigned.`}</p>{item.can_revert ? <button className="button button--secondary" type="button" disabled={revert.isPending} onClick={() => revert.mutate(item.id)}><RotateCcw size={13} /> Undo this correction</button> : null}</li>)}</ol> : <p>No manual corrections involve this history yet.</p>}
        </section>
      </div>
    </details>
  )
}

export function ConversationPage() {
  const { conversationId } = useParams({ from: '/conversations/$conversationId' })
  const { around, media, t } = useSearch({ from: '/conversations/$conversationId' })
  const navigate = useNavigate({ from: '/conversations/$conversationId' })
  const timezone = useArchiveTimezone()
  const [start, setStart] = useState<'latest' | 'earliest'>('latest')
  const [date, setDate] = useState('')
  const [dateJumping, setDateJumping] = useState(false)
  const [dateJumpError, setDateJumpError] = useState<string>()
  const [searchOpen, setSearchOpen] = useState(false)
  const [searchDraft, setSearchDraft] = useState('')
  const [searchQuery, setSearchQuery] = useState('')
  const [searchKind, setSearchKind] = useState('')
  const [searchSender, setSearchSender] = useState('')
  const detail = useQuery({ queryKey: ['conversation', conversationId], queryFn: () => api.conversation(conversationId) })
  const messagesQuery = useInfiniteQuery({
    queryKey: ['conversation-messages', conversationId, { around, start }],
    queryFn: ({ pageParam }) => api.conversationMessages(conversationId, pageParam ? { cursor: pageParam, limit: 100 } : { around, start, limit: 100 }),
    initialPageParam: undefined as string | undefined,
    getPreviousPageParam: (firstPage) => firstPage.page.before_cursor ?? undefined,
    getNextPageParam: (lastPage) => lastPage.page.after_cursor ?? undefined,
    maxPages: 6,
  })
  const messages = useMemo(() => {
    const seen = new Set<string>()
    return (messagesQuery.data?.pages.flatMap((page) => page.items) ?? [])
      .filter((message) => {
        if (seen.has(message.id)) return false
        seen.add(message.id)
        return true
      })
      .sort((left, right) => left.timestamp_ms - right.timestamp_ms || left.id.localeCompare(right.id))
  }, [messagesQuery.data])
  const messageTotal = messagesQuery.data?.pages[0]?.page.total ?? detail.data?.conversation.message_counts.total ?? 0
  const searchResults = useInfiniteQuery({
    queryKey: ['conversation-search', conversationId, searchQuery, searchKind, searchSender],
    queryFn: ({ pageParam }) => api.conversationSearch(conversationId, { q: searchQuery, kind: searchKind, sender: searchSender, cursor: pageParam, limit: 40 }),
    initialPageParam: undefined as string | undefined,
    getNextPageParam: (lastPage) => lastPage.page.next_cursor ?? undefined,
    enabled: searchQuery.length > 0,
  })
  const searchItems = useMemo(() => searchResults.data?.pages.flatMap((page) => page.items) ?? [], [searchResults.data])

  const jumpToMessage = async (messageId: string) => {
    setStart('latest')
    await navigate({ search: { around: messageId, media: undefined, t: undefined }, hash: `message-${messageId}`, replace: false })
  }
  const showEdge = async (edge: 'latest' | 'earliest') => {
    setStart(edge)
    await navigate({ search: { around: undefined, media: undefined, t: undefined }, hash: '', replace: true })
  }
  const submitDateJump = async (event: FormEvent) => {
    event.preventDefault()
    const atMs = localDateToTimestamp(date, timezone)
    if (atMs == null) return
    setDateJumping(true)
    setDateJumpError(undefined)
    try {
      const response = await api.conversationDateJump(conversationId, atMs)
      if (response.jump?.message_id) {
        await jumpToMessage(response.jump.message_id)
      } else {
        setDateJumpError('This conversation has no message near that date.')
      }
    } catch (error) {
      setDateJumpError(error instanceof Error ? error.message : 'The date jump could not be completed.')
    } finally {
      setDateJumping(false)
    }
  }
  const openCorrectedHistory = async (nextConversationId: string) => {
    await navigate({
      to: '/conversations/$conversationId',
      params: { conversationId: nextConversationId },
      search: { around: undefined, media: undefined, t: undefined },
      hash: '',
      replace: true,
    })
  }

  if (detail.isLoading || messagesQuery.isLoading) return <PageLoading label="Opening conversation history" />
  if (detail.isError) return <ErrorState title="Conversation could not be opened" error={detail.error} onRetry={() => detail.refetch()} />
  if (messagesQuery.isError) return <ErrorState title="Messages could not be loaded" error={messagesQuery.error} onRetry={() => messagesQuery.refetch()} />
  const conversation = detail.data?.conversation
  if (!conversation) return <EmptyState title="Conversation not found" description="This catalog entry is unavailable or was deleted." />

  return (
    <div className="page-stack conversation-reader-page">
      <nav className="reader-breadcrumb" aria-label="Breadcrumb"><Link to="/conversations"><ArrowLeft size={14} /> All conversations</Link><span>/</span><span aria-current="page">{conversation.title}</span></nav>
      <header className="reader-heading panel">
        <div className="reader-heading__title"><span className="reader-avatar" aria-hidden="true"><MessageSquareText size={22} /></span><div><p className="eyebrow">{sentenceCase(conversation.type)} conversation</p><h1>{conversation.title || 'Untitled conversation'}</h1><p>{conversation.source_categories.map(sentenceCase).join(' · ')}</p>{conversation.content_profile.kind === 'media_only' ? <StatusBadge status="media only" /> : null}{conversation.content_profile.kind === 'metadata_only' ? <StatusBadge status="metadata only" /> : null}{conversation.type === 'direct' ? <StatusBadge status={conversation.identity?.status ?? 'provisional'} /> : null}</div></div>
        <dl className="reader-heading__stats">
          <div><dt>Messages</dt><dd>{formatCount(conversation.message_counts.total)}</dd></div>
          <div><dt>Sent / received</dt><dd>{formatCount(conversation.message_counts.sent)} / {formatCount(conversation.message_counts.received)}</dd></div>
          <div><dt>People</dt><dd>{formatCount(conversation.participant_count)}</dd></div>
          <div><dt>Files</dt><dd>{formatCount(conversation.attachment_counts.occurrences)}</dd></div>
          <div><dt>Date range</dt><dd>{formatDateMs(conversation.first_activity_ms, timezone)} – {formatDateMs(conversation.last_activity_ms, timezone)}</dd></div>
          <div><dt>Coverage</dt><dd><StatusBadge status={conversation.completeness} /></dd></div>
        </dl>
      </header>

      <div className="reader-toolbar panel">
        <div className="reader-toolbar__group">
          <button className="button button--secondary" type="button" onClick={() => showEdge('earliest')}><ArrowUp size={14} /> Beginning</button>
          <button className="button button--secondary" type="button" onClick={() => showEdge('latest')}><ArrowDown size={14} /> Latest</button>
          <form className="date-jump" onSubmit={submitDateJump}><label><span className="sr-only">Jump to date</span><CalendarSearch size={15} aria-hidden="true" /><input type="date" value={date} onChange={(event) => setDate(event.target.value)} aria-describedby={dateJumpError ? 'date-jump-error' : undefined} /></label><button className="button button--secondary" type="submit" disabled={!date || dateJumping}>{dateJumping ? 'Finding…' : 'Jump'}</button>{dateJumpError ? <span id="date-jump-error" className="date-jump__error" role="alert">{dateJumpError}</span> : null}</form>
        </div>
        <div className="reader-toolbar__group">
          <Link className="button button--secondary" to="/media" search={{ conversation: conversationId }}><Files size={14} /> Conversation media</Link>
          <button className={`button ${searchOpen ? 'button--primary' : 'button--secondary'}`} type="button" aria-expanded={searchOpen} aria-controls="thread-search-panel" onClick={() => setSearchOpen((value) => !value)}><Search size={14} /> Search this conversation</button>
        </div>
      </div>

      {searchOpen ? (
        <section id="thread-search-panel" className="thread-search panel" aria-labelledby="thread-search-heading">
          <div className="section-header"><div><h2 id="thread-search-heading">Search this conversation</h2><p>Keyword matches cover message text, transcripts, and sender names. Private advanced fields stay excluded.</p></div><button className="icon-button" type="button" aria-label="Close conversation search" onClick={() => setSearchOpen(false)}><X size={18} /></button></div>
          <form onSubmit={(event) => { event.preventDefault(); setSearchQuery(searchDraft.trim()) }}>
            <label className="search-field"><span className="sr-only">Search words in this history</span><Search size={16} /><input type="search" value={searchDraft} onChange={(event) => setSearchDraft(event.target.value)} placeholder="Search words in this history" /></label>
            <label className="filter-field"><span>Message kind</span><input value={searchKind} onChange={(event) => setSearchKind(event.target.value)} placeholder="e.g. call, payment" /></label>
            <label className="filter-field"><span>Sender</span><input value={searchSender} onChange={(event) => setSearchSender(event.target.value)} placeholder="Exact sender name" /></label>
            <button className="button button--primary" type="submit" disabled={!searchDraft.trim()}>Search</button>
          </form>
          {searchQuery ? (
            searchResults.isLoading ? <div className="inline-loading" role="status">Searching local messages…</div> : searchResults.isError ? <ErrorState error={searchResults.error} onRetry={() => searchResults.refetch()} /> : searchItems.length === 0 ? <EmptyState title="No keyword matches" description={`No messages in this conversation contain “${searchQuery}”.`} /> : (
              <div className="thread-search-results">
                <p>{formatCount(searchResults.data?.pages[0]?.page.total)} matches for <strong>“{searchQuery}”</strong></p>
                <ul>{searchItems.map((message) => <li key={message.id}><button type="button" onClick={() => jumpToMessage(message.id)}><span><strong>{message.sender}</strong><time>{formatTimestampMs(message.timestamp_ms, timezone)}</time></span><p>{message.content || message.transcription || sentenceCase(message.kind)}</p><small>{message.matched_fields?.map(sentenceCase).join(', ')} · Jump to message</small></button></li>)}</ul>
                {searchResults.hasNextPage ? <button className="button button--secondary" type="button" disabled={searchResults.isFetchingNextPage} onClick={() => searchResults.fetchNextPage()}>{searchResults.isFetchingNextPage ? 'Loading…' : 'Load more matches'}</button> : null}
              </div>
            )
          ) : <p className="thread-search__hint"><LockKeyhole size={14} /> IP and payment fields are intentionally not searchable.</p>}
        </section>
      ) : null}

      <SegmentList segments={detail.data?.segments ?? []} timezone={timezone} />

      {conversation.type === 'direct' ? <IdentityCorrectionPanel conversation={conversation} segments={detail.data?.segments ?? []} onRedirect={openCorrectedHistory} /> : null}

      <section className="panel reader-panel" aria-labelledby="timeline-heading">
        <div className="table-toolbar"><div><h2 id="timeline-heading">Message history</h2><p>Chronological · virtualized to keep the rendered page bounded</p></div><span className="reader-count"><CheckCircle2 size={14} /> {formatCount(messages.length)} loaded</span></div>
        {messages.length === 0 ? <EmptyState title="No messages in this conversation" description={conversation.content_profile.kind === 'media_only' ? "This is a media-only source thread. Its attachments remain available from Conversation media; no message records were invented." : "The source thread contains metadata without message records."} /> : (
          <MessageTimeline
            messages={messages}
            total={messageTotal}
            conversationId={conversationId}
            timezone={timezone}
            focusId={around}
            focusAttachmentId={media}
            mediaStartMs={t}
            canLoadOlder={Boolean(messagesQuery.hasPreviousPage)}
            canLoadNewer={Boolean(messagesQuery.hasNextPage)}
            loadingOlder={messagesQuery.isFetchingPreviousPage}
            loadingNewer={messagesQuery.isFetchingNextPage}
            onLoadOlder={() => messagesQuery.fetchPreviousPage()}
            onLoadNewer={() => messagesQuery.fetchNextPage()}
            initialEdge={start}
          />
        )}
      </section>
      <p className="reader-privacy-note"><LockKeyhole size={14} /> Sensitive IP and payment metadata is hidden by default and fetched only after an explicit reveal.</p>
    </div>
  )
}
