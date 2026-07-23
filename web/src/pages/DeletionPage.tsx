import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { AlertOctagon, ArchiveRestore, DatabaseZap, ExternalLink, Eye, FileWarning, ShieldAlert, Trash2 } from 'lucide-react'
import { useMemo, useRef, useState, type ReactNode } from 'react'
import { CursorWindowControls } from '../components/CursorWindowControls'
import { api } from '../lib/api'
import type { DeletionEntityType, DeletionOperation, DeletionPreviewRequest, DeletionPreviewResponse, DeletionPurgeCount, DeletionPurgeSummary, DeletionRestoreRebuildSummary, DeletionSelectionItem, HardPurgePlan } from '../lib/contracts'
import { formatBytes, formatCount, formatDateTime, formatTimestampMs, sentenceCase, stringifyValue } from '../lib/format'
import { useCursorWindowQuery } from '../lib/useCursorWindowQuery'
import { EmptyState, ErrorState, PageHeading, SectionHeader, StatusBadge } from '../components/ui'
import './DeletionPage.css'

const entityOptions: { id: DeletionEntityType; label: string }[] = [
  { id: 'message', label: 'Messages' },
  { id: 'activity', label: 'Posts and activity' },
  { id: 'conversation', label: 'Whole conversations' },
  { id: 'attachment', label: 'Attachment occurrences' },
]
const mediaOptions = ['photo', 'video', 'audio', 'gif', 'sticker', 'document', 'other']
const availabilityOptions = ['resolved', 'missing', 'unlinked', 'external']
const purgeSummaryCountFields: { key: DeletionPurgeCount; label: string; bytes?: boolean }[] = [
  { key: 'normalized_rows', label: 'Normalized rows' },
  { key: 'workspace_files', label: 'Workspace files' },
  { key: 'workspace_bytes', label: 'Workspace bytes', bytes: true },
  { key: 'raw_source_files_rewritten', label: 'Source files rewritten' },
  { key: 'affected_messages', label: 'Messages removed' },
  { key: 'affected_activities', label: 'Posts and activity removed' },
  { key: 'affected_conversations', label: 'Conversations removed' },
  { key: 'affected_attachments', label: 'Attachment occurrences removed' },
  { key: 'affected_blobs', label: 'Blobs removed' },
  { key: 'affected_source_records', label: 'Source records removed' },
  { key: 'affected_search_windows', label: 'Search windows removed' },
  { key: 'deleted_vector_rows', label: 'Vector rows removed' },
]

function lines(value: string) {
  return [...new Set(value.split(/\r?\n|,/).map((item) => item.trim()).filter(Boolean))]
}

function objectEntries(value: unknown): [string, unknown][] {
  return value && typeof value === 'object' && !Array.isArray(value) ? Object.entries(value) : []
}

function ImpactView({ impact }: { impact: Record<string, unknown> }) {
  return <div className="deletion-impact">{Object.entries(impact).map(([section, value]) => {
    const nested = objectEntries(value)
    return <article key={section}><h3>{sentenceCase(section)}</h3>{nested.length ? <dl>{nested.map(([key, nestedValue]) => <div key={key}><dt>{sentenceCase(key)}</dt><dd>{Array.isArray(nestedValue) ? formatCount(nestedValue.length) : stringifyValue(nestedValue)}</dd></div>)}</dl> : <strong>{stringifyValue(value)}</strong>}</article>
  })}</div>
}

function PurgeImpactView({ impact }: { impact: Partial<Record<DeletionPurgeCount, number>> }) {
  const visibleCounts = purgeSummaryCountFields.filter(({ key }) => impact[key] != null)
  return <div className="deletion-impact"><article><h3>Allowlisted purge impact</h3>{visibleCounts.length ? <dl>{visibleCounts.map(({ key, label, bytes }) => <div key={key}><dt>{label}</dt><dd>{bytes ? formatBytes(impact[key]) : formatCount(impact[key])}</dd></div>)}</dl> : <strong>No public impact counts are available.</strong>}</article></div>
}

function ErrorBanner({ error }: { error: unknown }) {
  if (!error) return null
  return <div className="deletion-error" role="alert"><AlertOctagon size={18} aria-hidden="true" /><span>{error instanceof Error ? error.message : 'The operation could not be completed.'}</span></div>
}

function PurgeSummaryView({ summary }: { summary: DeletionPurgeSummary }) {
  const visibleCounts = purgeSummaryCountFields.filter(({ key }) => summary.counts[key] != null)
  return <details className="purge-summary">
    <summary>Safe purge summary</summary>
    <p>Only allowlisted status, counts, digests, and timestamps are exposed. Restore snapshots, paths, terms, tokens, and recovery internals remain private.</p>
    <dl>
      <div><dt>Operation status</dt><dd>{sentenceCase(summary.status.operation)}</dd></div>
      {summary.scope ? <div><dt>Scope</dt><dd>{summary.scope === 'workspace' ? 'Entire workspace copy' : 'Normalized datastore only'}</dd></div> : null}
      {summary.status.publication_action ? <div><dt>Manifest publication</dt><dd>{sentenceCase(summary.status.publication_action)} · {sentenceCase(summary.status.publication_phase ?? 'unknown')}</dd></div> : null}
      {summary.timestamps.planned_at ? <div><dt>Planned</dt><dd>{formatDateTime(summary.timestamps.planned_at)}</dd></div> : null}
      {summary.timestamps.completed_at ? <div><dt>Completed</dt><dd>{formatDateTime(summary.timestamps.completed_at)}</dd></div> : null}
      {visibleCounts.map(({ key, label, bytes }) => <div key={key}><dt>{label}</dt><dd>{bytes ? formatBytes(summary.counts[key]) : formatCount(summary.counts[key])}</dd></div>)}
      {summary.digests.purge_revision ? <div><dt>Purge revision</dt><dd><code>{summary.digests.purge_revision}</code></dd></div> : null}
      {summary.digests.logical_restore_state_sha256 ? <div><dt>Restore-state digest</dt><dd><code>{summary.digests.logical_restore_state_sha256}</code></dd></div> : null}
    </dl>
    {summary.status.blockers?.length ? <p>Safety blockers: {summary.status.blockers.map(sentenceCase).join(', ')}</p> : null}
  </details>
}

const restoreRebuildProductLabels: Record<DeletionRestoreRebuildSummary['product_kinds'][number], string> = {
  search_windows: 'Search windows',
  semantic_index: 'Semantic index',
  nlp_hourly_rollups: 'NLP hourly rollups',
  conflict_aggregates: 'Conflict aggregates',
  topic_clusters: 'Topic clusters',
}

function RestoreRebuildNotice({ summary, compact = false }: { summary: DeletionRestoreRebuildSummary; compact?: boolean }) {
  const incompatibility = summary.reason === 'intervening_hard_purge' ? 'an intervening hard purge' : 'a later quarantine that remains active'
  return <div className={compact ? 'restore-rebuild-notice restore-rebuild-notice--compact' : 'restore-rebuild-notice'} role="alert">
    <strong>Derived analytics rebuild required</strong>
    <p>The owned archive records were restored, but {incompatibility} means these global products remain invalid. They are intentionally not presented as exact or current until rebuilt.</p>
    <ul>{summary.product_kinds.map((productKind) => <li key={productKind}>{restoreRebuildProductLabels[productKind]}</li>)}</ul>
    <small>Detected invalidating operations: {formatCount(summary.invalidated_by_count)}</small>
  </div>
}

function AuditTrail({ audits }: { audits: NonNullable<DeletionOperation['audits']> }) {
  if (!audits.length) return <p>No mutation audits have been recorded for this operation.</p>
  return <ol className="deletion-audit-list">{audits.map((audit) => {
    const visibleCounts = purgeSummaryCountFields.filter(({ key }) => audit.detail.counts[key] != null)
    return <li key={audit.id}>
      <div><strong>{sentenceCase(audit.action)}</strong><StatusBadge status={audit.outcome} /></div>
      <small>{formatDateTime(audit.created_at)}{audit.scope ? ` · ${sentenceCase(audit.scope)}` : ''}</small>
      {visibleCounts.length ? <dl>{visibleCounts.map(({ key, label, bytes }) => <div key={key}><dt>{label}</dt><dd>{bytes ? formatBytes(audit.detail.counts[key]) : formatCount(audit.detail.counts[key])}</dd></div>)}</dl> : null}
      {audit.detail.digests.catalog_revision ? <p>Catalog revision: <code>{audit.detail.digests.catalog_revision}</code></p> : null}
      {audit.detail.digests.purge_revision ? <p>Purge revision: <code>{audit.detail.digests.purge_revision}</code></p> : null}
      {audit.detail.status.blockers?.length ? <p>Safety blockers: {audit.detail.status.blockers.map(sentenceCase).join(', ')}</p> : null}
      {audit.detail.status.recovered != null ? <p>Recovery attempted: {audit.detail.status.recovered ? 'yes' : 'no'}</p> : null}
      {audit.detail.restore_rebuild_required ? <RestoreRebuildNotice summary={audit.detail.restore_rebuild_required} compact /> : null}
    </li>
  })}</ol>
}

function SelectionItemCard({ item }: { item: DeletionSelectionItem }) {
  const metadata = [
    item.resource_kind ? sentenceCase(item.resource_kind) : null,
    item.availability ? sentenceCase(item.availability) : null,
    item.media_type,
    item.size_bytes != null ? formatBytes(item.size_bytes) : null,
  ].filter(Boolean)
  return <li className="deletion-selection-item">
    <div className="deletion-selection-item__heading">
      <span className="status-badge">{sentenceCase(item.entity_type)}</span>
      <strong>{item.label}{item.label_truncated ? '…' : ''}</strong>
    </div>
    {item.snippet ? <p>{item.snippet}{item.snippet_truncated ? '…' : ''}</p> : null}
    <dl>
      <div><dt>Selection reason</dt><dd>{sentenceCase(item.reason)}</dd></div>
      <div><dt>Exact record ID</dt><dd><code>{item.entity_id}</code></dd></div>
      {item.timestamp_ms != null ? <div><dt>Date</dt><dd>{formatTimestampMs(item.timestamp_ms)}</dd></div> : null}
      {metadata.length ? <div><dt>Resource</dt><dd>{metadata.join(' · ')}</dd></div> : null}
    </dl>
    {item.jump ? <a className="button button--secondary" href={item.jump.href}><ExternalLink size={14} aria-hidden="true" />{item.jump.label}</a> : item.quarantined ? <small>Quarantined records are hidden from browser pages, so no dead source link is shown.</small> : <small>No retained browser target is available for this record.</small>}
  </li>
}

function DeletionSelectionPanel({
  operationId,
  revision,
  expectedStatus,
  title = 'Exact records selected',
  footer,
}: {
  operationId: string
  revision: string
  expectedStatus: 'previewed' | 'committed'
  title?: string
  footer?: ReactNode
}) {
  const selection = useCursorWindowQuery({
    queryKey: ['deletion-selection', operationId, revision, expectedStatus],
    queryFn: (cursor) => api.deletionSelection(operationId, revision, { cursor, limit: 25 }),
  })
  const currentData = selection.data
    && !selection.isPlaceholderData
    && selection.data.operation_id === operationId
    && selection.data.revision === revision
    && selection.data.status === expectedStatus
    ? selection.data
    : undefined
  return <div className="deletion-selection" aria-busy={selection.isPending || selection.transitionPending}>
    <div className="deletion-selection__header"><div><h3>{title}</h3><p>Every selected record is listed by stable ID. Message and post text is whitespace-normalized and capped at 160 characters; raw and sensitive source fields are never returned.</p></div></div>
    {selection.isError ? <ErrorState error={selection.error} onRetry={() => void selection.refetch()} /> : currentData ? <>
      {currentData.items.length ? <ol className="deletion-selection-list">{currentData.items.map((item) => <SelectionItemCard key={`${item.entity_type}:${item.entity_id}`} item={item} />)}</ol> : <EmptyState title="No selected records" description="The operation receipt contains no records on this page." />}
      <CursorWindowControls
        label={`Exact selection for operation ${operationId}`}
        pageNumber={selection.pageNumber}
        targetPageNumber={selection.targetPageNumber}
        itemCount={currentData.items.length}
        total={currentData.page.total}
        hasPreviousPage={selection.hasPreviousPage}
        hasNextPage={selection.hasNextPage}
        loading={selection.transitionPending || selection.isPending}
        onPreviousPage={selection.previousPage}
        onNextPage={selection.nextPage}
      />
      {footer ? <div className="deletion-selection__footer">{footer}</div> : null}
    </> : <p role="status">Loading the exact selected records for this operation…</p>}
  </div>
}

export function DeletionPage() {
  const queryClient = useQueryClient()
  const [entities, setEntities] = useState<DeletionEntityType[]>(['message', 'activity'])
  const [text, setText] = useState('')
  const [textMode, setTextMode] = useState<'exact' | 'keyword'>('exact')
  const [caseSensitive, setCaseSensitive] = useState(true)
  const [wholeField, setWholeField] = useState(false)
  const [textVariant, setTextVariant] = useState<'display' | 'raw'>('display')
  const [fromDate, setFromDate] = useState('')
  const [toDate, setToDate] = useState('')
  const [timezone, setTimezone] = useState('America/New_York')
  const [participant, setParticipant] = useState('')
  const [conversationIds, setConversationIds] = useState('')
  const [mediaTypes, setMediaTypes] = useState<string[]>([])
  const [availability, setAvailability] = useState<string[]>([])
  const [allowAll, setAllowAll] = useState(false)
  const [allPhrase, setAllPhrase] = useState('')
  const [preview, setPreview] = useState<DeletionPreviewResponse | null>(null)
  const [commitConfirmation, setCommitConfirmation] = useState('')
  const [selectedOperationId, setSelectedOperationId] = useState<string | null>(null)
  const [actionConfirmation, setActionConfirmation] = useState('')
  const [purgeScope, setPurgeScope] = useState<'normalized' | 'workspace'>('workspace')
  const [purgePlan, setPurgePlan] = useState<HardPurgePlan | null>(null)
  const destructiveContextRef = useRef({ operationId: selectedOperationId, scope: purgeScope })
  destructiveContextRef.current = { operationId: selectedOperationId, scope: purgeScope }

  const history = useCursorWindowQuery({
    queryKey: ['deletion-operations'],
    queryFn: (cursor) => api.deletionOperations({ cursor, limit: 50 }),
  })
  const detail = useQuery({ queryKey: ['deletion-operation', selectedOperationId], queryFn: () => api.deletionOperation(selectedOperationId!), enabled: Boolean(selectedOperationId) })
  const invalidate = async (operationId?: string) => {
    await queryClient.invalidateQueries({ queryKey: ['deletion-operations'] })
    if (operationId) await queryClient.invalidateQueries({ queryKey: ['deletion-operation', operationId] })
  }
  const previewMutation = useMutation({
    mutationFn: api.previewDeletion,
    onSuccess: async (result) => {
      setPreview(result)
      setSelectedOperationId(result.operation_id)
      setCommitConfirmation('')
      setPurgePlan(null)
      setPurgeScope('workspace')
      await invalidate(result.operation_id)
    },
  })
  const commitMutation = useMutation({
    mutationFn: () => api.commitDeletion(preview!.operation_id, preview!.commit_token, preview!.catalog_revision),
    onSuccess: async (result) => {
      queryClient.removeQueries({ queryKey: ['deletion-selection', result.operation_id] })
      setPreview(null)
      setCommitConfirmation('')
      await invalidate(result.operation_id)
    },
  })
  const restoreMutation = useMutation({
    mutationFn: () => api.restoreDeletion(selectedOperationId!),
    onMutate: () => { setPurgePlan(null) },
    onSuccess: async (result) => {
      queryClient.removeQueries({ queryKey: ['deletion-selection', result.operation_id] })
      setActionConfirmation('')
      await invalidate(result.operation_id)
    },
  })
  const purgePreviewMutation = useMutation({
    mutationFn: async () => {
      const request = destructiveContextRef.current
      if (!request.operationId) throw new Error('Choose a deletion operation before planning a purge.')
      const result = await api.previewHardPurge(request.operationId, request.scope)
      return { request, result }
    },
    onMutate: () => { setPurgePlan(null) },
    onSuccess: ({ result, request }) => {
      const current = destructiveContextRef.current
      if (result.operation_id === request.operationId && result.scope === request.scope && current.operationId === request.operationId && current.scope === request.scope) {
        setPurgePlan(result)
        setActionConfirmation('')
      }
    },
  })
  const purgeMutation = useMutation({
    mutationFn: () => {
      const current = destructiveContextRef.current
      if (!purgePlan || purgePlan.operation_id !== current.operationId || purgePlan.scope !== current.scope) throw new Error('The purge plan no longer matches the selected operation and scope. Preview again.')
      return api.commitHardPurge(purgePlan)
    },
    onSuccess: async (result) => { setPurgePlan(null); setActionConfirmation(''); await invalidate(result.operation_id) },
  })
  const destructiveActionBusy = restoreMutation.isPending || purgePreviewMutation.isPending || purgeMutation.isPending

  const narrowed = Boolean(text.trim() || fromDate || toDate || participant.trim() || lines(conversationIds).length || mediaTypes.length || availability.length)
  const previewAllowed = entities.length > 0 && (narrowed || (allowAll && allPhrase === 'PREVIEW ENTIRE ARCHIVE'))
  const payload = useMemo<DeletionPreviewRequest>(() => ({
    entity_types: entities,
    ...(text.trim() ? { text: text } : {}),
    text_mode: textMode,
    case_sensitive: textMode === 'exact' ? caseSensitive : false,
    whole_field: textMode === 'exact' ? wholeField : false,
    text_variant: textMode === 'exact' ? textVariant : 'display',
    ...(fromDate ? { from_date: fromDate } : {}),
    ...(toDate ? { to_date: toDate } : {}),
    timezone,
    conversation_ids: lines(conversationIds),
    ...(participant.trim() ? { participant: participant.trim() } : {}),
    media_types: mediaTypes,
    availability,
    entity_ids: {},
    allow_all: allowAll && allPhrase === 'PREVIEW ENTIRE ARCHIVE',
  }), [entities, text, textMode, caseSensitive, wholeField, textVariant, fromDate, toDate, timezone, conversationIds, participant, mediaTypes, availability, allowAll, allPhrase])

  const toggle = <T extends string>(value: T, values: T[], setter: (next: T[]) => void) => setter(values.includes(value) ? values.filter((item) => item !== value) : [...values, value])
  const selected = detail.data?.id === selectedOperationId ? detail.data : undefined
  const exactActionConfirmed = Boolean(selectedOperationId && actionConfirmation === selectedOperationId)
  const clearHistorySelection = () => {
    setSelectedOperationId(null)
    setActionConfirmation('')
    setPurgePlan(null)
    setPurgeScope('workspace')
  }

  return (
    <div className="page-stack deletion-page">
      <PageHeading eyebrow="Stage 5 · Destructive tools" title="Deletion and quarantine" description="Preview exact selections, move them into reversible logical quarantine, restore them, or explicitly hard-purge every workspace copy." />
      <div className="deletion-warning" role="alert"><ShieldAlert size={22} aria-hidden="true" /><div><strong>Preview first. Tokens expire when the catalog changes.</strong><span>Commit is reversible until hard purge. Workspace purge rewrites source JSON and removes derived indexes; it cannot erase separate backups or storage-device remnants.</span></div></div>

      <section className="panel deletion-builder">
        <SectionHeader title="1. Build a selection" description="Exact-string matching is case-sensitive by default. End dates include the entire selected local day and are converted to a half-open UTC range." />
        <fieldset><legend>Record types</legend><div className="check-grid">{entityOptions.map((item) => {
          const keywordAttachment = textMode === 'keyword' && item.id === 'attachment'
          return <label key={item.id}><input type="checkbox" checked={entities.includes(item.id)} disabled={keywordAttachment} onChange={() => toggle(item.id, entities, setEntities)} /><span>{item.label}</span></label>
        })}</div></fieldset>
        <div className="deletion-form-grid">
          <label className="filter-field deletion-field--wide"><span>Text to match</span><textarea value={text} onChange={(event) => setText(event.target.value)} placeholder="Exact text or FTS phrase" rows={3} /></label>
          <label className="filter-field"><span>Match mode</span><select value={textMode} onChange={(event) => {
            const mode = event.target.value as 'exact' | 'keyword'
            setTextMode(mode)
            if (mode === 'keyword') {
              setEntities((current) => current.filter((entity) => entity !== 'attachment'))
              setTextVariant('display')
              setCaseSensitive(false)
              setWholeField(false)
              setMediaTypes([])
              setAvailability([])
            }
          }}><option value="exact">Exact literal substring</option><option value="keyword">FTS keyword / phrase</option></select></label>
          <label className="filter-field"><span>Text representation</span><select value={textVariant} disabled={textMode === 'keyword'} onChange={(event) => setTextVariant(event.target.value as 'display' | 'raw')}><option value="display">Repaired display text</option><option value="raw">Raw source text</option></select></label>
          <label className="filter-field"><span>Inclusive start date</span><input type="date" value={fromDate} onChange={(event) => setFromDate(event.target.value)} /></label>
          <label className="filter-field"><span>Inclusive end date</span><input type="date" value={toDate} onChange={(event) => setToDate(event.target.value)} /></label>
          <label className="filter-field"><span>Display timezone</span><input value={timezone} onChange={(event) => setTimezone(event.target.value)} /></label>
          <label className="filter-field"><span>Participant label contains</span><input value={participant} onChange={(event) => setParticipant(event.target.value)} /></label>
          <label className="filter-field deletion-field--wide"><span>Conversation IDs</span><textarea value={conversationIds} onChange={(event) => setConversationIds(event.target.value)} placeholder="One ID per line or comma-separated" rows={3} /></label>
        </div>
        <div className="deletion-options"><label><input type="checkbox" checked={caseSensitive} disabled={textMode === 'keyword'} onChange={(event) => setCaseSensitive(event.target.checked)} /><span>Case-sensitive</span></label><label><input type="checkbox" checked={wholeField} disabled={textMode === 'keyword'} onChange={(event) => setWholeField(event.target.checked)} /><span>Whole field only</span></label></div>
        {textMode === 'keyword' ? <p className="person-card__semantics">Keyword mode uses normalized FTS over retained message content/transcriptions and post titles/bodies. Selecting Whole conversations finds conversations with a matching retained message, then cascades to their retained messages and attachment occurrences. Attachment-occurrence selection is unavailable in keyword mode; use exact, date, conversation, or media filters instead. Sender and actor labels are never selected, and exact-only options do not apply.</p> : null}
        <fieldset disabled={textMode === 'keyword'}><legend>Attachment media types</legend><div className="check-grid">{mediaOptions.map((item) => <label key={item}><input type="checkbox" checked={mediaTypes.includes(item)} onChange={() => toggle(item, mediaTypes, setMediaTypes)} /><span>{sentenceCase(item)}</span></label>)}</div></fieldset>
        <fieldset disabled={textMode === 'keyword'}><legend>Attachment availability</legend><div className="check-grid">{availabilityOptions.map((item) => <label key={item}><input type="checkbox" checked={availability.includes(item)} onChange={() => toggle(item, availability, setAvailability)} /><span>{sentenceCase(item)}</span></label>)}</div></fieldset>
        {!narrowed ? <div className="deletion-all"><label><input type="checkbox" checked={allowAll} onChange={(event) => { setAllowAll(event.target.checked); setAllPhrase('') }} /><span>Allow an unbounded preview</span></label>{allowAll ? <label className="filter-field"><span>Type PREVIEW ENTIRE ARCHIVE</span><input value={allPhrase} onChange={(event) => setAllPhrase(event.target.value)} autoComplete="off" /></label> : null}</div> : null}
        <div className="deletion-builder__actions"><button className="button button--primary" type="button" disabled={!previewAllowed || previewMutation.isPending || destructiveActionBusy} onClick={() => previewMutation.mutate(payload)}><Eye size={16} aria-hidden="true" />{previewMutation.isPending ? 'Calculating impact…' : 'Preview selection'}</button><span>{narrowed ? 'The archive will not change during preview.' : 'Add a filter or explicitly authorize an unbounded preview.'}</span></div>
        <ErrorBanner error={previewMutation.error} />
      </section>

      {preview ? <section className="panel deletion-preview" aria-label="Deletion selection preview">
        <SectionHeader title="2. Review quarantine impact" description={`Preview operation ${preview.operation_id}. The commit token is held only in this page's memory.`} action={<StatusBadge status="preview" />} />
        <ImpactView impact={preview.impact} />
        <DeletionSelectionPanel
          operationId={preview.operation_id}
          revision={preview.selection_sha256}
          expectedStatus="previewed"
          footer={<>
            <div className="deletion-confirm"><label className="filter-field"><span>Type the operation ID to commit logical quarantine</span><input value={commitConfirmation} onChange={(event) => setCommitConfirmation(event.target.value)} autoComplete="off" /></label><button className="button button--danger" type="button" disabled={commitConfirmation !== preview.operation_id || commitMutation.isPending || previewMutation.isPending} onClick={() => commitMutation.mutate()}><Trash2 size={16} aria-hidden="true" />{commitMutation.isPending ? 'Committing…' : 'Commit to quarantine'}</button></div>
            <ErrorBanner error={commitMutation.error} />
          </>}
        />
      </section> : null}

      <section className="panel deletion-history">
        <SectionHeader title="3. Operation history, restore, and purge" description="Select an operation to inspect its selector, audit trail, and available next actions." action={<button className="button button--secondary" type="button" disabled={destructiveActionBusy} onClick={() => void history.refetch()}>Refresh history</button>} />
        {history.isError ? <ErrorState error={history.error} onRetry={() => void history.refetch()} /> : history.data?.items.length ? <div className="deletion-history__layout"><ul className="deletion-operation-list" aria-label="Deletion operations">{history.data.items.map((operation) => <li key={operation.id}><button type="button" className={selectedOperationId === operation.id ? 'deletion-operation deletion-operation--selected' : 'deletion-operation'} aria-pressed={selectedOperationId === operation.id} disabled={destructiveActionBusy} onClick={() => { setSelectedOperationId(operation.id); setActionConfirmation(''); setPurgePlan(null); setPurgeScope('workspace') }}><span><StatusBadge status={operation.status} /><strong>{formatCount(operation.item_count)} records</strong></span><code>{operation.id}</code><small>{formatDateTime(operation.created_at)}</small></button></li>)}</ul><div className="deletion-detail">{detail.isError ? <ErrorState error={detail.error} onRetry={() => void detail.refetch()} /> : detail.isPending ? <p>Loading operation detail…</p> : selected ? <><div className="deletion-detail__header"><div><h3>{selected.id}</h3><span>Created {formatDateTime(selected.created_at)}</span></div><StatusBadge status={selected.status} /></div><ImpactView impact={selected.impact} />{selected.error_text ? <p className="deletion-error">{selected.error_text}</p> : null}{selected.selection_sha256 && selected.id !== preview?.operation_id && (selected.status === 'previewed' || selected.status === 'committed') ? <DeletionSelectionPanel operationId={selected.id} revision={selected.selection_sha256} expectedStatus={selected.status} title="Exact operation records" /> : null}{selected.purge_summary ? <PurgeSummaryView summary={selected.purge_summary} /> : null}{selected.restore_rebuild_required ? <RestoreRebuildNotice summary={selected.restore_rebuild_required} /> : null}<details><summary>Selector and audit details</summary><p>Audit data is a public allowlist of counts, digests, status codes, and timestamps. Private restore state, paths, terms, tokens, and recovery internals are never rendered.</p><pre>{JSON.stringify({ selector: selected.selector }, null, 2)}</pre><AuditTrail audits={selected.audits ?? []} /></details>{['committed', 'quarantined'].includes(selected.status) ? <div className="deletion-actions"><label className="filter-field"><span>Type the operation ID to authorize an action</span><input value={actionConfirmation} onChange={(event) => setActionConfirmation(event.target.value)} autoComplete="off" /></label><div><button className="button button--secondary" type="button" disabled={!exactActionConfirmed || destructiveActionBusy} onClick={() => restoreMutation.mutate()}><ArchiveRestore size={16} aria-hidden="true" />Restore quarantine</button><label className="filter-field"><span>Purge scope</span><select value={purgeScope} disabled={destructiveActionBusy} onChange={(event) => { setPurgeScope(event.target.value as 'normalized' | 'workspace'); setPurgePlan(null) }}><option value="workspace">Entire workspace copy (recommended)</option><option value="normalized">Normalized datastore only</option></select></label><button className="button button--danger" type="button" disabled={!exactActionConfirmed || destructiveActionBusy} onClick={() => purgePreviewMutation.mutate()}><FileWarning size={16} aria-hidden="true" />Preview hard purge</button></div><p className={purgeScope === 'normalized' ? 'purge-scope-note purge-scope-note--warning' : 'purge-scope-note'} role={purgeScope === 'normalized' ? 'alert' : undefined}>{purgeScope === 'workspace' ? 'Workspace scope is the public default. It removes covered normalized records and derived copies and rewrites matching exported source JSON in this workspace.' : 'Normalized-only purge leaves matching records in exported source JSON and other workspace files. A later re-import can restore them; use this scope only for deliberate datastore maintenance.'}</p></div> : null}<ErrorBanner error={restoreMutation.error ?? purgePreviewMutation.error} />{purgePlan?.operation_id === selectedOperationId && purgePlan.scope === purgeScope ? <div className="purge-plan"><div><DatabaseZap size={20} aria-hidden="true" /><h3>{purgePlan.scope === 'workspace' ? 'Workspace hard purge' : 'Normalized hard purge'}</h3><StatusBadge status={purgePlan.executable ? 'executable' : 'blocked'} /></div><PurgeImpactView impact={purgePlan.impact} />{purgePlan.blockers.length ? <ul>{purgePlan.blockers.map((blocker) => <li key={blocker}>{blocker}</li>)}</ul> : purgePlan.executable ? <p>This irreversible plan passed its current safety preconditions. Type the operation ID again after every new preview.</p> : <p>One or more internal safety checks blocked this purge. Private diagnostics are not exposed; correct the workspace state and preview again.</p>}<button className="button button--danger" type="button" disabled={!purgePlan.executable || !exactActionConfirmed || destructiveActionBusy} onClick={() => purgeMutation.mutate()}><Trash2 size={16} aria-hidden="true" />{purgeMutation.isPending ? 'Purging…' : 'Permanently execute hard purge'}</button><ErrorBanner error={purgeMutation.error} /></div> : null}</> : <EmptyState title="Choose an operation" description="Its selector, audit trail, restore controls, and purge controls will appear here." />}</div></div> : history.isPending ? <p>Loading deletion history…</p> : <EmptyState title="No deletion operations" description="Build a filtered preview above. Previewing does not change the archive." />}
        {history.data ? (
          <CursorWindowControls
            label="Deletion operation history"
            pageNumber={history.pageNumber}
            targetPageNumber={history.targetPageNumber}
            itemCount={history.data.items.length}
            total={history.data.page.total}
            hasPreviousPage={history.hasPreviousPage}
            hasNextPage={history.hasNextPage}
            loading={history.transitionPending || history.isPending || destructiveActionBusy}
            onPreviousPage={() => {
              clearHistorySelection()
              history.previousPage()
            }}
            onNextPage={() => {
              clearHistorySelection()
              history.nextPage()
            }}
          />
        ) : null}
      </section>
    </div>
  )
}
