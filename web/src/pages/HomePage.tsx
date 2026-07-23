import { useQuery } from '@tanstack/react-query'
import { Link } from '@tanstack/react-router'
import {
  AlertTriangle,
  ArrowRight,
  CheckCircle2,
  Database,
  Files,
  HardDrive,
  Layers3,
  MessageSquareText,
  Paperclip,
  ShieldAlert,
} from 'lucide-react'
import { api } from '../lib/api'
import { formatBytes, formatCount, sentenceCase } from '../lib/format'
import {
  EmptyState,
  ErrorState,
  MetricCard,
  PageHeading,
  PageLoading,
  ProgressBar,
  SectionHeader,
  StatusBadge,
} from '../components/ui'

export function HomePage() {
  const overview = useQuery({ queryKey: ['overview'], queryFn: api.overview })

  if (overview.isLoading) return <PageLoading label="Loading archive overview" />
  if (overview.isError) return <ErrorState error={overview.error} onRetry={() => overview.refetch()} />

  const data = overview.data ?? {}
  const archive = data.archive ?? {}
  const content = data.content ?? {}
  const integrity = data.integrity ?? {}
  const lifecycle = data.lifecycle ?? {}
  const jsonTotal = archive.json_file_count
  const jsonProcessed = (archive.json_valid_count ?? 0) + (archive.json_unsupported_count ?? 0)
  const jsonPercent = jsonTotal ? (jsonProcessed / jsonTotal) * 100 : null
  const jsonPercentLabel =
    jsonPercent == null
      ? '—'
      : jsonPercent > 99 && jsonPercent < 100
        ? `${jsonPercent.toFixed(2)}%`
        : `${Math.round(jsonPercent)}%`
  const anomalyTone = (integrity.critical_anomalies ?? 0) > 0 ? 'danger' : (integrity.unresolved_anomalies ?? 0) > 0 ? 'warning' : 'positive'

  return (
    <div className="page-stack">
      <PageHeading
        eyebrow="Foundation & consolidation"
        title="Archive control center"
        description="A private operational view of what was found, what is trustworthy, and what the local importer is doing now."
        actions={
          <Link className="button button--secondary" to="/integrity">
            Review integrity
            <ArrowRight size={15} aria-hidden="true" />
          </Link>
        }
      />

      {lifecycle.ready !== true ? (
        <section className={`inline-alert ${lifecycle.state === 'failed' ? 'inline-alert--danger' : ''}`} role="status">
          <AlertTriangle size={18} aria-hidden="true" />
          <div>
            <strong>{lifecycle.state === 'processing' ? 'Catalog import is in progress' : 'Catalog is not fully verified yet'}</strong>
            <span>
              Totals below reflect the current checkpoint. {formatCount(lifecycle.pending_json_count)} JSON files remain pending and {formatCount(lifecycle.verification_pending_count)} verification checks remain incomplete.
            </span>
          </div>
        </section>
      ) : null}

      <section className="metric-grid" aria-label="Archive totals">
        <MetricCard
          icon={MessageSquareText}
          label="Imported messages"
          value={formatCount(content.message_count)}
          detail={`${formatCount(content.conversation_count)} logical conversations`}
          tone={lifecycle.ready ? 'positive' : undefined}
        />
        <MetricCard
          icon={Paperclip}
          label="Attachment references"
          value={formatCount(content.attachment_count)}
          detail={`${formatCount(integrity.missing_attachments)} currently unresolved`}
        />
        <MetricCard
          icon={Files}
          label="Source files"
          value={formatCount(archive.source_file_count)}
          detail={`${formatCount(archive.chunk_count)} export chunks`}
        />
        <MetricCard
          icon={ShieldAlert}
          label="Open anomalies"
          value={formatCount(integrity.unresolved_anomalies)}
          detail={`${formatCount(integrity.critical_anomalies)} critical`}
          tone={anomalyTone}
        />
      </section>

      <div className="home-layout">
        <section className="panel panel--archive">
          <SectionHeader
            title="Archive inventory"
            description="Read-only facts from the currently selected export."
          />
          <div className="archive-summary">
            <div className="archive-summary__identity">
              <span className="archive-summary__icon" aria-hidden="true">
                <Database size={25} />
              </span>
              <div>
                <span>Active archive</span>
                <strong title={archive.root_name}>{archive.root_name ?? 'Archive root not reported'}</strong>
              </div>
            </div>
            <dl className="definition-grid">
              <div>
                <dt>Total source size</dt>
                <dd>{formatBytes(archive.total_bytes)}</dd>
              </div>
              <div>
                <dt>JSON manifests</dt>
                <dd>{formatCount(archive.json_file_count)}</dd>
              </div>
              <div>
                <dt>Activity records</dt>
                <dd>{formatCount(content.activity_count)}</dd>
              </div>
              <div>
                <dt>Unlinked media</dt>
                <dd>{formatCount(integrity.unlinked_media)}</dd>
              </div>
            </dl>
          </div>
          <div className="validation-strip">
            <div className="validation-strip__copy">
              <span className="validation-strip__icon" aria-hidden="true">
                {(archive.json_invalid_count ?? 0) > 0 || (archive.json_pending_count ?? 0) > 0 ? <AlertTriangle size={18} /> : <CheckCircle2 size={18} />}
              </span>
              <div>
                <strong>JSON validation</strong>
                <span>
                  {formatCount(jsonProcessed)} processed · {formatCount(archive.json_pending_count)} pending · {formatCount(archive.json_invalid_count)} invalid
                </span>
              </div>
            </div>
            <div
              className="validation-strip__bar"
              role="progressbar"
              aria-label="JSON manifests validated"
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={jsonPercent ?? 0}
              aria-valuetext={jsonPercentLabel}
            >
              <span style={{ width: `${jsonPercent ?? 0}%` }} />
            </div>
            <strong>{jsonPercentLabel}</strong>
          </div>
          <Link className="panel-link" to="/coverage">
            See exactly which Facebook categories are present
            <ArrowRight size={15} aria-hidden="true" />
          </Link>
        </section>

        <section className="panel panel--job">
          <SectionHeader
            title="Latest job"
            description="Long-running work is resumable and stays on this device."
            action={<HardDrive size={19} aria-hidden="true" />}
          />
          {data.latest_job ? (
            <div className="job-spotlight">
              <div className="job-spotlight__top">
                <span className="job-spotlight__icon" aria-hidden="true">
                  <Layers3 size={20} />
                </span>
                <div>
                  <strong>{sentenceCase(data.latest_job.kind)}</strong>
                  <span>{data.latest_job.stage || 'Preparing archive operation'}</span>
                </div>
                <StatusBadge status={data.latest_job.status} />
              </div>
              <ProgressBar
                current={data.latest_job.progress_current}
                total={data.latest_job.progress_total}
                label={`${sentenceCase(data.latest_job.kind)} progress`}
              />
              {data.latest_job.error ? <p className="job-spotlight__error">{data.latest_job.error}</p> : null}
              <Link className="button button--secondary button--full" to="/jobs">
                Open job history
                <ArrowRight size={15} aria-hidden="true" />
              </Link>
            </div>
          ) : (
            <EmptyState
              title="No jobs have run yet"
              description="Scan, consolidation, and import operations will appear here with resumable progress."
              action={
                <Link className="button button--secondary" to="/jobs">
                  View jobs
                </Link>
              }
            />
          )}
        </section>
      </div>

      <section className="readiness-banner" aria-labelledby="readiness-title">
        <div className="readiness-banner__icon" aria-hidden="true">
          <HardDrive size={23} />
        </div>
        <div>
          <p className="eyebrow">Private by design</p>
          <h2 id="readiness-title">Your source export is never fetched from the browser.</h2>
          <p>
            The interface talks only to the localhost archive service. Cloud processing remains visibly off unless you explicitly change it in Settings.
          </p>
        </div>
        <Link className="button button--quiet" to="/settings">
          Privacy settings
          <ArrowRight size={15} aria-hidden="true" />
        </Link>
      </section>
    </div>
  )
}
