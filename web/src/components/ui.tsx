import type { LucideIcon } from 'lucide-react'
import { AlertTriangle, Inbox, RefreshCw } from 'lucide-react'
import type { ReactNode } from 'react'
import { sentenceCase } from '../lib/format'

export function PageHeading({
  eyebrow,
  title,
  description,
  actions,
}: {
  eyebrow?: string
  title: string
  description: string
  actions?: ReactNode
}) {
  return (
    <header className="page-heading">
      <div className="page-heading__copy">
        {eyebrow ? <p className="eyebrow">{eyebrow}</p> : null}
        <h1>{title}</h1>
        <p>{description}</p>
      </div>
      {actions ? <div className="page-heading__actions">{actions}</div> : null}
    </header>
  )
}

export function SectionHeader({
  title,
  description,
  action,
}: {
  title: string
  description?: string
  action?: ReactNode
}) {
  return (
    <div className="section-header">
      <div>
        <h2>{title}</h2>
        {description ? <p>{description}</p> : null}
      </div>
      {action}
    </div>
  )
}

export function MetricCard({
  icon: Icon,
  label,
  value,
  detail,
  tone = 'default',
}: {
  icon: LucideIcon
  label: string
  value: string
  detail?: string
  tone?: 'default' | 'positive' | 'warning' | 'danger'
}) {
  return (
    <article className={`metric-card metric-card--${tone}`}>
      <div className="metric-card__icon" aria-hidden="true">
        <Icon size={18} strokeWidth={1.8} />
      </div>
      <div className="metric-card__body">
        <p>{label}</p>
        <strong>{value}</strong>
        {detail ? <span>{detail}</span> : null}
      </div>
    </article>
  )
}

export function StatusBadge({ status }: { status: string | null | undefined }) {
  const normalized = (status ?? 'unknown').toLowerCase().replaceAll('_', '-')
  const tone =
    normalized === 'available' || normalized === 'ok' || normalized === 'passed' || normalized === 'complete'
      ? 'success'
      : normalized === 'partial' || normalized === 'warning' || normalized === 'running' || normalized === 'pending'
        ? 'warning'
        : normalized === 'critical' || normalized === 'error' || normalized === 'failed'
          ? 'danger'
          : normalized === 'unavailable' || normalized === 'cancelled'
            ? 'muted'
            : 'info'

  return (
    <span className={`status-badge status-badge--${tone}`}>
      <span className="status-badge__dot" aria-hidden="true" />
      {sentenceCase(status)}
    </span>
  )
}

export function PageLoading({ label = 'Loading archive data' }: { label?: string }) {
  return (
    <div className="state-panel state-panel--loading" role="status" aria-live="polite">
      <span className="spinner" aria-hidden="true" />
      <strong>{label}</strong>
      <span>The local service is reading the catalog.</span>
    </div>
  )
}

export function ErrorState({
  title = 'This local view could not be loaded',
  error,
  onRetry,
}: {
  title?: string
  error: unknown
  onRetry?: () => void
}) {
  const message = error instanceof Error ? error.message : 'An unexpected archive-service error occurred.'
  return (
    <div className="state-panel state-panel--error" role="alert">
      <span className="state-panel__icon" aria-hidden="true">
        <AlertTriangle size={22} />
      </span>
      <strong>{title}</strong>
      <span>{message}</span>
      {onRetry ? (
        <button className="button button--secondary" type="button" onClick={onRetry}>
          <RefreshCw size={15} aria-hidden="true" />
          Try again
        </button>
      ) : null}
    </div>
  )
}

export function EmptyState({
  title,
  description,
  action,
}: {
  title: string
  description: string
  action?: ReactNode
}) {
  return (
    <div className="state-panel state-panel--empty">
      <span className="state-panel__icon" aria-hidden="true">
        <Inbox size={22} />
      </span>
      <strong>{title}</strong>
      <span>{description}</span>
      {action}
    </div>
  )
}

export function ProgressBar({
  current,
  total,
  label,
}: {
  current?: number | null
  total?: number | null
  label: string
}) {
  const validTotal = typeof total === 'number' && total > 0
  const percentage = validTotal
    ? Math.max(0, Math.min(100, Math.round(((current ?? 0) / total) * 100)))
    : null

  return (
    <div className="progress">
      <div
        className={`progress__track ${percentage == null ? 'progress__track--indeterminate' : ''}`}
        role="progressbar"
        aria-label={label}
        aria-valuemin={0}
        aria-valuemax={validTotal ? total : undefined}
        aria-valuenow={validTotal ? current ?? 0 : undefined}
        aria-valuetext={percentage == null ? 'In progress' : `${percentage}%`}
      >
        <span className="progress__fill" style={percentage == null ? undefined : { width: `${percentage}%` }} />
      </div>
      <span>{percentage == null ? 'Working…' : `${percentage}%`}</span>
    </div>
  )
}
