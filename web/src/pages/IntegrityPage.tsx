import { useQuery } from '@tanstack/react-query'
import {
  AlertCircle,
  AlertTriangle,
  Check,
  CheckCircle2,
  CircleHelp,
  Info,
  Clock3,
  Search,
  ShieldCheck,
  XCircle,
} from 'lucide-react'
import { useMemo, useState } from 'react'
import { api } from '../lib/api'
import { formatCount, formatDateTime, sentenceCase, stringifyValue } from '../lib/format'
import { useArchiveTimezone } from '../lib/useArchiveTimezone'
import { EmptyState, ErrorState, MetricCard, PageHeading, PageLoading, StatusBadge } from '../components/ui'

type SeverityFilter = 'all' | 'critical' | 'warning' | 'info'

const passingCheckStatuses = ['ok', 'pass', 'passed', 'complete', 'completed', 'success']
const pendingCheckStatuses = ['pending', 'queued', 'waiting']

function IntegrityDetail({ detail }: { detail: unknown }) {
  if (!detail) return null
  if (typeof detail !== 'object' || Array.isArray(detail)) {
    return <p>{stringifyValue(detail)}</p>
  }

  const entries = Object.entries(detail as Record<string, unknown>)
  return (
    <details className="issue-detail">
      <summary>View source details</summary>
      <dl>
        {entries.map(([key, value]) => (
          <div key={key}>
            <dt>{sentenceCase(key)}</dt>
            <dd>{stringifyValue(value)}</dd>
          </div>
        ))}
      </dl>
    </details>
  )
}

export function IntegrityPage() {
  const timezone = useArchiveTimezone()
  const [severity, setSeverity] = useState<SeverityFilter>('all')
  const [showResolved, setShowResolved] = useState(false)
  const [search, setSearch] = useState('')
  const [pageIndex, setPageIndex] = useState(0)
  const pageSize = 25
  const integrity = useQuery({
    queryKey: ['integrity', { includeResolved: showResolved, severity, search, pageIndex }],
    queryFn: () => api.integrity({
      includeResolved: showResolved,
      severity,
      q: search,
      limit: pageSize,
      offset: pageIndex * pageSize,
    }),
  })

  const issues = useMemo(() => integrity.data?.items ?? [], [integrity.data?.items])
  const filteredIssues = useMemo(() => {
    const query = search.trim().toLowerCase()
    return issues.filter((issue) => {
      if (!showResolved && issue.resolved) return false
      if (severity !== 'all' && issue.severity.toLowerCase() !== severity) return false
      if (!query) return true
      return `${issue.code} ${issue.summary} ${stringifyValue(issue.detail)}`.toLowerCase().includes(query)
    })
  }, [issues, search, severity, showResolved])

  if (integrity.isLoading) return <PageLoading label="Checking archive integrity" />
  if (integrity.isError) return <ErrorState error={integrity.error} onRetry={() => integrity.refetch()} />

  const summary = integrity.data?.summary ?? {}
  const checks = integrity.data?.checks ?? []
  const filteredTotal = integrity.data?.page?.filtered_total ?? filteredIssues.length
  const checksPassed = checks.filter((check) => passingCheckStatuses.includes(check.status.toLowerCase())).length

  return (
    <div className="page-stack">
      <PageHeading
        eyebrow="Verification ledger"
        title="Integrity"
        description="Every gap stays visible. Reconciliation checks compare the normalized catalog with source facts before consolidation can be accepted."
      />

      <section className="metric-grid" aria-label="Integrity totals">
        <MetricCard
          icon={ShieldCheck}
          label="Total findings"
          value={formatCount(summary.total)}
          detail={`${formatCount(summary.unresolved)} unresolved`}
        />
        <MetricCard
          icon={AlertCircle}
          label="Critical"
          value={formatCount(summary.critical)}
          detail="Requires attention before promotion"
          tone={(summary.critical ?? 0) > 0 ? 'danger' : 'positive'}
        />
        <MetricCard
          icon={AlertTriangle}
          label="Warnings"
          value={formatCount(summary.warning)}
          detail="Preserved with provenance"
          tone={(summary.warning ?? 0) > 0 ? 'warning' : 'positive'}
        />
        <MetricCard
          icon={Info}
          label="Informational"
          value={formatCount(summary.info)}
          detail={`${checksPassed} of ${checks.length} checks passing`}
        />
      </section>

      <section className="panel" aria-labelledby="reconciliation-heading">
        <div className="section-header">
          <div>
            <h2 id="reconciliation-heading">Reconciliation checks</h2>
            <p>Expected archive facts compared with the current catalog.</p>
          </div>
          <span className="check-count">{checksPassed}/{checks.length} passing</span>
        </div>
        {checks.length === 0 ? (
          <EmptyState
            title="No reconciliation checks have run"
            description="Checks will appear after the inventory and importer establish a catalog baseline."
          />
        ) : (
          <details className="integrity-checks">
            <summary>
              Review all {checks.length} detailed checks
              <span>{checks.length - checksPassed} need attention or remain pending</span>
            </summary>
            <div className="check-grid">
              {checks.map((check) => {
                const normalizedStatus = check.status.toLowerCase()
                const passed = passingCheckStatuses.includes(normalizedStatus)
                const pending = pendingCheckStatuses.includes(normalizedStatus)
                return (
                  <article
                    className={`check-card ${passed ? 'check-card--passed' : pending ? 'check-card--pending' : 'check-card--failed'}`}
                    key={check.key}
                  >
                    <span className="check-card__icon" aria-hidden="true">
                      {passed ? <CheckCircle2 size={19} /> : pending ? <Clock3 size={19} /> : <XCircle size={19} />}
                    </span>
                    <div className="check-card__body">
                      <strong>{sentenceCase(check.key)}</strong>
                      <span>
                        {check.detail ||
                          (passed ? 'Expected and actual values agree.' : pending ? 'Awaiting a later import stage.' : 'Values need review.')}
                      </span>
                      <dl>
                        <div>
                          <dt>Expected</dt>
                          <dd>{stringifyValue(check.expected)}</dd>
                        </div>
                        <div>
                          <dt>Actual</dt>
                          <dd>{stringifyValue(check.actual)}</dd>
                        </div>
                      </dl>
                    </div>
                  </article>
                )
              })}
            </div>
          </details>
        )}
      </section>

      <section className="panel" aria-labelledby="findings-heading">
        <div className="table-toolbar table-toolbar--wrap">
          <div>
            <h2 id="findings-heading">Findings ledger</h2>
            <p>{filteredIssues.length} shown · {filteredTotal} matching</p>
          </div>
          <div className="table-toolbar__controls">
            <label className="search-field">
              <span className="sr-only">Search integrity findings</span>
              <Search size={16} aria-hidden="true" />
              <input
                type="search"
                value={search}
                onChange={(event) => {
                  setSearch(event.target.value)
                  setPageIndex(0)
                }}
                placeholder="Search code or detail"
              />
            </label>
            <label className="checkbox-control">
              <input
                type="checkbox"
                checked={showResolved}
                onChange={(event) => {
                  setShowResolved(event.target.checked)
                  setPageIndex(0)
                }}
              />
              <span aria-hidden="true"><Check size={13} /></span>
              Show resolved
            </label>
          </div>
        </div>
        <div className="segmented-control" role="group" aria-label="Filter findings by severity">
          {(['all', 'critical', 'warning', 'info'] as SeverityFilter[]).map((value) => (
            <button
              key={value}
              type="button"
              className={severity === value ? 'segmented-control__active' : ''}
              aria-pressed={severity === value}
              onClick={() => {
                setSeverity(value)
                setPageIndex(0)
              }}
            >
              {sentenceCase(value)}
            </button>
          ))}
        </div>

        {filteredIssues.length === 0 ? (
          <EmptyState
            title={issues.length === 0 ? 'No integrity findings recorded' : 'No findings match these filters'}
            description={
              issues.length === 0
                ? 'The ledger is empty. Re-run verification after the next archive operation.'
                : 'Try another severity, include resolved items, or clear the search.'
            }
            action={
              issues.length > 0 ? (
                <button
                  className="button button--secondary"
                  type="button"
                  onClick={() => {
                    setSeverity('all')
                    setSearch('')
                    setShowResolved(true)
                    setPageIndex(0)
                  }}
                >
                  Reset filters
                </button>
              ) : undefined
            }
          />
        ) : (
          <div className="issue-list">
            {filteredIssues.map((issue) => (
              <article className={`issue-card issue-card--${issue.severity.toLowerCase()}`} key={issue.id}>
                <span className="issue-card__icon" aria-hidden="true">
                  {issue.severity.toLowerCase() === 'critical' ? (
                    <AlertCircle size={19} />
                  ) : issue.severity.toLowerCase() === 'warning' ? (
                    <AlertTriangle size={19} />
                  ) : (
                    <CircleHelp size={19} />
                  )}
                </span>
                <div className="issue-card__body">
                  <div className="issue-card__topline">
                    <code>{issue.code}</code>
                    <StatusBadge status={issue.resolved ? 'resolved' : issue.severity} />
                    <time dateTime={issue.created_at ?? undefined}>{formatDateTime(issue.created_at, timezone)}</time>
                  </div>
                  <h3>{issue.summary}</h3>
                  <IntegrityDetail detail={issue.detail} />
                </div>
              </article>
            ))}
          </div>
        )}
        {filteredTotal > pageSize ? (
          <div className="pagination-controls" role="navigation" aria-label="Integrity findings pages">
            <button
              className="button button--secondary"
              type="button"
              disabled={pageIndex === 0}
              onClick={() => setPageIndex((value) => Math.max(0, value - 1))}
            >
              Previous
            </button>
            <span>Page {pageIndex + 1} of {Math.ceil(filteredTotal / pageSize)}</span>
            <button
              className="button button--secondary"
              type="button"
              disabled={(pageIndex + 1) * pageSize >= filteredTotal}
              onClick={() => setPageIndex((value) => value + 1)}
            >
              Next
            </button>
          </div>
        ) : null}
      </section>
    </div>
  )
}
