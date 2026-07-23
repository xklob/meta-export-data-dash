import { useQuery } from '@tanstack/react-query'
import { AlertTriangle, CheckCircle2, Clock3, Layers3, RefreshCw, Search, TimerReset } from 'lucide-react'
import { useMemo, useState } from 'react'
import { api } from '../lib/api'
import type { Job } from '../lib/contracts'
import { formatBytes, formatCount, formatDateTime, formatDuration, sentenceCase } from '../lib/format'
import { useArchiveTimezone } from '../lib/useArchiveTimezone'
import { EmptyState, ErrorState, PageHeading, PageLoading, ProgressBar, StatusBadge } from '../components/ui'

type JobFilter = 'all' | 'running' | 'complete' | 'failed'

function JobCard({ job, timezone }: { job: Job; timezone: string }) {
  const counters = Object.entries(job.counters ?? {})
  const formatCounter = (key: string, value: number) =>
    key.toLowerCase().includes('bytes') ? formatBytes(value) : formatCount(value)
  return (
    <article className="job-card">
      <div className="job-card__rail" aria-hidden="true">
        <Layers3 size={20} />
      </div>
      <div className="job-card__content">
        <div className="job-card__header">
          <div>
            <div className="job-card__titleline">
              <h3>{sentenceCase(job.kind)}</h3>
              <StatusBadge status={job.status} />
            </div>
            <p>{job.stage || 'No current stage reported'}</p>
          </div>
          <code title={job.id}>{job.id.slice(0, 12)}</code>
        </div>

        <ProgressBar current={job.progress_current} total={job.progress_total} label={`${sentenceCase(job.kind)} progress`} />

        <dl className="job-card__facts">
          <div>
            <dt>Started</dt>
            <dd>{formatDateTime(job.started_at, timezone)}</dd>
          </div>
          <div>
            <dt>Finished</dt>
            <dd>{job.finished_at ? formatDateTime(job.finished_at, timezone) : 'Still running'}</dd>
          </div>
          <div>
            <dt>Duration</dt>
            <dd>{formatDuration(job.started_at, job.finished_at)}</dd>
          </div>
          <div>
            <dt>Progress</dt>
            <dd>{formatCount(job.progress_current)} / {formatCount(job.progress_total)}</dd>
          </div>
        </dl>

        {counters.length > 0 ? (
          <div className="counter-list" role="group" aria-label="Job counters">
            {counters.map(([key, value]) => (
              <span key={key}>
                <strong>{formatCounter(key, value)}</strong> {sentenceCase(key)}
              </span>
            ))}
          </div>
        ) : null}

        {job.error ? (
          <div className="inline-alert inline-alert--danger" role="alert">
            <AlertTriangle size={17} aria-hidden="true" />
            <div>
              <strong>Job stopped with an error</strong>
              <span>{job.error}</span>
            </div>
          </div>
        ) : null}
      </div>
    </article>
  )
}

export function JobsPage() {
  const [status, setStatus] = useState<JobFilter>('all')
  const [search, setSearch] = useState('')
  const [pageIndex, setPageIndex] = useState(0)
  const pageSize = 10
  const timezone = useArchiveTimezone()
  const jobsQuery = useQuery({
    queryKey: ['jobs', { status, search, pageIndex }],
    queryFn: () => api.jobs({
      status,
      q: search,
      limit: pageSize,
      offset: pageIndex * pageSize,
    }),
    refetchInterval: (query) =>
      query.state.data?.items?.some((job) => ['running', 'pending', 'queued'].includes(job.status.toLowerCase()))
        ? 2_500
        : false,
  })

  const jobs = useMemo(() => jobsQuery.data?.items ?? [], [jobsQuery.data?.items])
  const filtered = useMemo(() => {
    const query = search.trim().toLowerCase()
    return jobs.filter((job) => {
      const normalizedStatus = job.status.toLowerCase()
      const matchesStatus =
        status === 'all' ||
        (status === 'complete' && ['complete', 'completed', 'completed_with_errors', 'success', 'succeeded'].includes(normalizedStatus)) ||
        (status === 'failed' && ['failed', 'error', 'cancelled'].includes(normalizedStatus)) ||
        (status === 'running' && ['running', 'pending', 'queued'].includes(normalizedStatus))
      const matchesSearch = !query || `${job.kind} ${job.stage ?? ''} ${job.id}`.toLowerCase().includes(query)
      return matchesStatus && matchesSearch
    })
  }, [jobs, search, status])

  if (jobsQuery.isLoading) return <PageLoading label="Loading archive jobs" />
  if (jobsQuery.isError) return <ErrorState error={jobsQuery.error} onRetry={() => jobsQuery.refetch()} />

  const summary = jobsQuery.data?.summary
  const running = summary?.running ?? jobs.filter((job) => ['running', 'pending', 'queued'].includes(job.status.toLowerCase())).length
  const completed = summary?.completed ?? jobs.filter((job) => ['complete', 'completed', 'completed_with_errors', 'success', 'succeeded'].includes(job.status.toLowerCase())).length
  const failed = summary?.failed ?? jobs.filter((job) => ['failed', 'error', 'cancelled'].includes(job.status.toLowerCase())).length
  const total = summary?.total ?? jobsQuery.data?.page?.total ?? jobs.length
  const filteredTotal = jobsQuery.data?.page?.filtered_total ?? filtered.length

  return (
    <div className="page-stack">
      <PageHeading
        eyebrow="Resumable operations"
        title="Jobs"
        description="Track scan, hash, consolidation, import, and verification work. Progress is checkpointed so safe restarts do not discard completed work."
        actions={
          <button
            className="button button--secondary"
            type="button"
            onClick={() => jobsQuery.refetch()}
            disabled={jobsQuery.isFetching}
          >
            <RefreshCw className={jobsQuery.isFetching ? 'spin' : ''} size={15} aria-hidden="true" />
            Refresh
          </button>
        }
      />

      <section className="job-summary" aria-label="Job summary">
        <div>
          <span className="job-summary__icon job-summary__icon--running" aria-hidden="true"><Clock3 size={18} /></span>
          <div><strong>{running}</strong><span>Active</span></div>
        </div>
        <div>
          <span className="job-summary__icon job-summary__icon--complete" aria-hidden="true"><CheckCircle2 size={18} /></span>
          <div><strong>{completed}</strong><span>Completed</span></div>
        </div>
        <div>
          <span className="job-summary__icon job-summary__icon--failed" aria-hidden="true"><AlertTriangle size={18} /></span>
          <div><strong>{failed}</strong><span>Stopped</span></div>
        </div>
        <div>
          <span className="job-summary__icon" aria-hidden="true"><TimerReset size={18} /></span>
          <div><strong>{total}</strong><span>Total jobs</span></div>
        </div>
      </section>

      <section className="panel" aria-labelledby="job-history-title">
        <div className="table-toolbar table-toolbar--wrap">
          <div>
            <h2 id="job-history-title">Operation history</h2>
            <p>{filtered.length} shown · {filteredTotal} matching</p>
          </div>
          <label className="search-field">
            <span className="sr-only">Search jobs</span>
            <Search size={16} aria-hidden="true" />
            <input
              type="search"
              value={search}
              onChange={(event) => {
                setSearch(event.target.value)
                setPageIndex(0)
              }}
              placeholder="Find a job or stage"
            />
          </label>
        </div>
        <div className="segmented-control" role="group" aria-label="Filter jobs by status">
          {(['all', 'running', 'complete', 'failed'] as JobFilter[]).map((value) => (
            <button
              type="button"
              key={value}
              className={status === value ? 'segmented-control__active' : ''}
              aria-pressed={status === value}
              onClick={() => {
                setStatus(value)
                setPageIndex(0)
              }}
            >
              {sentenceCase(value)}
            </button>
          ))}
        </div>

        {filtered.length === 0 ? (
          <EmptyState
            title={jobs.length === 0 ? 'No archive jobs yet' : 'No jobs match these filters'}
            description={
              jobs.length === 0
                ? 'Operations started from the CLI will appear here automatically.'
                : 'Clear the search or select another job state.'
            }
            action={
              jobs.length > 0 ? (
                <button
                  className="button button--secondary"
                  type="button"
                  onClick={() => {
                    setStatus('all')
                    setSearch('')
                    setPageIndex(0)
                  }}
                >
                  Clear filters
                </button>
              ) : undefined
            }
          />
        ) : (
          <div className="job-list">
            {filtered.map((job) => <JobCard job={job} timezone={timezone} key={job.id} />)}
          </div>
        )}
        {filteredTotal > pageSize ? (
          <div className="pagination-controls" role="navigation" aria-label="Job history pages">
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
