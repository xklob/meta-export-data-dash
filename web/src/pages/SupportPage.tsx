import { Link } from '@tanstack/react-router'
import { LifeBuoy, MessageSquareText, ShieldCheck } from 'lucide-react'
import { CoverageNotice } from '../components/CoverageNotice'
import { CursorWindowControls } from '../components/CursorWindowControls'
import { StarButton } from '../components/StarButton'
import { EmptyState, ErrorState, MetricCard, PageHeading, PageLoading } from '../components/ui'
import { api } from '../lib/api'
import { formatCount, formatDateMs } from '../lib/format'
import { useArchiveTimezone } from '../lib/useArchiveTimezone'
import { useCursorWindowQuery } from '../lib/useCursorWindowQuery'

export function SupportPage() {
  const timezone = useArchiveTimezone()
  const query = useCursorWindowQuery({
    queryKey: ['support-threads'],
    queryFn: (cursor) => api.support({ cursor, limit: 20 }),
  })

  if (query.isPending) return <PageLoading label="Loading Facebook support conversations" />
  if (query.isError || !query.data) {
    return <ErrorState error={query.error} onRetry={() => void query.refetch()} />
  }

  const messageCount = query.data.items.reduce((sum, thread) => sum + thread.message_count, 0)

  return (
    <div className="page-stack stage4-page">
      <PageHeading
        eyebrow="Facebook support history"
        title="Support conversations"
        description="Browse the support-message threads retained in this export. These are kept separate from ordinary Messenger conversations and non-message support activity."
      />
      <CoverageNotice coverage={query.data.coverage} />
      <section className="metric-grid" aria-label="Support conversation totals">
        <MetricCard icon={LifeBuoy} label="Support threads" value={formatCount(query.data.page.total)} detail="Source-backed conversations" />
        <MetricCard icon={MessageSquareText} label="Messages on this page" value={formatCount(messageCount)} detail="Across the threads shown below" />
        <MetricCard icon={ShieldCheck} label="Browsing mode" value="Local" detail="Opens the retained conversation reader" tone="positive" />
      </section>
      <section className="panel support-results" aria-labelledby="support-results-title">
        <div className="section-header">
          <div>
            <h2 id="support-results-title">Support thread history</h2>
            <p>Open a thread to read its complete exported message history.</p>
          </div>
        </div>
        {query.data.items.length ? (
          <div className="support-grid">
            {query.data.items.map((thread) => (
              <article className="support-card" key={thread.id}>
                <span className="support-card__icon" aria-hidden="true"><LifeBuoy size={19} /></span>
                <div>
                  <h3>{thread.title || 'Untitled support thread'}</h3>
                  <p>{formatCount(thread.message_count)} messages</p>
                  <span>{formatDateMs(thread.first_activity_ms, timezone)} – {formatDateMs(thread.last_activity_ms, timezone)}</span>
                </div>
                <StarButton target={thread.star_target ?? { target_type: 'conversation', target_id: thread.id }} star={thread.star} label={`support conversation ${thread.title || 'Untitled support thread'}`} />
                <Link
                  className="button button--secondary"
                  to="/conversations/$conversationId"
                  params={{ conversationId: thread.id }}
                  search={{ around: undefined }}
                >
                  Open conversation
                </Link>
              </article>
            ))}
          </div>
        ) : (
          <EmptyState
            title="Support-message history unavailable"
            description={query.data.coverage.description ?? 'No support-message threads were retained in this export.'}
          />
        )}
        <CursorWindowControls
          label="Support conversations"
          pageNumber={query.pageNumber}
          targetPageNumber={query.targetPageNumber}
          itemCount={query.data.items.length}
          total={query.data.page.total}
          hasPreviousPage={query.hasPreviousPage}
          hasNextPage={query.hasNextPage}
          loading={query.transitionPending || query.isPending}
          onPreviousPage={query.previousPage}
          onNextPage={query.nextPage}
        />
      </section>
    </div>
  )
}
