import { AlertTriangle, Database, Info } from 'lucide-react'
import type { CoverageSummary } from '../lib/contracts'
import { formatCount } from '../lib/format'
import { StatusBadge } from './ui'

export function CoverageNotice({ coverage }: { coverage: CoverageSummary }) {
  const unavailable = coverage.status === 'unavailable'
  const Icon = unavailable ? AlertTriangle : coverage.status === 'partial' ? Info : Database
  const count = coverage.logical_count ?? coverage.raw_count

  return (
    <aside className={`coverage-notice coverage-notice--${coverage.status}`} aria-label="Source coverage">
      <span className="coverage-notice__icon" aria-hidden="true">
        <Icon size={19} />
      </span>
      <div>
        <div className="coverage-notice__title">
          <strong>{coverage.label ?? 'Archive coverage'}</strong>
          <StatusBadge status={coverage.status} />
          {count != null ? <span>{formatCount(count)} records</span> : null}
        </div>
        {coverage.description ? <p>{coverage.description}</p> : null}
        {coverage.absence_is_not_zero_history ? (
          <p className="coverage-notice__caution">
            Missing source files mean this archive cannot establish that your history was empty.
          </p>
        ) : null}
      </div>
    </aside>
  )
}
