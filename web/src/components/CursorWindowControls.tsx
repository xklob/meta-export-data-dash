import { ArrowLeft, ArrowRight } from 'lucide-react'
import { formatCount } from '../lib/format'

interface CursorWindowControlsProps {
  label: string
  pageNumber: number
  targetPageNumber?: number
  itemCount: number
  total?: number | null
  hasPreviousPage: boolean
  hasNextPage: boolean
  loading: boolean
  onPreviousPage: () => void
  onNextPage: () => void
}

export function CursorWindowControls({
  label,
  pageNumber,
  targetPageNumber = pageNumber,
  itemCount,
  total,
  hasPreviousPage,
  hasNextPage,
  loading,
  onPreviousPage,
  onNextPage,
}: CursorWindowControlsProps) {
  const totalLabel = total == null ? 'total unavailable' : `${formatCount(total)} total`
  const previousDisabled = !hasPreviousPage || loading
  const nextDisabled = !hasNextPage || loading

  const visibleStatus = loading && itemCount > 0
    ? `${formatCount(itemCount)} current results remain shown · ${totalLabel}`
    : `${formatCount(itemCount)} shown · ${totalLabel}`
  const statusPageNumber = loading ? targetPageNumber : pageNumber

  return (
    <>
      <nav
        className="load-more-row cursor-window-controls"
        aria-label={`${label} pages`}
        aria-busy={loading}
        data-page-number={pageNumber}
        data-loaded-item-count={itemCount}
      >
        <span className="cursor-window-controls__status">
          <strong>{loading ? 'Loading page' : 'Page'} {formatCount(statusPageNumber)}</strong>
          <span>{visibleStatus}</span>
        </span>
        <span className="cursor-window-controls__buttons">
          <button
            className="button button--secondary"
            type="button"
            aria-disabled={previousDisabled}
            onClick={() => {
              if (!previousDisabled) onPreviousPage()
            }}
          >
            <ArrowLeft size={14} aria-hidden="true" />
            Previous page
          </button>
          <button
            className="button button--secondary"
            type="button"
            aria-disabled={nextDisabled}
            onClick={() => {
              if (!nextDisabled) onNextPage()
            }}
          >
            Next page
            <ArrowRight size={14} aria-hidden="true" />
          </button>
        </span>
      </nav>
      <span className="sr-only" role="status" aria-live="polite" aria-atomic="true">
        {loading ? `Loading page ${formatCount(statusPageNumber)}. ${visibleStatus}.` : ''}
      </span>
    </>
  )
}
