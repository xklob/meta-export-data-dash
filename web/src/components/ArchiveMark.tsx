export function ArchiveMark({ compact = false }: { compact?: boolean }) {
  return (
    <div className={`archive-mark ${compact ? 'archive-mark--compact' : ''}`} aria-hidden="true">
      <span />
      <span />
      <span />
    </div>
  )
}
