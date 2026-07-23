export function BoundedResultsHint({ id, subject }: { id: string; subject: string }) {
  return (
    <p id={id} className="bounded-results-hint">
      Scroll inside the framed pane to browse all loaded {subject}. Controls below load or move to another batch.
    </p>
  )
}
