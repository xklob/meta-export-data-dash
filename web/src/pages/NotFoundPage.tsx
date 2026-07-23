import { Link } from '@tanstack/react-router'
import { ArrowLeft, FileQuestion } from 'lucide-react'

export function NotFoundPage() {
  return (
    <div className="not-found">
      <span aria-hidden="true"><FileQuestion size={28} /></span>
      <p className="eyebrow">Page not found</p>
      <h1>This archive view does not exist.</h1>
      <p>The route may belong to a future stage or the address may be incomplete.</p>
      <Link className="button button--primary" to="/">
        <ArrowLeft size={15} aria-hidden="true" />
        Return home
      </Link>
    </div>
  )
}
