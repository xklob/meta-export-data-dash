import { useQueryClient } from '@tanstack/react-query'
import { Star } from 'lucide-react'
import {
  createContext,
  type ReactNode,
  useCallback,
  useContext,
  useMemo,
  useState,
} from 'react'
import { api } from '../lib/api'
import type { StarState, StarTarget } from '../lib/contracts'

function targetKey(target: StarTarget): string {
  return `${target.target_type}:${target.target_id}`
}

interface StarContextValue {
  stateFor: (target: StarTarget, serverState?: StarState | null) => StarState | null
  isPending: (target: StarTarget) => boolean
  toggle: (target: StarTarget, current: StarState | null, label: string) => Promise<boolean>
}

const StarContext = createContext<StarContextValue | null>(null)

export function StarProvider({ children }: { children: ReactNode }) {
  const queryClient = useQueryClient()
  const [overrides, setOverrides] = useState<Map<string, StarState | null>>(() => new Map())
  const [pending, setPending] = useState<Set<string>>(() => new Set())
  const [announcement, setAnnouncement] = useState('')

  const stateFor = useCallback(
    (target: StarTarget, serverState?: StarState | null) => {
      const key = targetKey(target)
      return overrides.has(key) ? overrides.get(key) ?? null : serverState ?? null
    },
    [overrides],
  )

  const isPending = useCallback((target: StarTarget) => pending.has(targetKey(target)), [pending])

  const toggle = useCallback(
    async (target: StarTarget, current: StarState | null, label: string) => {
      const key = targetKey(target)
      if (pending.has(key)) return Boolean(current)
      const nextIsStarred = !current
      const optimistic: StarState | null = nextIsStarred
        ? { id: `pending:${key}`, starred_at: new Date().toISOString() }
        : null
      setPending((value) => new Set(value).add(key))
      setOverrides((value) => new Map(value).set(key, optimistic))
      setAnnouncement('')

      try {
        if (nextIsStarred) {
          const result = await api.star(target)
          setOverrides((value) => new Map(value).set(key, result.star))
          setAnnouncement(`Starred ${label}.`)
        } else {
          await api.unstar(target)
          setAnnouncement(`Removed ${label} from starred items.`)
        }
        void queryClient.invalidateQueries({ queryKey: ['stars'] })
        return nextIsStarred
      } catch (error) {
        setOverrides((value) => new Map(value).set(key, current))
        setAnnouncement(
          error instanceof Error
            ? `${nextIsStarred ? 'Starring' : 'Unstarring'} ${label} failed: ${error.message}`
            : `${nextIsStarred ? 'Starring' : 'Unstarring'} ${label} failed.`,
        )
        throw error
      } finally {
        setPending((value) => {
          const next = new Set(value)
          next.delete(key)
          return next
        })
      }
    },
    [pending, queryClient],
  )

  const value = useMemo<StarContextValue>(
    () => ({ stateFor, isPending, toggle }),
    [isPending, stateFor, toggle],
  )

  return (
    <StarContext.Provider value={value}>
      {children}
      <span className="sr-only" role="status" aria-live="polite" aria-atomic="true">
        {announcement}
      </span>
    </StarContext.Provider>
  )
}

export function StarButton({
  target,
  star,
  label,
  className = '',
  compact = true,
  onChanged,
}: {
  target: StarTarget
  star?: StarState | null
  label: string
  className?: string
  compact?: boolean
  onChanged?: (starred: boolean) => void
}) {
  const context = useContext(StarContext)
  if (!context) throw new Error('StarButton must be rendered inside StarProvider.')
  const current = context.stateFor(target, star)
  const active = Boolean(current)
  const pending = context.isPending(target)
  const action = active ? 'Unstar' : 'Star'

  return (
    <button
      className={`star-button ${active ? 'star-button--active' : ''} ${compact ? 'star-button--compact' : ''} ${className}`.trim()}
      type="button"
      aria-label={`${action} ${label}`}
      aria-pressed={active}
      aria-busy={pending || undefined}
      disabled={pending}
      title={`${action} ${label}`}
      onClick={() => {
        void context.toggle(target, current, label)
          .then(onChanged)
          .catch(() => undefined)
      }}
    >
      <Star size={compact ? 16 : 18} fill={active ? 'currentColor' : 'none'} aria-hidden="true" />
      {!compact ? <span>{active ? 'Starred' : 'Star'}</span> : null}
    </button>
  )
}
