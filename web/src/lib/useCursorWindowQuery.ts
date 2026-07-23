import { keepPreviousData, useQuery, type QueryKey } from '@tanstack/react-query'
import { useEffect, useMemo, useState } from 'react'

interface CursorPage {
  page: {
    next_cursor?: string | null
  }
}

interface CursorNavigation {
  scopeKey: string
  cursors: (string | undefined)[]
  committedPageNumber: number
}

interface CursorWindowQueryOptions<TPage extends CursorPage> {
  queryKey: QueryKey
  queryFn: (cursor: string | undefined) => Promise<TPage>
  enabled?: boolean
}

/**
 * Traverse a forward-only keyset cursor without retaining every payload in memory.
 *
 * Cursor tokens are cheap enough to keep for Previous-page navigation, while
 * inactive React Query payloads are discarded immediately. At most the current
 * payload and a transient placeholder payload exist during a page transition.
 * The visible page number is committed only after that target request settles,
 * so consumers cannot mistake placeholder data for a completed navigation.
 */
export function useCursorWindowQuery<TPage extends CursorPage>({
  queryKey,
  queryFn,
  enabled = true,
}: CursorWindowQueryOptions<TPage>) {
  const scopeKey = useMemo(() => JSON.stringify(queryKey), [queryKey])
  const [navigation, setNavigation] = useState<CursorNavigation>(() => ({
    scopeKey,
    cursors: [undefined],
    committedPageNumber: 1,
  }))
  const cursors = navigation.scopeKey === scopeKey ? navigation.cursors : [undefined]
  const cursor = cursors.at(-1)
  const committedPageNumber = navigation.scopeKey === scopeKey
    ? navigation.committedPageNumber
    : 1

  useEffect(() => {
    setNavigation((current) => (
      current.scopeKey === scopeKey
        ? current
        : { scopeKey, cursors: [undefined], committedPageNumber: 1 }
    ))
  }, [scopeKey])

  const query = useQuery({
    queryKey: [...queryKey, 'cursor-window', cursor ?? null],
    queryFn: () => queryFn(cursor),
    enabled,
    gcTime: 0,
    placeholderData: keepPreviousData,
  })
  const nextCursor = query.data?.page.next_cursor ?? undefined
  const transitionPending = query.isFetching && query.isPlaceholderData
  const targetPageNumber = cursors.length

  useEffect(() => {
    if ((!query.isSuccess && !query.isError) || query.isPlaceholderData) return
    setNavigation((current) => {
      if (current.scopeKey !== scopeKey) return current
      return current.committedPageNumber === targetPageNumber
        ? current
        : { ...current, committedPageNumber: targetPageNumber }
    })
  }, [query.dataUpdatedAt, query.isError, query.isPlaceholderData, query.isSuccess, scopeKey, targetPageNumber])

  function nextPage() {
    if (!nextCursor || transitionPending) return
    setNavigation((current) => {
      const activeCursors = current.scopeKey === scopeKey ? current.cursors : [undefined]
      const activePageNumber = current.scopeKey === scopeKey
        ? current.committedPageNumber
        : 1
      return {
        scopeKey,
        cursors: [...activeCursors, nextCursor],
        committedPageNumber: activePageNumber,
      }
    })
  }

  function previousPage() {
    if (cursors.length <= 1 || transitionPending) return
    setNavigation((current) => {
      const activeCursors = current.scopeKey === scopeKey ? current.cursors : [undefined]
      const activePageNumber = current.scopeKey === scopeKey
        ? current.committedPageNumber
        : 1
      return {
        scopeKey,
        cursors: activeCursors.slice(0, -1),
        committedPageNumber: activePageNumber,
      }
    })
  }

  return {
    ...query,
    cursor,
    pageNumber: committedPageNumber,
    targetPageNumber,
    hasPreviousPage: cursors.length > 1,
    hasNextPage: Boolean(nextCursor),
    nextPage,
    previousPage,
    transitionPending,
  }
}
