import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { expect, it, vi } from 'vitest'
import { useCursorWindowQuery } from './useCursorWindowQuery'

interface FixturePage {
  items: { id: string }[]
  page: { total: number; limit: number; next_cursor: string | null }
}

function CursorHarness({ fetchPage }: { fetchPage: (cursor?: string) => Promise<FixturePage> }) {
  const query = useCursorWindowQuery({
    queryKey: ['cursor-window-test'],
    queryFn: fetchPage,
  })

  if (query.isPending) return <p>Loading first page</p>
  if (query.isError) return <p role="alert">Failed</p>

  return (
    <div>
      <p>Page {query.pageNumber}</p>
      <ul data-testid="current-window">
        {query.data.items.map((item) => <li key={item.id}>{item.id}</li>)}
      </ul>
      <button type="button" disabled={!query.hasPreviousPage || query.transitionPending} onClick={query.previousPage}>
        Previous page
      </button>
      <button type="button" disabled={!query.hasNextPage || query.transitionPending} onClick={query.nextPage}>
        Next page
      </button>
    </div>
  )
}

it('traverses beyond the former sixteen-page cap and back while retaining one payload window', async () => {
  const fetchPage = vi.fn(async (cursor?: string): Promise<FixturePage> => {
    const pageNumber = cursor ? Number(cursor.replace('cursor-', '')) : 1
    return {
      items: [{ id: `row-${pageNumber}` }],
      page: {
        total: 2_500,
        limit: 100,
        next_cursor: pageNumber < 25 ? `cursor-${pageNumber + 1}` : null,
      },
    }
  })
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false, staleTime: Infinity } },
  })
  const user = userEvent.setup()
  render(
    <QueryClientProvider client={queryClient}>
      <CursorHarness fetchPage={fetchPage} />
    </QueryClientProvider>,
  )

  expect(await screen.findByText('row-1')).toBeVisible()
  for (let pageNumber = 2; pageNumber <= 18; pageNumber += 1) {
    await user.click(screen.getByRole('button', { name: 'Next page' }))
    expect(await screen.findByText(`row-${pageNumber}`)).toBeVisible()
    expect(screen.getByTestId('current-window').children).toHaveLength(1)
    expect(screen.queryByText(`row-${pageNumber - 1}`)).not.toBeInTheDocument()
  }

  expect(screen.getByText('Page 18')).toBeVisible()
  await waitFor(() => {
    expect(queryClient.getQueryCache().findAll({ queryKey: ['cursor-window-test'] }).length).toBeLessThanOrEqual(1)
  })

  for (let pageNumber = 17; pageNumber >= 1; pageNumber -= 1) {
    await user.click(screen.getByRole('button', { name: 'Previous page' }))
    expect(await screen.findByText(`row-${pageNumber}`)).toBeVisible()
    expect(screen.getByTestId('current-window').children).toHaveLength(1)
  }

  expect(screen.getByText('Page 1')).toBeVisible()
  expect(screen.getByRole('button', { name: 'Previous page' })).toBeDisabled()
  expect(fetchPage).toHaveBeenCalledTimes(35)
})

it('commits a previous page number only after its bounded payload refetch settles', async () => {
  let firstPageRequests = 0
  let resolvePreviousPage!: (page: FixturePage) => void
  const pageOne: FixturePage = {
    items: [{ id: 'row-1' }],
    page: { total: 2, limit: 1, next_cursor: 'cursor-2' },
  }
  const fetchPage = vi.fn(async (cursor?: string): Promise<FixturePage> => {
    if (cursor === 'cursor-2') {
      return {
        items: [{ id: 'row-2' }],
        page: { total: 2, limit: 1, next_cursor: null },
      }
    }
    firstPageRequests += 1
    if (firstPageRequests === 1) return pageOne
    return new Promise((resolve) => {
      resolvePreviousPage = resolve
    })
  })
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false, staleTime: Infinity } },
  })
  const user = userEvent.setup()
  render(
    <QueryClientProvider client={queryClient}>
      <CursorHarness fetchPage={fetchPage} />
    </QueryClientProvider>,
  )

  expect(await screen.findByText('row-1')).toBeVisible()
  await user.click(screen.getByRole('button', { name: 'Next page' }))
  expect(await screen.findByText('row-2')).toBeVisible()
  expect(screen.getByText('Page 2')).toBeVisible()
  await waitFor(() => {
    expect(queryClient.getQueryCache().findAll({ queryKey: ['cursor-window-test'] })).toHaveLength(1)
  })

  await user.click(screen.getByRole('button', { name: 'Previous page' }))
  await waitFor(() => expect(fetchPage).toHaveBeenCalledTimes(3))
  expect(screen.getByText('Page 2')).toBeVisible()
  expect(screen.getByText('row-2')).toBeVisible()

  resolvePreviousPage(pageOne)
  expect(await screen.findByText('row-1')).toBeVisible()
  expect(screen.getByText('Page 1')).toBeVisible()
})
