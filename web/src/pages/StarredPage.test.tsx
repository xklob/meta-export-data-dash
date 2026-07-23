import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { type ReactNode } from 'react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { StarProvider } from '../components/StarButton'
import { StarredPage } from './StarredPage'

const starredItems = [
  {
    kind: 'message',
    star_target: { target_type: 'message', target_id: 'message-one' },
    star: { id: 'star-one', starred_at: '2026-07-22T20:00:00Z' },
    title: 'Argument message',
    subtitle: 'Alex Example · Direct message',
    excerpt: 'We should stop and talk about what happened.',
    timestamp_ms: 1_700_000_000_000,
    jump: { href: '/conversations/conv-one?around=message-one#message-message-one', conversation_id: 'conv-one', message_id: 'message-one' },
  },
  {
    kind: 'media',
    star_target: { target_type: 'blob', target_id: 'a'.repeat(64) },
    star: { id: 'star-two', starred_at: '2026-07-21T20:00:00Z' },
    title: 'sunset.jpg',
    subtitle: 'Alex Example',
    excerpt: null,
    timestamp_ms: 1_600_000_000_000,
    jump: { href: '/conversations/conv-one?around=message-two#message-message-two', conversation_id: 'conv-one', message_id: 'message-two' },
    media_scope: 'unique',
    media: {
      id: 'attachment-two',
      view_key: `blob:${'a'.repeat(64)}`,
      type: 'photo',
      kind: 'photos',
      filename: 'sunset.jpg',
      availability: 'resolved',
      blob_sha256: 'a'.repeat(64),
      blob_href: '/api/v1/media/blobs/a',
      unknown_date: false,
      occurrence_count: 2,
    },
  },
] as const

function wrapper({ children }: { children: ReactNode }) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  return <QueryClientProvider client={client}><StarProvider>{children}</StarProvider></QueryClientProvider>
}

describe('StarredPage', () => {
  beforeEach(() => {
    let messageRemoved = false
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = new URL(String(input), 'http://localhost')
      if (url.pathname === '/api/v1/settings') {
        return Response.json({ data: { timezone: 'America/New_York', cloud_enabled: false } })
      }
      if (url.pathname === '/api/v1/stars/message/message-one' && init?.method === 'DELETE') {
        messageRemoved = true
        return new Response(null, { status: 204 })
      }
      if (url.pathname === '/api/v1/stars') {
        const category = url.searchParams.get('category') ?? 'all'
        const retained = messageRemoved ? starredItems.slice(1) : starredItems
        const items = category === 'media' ? retained.filter((item) => item.kind === 'media') : retained
        return Response.json({ data: {
          items,
          page: { limit: 50, total: items.length, next_cursor: null },
          category,
          direction: url.searchParams.get('direction') ?? 'desc',
          facets: { all: retained.length, conversations: 0, messages: messageRemoved ? 0 : 1, media: 1, activity: 0, people: 0 },
        } })
      }
      return Response.json({ detail: 'Not found' }, { status: 404 })
    }))
  })

  it('renders heterogeneous cards, filters media, and preserves exact target semantics', async () => {
    const user = userEvent.setup()
    render(<StarredPage />, { wrapper })

    expect(await screen.findByRole('heading', { name: 'Starred' })).toBeVisible()
    expect(screen.getByText('Swipe to see Activity & People →')).toBeInTheDocument()
    expect(screen.getByText('Argument message')).toBeVisible()
    expect(screen.getByText('Unique File', { selector: '.status-badge' })).toBeVisible()
    expect(screen.getByRole('link', { name: 'Jump to message' })).toHaveAttribute(
      'href',
      '/conversations/conv-one?around=message-one#message-message-one',
    )

    await user.click(screen.getByRole('button', { name: 'Media' }))
    await waitFor(() => expect(screen.queryByText('Argument message')).not.toBeInTheDocument())
    expect(screen.getByRole('button', { name: 'Unstar sunset.jpg' })).toHaveAttribute('aria-pressed', 'true')
    const starsCall = vi.mocked(fetch).mock.calls
      .map(([input]) => new URL(String(input), 'http://localhost'))
      .find((url) => url.pathname === '/api/v1/stars' && url.searchParams.get('category') === 'media')
    expect(starsCall?.searchParams.get('direction')).toBe('desc')
  })

  it('removes an unstarred card and moves focus to the next item', async () => {
    const user = userEvent.setup()
    render(<StarredPage />, { wrapper })
    const remove = await screen.findByRole('button', { name: 'Unstar Argument message' })

    await user.click(remove)
    await waitFor(() => expect(screen.queryByText('Argument message')).not.toBeInTheDocument())
    await waitFor(() => expect(screen.getByRole('button', { name: 'Unstar sunset.jpg' })).toHaveFocus())
    expect(vi.mocked(fetch)).toHaveBeenCalledWith(
      '/api/v1/stars/message/message-one',
      expect.objectContaining({ method: 'DELETE' }),
    )
  })

  it('keeps loaded cards visible when the next page fails and offers an inline retry', async () => {
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => {
      const url = new URL(String(input), 'http://localhost')
      if (url.pathname === '/api/v1/settings') {
        return Response.json({ data: { timezone: 'America/New_York', cloud_enabled: false } })
      }
      if (url.pathname === '/api/v1/stars' && url.searchParams.has('cursor')) {
        return Response.json({ detail: 'Temporary catalog read failure' }, { status: 503 })
      }
      if (url.pathname === '/api/v1/stars') {
        return Response.json({ data: {
          items: [starredItems[0]],
          page: { limit: 1, total: 2, next_cursor: 'page-two' },
          category: 'all',
          direction: 'desc',
          facets: { all: 2, conversations: 0, messages: 1, media: 1, activity: 0, people: 0 },
        } })
      }
      return Response.json({ detail: 'Not found' }, { status: 404 })
    }))
    const user = userEvent.setup()
    render(<StarredPage />, { wrapper })

    expect(await screen.findByText('Argument message')).toBeVisible()
    await user.click(screen.getByRole('button', { name: 'Load more starred items' }))

    expect(await screen.findByRole('alert')).toHaveTextContent('Already loaded items remain available')
    expect(screen.getByText('Argument message')).toBeVisible()
    expect(screen.getByRole('button', { name: 'Try again' })).toBeEnabled()
  })
})
