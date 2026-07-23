import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { createMemoryHistory, createRouter, RouterProvider } from '@tanstack/react-router'
import { fireEvent, render, screen, within, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { routeTree } from './router'
import { fixtures } from './test/fixtures'

const SAVED_CONVERSATION_VIEWS_KEY = 'meta-archive:conversation-views:v1'

function stageTwoFetchMock() {
  const mock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = new URL(String(input), 'http://localhost')
    if (url.pathname === '/api/v1/health') return Response.json(fixtures.health)
    if (url.pathname === '/api/v1/settings') return Response.json(fixtures.settings)
    if (url.pathname === '/api/v1/conversation-corrections' && (init?.method ?? 'GET') === 'GET') {
      return Response.json({
        items: [],
        identity_revision: 'a'.repeat(64),
        page: { limit: 100, offset: 0, total: 0, next_offset: null },
        notice: 'Direct histories are provisional.',
      })
    }
    if (url.pathname === '/api/v1/conversation-corrections/split') {
      return Response.json({
        correction_id: 'correction-split',
        action: 'split',
        status: 'active',
        identity_revision: 'b'.repeat(64),
        redirect_conversation_id: 'conv-alex',
        effect: { segments: 1 },
      }, { status: 201 })
    }
    if (url.pathname === '/api/v1/conversations') return Response.json(fixtures.conversations)
    if (url.pathname === '/api/v1/conversations/conv-alex') return Response.json(fixtures.conversationDetail)
    if (url.pathname.endsWith('/messages/date-jump')) {
      return Response.json({
        requested_timestamp_ms: Number(url.searchParams.get('at_ms')),
        target_timestamp_ms: fixtures.conversationMessages.items[49]?.timestamp_ms,
        jump: {
          href: '/conversations/conv-alex?around=message-050#message-message-050',
          conversation_id: 'conv-alex',
          message_id: 'message-050',
          fragment: 'message-message-050',
        },
      })
    }
    if (url.pathname.endsWith('/message-search')) {
      const match = fixtures.conversationMessages.items[0]
      return Response.json({
        items: match ? [{ ...match, matched_fields: ['content'] }] : [],
        page: { limit: 40, total: match ? 1 : 0, next_cursor: null },
        query: url.searchParams.get('q'),
      })
    }
    if (url.pathname.endsWith('/advanced')) {
      return Response.json({
        message_id: 'message-001',
        has_sensitive_details: true,
        sensitive: { ip: '192.0.2.10', payment_info: { amount: 'fixture-only' } },
        notice: 'Advanced details can contain private network or payment data.',
      })
    }
    if (url.pathname === '/api/v1/conversations/conv-alex/messages') {
      const around = url.searchParams.get('around')
      return Response.json({
        ...fixtures.conversationMessages,
        focus: around
          ? { href: `/conversations/conv-alex?around=${around}`, message_id: around, fragment: `message-${around}` }
          : null,
      })
    }
    if (url.pathname.startsWith('/api/v1/media/blobs/')) {
      const repeated = { ...fixtures.media.items[0], occurrence_count: 2 }
      return Response.json({
        blob_sha256: repeated.blob_sha256,
        items: [repeated, { ...repeated, id: 'media-photo-2', view_key: 'occurrence:media-photo-2' }],
        page: { limit: 100, total: 2, next_cursor: null },
      })
    }
    if (url.pathname === '/api/v1/media') {
      const occurrence = url.searchParams.get('occurrence')
      if (occurrence) {
        const item = fixtures.media.items.find((candidate) => candidate.id === occurrence)
        return Response.json({
          ...fixtures.media,
          items: item ? [item] : [],
          page: { limit: 60, total: item ? 1 : 0, next_cursor: null },
          summary: {
            occurrences: item ? 1 : 0,
            unique_files: item ? 1 : 0,
            occurrence_bytes: item?.size_bytes ?? 0,
            unique_bytes: item?.size_bytes ?? 0,
          },
        })
      }
      if (url.searchParams.get('view') === 'unique') {
        const repeated = {
          ...fixtures.media.items[0],
          view_key: `blob:${'a'.repeat(64)}`,
          occurrence_count: 2,
          occurrences_href: `/api/v1/media/blobs/${'a'.repeat(64)}/occurrences`,
        }
        return Response.json({
          ...fixtures.media,
          view: 'unique',
          items: [repeated, ...fixtures.media.items.slice(1, 5)],
          page: { limit: 60, total: 5, next_cursor: null },
        })
      }
      return Response.json(fixtures.media)
    }
    if (url.pathname.startsWith('/api/v1/blobs/')) {
      return new Response(Uint8Array.from([137, 80, 78, 71]), { headers: { 'Content-Type': 'image/png' } })
    }
    return new Response(JSON.stringify({ detail: 'Not found' }), { status: 404 })
  })
  vi.stubGlobal('fetch', mock)
  return mock
}

async function renderAt(path: string) {
  const history = createMemoryHistory({ initialEntries: [path] })
  const router = createRouter({ routeTree, history })
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } })
  await router.load()
  return {
    router,
    ...render(
      <QueryClientProvider client={queryClient}>
        <RouterProvider router={router} />
      </QueryClientProvider>,
    ),
  }
}

describe('Stage 2 archive browsing', () => {
  beforeEach(() => {
    localStorage.clear()
    stageTwoFetchMock()
  })

  it.each([
    ['malformed JSON', '{malformed-json'],
    ['a null array member', '[null]'],
    ['an incomplete saved view', '[{"id":"broken","name":"Broken","view":{}}]'],
  ])('ignores saved-view storage containing %s', async (_case, storedValue) => {
    localStorage.setItem(SAVED_CONVERSATION_VIEWS_KEY, storedValue)
    await renderAt('/conversations')

    expect(await screen.findByRole('heading', { name: 'Conversations' })).toBeVisible()
    expect(screen.getByRole('option', { name: 'Saved views (0)' })).toBeInTheDocument()
  })

  it('filters, sorts, virtualizes, and saves conversation dashboard views', async () => {
    const fetchMock = stageTwoFetchMock()
    const user = userEvent.setup()
    await renderAt('/conversations')

    expect(await screen.findByRole('heading', { name: 'Conversations' })).toBeVisible()
    expect(screen.getByText('4')).toBeVisible()
    expect(screen.getByRole('link', { name: 'Alex Example' })).toHaveAttribute('href', '/conversations/conv-alex')
    expect(screen.getAllByText('2 photos, 1 videos, 1 audio files')).toHaveLength(2)
    expect(screen.getByText('Media-only thread · no message records')).toBeVisible()
    const conversationTable = screen.getByRole('table', { name: 'Conversations' })
    expect(conversationTable).toHaveAttribute('aria-rowcount', '5')
    expect(conversationTable).toHaveAttribute('data-loaded-row-count', '4')
    expect(conversationTable.querySelector('[role="row"][aria-rowindex="1"]')).not.toBeNull()
    expect(conversationTable.querySelectorAll('[role="row"]').length).toBeLessThan(20)
    const conversationViewport = conversationTable.querySelector('.conversation-table__viewport')
    expect(conversationViewport).toHaveAttribute('aria-describedby', 'conversation-results-scroll-hint')
    expect(document.getElementById('conversation-results-scroll-hint')).toHaveTextContent('Scroll inside the framed pane to browse all loaded conversations.')

    const filters = screen.getByRole('form', { name: 'Conversation filters' })
    await user.selectOptions(within(filters).getByLabelText('Type'), 'group')
    await user.type(within(filters).getByLabelText('Minimum messages'), '100')
    await user.type(within(filters).getByLabelText('Maximum messages'), '5000')
    await user.type(within(filters).getByLabelText('Minimum photos'), '10')
    await user.type(within(filters).getByLabelText('Maximum audio files'), '500')
    await user.selectOptions(within(filters).getByLabelText('Sort by'), 'photos')
    await user.selectOptions(within(filters).getByLabelText('Direction'), 'desc')
    await user.click(within(filters).getByRole('button', { name: 'Apply filters' }))

    await waitFor(() => {
      const calls = fetchMock.mock.calls.map(([input]) => new URL(String(input), 'http://localhost'))
      expect(calls.some((url) =>
        url.pathname === '/api/v1/conversations'
        && url.searchParams.get('conversation_type') === 'group'
        && url.searchParams.get('min_messages') === '100'
        && url.searchParams.get('max_messages') === '5000'
        && url.searchParams.get('min_photos') === '10'
        && url.searchParams.get('max_audio') === '500'
        && url.searchParams.get('sort') === 'photos',
      )).toBe(true)
    })

    const refreshedFilters = screen.getByRole('form', { name: 'Conversation filters' })
    await user.click(within(refreshedFilters).getByRole('button', { name: 'Save view' }))
    await user.type(within(refreshedFilters).getByLabelText('View name'), 'Big group chats')
    await user.click(within(refreshedFilters).getByRole('button', { name: 'Save current filters' }))
    expect(within(refreshedFilters).getByRole('option', { name: 'Big group chats' })).toBeInTheDocument()
    expect(JSON.parse(localStorage.getItem('meta-archive:conversation-views:v1') ?? '[]')).toHaveLength(1)
  })

  it('collapses detailed conversation filters and focuses results after mobile apply', async () => {
    const matchMedia = vi.spyOn(window, 'matchMedia').mockImplementation((query) => ({
      matches: query === '(max-width: 520px)',
      media: query,
      onchange: null,
      addListener: () => undefined,
      removeListener: () => undefined,
      addEventListener: () => undefined,
      removeEventListener: () => undefined,
      dispatchEvent: () => false,
    }))
    const user = userEvent.setup()
    await renderAt('/conversations')

    const toggle = await screen.findByRole('button', { name: 'Show detailed filters' })
    await user.click(toggle)
    await user.type(screen.getByLabelText('Conversation title'), 'Alex')
    expect(toggle).toHaveAccessibleName('Hide detailed filters')
    await user.click(screen.getByRole('button', { name: 'Apply filters' }))
    await waitFor(() => expect(screen.getByRole('heading', { name: 'Conversation ledger' })).toHaveFocus())
    const collapsedToggle = await screen.findByRole('button', { name: 'Show detailed filters (1 active)' })
    expect(collapsedToggle).toHaveAttribute('aria-expanded', 'false')
    matchMedia.mockRestore()
  })

  it('keeps the conversation DOM bounded and reveals private details only on request', async () => {
    const fetchMock = stageTwoFetchMock()
    const user = userEvent.setup()
    await renderAt('/conversations/conv-alex?around=message-002#message-message-002')

    expect(await screen.findByRole('heading', { name: 'Alex Example' })).toBeVisible()
    const feed = screen.getByRole('feed', { name: 'Conversation messages' })
    expect(Number(feed.getAttribute('data-loaded-message-count'))).toBe(140)
    expect(Number(feed.getAttribute('data-rendered-message-count'))).toBeLessThan(30)
    expect(within(feed).getAllByRole('article')[0]).toHaveAttribute('aria-setsize', '140')
    expect(screen.queryByText('192.0.2.10')).not.toBeInTheDocument()

    Object.defineProperties(feed, {
      scrollTop: { configurable: true, value: 0 },
      clientHeight: { configurable: true, value: 600 },
      scrollHeight: { configurable: true, value: 3_000 },
    })
    fireEvent.scroll(feed)
    await waitFor(() => {
      expect(fetchMock.mock.calls.some(([input]) => {
        const url = new URL(String(input), 'http://localhost')
        return url.pathname.endsWith('/messages') && url.searchParams.get('cursor') === 'older-page'
      })).toBe(true)
    })

    const reveal = screen.getAllByRole('button', { name: 'Show advanced details' })[0]
    expect(reveal).toBeDefined()
    await user.click(reveal!)
    expect(await screen.findByText('192.0.2.10')).toBeVisible()
    expect(screen.getByText('Private advanced details')).toBeVisible()

    await user.click(screen.getByRole('button', { name: 'Search this conversation' }))
    const search = screen.getByRole('searchbox', { name: /Search words in this history/i })
    await user.type(search, 'Français ABC')
    await user.click(within(screen.getByRole('region', { name: 'Search this conversation' })).getByRole('button', { name: 'Search' }))
    expect(await screen.findByText(/1 matches for/)).toBeVisible()
    expect(screen.getAllByText('Français ABC — the first searchable message.').length).toBeGreaterThan(0)

    const dateInput = screen.getByLabelText('Jump to date')
    await user.type(dateInput, '2023-01-02')
    await user.click(screen.getByRole('button', { name: /^Jump$/ }))
    await waitFor(() => expect(fetchMock.mock.calls.some(([input]) => String(input).includes('/messages/date-jump?at_ms='))).toBe(true))
  })

  it('opens the exact voice attachment and seeks its audio player from a transcript jump', async () => {
    await renderAt('/conversations/conv-alex?around=message-002&media=attachment-audio&t=1250#message-message-002')

    expect(await screen.findByRole('heading', { name: 'Alex Example' })).toBeVisible()
    const focused = document.getElementById('attachment-attachment-audio')
    expect(focused).toHaveClass('message-attachment--focused')
    await waitFor(() => expect(focused).toHaveFocus())
    const audio = focused?.querySelector('audio')
    expect(audio).not.toBeNull()
    expect(focused?.querySelector('video')).toBeNull()
    expect(audio).toHaveAttribute('data-seek-ms', '1250')
    fireEvent.loadedMetadata(audio!)
    expect(audio?.currentTime).toBe(1.25)
  })

  it('labels provisional identity and submits an audited source-segment split', async () => {
    const fetchMock = stageTwoFetchMock()
    const user = userEvent.setup()
    await renderAt('/conversations/conv-alex')

    expect(await screen.findByRole('heading', { name: 'Alex Example' })).toBeVisible()
    expect(screen.getAllByText(/provisional/i).length).toBeGreaterThan(0)
    expect(fetchMock.mock.calls.some(([input]) => (
      new URL(String(input), 'http://localhost').pathname === '/api/v1/conversation-corrections'
    ))).toBe(false)
    await user.click(screen.getByText('Direct-history identity'))
    expect(screen.getByText('This history is provisional.')).toBeVisible()
    await waitFor(() => expect(fetchMock.mock.calls.some(([input]) => (
      new URL(String(input), 'http://localhost').pathname === '/api/v1/conversation-corrections'
    ))).toBe(true))
    const archivedSegment = screen.getByRole('checkbox', { name: /archived_threads\/alex_example/i })
    await user.click(archivedSegment)
    await user.type(screen.getByLabelText('New history title'), 'Alex archived history')
    await user.type(screen.getByLabelText('Reason (optional, saved to audit)'), 'Reviewed source boundaries')
    await user.click(screen.getByRole('button', { name: 'Split selected segments' }))

    await waitFor(() => {
      const request = fetchMock.mock.calls.find(([input]) => new URL(String(input), 'http://localhost').pathname === '/api/v1/conversation-corrections/split')
      expect(request).toBeDefined()
      const payload = JSON.parse(String(request?.[1]?.body))
      expect(payload).toMatchObject({
        conversation_id: 'conv-alex',
        segment_ids: ['segment-old'],
        title: 'Alex archived history',
        identity_revision: 'a'.repeat(64),
        reason: 'Reviewed source boundaries',
      })
    })
  })

  it('switches media count and display modes while preserving placeholders and source jumps', async () => {
    const fetchMock = stageTwoFetchMock()
    const writeText = vi.fn().mockResolvedValue(undefined)
    const user = userEvent.setup()
    Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { writeText } })
    await renderAt('/media')

    expect(await screen.findByRole('heading', { name: 'Media' })).toBeVisible()
    expect(screen.getByText('missing-photo.jpg')).toBeVisible()
    expect(screen.getByText('File missing from export')).toBeVisible()
    expect(screen.getAllByText('orphan.bin').length).toBeGreaterThan(0)
    expect(screen.getAllByText('Unknown date').length).toBeGreaterThan(0)
    const gallery = screen.getByRole('list', { name: 'Media gallery' })
    expect(gallery).toHaveAttribute('data-loaded-item-count', '6')
    expect(gallery).toHaveAttribute('aria-describedby', 'media-results-scroll-hint')
    expect(document.getElementById('media-results-scroll-hint')).toHaveTextContent('Scroll inside the framed pane to browse all loaded media results.')
    expect(within(gallery).getAllByRole('listitem')[0]).toHaveAttribute('aria-posinset', '1')
    expect(within(gallery).getAllByRole('listitem')[0]).toHaveAttribute('aria-setsize', '6')
    expect(screen.getByText('Why these files have no source jump')).toBeVisible()
    expect(screen.getByText('library/media/unlinked/unknown-date/unlinked--orphan--media-unlinked.bin')).toBeVisible()
    await user.click(screen.getByRole('button', { name: 'Copy library path' }))
    expect(writeText).toHaveBeenCalledWith('library/media/unlinked/unknown-date/unlinked--orphan--media-unlinked.bin')
    expect(screen.getByRole('button', { name: 'Path copied' })).toBeVisible()
    expect(screen.getAllByRole('link', { name: 'Jump to message' })[0]).toHaveAttribute(
      'href',
      fixtures.media.items[0].jump.href,
    )
    expect(screen.getByRole('link', { name: 'Jump to activity' })).toHaveAttribute(
      'href',
      fixtures.media.items[1].jump.href,
    )
    expect(screen.getByText('Birthday album')).toBeVisible()

    await user.click(screen.getByRole('button', { name: 'Unique files' }))
    expect(await screen.findByText('2 occurrences')).toBeVisible()
    const occurrenceButtons = screen.getAllByRole('button', { name: /^Occurrences$/ })
    const occurrenceOpener = occurrenceButtons.at(-1)!
    await user.click(occurrenceOpener)
    expect(await screen.findByRole('dialog', { name: 'Every occurrence' })).toBeVisible()
    expect(screen.getByText('2 retained occurrences')).toBeVisible()
    const closeOccurrences = screen.getByRole('button', { name: 'Close occurrences' })
    await waitFor(() => expect(closeOccurrences).toHaveFocus())
    await user.keyboard('{Shift>}{Tab}{/Shift}')
    expect(screen.getByRole('dialog', { name: 'Every occurrence' })).toContainElement(document.activeElement as HTMLElement)
    await user.keyboard('{Escape}')
    await waitFor(() => expect(occurrenceOpener).toHaveFocus())

    await user.click(screen.getByRole('button', { name: 'Table' }))
    expect(screen.getByRole('table', { name: 'Media files' })).toBeVisible()
    expect(screen.getByRole('table', { name: 'Media files' })).toHaveAttribute('aria-rowcount', '6')
    expect(screen.getByRole('table', { name: 'Media files' }).querySelector('.media-table__viewport')).toHaveAttribute('aria-describedby', 'media-results-scroll-hint')
    const filters = screen.getByRole('form', { name: 'Media filters' })
    await user.type(within(filters).getByLabelText('Conversation title'), 'Alex')
    await user.selectOptions(within(filters).getByLabelText('Media type'), 'audio')
    await user.click(within(filters).getByRole('button', { name: 'Apply filters' }))
    await waitFor(() => {
      const calls = fetchMock.mock.calls.map(([input]) => new URL(String(input), 'http://localhost'))
      expect(calls.some((url) => url.pathname === '/api/v1/media' && url.searchParams.get('conversation_q') === 'Alex' && url.searchParams.get('type') === 'audio')).toBe(true)
    })
  })

  it('shows resolved media first by default while keeping all and missing-only views explicit', async () => {
    const fallbackFetch = stageTwoFetchMock()
    const missingFirst = [fixtures.media.items[4], ...fixtures.media.items.slice(0, 4), fixtures.media.items[5]]
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = new URL(String(input), 'http://localhost')
      if (url.pathname === '/api/v1/media' && !url.searchParams.has('occurrence')) {
        return Response.json({ ...fixtures.media, items: missingFirst })
      }
      return fallbackFetch(input, init)
    })
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()
    await renderAt('/media')

    const visibility = await screen.findByRole('group', { name: 'Media visibility' })
    expect(within(visibility).getByRole('button', { name: 'Resolved first' })).toHaveAttribute('aria-pressed', 'true')
    const gallery = screen.getByRole('list', { name: 'Media gallery' })
    expect(gallery).toHaveAttribute('data-loaded-item-count', '6')
    expect(within(gallery).getAllByRole('listitem')[0]).not.toHaveTextContent('missing-photo.jpg')
    expect(screen.getByText('missing-photo.jpg')).toBeVisible()

    await user.click(within(visibility).getByRole('button', { name: 'Show all' }))
    expect(within(gallery).getAllByRole('listitem')[0]).toHaveTextContent('missing-photo.jpg')
    await user.click(within(visibility).getByRole('button', { name: 'Missing only' }))
    await waitFor(() => expect(fetchMock.mock.calls.some(([input]) => {
      const url = new URL(String(input), 'http://localhost')
      return url.pathname === '/api/v1/media' && url.searchParams.get('availability') === 'missing'
    })).toBe(true))
  })

  it('resolves an unlinked enrichment jump to one exact media occurrence', async () => {
    const fetchMock = stageTwoFetchMock()
    await renderAt('/media?occurrence=media-audio&t=2200')

    expect(await screen.findByText('Showing the requested attachment')).toBeVisible()
    await waitFor(() => {
      expect(fetchMock.mock.calls.some(([input]) => {
        const url = new URL(String(input), 'http://localhost')
        return url.pathname === '/api/v1/media' && url.searchParams.get('occurrence') === 'media-audio'
      })).toBe(true)
    })
    const focused = document.getElementById('media-occurrence-media-audio')
    expect(focused).toHaveClass('media-card--focused')
    await waitFor(() => expect(focused).toHaveFocus())
    expect(focused?.querySelector('audio')).toHaveAttribute('data-seek-ms', '2200')
    expect(screen.getByRole('button', { name: 'Show all' })).toHaveAttribute('aria-pressed', 'true')
    expect(screen.getByRole('link', { name: 'Browse all media' })).toHaveAttribute('href', '/media')
  })

  it('keeps a conversation-only media jump identical to the API source href', async () => {
    const fallbackFetch = stageTwoFetchMock()
    const conversationOnly = {
      ...fixtures.media.items[0],
      id: 'media-conversation-only',
      view_key: 'occurrence:media-conversation-only',
      message_id: null,
      jump: {
        href: '/conversations/conv-alex',
        conversation_id: 'conv-alex',
      },
    }
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = new URL(String(input), 'http://localhost')
      if (url.pathname === '/api/v1/media') {
        return Response.json({
          ...fixtures.media,
          items: [conversationOnly],
          page: { limit: 60, total: 1, next_cursor: null },
          summary: {
            occurrences: 1,
            unique_files: 1,
            occurrence_bytes: conversationOnly.size_bytes,
            unique_bytes: conversationOnly.size_bytes,
          },
        })
      }
      return fallbackFetch(input, init)
    }))

    await renderAt('/media')

    expect(await screen.findByRole('link', { name: 'Open conversation' })).toHaveAttribute(
      'href',
      conversationOnly.jump.href,
    )
  })
})
