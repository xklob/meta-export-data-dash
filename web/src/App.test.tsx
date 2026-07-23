import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { createMemoryHistory, createRouter, RouterProvider } from '@tanstack/react-router'
import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { routeTree } from './router'
import { stageForPathname } from './components/AppShell'
import { fixtures } from './test/fixtures'

type RouteName = keyof typeof fixtures

const endpointFixture: Record<string, RouteName> = {
  '/api/v1/health': 'health',
  '/api/v1/overview': 'overview',
  '/api/v1/coverage': 'coverage',
  '/api/v1/integrity': 'integrity',
  '/api/v1/jobs': 'jobs',
  '/api/v1/settings': 'settings',
}

function installFetchMock(overrides: Partial<Record<string, unknown>> = {}) {
  const mock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = new URL(String(input), 'http://localhost')
    const override = overrides[url.pathname]
    if (override instanceof Response) return override

    const fixtureName = endpointFixture[url.pathname]
    if (!fixtureName) return new Response(JSON.stringify({ detail: 'Not found' }), { status: 404 })

    if (url.pathname === '/api/v1/settings' && init?.method === 'PATCH') {
      const patch = JSON.parse(String(init.body)) as { timezone: string; cloud_enabled: boolean }
      return Response.json({ ...fixtures.settings, ...patch })
    }

    return Response.json(override ?? fixtures[fixtureName])
  })
  vi.stubGlobal('fetch', mock)
  return mock
}

async function renderAt(path: string) {
  const history = createMemoryHistory({ initialEntries: [path] })
  const router = createRouter({ routeTree, history })
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false, staleTime: Infinity } },
  })
  await router.load()
  return {
    router,
    queryClient,
    ...render(
      <QueryClientProvider client={queryClient}>
        <RouterProvider router={router} />
      </QueryClientProvider>,
    ),
  }
}

describe('Stage 1 archive application', () => {
  beforeEach(() => installFetchMock())

  it('renders source-backed home totals and permanent privacy state', async () => {
    await renderAt('/')

    expect(await screen.findByRole('heading', { name: 'Archive control center' })).toBeVisible()
    expect(screen.getByText('240')).toBeVisible()
    expect(screen.getByText('Local archive only')).toBeVisible()
    expect(screen.getByText('Cloud off')).toBeVisible()
    expect(screen.getByRole('link', { name: 'Conversations' })).toHaveAttribute('href', '/conversations')
    expect(screen.getByText('Hashing source files')).toBeVisible()
  })

  it('filters coverage without interpreting missing categories as zero', async () => {
    const user = userEvent.setup()
    await renderAt('/coverage')

    const main = await screen.findByRole('main')
    expect(await within(main).findByRole('heading', { name: 'Data coverage' })).toBeVisible()
    expect(within(main).getByText('Messages')).toBeVisible()

    await user.click(within(main).getByRole('button', { name: 'Unavailable' }))
    expect(within(main).getByText('Friends')).toBeVisible()
    expect(within(main).queryByText('Messages')).not.toBeInTheDocument()
    expect(within(main).getAllByText('—')).toHaveLength(2)
  })

  it('shows integrity checks, unresolved findings, and opt-in resolved findings', async () => {
    const fetchMock = installFetchMock()
    const user = userEvent.setup()
    await renderAt('/integrity')

    expect(await screen.findByText('Five message threads begin after segment one.')).toBeVisible()
    expect(screen.queryByText('Five duplicate reaction rows were identified.')).not.toBeInTheDocument()

    await user.click(screen.getByRole('checkbox', { name: 'Show resolved' }))
    expect(screen.getByText('Five duplicate reaction rows were identified.')).toBeVisible()
    expect(screen.getAllByText('240', { selector: 'dd' })).toHaveLength(2)
    const integrityCalls = fetchMock.mock.calls
      .map(([input]) => new URL(String(input), 'http://localhost'))
      .filter((url) => url.pathname === '/api/v1/integrity')
    expect(integrityCalls.some((url) => url.searchParams.get('unresolved_only') === 'true')).toBe(true)
    expect(integrityCalls.some((url) => url.searchParams.get('unresolved_only') === 'false')).toBe(true)
  })

  it('surfaces API failures with a retry action', async () => {
    installFetchMock({
      '/api/v1/jobs': new Response(JSON.stringify({ detail: 'Catalog database is unavailable.' }), {
        status: 503,
        headers: { 'Content-Type': 'application/json' },
      }),
    })
    await renderAt('/jobs')

    expect(await screen.findByRole('alert')).toHaveTextContent('Catalog database is unavailable.')
    expect(screen.getByRole('button', { name: 'Try again' })).toBeEnabled()
  })

  it('persists only the allowed settings and updates the cloud indicator', async () => {
    const fetchMock = installFetchMock()
    document.cookie = 'meta_archive_csrf=test-csrf-token; path=/'
    const user = userEvent.setup()
    await renderAt('/settings')

    const cloudToggle = await screen.findByRole('checkbox', { name: 'Allow cloud-assisted features' })
    await user.click(cloudToggle)
    expect(screen.getByText('Cloud access will be enabled after you save.')).toBeVisible()
    await user.click(screen.getByRole('button', { name: 'Save settings' }))

    expect(await screen.findByText('Cloud on')).toBeVisible()
    const patchCall = fetchMock.mock.calls.find(([, init]) => init?.method === 'PATCH')
    expect(patchCall).toBeDefined()
    expect(JSON.parse(String(patchCall?.[1]?.body))).toEqual({
      timezone: 'America/New_York',
      cloud_enabled: true,
    })
    expect(patchCall?.[1]?.headers).toMatchObject({ 'X-CSRF-Token': 'test-csrf-token' })
    expect(screen.getByRole('button', { name: /Approval required/ })).toBeDisabled()
  })

  it('announces SPA route changes, updates the title, and moves focus to main content', async () => {
    const user = userEvent.setup()
    await renderAt('/')

    expect(document.title).toBe('Home · Meta Archive Explorer')
    await user.click(screen.getByRole('link', { name: 'Data Coverage' }))

    expect(await screen.findByRole('heading', { name: 'Data coverage' })).toBeVisible()
    await waitFor(() => expect(screen.getByRole('main')).toHaveFocus())
    expect(document.title).toBe('Data Coverage · Meta Archive Explorer')
    expect(screen.getByTestId('route-announcement')).toHaveTextContent('Data Coverage page loaded')
  })
})

describe('application stage labels', () => {
  it.each([
    ['/', 1],
    ['/integrity', 1],
    ['/conversations', 2],
    ['/conversations/conv-alex', 2],
    ['/media', 2],
    ['/search', 3],
    ['/enrichment', 3],
    ['/people', 4],
    ['/events', 4],
    ['/activity', 4],
    ['/timeline', 4],
    ['/support', 4],
    ['/analytics', 4],
    ['/deletion', 5],
  ])('maps %s to Stage %i', (path, stage) => {
    expect(stageForPathname(path)).toBe(stage)
  })
})
