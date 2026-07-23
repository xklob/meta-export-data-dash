import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import type { ReactNode } from 'react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { StarButton, StarProvider } from './StarButton'

function wrapper({ children }: { children: ReactNode }) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  return <QueryClientProvider client={client}><StarProvider>{children}</StarProvider></QueryClientProvider>
}

function response(body: unknown, status = 200) {
  const payload = status >= 400 ? body : { data: body }
  return Promise.resolve(new Response(status === 204 ? null : JSON.stringify(payload), {
    status,
    headers: { 'Content-Type': 'application/json' },
  }))
}

describe('StarButton', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn())
  })

  it('optimistically stars one target everywhere and uses the canonical PUT endpoint', async () => {
    const fetchMock = vi.mocked(fetch)
    fetchMock.mockImplementation(() => response({
      star_target: { target_type: 'message', target_id: 'message-1' },
      star: { id: 'star-1', starred_at: '2026-07-23T01:00:00Z' },
      created: true,
    }))
    render(
      <>
        <StarButton target={{ target_type: 'message', target_id: 'message-1' }} label="message from Alex" />
        <StarButton target={{ target_type: 'message', target_id: 'message-1' }} label="duplicate message projection" />
      </>,
      { wrapper },
    )

    await userEvent.click(screen.getByRole('button', { name: 'Star message from Alex' }))
    expect(screen.getByRole('button', { name: 'Unstar message from Alex' })).toHaveAttribute('aria-pressed', 'true')
    expect(screen.getByRole('button', { name: 'Unstar duplicate message projection' })).toHaveAttribute('aria-pressed', 'true')
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(
      '/api/v1/stars/message/message-1',
      expect.objectContaining({ method: 'PUT' }),
    ))
    expect(await screen.findByText('Starred message from Alex.')).toBeInTheDocument()
  })

  it('rolls back a failed optimistic update and announces the API error', async () => {
    vi.mocked(fetch).mockImplementation(() => response({ detail: 'Archive is read only' }, 405))
    render(
      <StarButton target={{ target_type: 'activity', target_id: 'activity-1' }} label="activity record" />,
      { wrapper },
    )

    await userEvent.click(screen.getByRole('button', { name: 'Star activity record' }))
    expect(await screen.findByRole('button', { name: 'Star activity record' })).toHaveAttribute('aria-pressed', 'false')
    expect(await screen.findByText(/Starring activity record failed: Archive is read only/)).toBeInTheDocument()
  })

  it('removes an existing star with idempotent DELETE by target pair', async () => {
    const fetchMock = vi.mocked(fetch)
    fetchMock.mockImplementation(() => response(undefined, 204))
    render(
      <StarButton
        target={{ target_type: 'blob', target_id: 'abc123' }}
        star={{ id: 'star-2', starred_at: '2026-07-23T01:00:00Z' }}
        label="unique file sunset.jpg"
      />,
      { wrapper },
    )

    await userEvent.click(screen.getByRole('button', { name: 'Unstar unique file sunset.jpg' }))
    expect(screen.getByRole('button', { name: 'Star unique file sunset.jpg' })).toHaveAttribute('aria-pressed', 'false')
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(
      '/api/v1/stars/blob/abc123',
      expect.objectContaining({ method: 'DELETE' }),
    ))
  })
})
