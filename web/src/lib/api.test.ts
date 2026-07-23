import { afterEach, expect, it, vi } from 'vitest'
import { api } from './api'

afterEach(() => {
  vi.unstubAllGlobals()
})

it('bounds Analytics reads until each response body settles before draining the queue', async () => {
  let active = 0
  let peak = 0
  const releaseBodies: Array<() => void> = []
  const fetchMock = vi.fn(async () => {
    active += 1
    peak = Math.max(peak, active)
    return {
      ok: true,
      status: 200,
      json: () => new Promise<Record<string, never>>((resolve) => {
        releaseBodies.push(() => {
          active -= 1
          resolve({})
        })
      }),
    } as Response
  })
  vi.stubGlobal('fetch', fetchMock)

  const reads = [
    api.analyticsNetwork(40),
    api.analyticsStorage(),
    api.analyticsCoverage(),
    api.analyticsConversations(),
    api.analyticsTopics(),
  ]

  await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2))
  expect(active).toBe(2)
  for (let completed = 0; completed < reads.length; completed += 1) {
    await vi.waitFor(() => expect(releaseBodies.length).toBeGreaterThan(0))
    releaseBodies.shift()?.()
    const expectedStarted = Math.min(reads.length, completed + 3)
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(expectedStarted))
    expect(active).toBeLessThanOrEqual(2)
  }
  await Promise.all(reads)

  expect(fetchMock).toHaveBeenCalledTimes(5)
  expect(peak).toBe(2)
})
