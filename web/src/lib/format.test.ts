import { describe, expect, it } from 'vitest'
import { formatBytes, formatCount, formatDateTime, localDateToTimestamp, sentenceCase } from './format'

describe('archive formatters', () => {
  it('keeps absent values distinct from real zeroes', () => {
    expect(formatCount(undefined)).toBe('—')
    expect(formatCount(null)).toBe('—')
    expect(formatCount(0)).toBe('0')
  })

  it('formats byte totals for dense metrics', () => {
    expect(formatBytes(0)).toBe('0 B')
    expect(formatBytes(1024)).toBe('1.00 KiB')
    expect(formatBytes(125_829_120)).toBe('120 MiB')
  })

  it('turns machine status keys into readable labels', () => {
    expect(sentenceCase('localhost-only')).toBe('Localhost Only')
    expect(sentenceCase('archive_inventory')).toBe('Archive Inventory')
  })

  it('formats UTC instants in the configured archive timezone', () => {
    expect(formatDateTime('2026-07-20T12:00:00+00:00', 'America/New_York')).toContain('8:00 AM')
    expect(formatDateTime('2026-07-20T12:00:00+00:00', 'UTC')).toContain('12:00 PM')
  })

  it('converts archive-local date filters across standard and daylight time', () => {
    expect(new Date(localDateToTimestamp('2023-01-02', 'America/New_York')!).toISOString()).toBe(
      '2023-01-02T05:00:00.000Z',
    )
    expect(new Date(localDateToTimestamp('2023-07-02', 'America/New_York')!).toISOString()).toBe(
      '2023-07-02T04:00:00.000Z',
    )
    expect(new Date(localDateToTimestamp('2023-07-02', 'America/New_York', true)!).toISOString()).toBe(
      '2023-07-03T03:59:59.999Z',
    )
    expect(localDateToTimestamp('not-a-date', 'America/New_York')).toBeUndefined()
  })
})
