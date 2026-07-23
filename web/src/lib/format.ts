const integerFormatter = new Intl.NumberFormat('en-US', { maximumFractionDigits: 0 })
const compactFormatter = new Intl.NumberFormat('en-US', {
  notation: 'compact',
  maximumFractionDigits: 1,
})

export function formatCount(value: number | null | undefined): string {
  return value == null || !Number.isFinite(value) ? '—' : integerFormatter.format(value)
}

export function formatCompactCount(value: number | null | undefined): string {
  return value == null || !Number.isFinite(value) ? '—' : compactFormatter.format(value)
}

export function formatBytes(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return '—'
  if (value === 0) return '0 B'

  const units = ['B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB']
  const index = Math.min(Math.floor(Math.log(Math.abs(value)) / Math.log(1024)), units.length - 1)
  const amount = value / 1024 ** index
  const digits = amount >= 100 || index === 0 ? 0 : amount >= 10 ? 1 : 2
  return `${amount.toFixed(digits)} ${units[index]}`
}

export function formatDateTime(
  value: string | null | undefined,
  timeZone = 'America/New_York',
): string {
  if (!value) return 'Not recorded'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return 'Not recorded'
  return new Intl.DateTimeFormat('en-US', {
    dateStyle: 'medium',
    timeStyle: 'short',
    timeZone,
  }).format(date)
}

export function formatTimestampMs(
  value: number | null | undefined,
  timeZone = 'America/New_York',
): string {
  if (value == null || !Number.isFinite(value)) return 'Unknown date'
  return formatDateTime(new Date(value).toISOString(), timeZone)
}

export function formatDateMs(
  value: number | null | undefined,
  timeZone = 'America/New_York',
): string {
  if (value == null || !Number.isFinite(value)) return 'Unknown date'
  return new Intl.DateTimeFormat('en-US', { dateStyle: 'medium', timeZone }).format(new Date(value))
}

export function formatSeconds(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return 'Unknown duration'
  const seconds = Math.max(0, Math.floor(value))
  const minutes = Math.floor(seconds / 60)
  return minutes > 0 ? `${minutes}m ${seconds % 60}s` : `${seconds}s`
}

export function localDateToTimestamp(
  value: string,
  timeZone: string,
  endOfDay = false,
): number | undefined {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(value)
  if (!match) return undefined
  const desired = {
    year: Number(match[1]),
    month: Number(match[2]),
    day: Number(match[3]),
    hour: endOfDay ? 23 : 0,
    minute: endOfDay ? 59 : 0,
    second: endOfDay ? 59 : 0,
  }
  const desiredUtc = Date.UTC(
    desired.year,
    desired.month - 1,
    desired.day,
    desired.hour,
    desired.minute,
    desired.second,
    endOfDay ? 999 : 0,
  )
  const formatter = new Intl.DateTimeFormat('en-US', {
    timeZone,
    hour12: false,
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  })
  let result = desiredUtc
  for (let pass = 0; pass < 2; pass += 1) {
    const parts = Object.fromEntries(
      formatter.formatToParts(new Date(result)).map((part) => [part.type, part.value]),
    )
    const renderedHour = Number(parts.hour) % 24
    const renderedUtc = Date.UTC(
      Number(parts.year),
      Number(parts.month) - 1,
      Number(parts.day),
      renderedHour,
      Number(parts.minute),
      Number(parts.second),
      endOfDay ? 999 : 0,
    )
    result += desiredUtc - renderedUtc
  }
  return result
}

export function formatDuration(start: string | null | undefined, end: string | null | undefined): string {
  if (!start) return 'Not started'
  const startedAt = new Date(start).getTime()
  const finishedAt = end ? new Date(end).getTime() : Date.now()
  if (!Number.isFinite(startedAt) || !Number.isFinite(finishedAt)) return 'Unknown duration'

  const seconds = Math.max(0, Math.floor((finishedAt - startedAt) / 1000))
  if (seconds < 60) return `${seconds}s`
  const minutes = Math.floor(seconds / 60)
  if (minutes < 60) return `${minutes}m ${seconds % 60}s`
  const hours = Math.floor(minutes / 60)
  return `${hours}h ${minutes % 60}m`
}

export function sentenceCase(value: string | null | undefined): string {
  if (!value) return 'Unknown'
  return value
    .replace(/[._-]+/g, ' ')
    .replace(/\b\w/g, (letter) => letter.toUpperCase())
}

export function stringifyValue(value: unknown): string {
  if (value == null || value === '') return '—'
  if (typeof value === 'boolean') return value ? 'Yes' : 'No'
  if (typeof value === 'number') return formatCount(value)
  if (typeof value === 'object') return JSON.stringify(value)
  return String(value)
}
