import { useMutation, useQuery } from '@tanstack/react-query'
import type { EChartsOption } from 'echarts'
import { Activity, BarChart3, CalendarRange, Database, Files, Flame, GitBranch, Layers3, MessageSquareText, TimerReset, Users } from 'lucide-react'
import { type ReactNode, useEffect, useMemo, useState } from 'react'
import { ArchiveChart } from '../components/ArchiveChart'
import { CoverageNotice } from '../components/CoverageNotice'
import { ErrorState, MetricCard, PageHeading, PageLoading, StatusBadge } from '../components/ui'
import { api } from '../lib/api'
import type { AnalyticsConflictRateResponse, AnalyticsInteractionPatternsResponse, AnalyticsInteractionSeriesResponse, AnalyticsMessagesResponse, AnalyticsNlpTrendsResponse, AnalyticsSeriesPage, NlpTrendPoint } from '../lib/contracts'
import { formatBytes, formatCompactCount, formatCount, formatSeconds, localDateToTimestamp, sentenceCase } from '../lib/format'
import { useArchiveTimezone } from '../lib/useArchiveTimezone'
import { useCursorWindowQuery } from '../lib/useCursorWindowQuery'
import './AnalyticsPage.css'

const chartColors = ['#176b67', '#7bc3b9', '#d28a22', '#326d91', '#8c70a8', '#a85a5a']
const axis = { axisLine: { lineStyle: { color: '#cbd5d0' } }, axisLabel: { color: '#60706d' } }
const MOBILE_ANALYTICS_QUERY = '(max-width: 680px)'

function useCompactAnalyticsLayout() {
  const [compact, setCompact] = useState(() => (
    Boolean(typeof window !== 'undefined' && window.matchMedia?.(MOBILE_ANALYTICS_QUERY).matches)
  ))

  useEffect(() => {
    const media = window.matchMedia?.(MOBILE_ANALYTICS_QUERY)
    if (!media) return undefined
    const update = () => setCompact(media.matches)
    update()
    media.addEventListener('change', update)
    return () => media.removeEventListener('change', update)
  }, [])

  return compact
}

function AnalyticsDrilldown({
  sectionId,
  title,
  description,
  children,
  compact,
  defaultOpen = false,
}: {
  sectionId: string
  title: string
  description: string
  children: ReactNode
  compact: boolean
  defaultOpen?: boolean
}) {
  const [expanded, setExpanded] = useState(defaultOpen)
  if (!compact) return <>{children}</>
  return (
    <details
      className="analytics-drilldown"
      data-analytics-section={sectionId}
      open={expanded}
      onToggle={(event) => setExpanded(event.currentTarget.open)}
    >
      <summary>
        <span><strong>{title}</strong><small>{description}</small></span>
        <span className="analytics-drilldown__action">{expanded ? 'Hide' : 'Explore'}</span>
      </summary>
      {expanded ? <div className="analytics-drilldown__body">{children}</div> : null}
    </details>
  )
}

function AnalyticsChartKey({
  label,
  items,
  formatValue,
}: {
  label: string
  items: Array<{ label: string; value: number }>
  formatValue: (value: number) => string
}) {
  return (
    <ul className="analytics-chart-key" aria-label={label}>
      {items.map((item, index) => (
        <li key={item.label}>
          <span
            className="analytics-chart-key__swatch"
            style={{ backgroundColor: chartColors[index % chartColors.length] }}
            aria-hidden="true"
          />
          <span>{item.label}</span>
          <strong>{formatValue(item.value)}</strong>
        </li>
      ))}
    </ul>
  )
}

function SeriesPager({
  label,
  page,
  pageNumber,
  hasPreviousPage,
  hasNextPage,
  transitionPending,
  previousPage,
  nextPage,
}: {
  label: string
  page: AnalyticsSeriesPage
  pageNumber: number
  hasPreviousPage: boolean
  hasNextPage: boolean
  transitionPending: boolean
  previousPage: () => void
  nextPage: () => void
}) {
  return (
    <div className="pagination-controls" role="navigation" aria-label={`${label} pages`} data-page-number={pageNumber}>
      <button className="button button--secondary" type="button" disabled={!hasPreviousPage || transitionPending} aria-disabled={!hasPreviousPage || transitionPending} onClick={previousPage}>Previous page</button>
      <span>
        Page {pageNumber} · {formatCount(page.returned)} periods
        {page.total == null ? (hasNextPage ? ' · more available' : '') : ` of ${formatCount(page.total)}`}
      </span>
      <button className="button button--secondary" type="button" disabled={!hasNextPage || transitionPending} aria-disabled={!hasNextPage || transitionPending} onClick={nextPage}>Next page</button>
    </div>
  )
}

function messageChartOption(data: AnalyticsMessagesResponse): EChartsOption {
  return {
    color: chartColors,
    tooltip: { trigger: 'axis' },
    legend: { top: 0, textStyle: { color: '#526164' } },
    grid: { left: 48, right: 24, top: 42, bottom: 52 },
    xAxis: { type: 'category', data: data.series.map((row) => row.period), ...axis, axisLabel: { color: '#60706d', rotate: data.series.length > 18 ? 45 : 0 } },
    yAxis: { type: 'value', ...axis },
    series: [
      { name: 'Sent', type: 'bar', stack: 'messages', data: data.series.map((row) => row.sent), emphasis: { focus: 'series' } },
      { name: 'Received', type: 'bar', stack: 'messages', data: data.series.map((row) => row.received), emphasis: { focus: 'series' } },
      { name: 'Active conversations', type: 'line', yAxisIndex: 0, smooth: true, symbolSize: 5, data: data.series.map((row) => row.active_conversations) },
    ],
    dataZoom: data.series.length > 24 ? [{ type: 'inside' }, { type: 'slider', height: 18, bottom: 4 }] : undefined,
  }
}

function responseDuration(value: number | null | undefined): string {
  return value == null ? 'Unavailable' : formatSeconds(value / 1000)
}

const responseBucketLabels: Record<string, string> = {
  under_1_minute: '<1m',
  '1_to_5_minutes': '1–5m',
  '5_to_30_minutes': '5–30m',
  '30_minutes_to_2_hours': '30m–2h',
  '2_to_24_hours': '2–24h',
  '1_to_7_days': '1–7d',
  over_7_days: '>7d',
}

function responseChartOption(data: AnalyticsInteractionPatternsResponse): EChartsOption {
  const keys = Object.keys(data.responses.owner_replied.buckets)
  return {
    color: chartColors,
    tooltip: { trigger: 'axis' },
    legend: { top: 0, textStyle: { color: '#526164' } },
    grid: { left: 48, right: 18, top: 42, bottom: 42 },
    xAxis: { type: 'category', data: keys.map((key) => responseBucketLabels[key] ?? sentenceCase(key)), ...axis },
    yAxis: { type: 'value', ...axis },
    series: [
      { name: 'You replied', type: 'bar', data: keys.map((key) => data.responses.owner_replied.buckets[key] ?? 0) },
      { name: 'Others replied', type: 'bar', data: keys.map((key) => data.responses.other_replied.buckets[key] ?? 0) },
    ],
  }
}

function interactionSeriesChartOption(data: AnalyticsInteractionSeriesResponse): EChartsOption {
  return {
    color: chartColors,
    tooltip: { trigger: 'axis' },
    legend: { top: 0, textStyle: { color: '#526164' } },
    grid: { left: 58, right: 68, top: 48, bottom: 52 },
    xAxis: { type: 'category', data: data.series.map((row) => row.period), ...axis, axisLabel: { color: '#60706d', rotate: data.series.length > 18 ? 45 : 0 } },
    yAxis: [
      { type: 'value', name: 'Calls / audio', ...axis },
      { type: 'value', name: 'Reactions / links', ...axis },
    ],
    series: [
      { name: 'Calls', type: 'bar', data: data.series.map((row) => row.calls) },
      { name: 'Audio attachments', type: 'bar', data: data.series.map((row) => row.audio_attachments) },
      { name: 'Reactions', type: 'line', yAxisIndex: 1, smooth: true, showSymbol: data.series.length < 40, data: data.series.map((row) => row.reactions) },
      { name: 'Linked URLs', type: 'line', yAxisIndex: 1, smooth: true, showSymbol: data.series.length < 40, data: data.series.map((row) => row.link_occurrences) },
    ],
    dataZoom: data.series.length > 24 ? [{ type: 'inside' }, { type: 'slider', height: 18, bottom: 4 }] : undefined,
  }
}

function domainHistoryChartOption(data: AnalyticsInteractionSeriesResponse): EChartsOption {
  const periods = data.series.map((row) => row.period)
  return {
    color: chartColors,
    tooltip: { trigger: 'axis' },
    legend: { top: 0, type: 'scroll', textStyle: { color: '#526164' } },
    grid: { left: 58, right: 20, top: 48, bottom: 52 },
    xAxis: { type: 'category', data: periods, ...axis, axisLabel: { color: '#60706d', rotate: periods.length > 18 ? 45 : 0 } },
    yAxis: { type: 'value', name: 'Links', ...axis },
    series: data.domains.map((domain) => {
      const occurrences = new Map(domain.periods.map((row) => [row.period, row.occurrences]))
      return {
        name: domain.domain,
        type: 'line' as const,
        smooth: true,
        showSymbol: periods.length < 40,
        data: periods.map((period) => occurrences.get(period) ?? 0),
      }
    }),
    dataZoom: periods.length > 24 ? [{ type: 'inside' }, { type: 'slider', height: 18, bottom: 4 }] : undefined,
  }
}

const directionLabels: Record<string, string> = { all: 'All sources', owner: 'You', other: 'Others', activity: 'Activity', audio: 'Audio', attachment: 'Attachments' }

function nlpTooltip(params: unknown): string {
  const rows = Array.isArray(params) ? params as Array<{ axisValueLabel?: string; marker?: string; seriesName?: string; data?: { value?: number | null; sample_count?: number; mean_confidence?: number } }> : []
  const heading = rows[0]?.axisValueLabel ?? ''
  return [heading, ...rows.map((row) => {
    const value = row.data?.value
    const score = value == null ? 'Unavailable' : Number(value).toFixed(3)
    const sample = formatCount(row.data?.sample_count)
    const confidence = row.data?.mean_confidence == null ? '—' : `${(row.data.mean_confidence * 100).toFixed(1)}%`
    return `${row.marker ?? ''}${row.seriesName ?? 'Estimate'}: ${score} · n=${sample} · mean confidence ${confidence}`
  })].join('<br/>')
}

function scoreTrendOption(data: AnalyticsNlpTrendsResponse, task: 'sentiment' | 'toxicity'): EChartsOption {
  const rows = data.series[task]
  const periods = [...new Set(rows.map((row) => row.period).filter((value): value is string => Boolean(value)))]
  const directions = [...new Set(rows.map((row) => row.direction))]
  return {
    color: chartColors,
    tooltip: { trigger: 'axis', formatter: nlpTooltip }, legend: { top: 0, textStyle: { color: '#526164' } },
    grid: { left: 50, right: 20, top: 42, bottom: 52 },
    xAxis: { type: 'category', data: periods, ...axis, axisLabel: { color: '#60706d', rotate: periods.length > 18 ? 45 : 0 } },
    yAxis: { type: 'value', min: task === 'sentiment' ? -1 : 0, max: 1, ...axis },
    series: directions.map((direction) => {
      const values = new Map(rows.filter((row) => row.direction === direction).map((row) => [row.period, row]))
      return { name: directionLabels[direction] ?? sentenceCase(direction), type: 'line' as const, showSymbol: periods.length < 40, smooth: true, connectNulls: false, data: periods.map((period) => {
        const row = values.get(period)
        return row ? { value: row.score, sample_count: row.sample_count, mean_confidence: row.mean_confidence } : null
      }) }
    }),
    dataZoom: periods.length > 24 ? [{ type: 'inside' }, { type: 'slider', height: 18, bottom: 4 }] : undefined,
  }
}

function sentimentHourOption(rows: NlpTrendPoint[]): EChartsOption {
  const allSources = rows.filter((row) => row.direction === 'all')
  const combined = allSources.length ? allSources : rows
  const byHour = new Map(combined.map((row) => [row.hour, row]))
  const hours = Array.from({ length: 24 }, (_, hour) => String(hour).padStart(2, '0'))
  return {
    color: chartColors,
    tooltip: { trigger: 'axis', formatter: nlpTooltip },
    grid: { left: 50, right: 18, top: 20, bottom: 38 },
    xAxis: { type: 'category', data: hours.map((hour) => `${hour}:00`), ...axis },
    yAxis: { type: 'value', min: -1, max: 1, ...axis },
    series: [{ name: 'Expected sentiment', type: 'line', smooth: true, areaStyle: { opacity: 0.12 }, data: hours.map((hour) => {
      const row = byHour.get(hour)
      return row ? { value: row.score, sample_count: row.sample_count, mean_confidence: row.mean_confidence } : null
    }) }],
  }
}

function emotionTrendOption(data: AnalyticsNlpTrendsResponse): EChartsOption {
  const allSources = data.series.emotion.filter((row) => row.direction === 'all')
  const rows = allSources.length ? allSources : data.series.emotion
  const periods = rows.map((row) => row.period).filter((value): value is string => Boolean(value))
  const labelTotals = new Map<string, number>()
  for (const row of rows) for (const [label, probability] of Object.entries(row.mean_probabilities)) labelTotals.set(label, (labelTotals.get(label) ?? 0) + probability * row.sample_count)
  const labels = [...labelTotals].sort((left, right) => right[1] - left[1]).slice(0, 6).map(([label]) => label)
  return {
    color: chartColors,
    tooltip: { trigger: 'axis' }, legend: { top: 0, type: 'scroll', textStyle: { color: '#526164' } },
    grid: { left: 50, right: 18, top: 52, bottom: 52 },
    xAxis: { type: 'category', data: periods, ...axis, axisLabel: { color: '#60706d', rotate: periods.length > 18 ? 45 : 0 } },
    yAxis: { type: 'value', min: 0, max: 1, ...axis },
    series: labels.map((label) => ({ name: sentenceCase(label), type: 'line' as const, smooth: true, showSymbol: periods.length < 40, data: rows.map((row) => row.mean_probabilities[label] ?? null) })),
    dataZoom: periods.length > 24 ? [{ type: 'inside' }, { type: 'slider', height: 18, bottom: 4 }] : undefined,
  }
}

function conflictRateOption(data: AnalyticsConflictRateResponse): EChartsOption {
  return {
    color: ['#a85a5a', '#cbd5d0'],
    tooltip: {
      trigger: 'axis',
      formatter: (params: unknown) => {
        const rows = Array.isArray(params) ? params as Array<{ axisValueLabel?: string; marker?: string; seriesName?: string; data?: number }> : []
        const point = data.series.find((item) => item.period === rows[0]?.axisValueLabel)
        if (!point) return ''
        const warning = point.low_sample_warning ? '<br/>Low sample count: interpret cautiously' : ''
        return `${point.period}<br/>${rows[0]?.marker ?? ''}Estimated direct-conflict windows: ${(Number(point.direct_conflict_window_rate ?? 0) * 100).toFixed(1)}%<br/>${rows[1]?.marker ?? ''}Samples: ${formatCount(point.sample_count)}${warning}`
      },
    },
    legend: { top: 0, textStyle: { color: '#526164' } },
    grid: { left: 56, right: 64, top: 42, bottom: 52 },
    xAxis: { type: 'category', data: data.series.map((row) => row.period), ...axis, axisLabel: { color: '#60706d', rotate: data.series.length > 18 ? 45 : 0 } },
    yAxis: [
      { type: 'value', name: 'Rate', min: 0, max: 1, ...axis, axisLabel: { formatter: (value: number) => `${Math.round(value * 100)}%`, color: '#60706d' } },
      { type: 'value', name: 'Samples', min: 0, ...axis },
    ],
    series: [
      { name: 'Estimated direct-conflict-window rate', type: 'line', smooth: true, showSymbol: data.series.length < 40, data: data.series.map((row) => row.direct_conflict_window_rate ?? null), z: 2 },
      { name: 'Classified windows', type: 'bar', yAxisIndex: 1, data: data.series.map((row) => row.sample_count), itemStyle: { opacity: 0.45 }, z: 1 },
    ],
    dataZoom: data.series.length > 24 ? [{ type: 'inside' }, { type: 'slider', height: 18, bottom: 4 }] : undefined,
  }
}

export function AnalyticsPage() {
  const compactLayout = useCompactAnalyticsLayout()
  const archiveTimezone = useArchiveTimezone()
  const [timezone, setTimezone] = useState(archiveTimezone)
  const [bucket, setBucket] = useState<AnalyticsMessagesResponse['bucket']>('month')
  const [nlpBucket, setNlpBucket] = useState<AnalyticsNlpTrendsResponse['bucket']>('month')
  const [nlpDirection, setNlpDirection] = useState('all')
  const [nlpFrom, setNlpFrom] = useState('')
  const [nlpTo, setNlpTo] = useState('')
  const [conflictBucket, setConflictBucket] = useState<AnalyticsConflictRateResponse['bucket']>('month')
  const [conflictFrom, setConflictFrom] = useState('')
  const [conflictTo, setConflictTo] = useState('')
  const [calendarYear, setCalendarYear] = useState('')
  const overview = useQuery({ queryKey: ['analytics-overview', timezone], queryFn: () => api.analyticsOverview(timezone) })
  const messages = useCursorWindowQuery({ queryKey: ['analytics-messages', bucket, timezone], queryFn: (cursor) => api.analyticsMessages({ bucket, timezone, limit: 120, cursor }) })
  const patterns = useCursorWindowQuery({ queryKey: ['analytics-patterns', timezone], queryFn: (cursor) => api.analyticsPatterns(timezone, { limit: 120, cursor }) })
  const interactions = useCursorWindowQuery({ queryKey: ['analytics-interactions', bucket, timezone], queryFn: (cursor) => api.analyticsInteractionPatterns(bucket, timezone, { limit: 120, cursor }) })
  const interactionSeries = useCursorWindowQuery({ queryKey: ['analytics-interaction-series', bucket, timezone], queryFn: (cursor) => api.analyticsInteractionSeries(bucket, timezone, { limit: 120, cursor }) })
  const nlpTrends = useCursorWindowQuery({
    queryKey: ['analytics-nlp-trends', nlpBucket, timezone, nlpDirection, nlpFrom, nlpTo],
    queryFn: (cursor) => api.analyticsNlpTrends(nlpBucket, timezone, {
      direction: nlpDirection,
      fromMs: localDateToTimestamp(nlpFrom, timezone),
      toMs: localDateToTimestamp(nlpTo, timezone, true),
      limit: 24,
      cursor,
    }),
  })
  const conflictRate = useCursorWindowQuery({
    queryKey: ['analytics-conflict-rate', conflictBucket, timezone, conflictFrom, conflictTo],
    queryFn: (cursor) => api.analyticsConflictRate(conflictBucket, timezone, {
      fromMs: localDateToTimestamp(conflictFrom, timezone),
      toMs: localDateToTimestamp(conflictTo, timezone, true),
      limit: 24,
      cursor,
    }),
  })
  const network = useQuery({ queryKey: ['analytics-network'], queryFn: () => api.analyticsNetwork(40) })
  const topics = useQuery({
    queryKey: ['analytics-topics'],
    queryFn: api.analyticsTopics,
    refetchInterval: (query) => query.state.data?.status === 'building' ? 1_500 : false,
  })
  const buildTopics = useMutation({
    mutationFn: () => api.buildAnalyticsTopics(),
    onSuccess: () => void topics.refetch(),
  })
  const attachments = useCursorWindowQuery({
    queryKey: ['analytics-attachments', bucket, timezone],
    queryFn: (cursor) => api.analyticsAttachments(bucket, timezone, { limit: 120, cursor }),
  })
  const conversations = useQuery({ queryKey: ['analytics-conversations'], queryFn: api.analyticsConversations })
  const storage = useQuery({ queryKey: ['analytics-storage'], queryFn: api.analyticsStorage })
  const coverage = useQuery({ queryKey: ['analytics-coverage'], queryFn: api.analyticsCoverage })
  const queries = [overview, messages, patterns, interactions, interactionSeries, nlpTrends, conflictRate, network, topics, attachments, conversations, storage, coverage]
  // `isPending` alone excludes cached and placeholder-backed refetches. Those
  // reads can still be active or queued behind the two-request Analytics
  // limiter, so the route is busy until every enabled query is fully idle.
  const isBusy = queries.some((query) => query.isPending || query.fetchStatus !== 'idle')
  const failed = queries.find((query) => query.isError)

  const messagesOption = useMemo(() => messages.data ? messageChartOption(messages.data) : {}, [messages.data])
  const hourOption = useMemo<EChartsOption>(() => ({
    color: chartColors,
    tooltip: { trigger: 'axis' }, legend: { top: 0 },
    grid: { left: 45, right: 18, top: 40, bottom: 35 },
    xAxis: { type: 'category', data: patterns.data?.hours.map((row) => `${String(row.hour).padStart(2, '0')}:00`) ?? [], ...axis },
    yAxis: { type: 'value', ...axis },
    series: [
      { name: 'Messages', type: 'bar', data: patterns.data?.hours.map((row) => row.messages) ?? [] },
      { name: 'Facebook activity', type: 'bar', data: patterns.data?.hours.map((row) => row.activities) ?? [] },
    ],
  }), [patterns.data])
  const weekdayOption = useMemo<EChartsOption>(() => ({
    color: chartColors,
    tooltip: { trigger: 'axis' }, legend: { top: 0 },
    grid: { left: 45, right: 18, top: 40, bottom: 35 },
    xAxis: { type: 'category', data: patterns.data?.weekdays.map((row) => row.label.slice(0, 3)) ?? [], ...axis },
    yAxis: { type: 'value', ...axis },
    series: [
      { name: 'Messages', type: 'line', areaStyle: { opacity: 0.12 }, smooth: true, data: patterns.data?.weekdays.map((row) => row.messages) ?? [] },
      { name: 'Activity', type: 'line', areaStyle: { opacity: 0.1 }, smooth: true, data: patterns.data?.weekdays.map((row) => row.activities) ?? [] },
    ],
  }), [patterns.data])
  const calendarYears = useMemo(() => [...new Set(
    (patterns.data?.heatmap ?? [])
      .map((row) => row.date.slice(0, 4))
      .filter((year) => /^\d{4}$/.test(year)),
  )].sort(), [patterns.data])
  const selectedCalendarYear = calendarYears.includes(calendarYear)
    ? calendarYear
    : calendarYears.at(-1) ?? ''
  const selectedCalendarRows = useMemo(
    () => (patterns.data?.heatmap ?? []).filter((row) => row.date.startsWith(`${selectedCalendarYear}-`)),
    [patterns.data, selectedCalendarYear],
  )
  const heatmapOption = useMemo<EChartsOption>(() => {
    const rows = selectedCalendarRows
    const range = selectedCalendarYear || '2000'
    return {
      tooltip: { formatter: (params: unknown) => {
        const value = (params as { value?: [string, number] }).value
        return value ? `${value[0]}: ${formatCount(value[1])} retained records` : ''
      } },
      visualMap: { min: 0, max: Math.max(1, ...rows.map((row) => row.total)), orient: 'horizontal', left: 'center', top: 0, calculable: true, inRange: { color: ['#edf5f2', '#86c7bd', '#176b67'] } },
      calendar: { top: 78, left: 36, right: 18, bottom: 12, range, cellSize: ['auto', 15], itemStyle: { borderColor: '#fff', borderWidth: 2 }, yearLabel: { show: false }, dayLabel: { color: '#60706d' }, monthLabel: { color: '#60706d' } },
      series: [{ type: 'heatmap', coordinateSystem: 'calendar', data: rows.map((row) => [row.date, row.total]) }],
    }
  }, [selectedCalendarRows, selectedCalendarYear])
  const responseOption = useMemo<EChartsOption>(() => interactions.data ? responseChartOption(interactions.data) : {}, [interactions.data])
  const participantOption = useMemo<EChartsOption>(() => ({
    color: chartColors,
    tooltip: { trigger: 'axis' }, legend: { top: 0, textStyle: { color: '#526164' } },
    grid: { left: 48, right: 20, top: 42, bottom: 50 },
    xAxis: { type: 'category', data: interactions.data?.active_participants.map((row) => row.period) ?? [], ...axis, axisLabel: { color: '#60706d', rotate: (interactions.data?.active_participants.length ?? 0) > 18 ? 45 : 0 } },
    yAxis: { type: 'value', ...axis },
    series: [
      { name: 'Participant labels', type: 'line', smooth: true, data: interactions.data?.active_participants.map((row) => row.participants) ?? [] },
      { name: 'Conversations', type: 'line', smooth: true, data: interactions.data?.active_participants.map((row) => row.conversations) ?? [] },
    ],
    dataZoom: (interactions.data?.active_participants.length ?? 0) > 24 ? [{ type: 'inside' }, { type: 'slider', height: 18, bottom: 4 }] : undefined,
  }), [interactions.data])
  const initiationOption = useMemo<EChartsOption>(() => ({
    color: chartColors,
    tooltip: { trigger: 'item' },
    series: [{ name: 'First retained message', type: 'pie', radius: ['46%', '72%'], data: [{ name: 'You', value: interactions.data?.initiations.owner ?? 0 }, { name: 'Others', value: interactions.data?.initiations.other ?? 0 }], label: { formatter: '{b}: {c}' } }],
  }), [interactions.data])
  const interactionSeriesOption = useMemo<EChartsOption>(() => interactionSeries.data ? interactionSeriesChartOption(interactionSeries.data) : {}, [interactionSeries.data])
  const domainHistoryOption = useMemo<EChartsOption>(() => interactionSeries.data ? domainHistoryChartOption(interactionSeries.data) : {}, [interactionSeries.data])
  const sentimentOption = useMemo<EChartsOption>(() => nlpTrends.data ? scoreTrendOption(nlpTrends.data, 'sentiment') : {}, [nlpTrends.data])
  const sentimentByHourOption = useMemo<EChartsOption>(() => sentimentHourOption(nlpTrends.data?.hour_of_day.sentiment ?? []), [nlpTrends.data])
  const emotionOption = useMemo<EChartsOption>(() => nlpTrends.data ? emotionTrendOption(nlpTrends.data) : {}, [nlpTrends.data])
  const toxicityOption = useMemo<EChartsOption>(() => nlpTrends.data ? scoreTrendOption(nlpTrends.data, 'toxicity') : {}, [nlpTrends.data])
  const conflictRateChart = useMemo<EChartsOption>(() => conflictRate.data ? conflictRateOption(conflictRate.data) : {}, [conflictRate.data])
  const networkOption = useMemo<EChartsOption>(() => ({
    color: chartColors,
    tooltip: { trigger: 'item' },
    legend: [{ data: ['Conversation', 'Participant label'], textStyle: { color: '#526164' } }],
    series: [{
      name: 'Archive network', type: 'graph', layout: 'force', roam: true,
      categories: [{ name: 'Conversation' }, { name: 'Participant label' }],
      data: network.data?.nodes.map((node) => ({ id: node.id, name: node.label, value: node.value, category: node.kind === 'conversation' ? 0 : 1, symbolSize: Math.min(54, 10 + Math.log10(node.value + 1) * 7), itemStyle: node.is_owner ? { borderColor: '#d28a22', borderWidth: 3 } : undefined })) ?? [],
      links: network.data?.edges.map((edge) => ({ source: edge.source, target: edge.target, value: edge.value })) ?? [],
      force: { repulsion: 125, edgeLength: [45, 115], gravity: 0.08 },
      lineStyle: { color: 'source', opacity: 0.25, curveness: 0.08 },
      label: { show: false }, emphasis: { focus: 'adjacency', label: { show: true } },
    }],
  }), [network.data])
  const topicsOption = useMemo<EChartsOption>(() => ({
    color: chartColors,
    tooltip: { trigger: 'axis', valueFormatter: (value) => `${(Number(value) * 100).toFixed(1)}%` },
    grid: { left: 150, right: 26, top: 15, bottom: 30 },
    xAxis: { type: 'value', max: 1, ...axis, axisLabel: { formatter: (value: number) => `${Math.round(value * 100)}%`, color: '#60706d' } },
    yAxis: { type: 'category', inverse: true, data: topics.data?.result?.clusters.map((cluster) => cluster.terms.slice(0, 2).join(' · ') || `Topic ${cluster.ordinal}`) ?? [], ...axis },
    series: [{ name: 'Share of sample', type: 'bar', data: topics.data?.result?.clusters.map((cluster) => cluster.sample_share) ?? [], label: { show: true, position: 'right', formatter: (params) => `${(Number(params.value ?? 0) * 100).toFixed(1)}%` } }],
  }), [topics.data])
  const attachmentOption = useMemo<EChartsOption>(() => ({
    color: chartColors,
    tooltip: { trigger: 'axis' },
    legend: { top: 0, textStyle: { color: '#526164' } },
    grid: { left: 55, right: 62, top: 42, bottom: 38 },
    xAxis: { type: 'category', data: attachments.data?.types.map((row) => sentenceCase(row.type)) ?? [], ...axis },
    yAxis: [{ type: 'value', name: 'Files', ...axis }, { type: 'value', name: 'Bytes', ...axis, axisLabel: { formatter: (value: number) => formatBytes(value), color: '#60706d' } }],
    series: [
      { name: 'Occurrences', type: 'bar', data: attachments.data?.types.map((row) => row.occurrences) ?? [] },
      { name: 'Unique files', type: 'bar', data: attachments.data?.types.map((row) => row.unique_files) ?? [] },
      { name: 'Occurrence bytes', type: 'line', yAxisIndex: 1, smooth: true, data: attachments.data?.types.map((row) => row.occurrence_bytes) ?? [] },
    ],
  }), [attachments.data])
  const attachmentAvailabilityOption = useMemo<EChartsOption>(() => ({
    color: chartColors,
    tooltip: { trigger: 'axis' },
    grid: { left: 55, right: 18, top: 20, bottom: 38 },
    xAxis: { type: 'category', data: attachments.data?.availability.map((row) => sentenceCase(row.availability)) ?? [], ...axis },
    yAxis: { type: 'value', ...axis },
    series: [{ name: 'Occurrences', type: 'bar', data: attachments.data?.availability.map((row) => row.occurrences) ?? [] }],
  }), [attachments.data])
  const attachmentTrendOption = useMemo<EChartsOption>(() => {
    const periods = attachments.data?.trends?.map((row) => row.period) ?? []
    const types = attachments.data?.types.map((row) => row.type) ?? []
    return {
      color: chartColors,
      tooltip: { trigger: 'axis' },
      legend: { top: 0, type: 'scroll', textStyle: { color: '#526164' } },
      grid: { left: 58, right: 72, top: 48, bottom: 52 },
      xAxis: { type: 'category', data: periods, ...axis, axisLabel: { color: '#60706d', rotate: periods.length > 18 ? 45 : 0 } },
      yAxis: [
        { type: 'value', name: 'Files', ...axis },
        { type: 'value', name: 'Bytes', ...axis, axisLabel: { formatter: (value: number) => formatBytes(value), color: '#60706d' } },
      ],
      series: [
        ...types.map((type) => ({
          name: sentenceCase(type),
          type: 'bar' as const,
          stack: 'attachment-occurrences',
          data: attachments.data?.trends?.map((row) => row.by_type[type]?.occurrences ?? 0) ?? [],
        })),
        {
          name: 'Occurrence bytes',
          type: 'line' as const,
          yAxisIndex: 1,
          smooth: true,
          showSymbol: periods.length < 40,
          data: attachments.data?.trends?.map((row) => row.occurrence_bytes) ?? [],
        },
      ],
      dataZoom: periods.length > 24 ? [{ type: 'inside' }, { type: 'slider', height: 18, bottom: 4 }] : undefined,
    }
  }, [attachments.data])
  const storageOption = useMemo<EChartsOption>(() => ({
    color: chartColors,
    tooltip: { trigger: 'item', valueFormatter: (value) => formatBytes(Number(value)) },
    legend: { show: false },
    series: [{
      name: 'Unique blob bytes', type: 'pie', radius: ['38%', '64%'], center: ['50%', '50%'],
      label: { show: false }, labelLine: { show: false },
      data: storage.data?.by_media_class.map((row) => ({ name: sentenceCase(row.class), value: row.bytes })) ?? [],
    }],
  }), [storage.data])
  const conversationOption = useMemo<EChartsOption>(() => ({
    color: chartColors,
    tooltip: { trigger: 'item' },
    legend: { show: false },
    series: [{
      name: 'Conversations', type: 'pie', radius: ['38%', '64%'], center: ['50%', '50%'],
      data: Object.entries(conversations.data?.by_type ?? {}).map(([name, value]) => ({ name: sentenceCase(name), value })),
      label: { show: false }, labelLine: { show: false },
    }],
  }), [conversations.data])

  return (
    <div
      className="page-stack stage4-page analytics-page"
      aria-busy={isBusy}
      data-analytics-route-state={isBusy ? 'loading' : 'settled'}
    >
      <PageHeading
        eyebrow="Patterns & provenance"
        title="Archive analytics"
        description="Explore source-backed messaging, activity, attachment, conversation, and storage trends without ranking people or relationships."
        actions={(
          <div className="analytics-controls">
            <label><span>Bucket</span><select value={bucket} onChange={(event) => setBucket(event.target.value as AnalyticsMessagesResponse['bucket'])}><option value="day">Day</option><option value="week">Week</option><option value="month">Month</option><option value="year">Year</option></select></label>
            <label><span>Timezone</span><input value={timezone} onChange={(event) => setTimezone(event.target.value)} /></label>
          </div>
        )}
      />
      {isBusy ? <PageLoading label="Computing archive analytics" /> : null}
      {failed ? <ErrorState error={failed.error} onRetry={() => void Promise.all(queries.map((query) => query.refetch()))} /> : null}

      <AnalyticsDrilldown
        compact={compactLayout}
        defaultOpen
        sectionId="snapshot"
        title="Archive snapshot"
        description="Headline totals and source coverage"
      >
        {overview.data ? (
          <>
            <CoverageNotice coverage={overview.data.coverage} />
            <section className="metric-grid metric-grid--four" aria-label="Analytics overview">
              <MetricCard icon={MessageSquareText} label="Messages" value={formatCompactCount(overview.data.messages.total)} detail={`${formatCount(overview.data.messages.sent)} sent · ${formatCount(overview.data.messages.received)} received`} />
              <MetricCard icon={Activity} label="Average per day" value={formatCount(overview.data.messages.average_per_calendar_day)} detail={`${formatCount(overview.data.messages.calendar_days)} local calendar days in ${overview.data.timezone}`} />
              <MetricCard icon={Files} label="Attachments" value={formatCompactCount(overview.data.attachments.occurrences)} detail={formatBytes(overview.data.attachments.occurrence_bytes)} />
              <MetricCard icon={TimerReset} label="Call time" value={formatSeconds(overview.data.interactions.call_duration_seconds)} detail={`${formatCount(overview.data.interactions.calls)} calls · ${formatCount(overview.data.interactions.missed_calls)} missed`} />
            </section>
            <section className="metric-grid metric-grid--four" aria-label="Interaction and activity totals">
              <MetricCard icon={MessageSquareText} label="Message reactions" value={formatCompactCount(overview.data.interactions.message_reactions)} detail={`${formatCount(Object.keys(overview.data.interactions.reaction_types).length)} retained reaction types`} />
              <MetricCard icon={Files} label="Audio attachments" value={formatCompactCount(overview.data.interactions.audio_attachments)} detail="Attachment occurrences" />
              <MetricCard icon={Activity} label="Facebook activity records" value={formatCompactCount(overview.data.activity_records)} detail="Active source-backed records" />
              <MetricCard icon={GitBranch} label="Linked domains" value={formatCount(overview.data.link_domains.length)} detail="Most frequent domains listed below" />
            </section>
          </>
        ) : null}
      </AnalyticsDrilldown>

      <AnalyticsDrilldown
        compact={compactLayout}
        sectionId="messages"
        title="Messages and daily activity"
        description="Volume, streaks, calendar, hour, and weekday charts"
      >

      {messages.data ? (
        <section className="panel analytics-chart-panel">
          <div className="section-header"><div><h2>Message volume over time</h2><p>{formatCount(messages.data.summary.messages)} messages across {formatCount(messages.data.summary.observed_periods)} observed {bucket} buckets; average {formatCount(messages.data.summary.average_per_observed_period)}.</p></div></div>
          <ArchiveChart option={messagesOption} label={`Sent, received, and active conversations by ${bucket}`} description={`A ${bucket} series containing ${messages.data.series.length} observed periods.`} height={390} />
          <SeriesPager {...messages} label="Message volume periods" page={messages.data.page} />
        </section>
      ) : null}

      {patterns.data ? (
        <>
          <section className="metric-grid metric-grid--four" aria-label="Activity streaks">
            <MetricCard icon={Flame} label="Longest streak" value={`${formatCount(patterns.data.streaks.longest_days)} days`} detail="Consecutive local calendar days with retained activity" />
            <MetricCard icon={Activity} label="Latest streak" value={`${formatCount(patterns.data.streaks.latest_days)} days`} detail={`Ending ${patterns.data.streaks.last_active_date ?? 'on an unknown date'}`} />
            <MetricCard icon={CalendarRange} label="Active days" value={formatCount(patterns.data.streaks.active_days)} detail={`Since ${patterns.data.streaks.first_active_date ?? 'an unknown date'}`} />
            <MetricCard icon={BarChart3} label="Peak day" value={formatCount(patterns.data.peak_days[0]?.total)} detail={patterns.data.peak_days[0]?.date ?? 'No dated activity'} />
          </section>
          <section className="panel analytics-chart-panel">
            <div className="section-header">
              <div><h2>Archive activity calendar</h2><p>Messages and dated Facebook activity by local calendar day in {patterns.data.timezone}. {formatCount(patterns.data.summary?.unknown_date_activity_records)} archive-wide Facebook activity records have unknown dates and are excluded from time buckets.</p></div>
              <div className="analytics-nlp-controls analytics-calendar-controls" role="group" aria-label="Activity calendar year">
                <label>
                  <span>Calendar year</span>
                  <select value={selectedCalendarYear} onChange={(event) => setCalendarYear(event.target.value)} disabled={calendarYears.length < 2}>
                    {calendarYears.length ? calendarYears.map((year) => <option value={year} key={year}>{year}</option>) : <option value="">No dated activity</option>}
                  </select>
                </label>
              </div>
            </div>
            <ArchiveChart option={heatmapOption} label={`Calendar heatmap of retained archive activity in ${selectedCalendarYear || 'the selected year'}`} description={`${formatCount(selectedCalendarRows.length)} active calendar days in ${selectedCalendarYear || 'the selected year'}; archive-wide longest consecutive streak ${formatCount(patterns.data.streaks.longest_days)} days.`} height={310} />
            <SeriesPager {...patterns} label="Activity calendar periods" page={patterns.data.page} />
            <p className="analytics-topics__note">{patterns.data.range_provenance}</p>
          </section>
          <div className="analytics-grid analytics-grid--two">
            <section className="panel analytics-chart-panel"><div className="section-header"><div><h2>Activity by hour</h2><p>Local hour in {patterns.data.timezone}.</p></div></div><ArchiveChart option={hourOption} label="Messages and Facebook activity by hour" description="Twenty-four hourly buckets for source-backed messages and activity." /></section>
            <section className="panel analytics-chart-panel"><div className="section-header"><div><h2>Weekday rhythm</h2><p>Messages and activity by local weekday.</p></div></div><ArchiveChart option={weekdayOption} label="Messages and Facebook activity by weekday" description="Seven weekday buckets for messages and activity." /></section>
          </div>
        </>
      ) : null}

      </AnalyticsDrilldown>

      <AnalyticsDrilldown
        compact={compactLayout}
        sectionId="interactions"
        title="Conversation interactions"
        description="Reply timing, initiations, calls, reactions, audio, and links"
      >

      {interactions.data ? (
        <>
          <section className="metric-grid metric-grid--four" aria-label="Conversation timing summary">
            <MetricCard icon={TimerReset} label="Median time for you to reply" value={responseDuration(interactions.data.responses.owner_replied.median_ms)} detail={`${formatCount(interactions.data.responses.owner_replied.count)} direction changes`} />
            <MetricCard icon={TimerReset} label="Median time for others to reply" value={responseDuration(interactions.data.responses.other_replied.median_ms)} detail={`${formatCount(interactions.data.responses.other_replied.count)} direction changes`} />
            <MetricCard icon={MessageSquareText} label="Conversations you initiated" value={formatCount(interactions.data.initiations.owner)} detail={interactions.data.initiations.definition} />
            <MetricCard icon={Users} label="Conversations others initiated" value={formatCount(interactions.data.initiations.other)} detail="First retained message direction; missing history can affect this." />
          </section>
          <div className="analytics-grid analytics-grid--two">
            <section className="panel analytics-chart-panel"><div className="section-header"><div><h2>Response-time distribution</h2><p>{interactions.data.responses.definition}</p></div></div><ArchiveChart option={responseOption} label="Response time distributions by speaker direction" description={`You replied in ${formatCount(interactions.data.responses.owner_replied.count)} measured direction changes; others replied in ${formatCount(interactions.data.responses.other_replied.count)}.`} /></section>
            <section className="panel analytics-chart-panel"><div className="section-header"><div><h2>Active participant labels</h2><p>Distinct exported sender labels and active conversations per {interactions.data.bucket}.</p></div></div><ArchiveChart option={participantOption} label={`Active participant labels and conversations by ${interactions.data.bucket}`} description={`${interactions.data.active_participants.length} observed periods. Participant labels are not authoritative Facebook IDs.`} /><SeriesPager {...interactions} label="Active participant periods" page={interactions.data.page} /></section>
          </div>
          <section className="panel analytics-chart-panel analytics-chart-panel--compact"><div className="section-header"><div><h2>Conversation initiations</h2><p>{interactions.data.ranking_policy}</p></div></div><ArchiveChart option={initiationOption} label="First retained message direction by conversation" description={`${formatCount(interactions.data.initiations.owner)} conversations begin with an owner message and ${formatCount(interactions.data.initiations.other)} with another sender in the selected scope.`} height={260} /></section>
        </>
      ) : null}

      {interactionSeries.data ? (
        <>
          <section className="metric-grid metric-grid--four" aria-label="Interaction event totals">
            <MetricCard icon={TimerReset} label="Calls over time" value={formatCount(interactionSeries.data.summary.calls)} detail={`${formatSeconds(interactionSeries.data.summary.call_duration_seconds)} retained call time`} />
            <MetricCard icon={Activity} label="Reactions over time" value={formatCount(interactionSeries.data.summary.reactions)} detail={`${formatCompactCount(interactionSeries.data.summary.reaction_message_timestamp_fallback)} used message time`} />
            <MetricCard icon={Files} label="Dated audio over time" value={formatCount(interactionSeries.data.summary.audio_attachments)} detail={`${formatCount(interactionSeries.data.summary.audio_unknown_date)} unknown-date excluded`} />
            <MetricCard icon={GitBranch} label="Linked URLs over time" value={formatCount(interactionSeries.data.summary.link_occurrences)} detail={`${formatCount(interactionSeries.data.summary.domain_count)} retained domains`} />
          </section>
          <section className="panel analytics-chart-panel">
            <div className="section-header">
              <div>
                <h2>Interaction events over time</h2>
                <p>Calls, reactions, dated audio attachments, and linked URLs grouped by {interactionSeries.data.bucket} in {interactionSeries.data.timezone}. Unknown-date audio is reported separately and never assigned to a chart bucket.</p>
              </div>
            </div>
            <ArchiveChart
              option={interactionSeriesOption}
              label={`Calls, reactions, audio attachments, and linked URLs by ${interactionSeries.data.bucket}`}
              description={`${formatCount(interactionSeries.data.summary.calls)} calls, ${formatCount(interactionSeries.data.summary.reactions)} reactions, ${formatCount(interactionSeries.data.summary.audio_attachments)} dated audio attachments, and ${formatCount(interactionSeries.data.summary.link_occurrences)} linked URLs across ${formatCount(interactionSeries.data.series.length)} observed periods.`}
              height={390}
            />
            <SeriesPager {...interactionSeries} label="Interaction event periods" page={interactionSeries.data.page} />
            {interactionSeries.data.domains.length ? (
              <ArchiveChart
                option={domainHistoryOption}
                label={`Top linked-domain occurrences by ${interactionSeries.data.bucket}`}
                description={interactionSeries.data.domains.map((row) => `${row.domain}: ${formatCount(row.occurrences)}`).join(', ')}
                height={330}
              />
            ) : <p className="analytics-topics__note">No linked domains have dated occurrences in this scope.</p>}
            <details className="analytics-nlp-evidence">
              <summary>How interaction timestamps were assigned</summary>
              <ul className="analytics-warning-list">
                <li>{interactionSeries.data.provenance.calls}</li>
                <li>{interactionSeries.data.provenance.reactions}</li>
                <li>{interactionSeries.data.provenance.audio}</li>
                <li>{interactionSeries.data.provenance.links}</li>
              </ul>
            </details>
          </section>
        </>
      ) : null}

      </AnalyticsDrilldown>

      <AnalyticsDrilldown
        compact={compactLayout}
        sectionId="language"
        title="Language-model signals"
        description="Sentiment, emotion, toxicity, and calibrated time-only conflict trends"
      >

      {nlpTrends.data ? (
        <section className="panel analytics-nlp-trends">
          <div className="section-header">
            <div><h2>Local language-model trends</h2><p>{formatCount(nlpTrends.data.source_sample_count)} versioned annotation samples, grouped by {nlpBucket} in {nlpTrends.data.timezone}. Every chart shows model estimates—not facts about a person or relationship.</p></div>
            <div className="analytics-nlp-controls" role="group" aria-label="Language-model trend filters">
              <label><span>Time bucket</span><select value={nlpBucket} onChange={(event) => setNlpBucket(event.target.value as AnalyticsNlpTrendsResponse['bucket'])}><option value="hour">Hour</option><option value="day">Day</option><option value="week">Week</option><option value="month">Month</option><option value="year">Year</option></select></label>
              <label><span>Speaker/source</span><select value={nlpDirection} onChange={(event) => setNlpDirection(event.target.value)}>{Object.entries(directionLabels).map(([value, label]) => <option value={value} key={value}>{label}</option>)}</select></label>
              <label><span>From</span><input type="date" value={nlpFrom} onChange={(event) => setNlpFrom(event.target.value)} /></label>
              <label><span>Through</span><input type="date" value={nlpTo} onChange={(event) => setNlpTo(event.target.value)} /></label>
            </div>
          </div>
          {nlpTrends.data.source_sample_count ? (
            <>
              <div className="analytics-grid analytics-grid--two">
                <div className="analytics-chart-panel"><h3>Sentiment over time</h3><p>Expected score: mean positive probability minus mean negative probability. Hover points for sample size and mean model confidence.</p><ArchiveChart option={sentimentOption} label={`Model-estimated sentiment by ${nlpBucket} and speaker direction`} description={`Expected sentiment across ${nlpTrends.data.series.sentiment.length} observed period/direction buckets, with sample counts and mean confidence in each point.`} /></div>
                <div className="analytics-chart-panel"><h3>Sentiment by local hour</h3><p>All retained annotated sources averaged by hour of day in {nlpTrends.data.timezone}.</p><ArchiveChart option={sentimentByHourOption} label="Model-estimated sentiment by local hour of day" description="Twenty-four local-hour buckets with missing hours left blank." /></div>
                <div className="analytics-chart-panel"><h3>Emotion probabilities</h3><p>Top mean model-label probabilities over time. Labels describe text patterns, not internal emotional state.</p><ArchiveChart option={emotionOption} label={`Top model-estimated emotion probabilities by ${nlpBucket}`} description={`${nlpTrends.data.series.emotion.length} observed period/direction buckets with full probability vectors.`} /></div>
                <div className="analytics-chart-panel"><h3>Toxicity dimensions</h3><p>Highest mean modeled toxicity-dimension score. Context, dialect, quotation, and banter can produce false positives.</p><ArchiveChart option={toxicityOption} label={`Model-estimated toxicity dimension by ${nlpBucket} and speaker direction`} description={`${nlpTrends.data.series.toxicity.length} observed period/direction buckets; no people or relationships are ranked.`} /></div>
              </div>
              <details className="analytics-nlp-evidence">
                <summary>Inspect sentiment sample sizes and confidence</summary>
                <div className="data-table-wrap"><table className="data-table"><thead><tr><th>Period</th><th>Source</th><th>Samples</th><th>Mean confidence</th><th>Expected score</th></tr></thead><tbody>{nlpTrends.data.series.sentiment.slice(-100).map((row, index) => <tr key={`${row.period ?? row.hour}-${row.direction}-${index}`}><td>{row.period ?? (row.hour != null ? `${row.hour}:00` : 'Unknown')}</td><td>{directionLabels[row.direction] ?? sentenceCase(row.direction)}</td><td className="number-cell">{formatCount(row.sample_count)}</td><td className="number-cell">{(row.mean_confidence * 100).toFixed(1)}%</td><td className="number-cell">{row.score == null ? '—' : row.score.toFixed(3)}</td></tr>)}</tbody></table></div>
              </details>
            </>
          ) : (
            <p className="analytics-topics__empty" role="status">No versioned language-model annotations match these filters. Clear or widen the date and source filters to continue.</p>
          )}
          <p className="analytics-topics__note">
            Rollup provenance: {nlpTrends.data.rollup_provenance.range_filter_semantics}; {nlpTrends.data.rollup_provenance.local_bucket_label}.
            {nlpTrends.data.rollup_provenance.partial_boundary_hours_included ? ' Selected dates include overlapping whole UTC hours, so partial boundary hours are approximate.' : ''}
          </p>
          <p className="analytics-topics__note">Hour-of-day estimates use the {formatCount(nlpTrends.data.page_sample_count)} source samples on this visible period page; archive/filter-scope totals above remain complete.</p>
          <SeriesPager {...nlpTrends} label="Language-model trend periods" page={nlpTrends.data.page} />
          <ul className="analytics-warning-list">{nlpTrends.data.warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul>
        </section>
      ) : null}

      {conflictRate.data ? (
        <section className="panel analytics-nlp-trends analytics-conflict-rate">
          <div className="section-header">
            <div><h2>Estimated direct-conflict-window rate</h2><p>Time-only, calibration-gated classifications across the complete active message-window corpus. This never ranks people, conversations, or relationships.</p></div>
            <StatusBadge status={conflictRate.data.status} />
          </div>
          <div className="analytics-nlp-controls" role="group" aria-label="Conflict rate filters">
            <label><span>Conflict time bucket</span><select value={conflictBucket} onChange={(event) => setConflictBucket(event.target.value as AnalyticsConflictRateResponse['bucket'])}><option value="hour">Hour</option><option value="day">Day</option><option value="week">Week</option><option value="month">Month</option><option value="year">Year</option></select></label>
            <label><span>From</span><input type="date" value={conflictFrom} onChange={(event) => setConflictFrom(event.target.value)} /></label>
            <label><span>Through</span><input type="date" value={conflictTo} onChange={(event) => setConflictTo(event.target.value)} /></label>
          </div>
          {conflictRate.data.available ? (
            <>
              <section className="metric-grid metric-grid--four" aria-label="Conflict aggregate summary">
                <MetricCard icon={MessageSquareText} label="Classified windows" value={formatCount(conflictRate.data.summary.sample_count)} detail="Overlapping text windows in the selected period" />
                <MetricCard icon={BarChart3} label="Estimated direct-conflict rate" value={conflictRate.data.summary.direct_conflict_window_rate == null ? 'Unavailable' : `${(conflictRate.data.summary.direct_conflict_window_rate * 100).toFixed(1)}%`} detail={`${formatCount(conflictRate.data.summary.direct_conflict_window_count)} top-labelled windows`} tone="warning" />
              </section>
              <ArchiveChart option={conflictRateChart} label={`Model-estimated direct-conflict-window rate by ${conflictBucket}`} description={`${formatCount(conflictRate.data.summary.sample_count)} classified windows across ${formatCount(conflictRate.data.series.length)} observed time buckets; bars show sample sizes.`} height={390} />
              <details className="analytics-nlp-evidence">
                <summary>Inspect conflict-rate samples</summary>
                <div className="data-table-wrap"><table className="data-table"><thead><tr><th>Period</th><th>Samples</th><th>Direct-conflict label</th><th>Estimated rate</th><th>Note</th></tr></thead><tbody>{conflictRate.data.series.slice(-200).map((row) => <tr key={row.period}><td>{row.period}</td><td className="number-cell">{formatCount(row.sample_count)}</td><td className="number-cell">{formatCount(row.direct_conflict_window_count)}</td><td className="number-cell">{row.direct_conflict_window_rate == null ? '—' : `${(row.direct_conflict_window_rate * 100).toFixed(1)}%`}</td><td>{row.low_sample_warning ? 'Low sample count' : '—'}</td></tr>)}</tbody></table></div>
              </details>
            </>
          ) : (
            <div className="analytics-topics__empty"><p>{conflictRate.data.reason ?? 'Conflict aggregates are not available.'}</p><a className="button button--secondary" href="/enrichment">Review calibration and build status</a></div>
          )}
          {typeof conflictRate.data.provenance?.range_filter_semantics === 'string' ? (
            <p className="analytics-topics__note">
              Rollup provenance: {conflictRate.data.provenance.range_filter_semantics}.
              {conflictFrom || conflictTo ? ' Selected dates can include partial boundary hours.' : ''}
            </p>
          ) : null}
          <SeriesPager {...conflictRate} label="Conflict-rate periods" page={conflictRate.data.page} />
          <ul className="analytics-warning-list">{conflictRate.data.warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul>
        </section>
      ) : null}

      </AnalyticsDrilldown>

      <AnalyticsDrilldown
        compact={compactLayout}
        sectionId="themes"
        title="Themes and archive network"
        description="Conversation-participant structure and unsupervised topic clusters"
      >

      {network.data ? (
        <section className="panel analytics-chart-panel">
          <div className="section-header"><div><h2>Conversation and participant-label network</h2><p>{network.data.scope} {network.data.caveat}</p></div><GitBranch size={19} aria-hidden="true" /></div>
          <ArchiveChart option={networkOption} label="Bipartite graph of conversations and participant labels" description={`${formatCount(network.data.conversation_count)} conversation nodes, ${formatCount(network.data.participant_label_count)} provisional participant-label nodes, and ${formatCount(network.data.edges.length)} source-membership edges.`} height={520} />
          <div className="analytics-network-links" role="group" aria-label="Largest conversations represented in the graph">
            {network.data.nodes.filter((node) => node.kind === 'conversation' && node.href).slice(0, 12).map((node) => <a key={node.id} href={node.href}>{node.label}<span>{formatCount(node.value)} messages</span></a>)}
          </div>
        </section>
      ) : null}

      {topics.data ? (
        <section className="panel analytics-chart-panel analytics-topics">
          <div className="section-header">
            <div><h2>Topic clusters</h2><p>Unsupervised themes from a bounded sample of the active semantic text index. Labels are descriptive model estimates, never people or relationship rankings.</p></div>
            <div className="analytics-topics__actions"><StatusBadge status={topics.data.status} /><button className="button button--secondary" type="button" disabled={!topics.data.semantic_index || topics.data.status === 'building' || buildTopics.isPending} onClick={() => buildTopics.mutate()}><Layers3 size={15} aria-hidden="true" />{topics.data.ready ? 'Rebuild topics' : topics.data.status === 'building' ? 'Building…' : 'Build topics'}</button></div>
          </div>
          {buildTopics.isError ? <p className="inline-error" role="alert">{buildTopics.error.message}</p> : null}
          {topics.data.status === 'building' && topics.data.latest_job ? <p className="analytics-topics__progress" role="status">{topics.data.latest_job.stage ?? 'Clustering'} · {formatCount(topics.data.latest_job.progress_current)} of {formatCount(topics.data.latest_job.progress_total)}</p> : null}
          {!topics.data.ready ? <p className="analytics-topics__empty">{topics.data.reason ?? topics.data.latest_job?.error ?? 'Topic clusters are not ready.'}</p> : null}
          {topics.data.result ? (
            <>
              <ArchiveChart option={topicsOption} label="Estimated topic share in the sampled semantic windows" description={`${formatCount(topics.data.result.clusters.length)} clusters from ${formatCount(topics.data.result.sample_count)} of ${formatCount(topics.data.result.corpus_window_count)} searchable windows.`} height={Math.max(300, topics.data.result.clusters.length * 38)} />
              <p className="analytics-topics__note">{topics.data.result.sampling} {topics.data.result.warning}</p>
              <div className="analytics-topic-cards">
                {topics.data.result.clusters.map((cluster) => (
                  <article key={cluster.id}>
                    <div><span>Topic {cluster.ordinal}</span><strong>{cluster.terms.slice(0, 3).join(' · ') || 'Unlabelled theme'}</strong><small>{formatCount(cluster.sample_windows)} sampled windows · {(cluster.sample_share * 100).toFixed(1)}%</small></div>
                    <ul>
                      {cluster.representatives.map((item) => <li key={item.id}><p>{item.excerpt}</p>{item.jump?.href ? <a href={item.jump.href}>Jump to source</a> : <span>Source no longer retained</span>}</li>)}
                    </ul>
                  </article>
                ))}
              </div>
            </>
          ) : null}
        </section>
      ) : null}

      </AnalyticsDrilldown>

      <AnalyticsDrilldown
        compact={compactLayout}
        sectionId="attachments"
        title="Attachments and storage"
        description="File trends, availability, media mix, storage, and conversation types"
      >

      {attachments.data?.trends?.length ? (
        <section className="panel analytics-chart-panel">
          <div className="section-header">
            <div>
              <h2>Attachment trends over time</h2>
              <p>
                Dated attachment occurrences and bytes by {attachments.data.bucket} in {attachments.data.timezone}.{' '}
                {formatCount(attachments.data.summary.unknown_date_occurrences)} unknown-date occurrences {attachments.data.summary.unknown_date_included_in_totals ? 'remain in totals but not the chart' : 'are archive-wide and excluded from date-filtered totals'}.
              </p>
            </div>
          </div>
          <ArchiveChart
            option={attachmentTrendOption}
            label={`Attachment occurrences and bytes by ${attachments.data.bucket}`}
            description={`${formatCount(attachments.data.summary.dated_occurrences)} dated occurrences across ${formatCount(attachments.data.trends.length)} observed periods. Missing rate ${attachments.data.summary.missing_rate == null ? 'unavailable' : `${(attachments.data.summary.missing_rate * 100).toFixed(1)}%`}; unlinked rate ${attachments.data.summary.unlinked_rate == null ? 'unavailable' : `${(attachments.data.summary.unlinked_rate * 100).toFixed(1)}%`}.`}
            height={390}
          />
          <SeriesPager {...attachments} label="Attachment trend periods" page={attachments.data.page} />
          <p className="analytics-topics__note">{attachments.data.provenance}</p>
        </section>
      ) : null}

      <div className="analytics-grid analytics-grid--three">
        {attachments.data ? <section className="panel analytics-chart-panel"><div className="section-header"><div><h2>Attachment mix and availability</h2><p>{attachments.data.count_semantics}</p></div></div><ArchiveChart option={attachmentOption} label="Attachment occurrences, unique files, and bytes by type" description={attachments.data.types.map((row) => `${row.type}: ${row.occurrences} occurrences, ${row.unique_files} unique files, ${formatBytes(row.occurrence_bytes)}`).join(', ')} /><ArchiveChart option={attachmentAvailabilityOption} label="Attachment occurrences by availability" description={attachments.data.availability.map((row) => `${row.availability}: ${row.occurrences} occurrences and ${formatBytes(row.bytes)}`).join(', ')} height={260} /></section> : null}
        {storage.data ? <section className="panel analytics-chart-panel"><div className="section-header"><div><h2>Storage composition</h2><p>{formatBytes(storage.data.unique_blobs.bytes)} in {formatCount(storage.data.unique_blobs.files)} unique blobs.</p></div></div><ArchiveChart option={storageOption} label="Unique archive bytes by media class" description={storage.data.by_media_class.map((row) => `${row.class}: ${formatBytes(row.bytes)}`).join(', ')} /><AnalyticsChartKey label="Storage composition key" items={storage.data.by_media_class.map((row) => ({ label: sentenceCase(row.class), value: row.bytes }))} formatValue={formatBytes} /></section> : null}
        {conversations.data ? <section className="panel analytics-chart-panel"><div className="section-header"><div><h2>Conversation types</h2><p>{formatCount(conversations.data.conversation_count)} conversations; mean active span {conversations.data.longevity_days.mean == null ? 'unknown' : `${conversations.data.longevity_days.mean.toFixed(1)} days`}.</p></div></div><ArchiveChart option={conversationOption} label="Conversation count by type" description={Object.entries(conversations.data.by_type).map(([key, value]) => `${key}: ${value}`).join(', ')} /><AnalyticsChartKey label="Conversation types key" items={Object.entries(conversations.data.by_type).map(([name, value]) => ({ label: sentenceCase(name), value }))} formatValue={formatCount} /></section> : null}
      </div>

      </AnalyticsDrilldown>

      <AnalyticsDrilldown
        compact={compactLayout}
        sectionId="coverage"
        title="Coverage and provenance"
        description="Derived-signal availability, linked domains, source coverage, and deduplication"
      >

      {overview.data ? (
        <section className="panel nlp-availability">
          <div className="section-header"><div><h2>Derived language signals</h2><p>Sentiment, emotion, toxicity, and conflict only appear after versioned local annotation jobs finish.</p></div></div>
          <div>
            {Object.entries(overview.data.derived_nlp).map(([name, status]) => (
              <article key={name}><div><strong>{sentenceCase(name)}</strong><StatusBadge status={status.status} /></div><p>{status.reason}</p>{status.relationship_ranking_enabled === false ? <small>Relationship ranking is disabled.</small> : null}</article>
            ))}
          </div>
        </section>
      ) : null}

      <div className="analytics-grid analytics-grid--two">
        {overview.data ? (
          <section className="panel compact-list-panel"><div className="section-header"><div><h2>Most linked domains</h2><p>Domains found in retained message text and structured metadata.</p></div></div><ol aria-label="Most linked domains">{overview.data.link_domains.slice(0, 12).map((row) => <li key={row.domain}><span>{row.domain}</span><strong>{formatCount(row.occurrences)}</strong></li>)}</ol></section>
        ) : null}
        {coverage.data ? (
          <section className="panel compact-list-panel">
            <div className="section-header"><div><h2>Analytics coverage</h2><p>Unavailable means the export cannot prove zero history.</p></div><BarChart3 size={19} /></div>
            <ul aria-label="Analytics coverage statuses">{coverage.data.items.map((item) => <li key={item.key ?? item.label}><span>{item.label}</span><StatusBadge status={item.status} /></li>)}</ul>
            {coverage.data.history ? (
              <>
                <p className="analytics-topics__note">{coverage.data.history.description}</p>
                <p className="analytics-topics__note">
                  {coverage.data.history.historical_comparison_available
                    ? `${formatCount(coverage.data.history.snapshots.length)} observed export snapshots are available for comparison.`
                    : 'This is one observed export snapshot. Prior coverage snapshots were not captured, so no historical trend is inferred.'}
                  {coverage.data.history.observed_at ? ` Observed ${coverage.data.history.observed_at.slice(0, 10)}.` : ''}
                </p>
                {coverage.data.history.observed_at_basis ? <p className="analytics-topics__note">{coverage.data.history.observed_at_basis}</p> : null}
              </>
            ) : null}
          </section>
        ) : null}
      </div>

      {storage.data ? (
        <section className="panel analytics-provenance">
          <Database size={19} aria-hidden="true" />
          <div><strong>Deduplication impact</strong><p>{formatCount(storage.data.source_occurrences.files)} source file occurrences become {formatCount(storage.data.unique_blobs.files)} unique blobs, avoiding {formatBytes(storage.data.deduplication.duplicate_bytes)} of duplicate storage.</p></div>
        </section>
      ) : null}
      </AnalyticsDrilldown>
    </div>
  )
}
