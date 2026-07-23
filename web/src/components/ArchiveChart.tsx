import { BarChart, GraphChart, HeatmapChart, LineChart, PieChart } from 'echarts/charts'
import {
  CalendarComponent,
  DataZoomComponent,
  GridComponent,
  LegendComponent,
  TooltipComponent,
  VisualMapComponent,
} from 'echarts/components'
import * as echarts from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import type { EChartsOption } from 'echarts'
import { useEffect, useId, useRef } from 'react'

echarts.use([
  BarChart,
  GraphChart,
  HeatmapChart,
  LineChart,
  PieChart,
  CalendarComponent,
  DataZoomComponent,
  GridComponent,
  LegendComponent,
  TooltipComponent,
  VisualMapComponent,
  CanvasRenderer,
])

export function ArchiveChart({
  option,
  label,
  description,
  height = 320,
}: {
  option: EChartsOption
  label: string
  description: string
  height?: number
}) {
  const chartRef = useRef<HTMLDivElement>(null)
  const descriptionId = useId()

  useEffect(() => {
    const element = chartRef.current
    if (!element) return
    const chart = echarts.init(element, undefined, { renderer: 'canvas' })
    chart.setOption(option, true)
    const observer = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(() => chart.resize())
    observer?.observe(element)
    const resize = () => chart.resize()
    window.addEventListener('resize', resize)
    return () => {
      observer?.disconnect()
      window.removeEventListener('resize', resize)
      chart.dispose()
    }
  }, [option])

  return (
    <div className="archive-chart" role="img" aria-label={label} aria-describedby={descriptionId}>
      <div ref={chartRef} style={{ height }} aria-hidden="true" />
      <p id={descriptionId} className="sr-only">
        {description}
      </p>
    </div>
  )
}
