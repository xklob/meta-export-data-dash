import { useQuery } from '@tanstack/react-query'
import {
  type ColumnDef,
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  type SortingState,
  useReactTable,
} from '@tanstack/react-table'
import { ArrowDown, ArrowUp, ArrowUpDown, Database, Search } from 'lucide-react'
import { useMemo, useState } from 'react'
import { api } from '../lib/api'
import type { CoverageItem } from '../lib/contracts'
import { formatCount, formatDateTime } from '../lib/format'
import { useArchiveTimezone } from '../lib/useArchiveTimezone'
import { EmptyState, ErrorState, PageHeading, PageLoading, StatusBadge } from '../components/ui'

type CoverageFilter = 'all' | 'available' | 'partial' | 'unavailable'

const filterOptions: Array<{ value: CoverageFilter; label: string }> = [
  { value: 'all', label: 'All categories' },
  { value: 'available', label: 'Available' },
  { value: 'partial', label: 'Partial' },
  { value: 'unavailable', label: 'Unavailable' },
]

const coverageColumnLabels: Record<string, string> = {
  label: 'Category',
  status: 'Coverage',
  raw_count: 'Raw records',
  logical_count: 'Logical records',
  updated_at: 'Last checked',
}

function SortLabel({ label, sorted }: { label: string; sorted: false | 'asc' | 'desc' }) {
  return (
    <span className="sort-label">
      {label}
      {sorted === 'asc' ? <ArrowUp size={14} /> : sorted === 'desc' ? <ArrowDown size={14} /> : <ArrowUpDown size={14} />}
    </span>
  )
}

export function CoveragePage() {
  const coverage = useQuery({ queryKey: ['coverage'], queryFn: api.coverage })
  const timezone = useArchiveTimezone()
  const [statusFilter, setStatusFilter] = useState<CoverageFilter>('all')
  const [search, setSearch] = useState('')
  const [sorting, setSorting] = useState<SortingState>([{ id: 'status', desc: false }])

  const items = useMemo(() => coverage.data?.items ?? [], [coverage.data?.items])
  const filtered = useMemo(() => {
    const query = search.trim().toLowerCase()
    return items.filter((item) => {
      const matchesStatus = statusFilter === 'all' || item.status.toLowerCase() === statusFilter
      const matchesSearch =
        !query ||
        item.label.toLowerCase().includes(query) ||
        item.key.toLowerCase().includes(query) ||
        item.description?.toLowerCase().includes(query)
      return matchesStatus && matchesSearch
    })
  }, [items, search, statusFilter])

  const columns = useMemo<ColumnDef<CoverageItem>[]>(
    () => [
      {
        accessorKey: 'label',
        header: ({ column }) => (
          <button type="button" onClick={column.getToggleSortingHandler()}>
            <SortLabel label="Category" sorted={column.getIsSorted()} />
          </button>
        ),
        cell: ({ row }) => (
          <div className="table-primary">
            <strong>{row.original.label}</strong>
            <span>{row.original.description || 'No coverage note was supplied.'}</span>
          </div>
        ),
      },
      {
        accessorKey: 'status',
        header: ({ column }) => (
          <button type="button" onClick={column.getToggleSortingHandler()}>
            <SortLabel label="Coverage" sorted={column.getIsSorted()} />
          </button>
        ),
        cell: ({ getValue }) => <StatusBadge status={getValue<string>()} />,
      },
      {
        accessorKey: 'raw_count',
        header: ({ column }) => (
          <button type="button" onClick={column.getToggleSortingHandler()}>
            <SortLabel label="Raw records" sorted={column.getIsSorted()} />
          </button>
        ),
        cell: ({ row }) => <span className="number-cell">{formatCount(row.original.raw_count)}</span>,
        sortUndefined: 'last',
      },
      {
        accessorKey: 'logical_count',
        header: ({ column }) => (
          <button type="button" onClick={column.getToggleSortingHandler()}>
            <SortLabel label="Logical records" sorted={column.getIsSorted()} />
          </button>
        ),
        cell: ({ row }) => <span className="number-cell">{formatCount(row.original.logical_count)}</span>,
        sortUndefined: 'last',
      },
      {
        accessorKey: 'updated_at',
        header: 'Last checked',
        cell: ({ row }) => <span className="muted-cell">{formatDateTime(row.original.updated_at, timezone)}</span>,
      },
    ],
    [timezone],
  )

  const table = useReactTable({
    data: filtered,
    columns,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
  })

  if (coverage.isLoading) return <PageLoading label="Auditing data coverage" />
  if (coverage.isError) return <ErrorState error={coverage.error} onRetry={() => coverage.refetch()} />

  const counts = {
    available: items.filter((item) => item.status.toLowerCase() === 'available').length,
    partial: items.filter((item) => item.status.toLowerCase() === 'partial').length,
    unavailable: items.filter((item) => item.status.toLowerCase() === 'unavailable').length,
  }

  return (
    <div className="page-stack">
      <PageHeading
        eyebrow="Source truth"
        title="Data coverage"
        description="See what this export can support before drawing conclusions. Unavailable means the category was not included—not that the history was empty."
      />

      <section className="coverage-overview" aria-label="Coverage summary">
        <div>
          <span className="coverage-overview__dot coverage-overview__dot--available" aria-hidden="true" />
          <strong>{counts.available}</strong>
          <span>Available</span>
        </div>
        <div>
          <span className="coverage-overview__dot coverage-overview__dot--partial" aria-hidden="true" />
          <strong>{counts.partial}</strong>
          <span>Partial</span>
        </div>
        <div>
          <span className="coverage-overview__dot coverage-overview__dot--unavailable" aria-hidden="true" />
          <strong>{counts.unavailable}</strong>
          <span>Unavailable</span>
        </div>
        <p>
          <Database size={17} aria-hidden="true" />
          Counts reflect source records and normalized logical records separately.
        </p>
      </section>

      <section className="panel table-panel" aria-labelledby="coverage-table-title">
        <div className="table-toolbar">
          <div>
            <h2 id="coverage-table-title">Facebook categories</h2>
            <p>{filtered.length} of {items.length} categories shown</p>
          </div>
          <div className="table-toolbar__controls">
            <label className="search-field">
              <span className="sr-only">Search coverage categories</span>
              <Search size={16} aria-hidden="true" />
              <input
                type="search"
                value={search}
                onChange={(event) => setSearch(event.target.value)}
                placeholder="Find a category"
              />
            </label>
          </div>
        </div>
        <div className="segmented-control" role="group" aria-label="Filter by coverage status">
          {filterOptions.map((option) => (
            <button
              key={option.value}
              type="button"
              className={statusFilter === option.value ? 'segmented-control__active' : ''}
              aria-pressed={statusFilter === option.value}
              onClick={() => setStatusFilter(option.value)}
            >
              {option.label}
            </button>
          ))}
        </div>

        {filtered.length === 0 ? (
          <EmptyState
            title={items.length === 0 ? 'No coverage audit is available yet' : 'No categories match these filters'}
            description={
              items.length === 0
                ? 'Run the archive scan to populate explicit coverage states.'
                : 'Clear the search or choose a different coverage state.'
            }
            action={
              items.length > 0 ? (
                <button
                  type="button"
                  className="button button--secondary"
                  onClick={() => {
                    setSearch('')
                    setStatusFilter('all')
                  }}
                >
                  Clear filters
                </button>
              ) : undefined
            }
          />
        ) : (
          <>
            <p className="table-scroll-hint">Each category includes source counts, normalized logical counts, and its last-checked date.</p>
            <div className="data-table-wrap data-table-wrap--coverage" role="region" tabIndex={0} aria-label="Coverage categories table">
            <table className="data-table data-table--coverage">
              <thead>
                {table.getHeaderGroups().map((headerGroup) => (
                  <tr key={headerGroup.id}>
                    {headerGroup.headers.map((header) => {
                      const sorted = header.column.getIsSorted()
                      return (
                        <th
                          key={header.id}
                          scope="col"
                          aria-sort={sorted === 'asc' ? 'ascending' : sorted === 'desc' ? 'descending' : 'none'}
                        >
                          {header.isPlaceholder ? null : flexRender(header.column.columnDef.header, header.getContext())}
                        </th>
                      )
                    })}
                  </tr>
                ))}
              </thead>
              <tbody>
                {table.getRowModel().rows.map((row) => (
                  <tr key={row.original.key}>
                    {row.getVisibleCells().map((cell) => (
                      <td key={cell.id} data-label={coverageColumnLabels[cell.column.id]}>{flexRender(cell.column.columnDef.cell, cell.getContext())}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
            </div>
          </>
        )}
      </section>
    </div>
  )
}
