import { useQuery } from '@tanstack/react-query'
import { Link, Outlet, useRouterState } from '@tanstack/react-router'
import {
  Activity,
  AudioLines,
  BarChart3,
  BriefcaseBusiness,
  CalendarDays,
  CheckCircle2,
  Database,
  FileSearch,
  FolderClock,
  Gauge,
  HardDrive,
  History,
  Images,
  Menu,
  MessageSquareText,
  Search,
  Settings,
  ShieldCheck,
  Trash2,
  Users,
  WifiOff,
  X,
  type LucideIcon,
} from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { api } from '../lib/api'
import { ArchiveMark } from './ArchiveMark'

interface NavItem {
  label: string
  icon: LucideIcon
  stage?: number
}

interface CurrentNavItem extends NavItem {
  path:
    | '/'
    | '/coverage'
    | '/integrity'
    | '/jobs'
    | '/settings'
    | '/conversations'
    | '/media'
    | '/search'
    | '/people'
    | '/events'
    | '/activity'
    | '/timeline'
    | '/analytics'
    | '/support'
    | '/enrichment'
    | '/deletion'
}

const primaryNav: CurrentNavItem[] = [
  { label: 'Home', path: '/', icon: Gauge },
  { label: 'Data Coverage', path: '/coverage', icon: Database },
  { label: 'Integrity', path: '/integrity', icon: ShieldCheck },
  { label: 'Jobs', path: '/jobs', icon: FolderClock },
  { label: 'Deletion', path: '/deletion', icon: Trash2, stage: 5 },
  { label: 'Settings', path: '/settings', icon: Settings },
]

const futureNav: NavItem[] = []

const browseNav: CurrentNavItem[] = [
  { label: 'Conversations', path: '/conversations', icon: MessageSquareText, stage: 2 },
  { label: 'Media', path: '/media', icon: Images, stage: 2 },
  { label: 'Search', path: '/search', icon: Search, stage: 3 },
  { label: 'Enrichment', path: '/enrichment', icon: AudioLines, stage: 3 },
  { label: 'People', path: '/people', icon: Users, stage: 4 },
  { label: 'Events', path: '/events', icon: CalendarDays, stage: 4 },
  { label: 'Activity', path: '/activity', icon: Activity, stage: 4 },
  { label: 'Timeline', path: '/timeline', icon: History, stage: 4 },
  { label: 'Support', path: '/support', icon: BriefcaseBusiness, stage: 4 },
  { label: 'Analytics', path: '/analytics', icon: BarChart3, stage: 4 },
]

export function stageForPathname(pathname: string): number {
  const item = [...primaryNav, ...browseNav].find(({ path }) =>
    path === '/' ? pathname === '/' : pathname === path || pathname.startsWith(`${path}/`),
  )
  return item?.stage ?? 1
}

const pageLabels: Record<string, string> = {
  '/': 'Home',
  '/coverage': 'Data Coverage',
  '/integrity': 'Integrity',
  '/jobs': 'Jobs',
  '/conversations': 'Conversations',
  '/media': 'Media',
  '/search': 'Search',
  '/people': 'People',
  '/events': 'Events',
  '/activity': 'Activity',
  '/timeline': 'Timeline',
  '/analytics': 'Analytics',
  '/support': 'Support',
  '/enrichment': 'Enrichment',
  '/deletion': 'Deletion',
  '/settings': 'Settings',
}

export function pageLabelForPathname(pathname: string): string {
  if (pathname.startsWith('/conversations/')) return 'Conversation reader'
  return pageLabels[pathname] ?? 'Page not found'
}

export function AppShell() {
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const menuButtonRef = useRef<HTMLButtonElement>(null)
  const closeButtonRef = useRef<HTMLButtonElement>(null)
  const pathname = useRouterState({ select: (state) => state.location.pathname })
  const previousPathnameRef = useRef(pathname)
  const health = useQuery({
    queryKey: ['health'],
    queryFn: api.health,
    refetchInterval: 30_000,
    retry: 1,
  })
  const settings = useQuery({
    queryKey: ['settings'],
    queryFn: api.settings,
    staleTime: 30_000,
    retry: 1,
  })

  useEffect(() => {
    if (!sidebarOpen) return
    setSidebarOpen(false)
    // Route changes are the only trigger; sidebarOpen is intentionally read as the pre-navigation state.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pathname])
  const currentPageLabel = pageLabelForPathname(pathname)
  useEffect(() => {
    document.title = `${currentPageLabel} · Meta Archive Explorer`
    if (previousPathnameRef.current === pathname) return
    previousPathnameRef.current = pathname
    requestAnimationFrame(() => document.getElementById('main-content')?.focus())
  }, [currentPageLabel, pathname])
  useEffect(() => {
    if (!sidebarOpen) return
    const previousOverflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    requestAnimationFrame(() => closeButtonRef.current?.focus())
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setSidebarOpen(false)
        requestAnimationFrame(() => menuButtonRef.current?.focus())
      }
      if (event.key === 'Tab') {
        const sidebar = document.getElementById('primary-navigation')
        const focusable = sidebar
          ? Array.from(
              sidebar.querySelectorAll<HTMLElement>(
                'a[href], button:not([disabled]), input:not([disabled]), [tabindex]:not([tabindex="-1"])',
              ),
            ).filter((element) => element.getClientRects().length > 0)
          : []
        const first = focusable[0]
        const last = focusable.at(-1)
        if (!first || !last) return
        if (event.shiftKey && document.activeElement === first) {
          event.preventDefault()
          last.focus()
        } else if (!event.shiftKey && document.activeElement === last) {
          event.preventDefault()
          first.focus()
        }
      }
    }
    window.addEventListener('keydown', closeOnEscape)
    return () => {
      document.body.style.overflow = previousOverflow
      window.removeEventListener('keydown', closeOnEscape)
    }
  }, [sidebarOpen])

  const cloudEnabled = settings.data?.cloud_enabled === true
  const serviceReady = health.isSuccess && health.data.status === 'ok'
  const catalogReady = health.isSuccess && health.data.catalog_ready === true
  const catalogState = health.data?.catalog_state
  const catalogLabel = catalogReady
    ? 'Catalog ready'
    : catalogState === 'processing'
      ? 'Import in progress'
      : catalogState === 'failed'
        ? 'Catalog needs attention'
        : catalogState === 'verification_pending'
          ? 'Verification pending'
          : 'Catalog partial'
  const currentStage = stageForPathname(pathname)

  return (
    <div className="app-shell">
      <a className="skip-link" href="#main-content">
        Skip to main content
      </a>
      <div
        className="sr-only"
        role="status"
        aria-live="polite"
        aria-atomic="true"
        data-testid="route-announcement"
      >
        {currentPageLabel} page loaded
      </div>
      <aside
        id="primary-navigation"
        className={`sidebar ${sidebarOpen ? 'sidebar--open' : ''}`}
        aria-label="Archive navigation"
        aria-modal={sidebarOpen ? true : undefined}
        role={sidebarOpen ? 'dialog' : undefined}
      >
        <div className="sidebar__brand">
          <ArchiveMark />
          <div>
            <strong>Meta Archive</strong>
            <span>Explorer</span>
          </div>
          <button
            ref={closeButtonRef}
            className="icon-button sidebar__close"
            type="button"
            aria-label="Close navigation"
            onClick={() => {
              setSidebarOpen(false)
              requestAnimationFrame(() => menuButtonRef.current?.focus())
            }}
          >
            <X size={19} />
          </button>
        </div>

        <div className={`privacy-card ${cloudEnabled ? 'privacy-card--cloud' : ''}`}>
          <span className="privacy-card__icon" aria-hidden="true">
            {cloudEnabled ? <HardDrive size={17} /> : <WifiOff size={17} />}
          </span>
          <div>
            <strong>{cloudEnabled ? 'Cloud access enabled' : 'Local archive only'}</strong>
            <span>{cloudEnabled ? 'Selected excerpts may leave this device' : 'Cloud processing is off'}</span>
          </div>
        </div>

        <nav className="sidebar-nav" aria-label="Stage 1">
          <p className="sidebar-nav__label">Archive control</p>
          {primaryNav.map(({ label, path, icon: Icon }) => (
            <Link
              key={label}
              to={path}
              className="sidebar-nav__link"
              activeProps={{ className: 'sidebar-nav__link sidebar-nav__link--active' }}
              activeOptions={{ exact: path === '/' }}
            >
              <Icon size={18} strokeWidth={1.8} aria-hidden="true" />
              <span>{label}</span>
            </Link>
          ))}
        </nav>


        <nav className="sidebar-nav" aria-label="Browse archive">
          <p className="sidebar-nav__label">Browse archive</p>
          {browseNav.map(({ label, path, icon: Icon }) => (
            <Link
              key={label}
              to={path}
              className="sidebar-nav__link"
              activeProps={{ className: 'sidebar-nav__link sidebar-nav__link--active' }}
            >
              <Icon size={18} strokeWidth={1.8} aria-hidden="true" />
              <span>{label}</span>
            </Link>
          ))}
        </nav>

        {futureNav.length ? <nav className="sidebar-nav sidebar-nav--future" aria-label="Future stages">
          <p className="sidebar-nav__label">Coming next</p>
          {futureNav.map(({ label, icon: Icon, stage }) => (
            <button
              key={label}
              className="sidebar-nav__link sidebar-nav__link--disabled"
              type="button"
              disabled
              aria-label={`${label}, available in stage ${stage}`}
              title={`Available in Stage ${stage}`}
            >
              <Icon size={18} strokeWidth={1.8} aria-hidden="true" />
              <span>{label}</span>
              <small>S{stage}</small>
            </button>
          ))}
        </nav> : null}

        <div className="sidebar__footer">
          <span className={`service-dot ${serviceReady ? 'service-dot--ok' : ''}`} aria-hidden="true" />
          <div>
            <strong>{serviceReady ? 'Local service connected' : 'Checking local service'}</strong>
            <span>{health.data?.version ? `Version ${health.data.version}` : 'Localhost only'}</span>
          </div>
        </div>
      </aside>

      {sidebarOpen ? (
        <button
          className="sidebar-backdrop"
          type="button"
          aria-label="Close navigation"
          onClick={() => {
            setSidebarOpen(false)
            requestAnimationFrame(() => menuButtonRef.current?.focus())
          }}
        />
      ) : null}

      <div className="app-frame">
        <header className="topbar">
          <div className="topbar__start">
            <button
              ref={menuButtonRef}
              className="icon-button topbar__menu"
              type="button"
              aria-label="Open navigation"
              aria-controls="primary-navigation"
              aria-expanded={sidebarOpen}
              onClick={() => setSidebarOpen(true)}
            >
              <Menu size={20} />
            </button>
            <div className="topbar__title">
              <span>{`Stage ${currentStage}`}</span>
              <strong>{currentPageLabel}</strong>
            </div>
          </div>
          <div className="topbar__status" role="group" aria-label="Archive status">
            <span className={`catalog-pill ${catalogReady ? '' : 'catalog-pill--warning'}`}>
              {catalogReady ? <CheckCircle2 size={15} /> : <FileSearch size={15} />}
              {catalogLabel}
            </span>
            <span className={`cloud-pill ${cloudEnabled ? 'cloud-pill--on' : ''}`}>
              <span aria-hidden="true" />
              {cloudEnabled ? 'Cloud on' : 'Cloud off'}
            </span>
          </div>
        </header>

        <main id="main-content" className="main-content" tabIndex={-1}>
          <Outlet />
        </main>
      </div>
    </div>
  )
}
