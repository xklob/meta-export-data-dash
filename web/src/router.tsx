import { createRootRoute, createRoute, createRouter, lazyRouteComponent } from '@tanstack/react-router'
import { AppShell } from './components/AppShell'
import { CoveragePage } from './pages/CoveragePage'
import { HomePage } from './pages/HomePage'
import { IntegrityPage } from './pages/IntegrityPage'
import { JobsPage } from './pages/JobsPage'
import { ConversationsPage } from './pages/ConversationsPage'
import { ConversationPage } from './pages/ConversationPage'
import { MediaPage } from './pages/MediaPage'
import { NotFoundPage } from './pages/NotFoundPage'
import { SettingsPage } from './pages/SettingsPage'

const rootRoute = createRootRoute({
  component: AppShell,
  notFoundComponent: NotFoundPage,
})

function optionalNonNegativeInteger(value: unknown): number | undefined {
  if (typeof value === 'number' && Number.isSafeInteger(value) && value >= 0) return value
  if (typeof value !== 'string' || !/^\d+$/.test(value)) return undefined
  const parsed = Number(value)
  return Number.isSafeInteger(parsed) ? parsed : undefined
}

interface ConversationRouteSearch {
  around?: string
  media?: string
  t?: number
}

interface MediaRouteSearch {
  conversation?: string
  occurrence?: string
  t?: number
}

const homeRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/',
  component: HomePage,
})

const coverageRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/coverage',
  component: CoveragePage,
})

const integrityRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/integrity',
  component: IntegrityPage,
})

const jobsRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/jobs',
  component: JobsPage,
})

const settingsRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/settings',
  component: SettingsPage,
})

const conversationsRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/conversations',
  component: ConversationsPage,
})

const conversationRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/conversations/$conversationId',
  validateSearch: (search: Record<string, unknown>): ConversationRouteSearch => {
    const validated: ConversationRouteSearch = {}
    if (typeof search.around === 'string') validated.around = search.around
    if (typeof search.media === 'string') validated.media = search.media
    const startMs = optionalNonNegativeInteger(search.t)
    if (startMs != null) validated.t = startMs
    return validated
  },
  component: ConversationPage,
})

const mediaRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/media',
  validateSearch: (search: Record<string, unknown>): MediaRouteSearch => {
    const validated: MediaRouteSearch = {}
    if (typeof search.conversation === 'string') validated.conversation = search.conversation
    if (typeof search.occurrence === 'string') validated.occurrence = search.occurrence
    const startMs = optionalNonNegativeInteger(search.t)
    if (startMs != null) validated.t = startMs
    return validated
  },
  component: MediaPage,
})

const searchRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/search',
  component: lazyRouteComponent(() => import('./pages/SearchPage'), 'SearchPage'),
})

const peopleRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/people',
  validateSearch: (search: Record<string, unknown>) => ({
    view: typeof search.view === 'string' ? search.view : undefined,
    around: typeof search.around === 'string' ? search.around : undefined,
  }),
  component: lazyRouteComponent(() => import('./pages/PeoplePage'), 'PeoplePage'),
})

const eventsRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/events',
  validateSearch: (search: Record<string, unknown>) => ({
    around: typeof search.around === 'string' ? search.around : undefined,
  }),
  component: lazyRouteComponent(() => import('./pages/EventsPage'), 'EventsPage'),
})

const activityRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/activity',
  component: lazyRouteComponent(() => import('./pages/ActivityPage'), 'ActivityPage'),
})

const timelineRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/timeline',
  validateSearch: (search: Record<string, unknown>) => ({
    around: typeof search.around === 'string' ? search.around : undefined,
  }),
  component: lazyRouteComponent(() => import('./pages/TimelinePage'), 'TimelinePage'),
})

const analyticsRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/analytics',
  component: lazyRouteComponent(() => import('./pages/AnalyticsPage'), 'AnalyticsPage'),
})

const supportRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/support',
  component: lazyRouteComponent(() => import('./pages/SupportPage'), 'SupportPage'),
})

const enrichmentRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/enrichment',
  component: lazyRouteComponent(() => import('./pages/EnrichmentPage'), 'EnrichmentPage'),
})

const deletionRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/deletion',
  component: lazyRouteComponent(() => import('./pages/DeletionPage'), 'DeletionPage'),
})

export const routeTree = rootRoute.addChildren([
  homeRoute,
  coverageRoute,
  integrityRoute,
  jobsRoute,
  conversationsRoute,
  conversationRoute,
  mediaRoute,
  searchRoute,
  peopleRoute,
  eventsRoute,
  activityRoute,
  timelineRoute,
  analyticsRoute,
  supportRoute,
  enrichmentRoute,
  deletionRoute,
  settingsRoute,
])

export const router = createRouter({
  routeTree,
  defaultPreload: 'intent',
  defaultPreloadStaleTime: 10_000,
  scrollRestoration: true,
})

declare module '@tanstack/react-router' {
  interface Register {
    router: typeof router
  }
}
