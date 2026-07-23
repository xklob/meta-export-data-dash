#!/usr/bin/env node

import AxeBuilder from '@axe-core/playwright'
import { chromium } from '@playwright/test'
import { createHash } from 'node:crypto'
import { constants as fsConstants } from 'node:fs'
import fs from 'node:fs/promises'
import path from 'node:path'
import process from 'node:process'
import { fileURLToPath } from 'node:url'

process.umask(0o077)

const OWNER_DIRECTORY_MODE = 0o700
const OWNER_FILE_MODE = 0o600
const INITIAL_CONTENT_LIMIT_MS = 2_500
const DEFAULT_ROUTE_SETTLE_LIMIT_MS = 120_000
const ANALYTICS_ROUTE_SETTLE_LIMIT_MS = 300_000
const MAX_RENDERED_MESSAGES = 100
const MAX_LOADED_MESSAGES = 250
const WCAG_TAGS = ['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'wcag22a', 'wcag22aa']
const SESSION_COOKIE_NAME = 'meta_archive_session'
const REPORT_SCHEMA_VERSION = 5
const INTERACTION_PROOF_METRIC_MAX = 100
export const COMPLETED_GATE_FAILURE_CATEGORIES = [
  'accessibility',
  'browser_errors',
  'evidence_completeness',
  'interaction_evidence',
  'network_read_only',
  'performance',
  'reader_bounds',
]
export const MEDIA_SOURCE_JUMP_SELECTOR = 'a.media-jump'
export const PAGINATION_LANDMARK_SELECTOR = 'nav, [role="navigation"]'
const ACCEPTANCE_IDENTITY_KEYS = [
  'backend_source_sha256',
  'backend_version',
  'catalog_source_revision',
  'frontend_manifest_sha256',
  'server_nonce',
]
const SHA256_PATTERN = /^[a-f0-9]{64}$/

const scriptDirectory = path.dirname(fileURLToPath(import.meta.url))
const projectRoot = path.resolve(scriptDirectory, '..', '..')
export const reviewRoot = path.join(projectRoot, '.private-review')

export const topLevelRoutes = [
  ['home', '/', 'Archive control center'],
  ['coverage', '/coverage', 'Data coverage'],
  ['integrity', '/integrity', 'Integrity'],
  ['jobs', '/jobs', 'Jobs'],
  ['settings', '/settings', 'Settings'],
  ['conversations', '/conversations', 'Conversations'],
  ['media', '/media', 'Media'],
  ['search', '/search', 'Search the archive'],
  ['enrichment', '/enrichment', 'Transcripts and language signals'],
  ['people', '/people', 'People'],
  ['events', '/events', 'Events'],
  ['activity', '/activity', 'Posts, comments & Facebook activity'],
  ['timeline', '/timeline', 'Timeline'],
  ['support', '/support', 'Support conversations'],
  ['analytics', '/analytics', 'Archive analytics'],
  ['deletion', '/deletion', 'Deletion and quarantine'],
]

const viewports = [
  ['desktop', { width: 1440, height: 1000 }],
  ['mobile', { width: 390, height: 844 }],
]

export const interactionNames = [
  'coverage_status_filter',
  'conversation_sort_and_page',
  'conversation_reader_edges_bounded',
  'conversation_types_source_bound',
  'media_modes_and_sort',
  'media_message_exact_jump',
  'media_activity_exact_jump',
  'media_representative_kinds_and_jumps',
  'keyword_search_exact_jump',
  'exact_search_exact_jump',
  'semantic_search_exact_jump',
  'semantic_argument_search_exact_jump',
  'hybrid_search_exact_jump',
  'enrichment_exact_source_jump',
  'people_views_sort_and_page',
  'events_filter_sort_and_page',
  'activity_filter_sort_and_page',
  'activity_post_comment_source_jumps',
  'timeline_mode_filter_and_page',
  'analytics_bucket_queries',
  'deletion_preview_guard',
]

const interactionRouteSlugs = {
  coverage_status_filter: 'coverage',
  conversation_sort_and_page: 'conversations',
  conversation_reader_edges_bounded: 'largest-conversation',
  conversation_types_source_bound: 'conversations',
  media_modes_and_sort: 'media',
  media_message_exact_jump: 'media',
  media_activity_exact_jump: 'media',
  media_representative_kinds_and_jumps: 'media',
  keyword_search_exact_jump: 'search',
  exact_search_exact_jump: 'search',
  semantic_search_exact_jump: 'search',
  semantic_argument_search_exact_jump: 'search',
  hybrid_search_exact_jump: 'search',
  enrichment_exact_source_jump: 'enrichment',
  people_views_sort_and_page: 'people',
  events_filter_sort_and_page: 'events',
  activity_filter_sort_and_page: 'activity',
  activity_post_comment_source_jumps: 'activity',
  timeline_mode_filter_and_page: 'timeline',
  analytics_bucket_queries: 'analytics',
  deletion_preview_guard: 'deletion',
}

export const interactionProofKeys = [
  'assertion_count',
  'dom_api_reconciliation_count',
  'nonempty_dataset_count',
  'source_bound_empty_dataset_count',
  'pagination_bound_count',
  'pagination_exercised_count',
  'source_jump_count',
  'representative_kind_count',
  'conversation_type_count',
]

const interactionProofMinimums = {
  coverage_status_filter: { assertion_count: 1, nonempty_dataset_count: 1 },
  conversation_sort_and_page: {
    assertion_count: 8,
    dom_api_reconciliation_count: 10,
    nonempty_dataset_count: 4,
    pagination_bound_count: 1,
    pagination_exercised_count: 1,
  },
  conversation_reader_edges_bounded: {
    assertion_count: 8,
    dom_api_reconciliation_count: 8,
    nonempty_dataset_count: 1,
    pagination_bound_count: 2,
    pagination_exercised_count: 2,
  },
  conversation_types_source_bound: {
    assertion_count: 4,
    dom_api_reconciliation_count: 2,
    nonempty_dataset_count: 2,
    conversation_type_count: 2,
  },
  media_modes_and_sort: {
    assertion_count: 6,
    dom_api_reconciliation_count: 4,
    nonempty_dataset_count: 4,
  },
  media_message_exact_jump: {
    assertion_count: 2,
    dom_api_reconciliation_count: 1,
    nonempty_dataset_count: 1,
    source_jump_count: 1,
  },
  media_activity_exact_jump: {
    assertion_count: 2,
    dom_api_reconciliation_count: 1,
    nonempty_dataset_count: 1,
    source_jump_count: 1,
  },
  media_representative_kinds_and_jumps: {
    assertion_count: 8,
    dom_api_reconciliation_count: 4,
    nonempty_dataset_count: 4,
    source_jump_count: 4,
    representative_kind_count: 4,
  },
  keyword_search_exact_jump: {
    assertion_count: 2,
    dom_api_reconciliation_count: 1,
    nonempty_dataset_count: 1,
    source_jump_count: 1,
  },
  exact_search_exact_jump: {
    assertion_count: 2,
    dom_api_reconciliation_count: 1,
    nonempty_dataset_count: 1,
    source_jump_count: 1,
  },
  semantic_search_exact_jump: {
    assertion_count: 2,
    dom_api_reconciliation_count: 1,
    nonempty_dataset_count: 1,
    source_jump_count: 1,
  },
  semantic_argument_search_exact_jump: {
    assertion_count: 2,
    dom_api_reconciliation_count: 1,
    nonempty_dataset_count: 1,
    source_jump_count: 1,
  },
  hybrid_search_exact_jump: {
    assertion_count: 2,
    dom_api_reconciliation_count: 1,
    nonempty_dataset_count: 1,
    source_jump_count: 1,
  },
  enrichment_exact_source_jump: { assertion_count: 1, source_jump_count: 1 },
  people_views_sort_and_page: {
    assertion_count: 8,
    dom_api_reconciliation_count: 7,
    nonempty_dataset_count: 1,
    pagination_bound_count: 1,
    pagination_exercised_count: 1,
  },
  events_filter_sort_and_page: {
    assertion_count: 2,
    dom_api_reconciliation_count: 1,
    pagination_bound_count: 1,
  },
  activity_filter_sort_and_page: {
    assertion_count: 3,
    dom_api_reconciliation_count: 2,
    nonempty_dataset_count: 1,
    pagination_bound_count: 1,
  },
  activity_post_comment_source_jumps: {
    assertion_count: 4,
    dom_api_reconciliation_count: 2,
    nonempty_dataset_count: 2,
    pagination_bound_count: 1,
    pagination_exercised_count: 1,
    source_jump_count: 2,
    representative_kind_count: 2,
  },
  timeline_mode_filter_and_page: {
    assertion_count: 4,
    dom_api_reconciliation_count: 2,
    nonempty_dataset_count: 1,
    pagination_bound_count: 2,
    pagination_exercised_count: 1,
  },
  analytics_bucket_queries: { assertion_count: 4, dom_api_reconciliation_count: 2 },
  deletion_preview_guard: { assertion_count: 3 },
}

const PREVIEW_PATH_PATTERN = /^\/api\/v1\/(?:blobs|derivatives)\/[a-f0-9]{64}$/

export function isExpectedBrowserAbort(baseUrl, requestDetails) {
  try {
    const target = new URL(requestDetails.url)
    if (
      target.origin !== new URL(baseUrl).origin ||
      requestDetails.method !== 'GET' ||
      requestDetails.errorText !== 'net::ERR_ABORTED'
    ) {
      return false
    }
    const previewWasCancelled =
      ['image', 'media'].includes(requestDetails.resourceType) &&
      PREVIEW_PATH_PATTERN.test(target.pathname)
    const navigationCancelledHealthPoll =
      requestDetails.resourceType === 'fetch' && target.pathname === '/api/v1/health'
    return previewWasCancelled || navigationCancelledHealthPoll
  } catch {
    return false
  }
}

export function parseArguments(argv) {
  const options = {
    baseUrl: process.env.META_ARCHIVE_ACCEPTANCE_BASE_URL,
    tokenFile: process.env.META_ARCHIVE_ACCEPTANCE_TOKEN_FILE,
    output: process.env.META_ARCHIVE_UI_ACCEPTANCE_OUTPUT,
  }
  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index]
    if (argument === '--') continue
    else if (argument === '--base-url') options.baseUrl = argv[++index]
    else if (argument === '--token-file') options.tokenFile = argv[++index]
    else if (argument === '--output') options.output = argv[++index]
    else if (argument === '--help') {
      process.stdout.write(
        'Usage: pnpm run test:real-ui -- --base-url URL --token-file FILE [--output .private-review/.../report.json]\n',
      )
      process.exit(0)
    } else {
      throw new Error('Unsupported acceptance-runner option')
    }
  }
  if (!options.baseUrl || !options.tokenFile) {
    throw new Error('A loopback base URL and owner-only startup-token file are required')
  }
  return options
}

export function validateLoopbackBaseUrl(rawUrl) {
  const parsed = new URL(rawUrl)
  const hostname = parsed.hostname.toLowerCase()
  if (parsed.protocol !== 'http:' || !['127.0.0.1', 'localhost', '[::1]'].includes(hostname)) {
    throw new Error('The real UI runner accepts only an HTTP loopback server')
  }
  if (parsed.username || parsed.password || parsed.search || parsed.hash) {
    throw new Error('The base URL must not contain credentials, a query, or a fragment')
  }
  if (parsed.pathname !== '/') {
    throw new Error('The base URL must be a loopback origin without a path')
  }
  return parsed.toString().replace(/\/$/, '')
}

async function lstatOrNull(target) {
  try {
    return await fs.lstat(target)
  } catch (error) {
    if (error && error.code === 'ENOENT') return null
    throw error
  }
}

export async function readStartupToken(rawPath) {
  const tokenPath = path.isAbsolute(rawPath)
    ? path.resolve(rawPath)
    : path.resolve(projectRoot, rawPath)
  let handle
  try {
    handle = await fs.open(
      tokenPath,
      fsConstants.O_RDONLY | fsConstants.O_NOFOLLOW | fsConstants.O_NONBLOCK,
    )
    const tokenStat = await handle.stat()
    if (!tokenStat.isFile()) {
      throw new Error('The startup-token path must be a non-symlink regular file')
    }
    if ((tokenStat.mode & 0o077) !== 0 || tokenStat.size > 4096) {
      throw new Error('The startup-token file must be owner-only and bounded')
    }
    const token = (await handle.readFile({ encoding: 'utf8' })).trim()
    if (token.length < 20 || /\s/.test(token)) {
      throw new Error('The startup token is missing or malformed')
    }
    return token
  } catch (error) {
    if (error && ['ELOOP', 'ENOENT', 'ENXIO'].includes(error.code)) {
      throw new Error('The startup-token path must be a non-symlink regular file')
    }
    throw error
  } finally {
    await handle?.close().catch(() => {})
  }
}

export function outputPath(rawOutput) {
  if (rawOutput) {
    const candidate = path.resolve(projectRoot, rawOutput)
    const relative = path.relative(reviewRoot, candidate)
    if (!relative || relative.startsWith('..') || path.isAbsolute(relative)) {
      throw new Error('The UI report must be a file below .private-review/')
    }
    return candidate
  }
  const stamp = new Date().toISOString().replaceAll(/[-:.]/g, '').replace('Z', 'Z')
  return path.join(reviewRoot, `real-ui-acceptance-${stamp}-${process.pid}`, 'report.json')
}

export async function ensurePrivateDirectory(directory) {
  const relative = path.relative(reviewRoot, directory)
  if (relative.startsWith('..') || path.isAbsolute(relative)) {
    throw new Error('Private artifact directories must stay below .private-review/')
  }
  const parts = relative ? relative.split(path.sep) : []
  let current = reviewRoot
  for (const part of ['', ...parts]) {
    if (part) current = path.join(current, part)
    let currentStat = await lstatOrNull(current)
    if (!currentStat) {
      await fs.mkdir(current, { mode: OWNER_DIRECTORY_MODE })
      currentStat = await fs.lstat(current)
    }
    if (currentStat.isSymbolicLink() || !currentStat.isDirectory()) {
      throw new Error('A private artifact directory is unsafe')
    }
    await fs.chmod(current, OWNER_DIRECTORY_MODE)
  }
}

async function securePrivateTree(root) {
  const rootStat = await lstatOrNull(root)
  if (!rootStat) return
  if (rootStat.isSymbolicLink() || !rootStat.isDirectory()) {
    throw new Error('A private artifact root is unsafe')
  }
  await fs.chmod(root, OWNER_DIRECTORY_MODE)
  for (const entry of await fs.readdir(root, { withFileTypes: true })) {
    const child = path.join(root, entry.name)
    const childStat = await fs.lstat(child)
    if (childStat.isSymbolicLink()) throw new Error('A private artifact symlink is unsafe')
    if (childStat.isDirectory()) await securePrivateTree(child)
    else if (childStat.isFile()) await fs.chmod(child, OWNER_FILE_MODE)
    else throw new Error('A private artifact is neither a file nor a directory')
  }
}

function sha256Text(value) {
  return createHash('sha256').update(String(value)).digest('hex')
}

async function privateArtifactRecord(root, filename) {
  const target = path.join(root, filename)
  let handle
  try {
    handle = await fs.open(target, fsConstants.O_RDONLY | fsConstants.O_NOFOLLOW)
    const targetStat = await handle.stat()
    if (
      !targetStat.isFile() ||
      targetStat.size <= 0 ||
      (targetStat.mode & 0o777) !== OWNER_FILE_MODE
    ) {
      throw new Error('A private acceptance artifact is missing, empty, or not owner-only')
    }
    const digest = createHash('sha256')
    const buffer = Buffer.allocUnsafe(1024 * 1024)
    let position = 0
    while (true) {
      const { bytesRead } = await handle.read(buffer, 0, buffer.length, position)
      if (!bytesRead) break
      digest.update(buffer.subarray(0, bytesRead))
      position += bytesRead
    }
    return {
      relative_path: `artifacts/${filename}`,
      size_bytes: targetStat.size,
      sha256: digest.digest('hex'),
    }
  } finally {
    await handle?.close().catch(() => {})
  }
}

export async function buildArtifactManifest(root) {
  const expected = expectedArtifactFilenames()
  const actual = new Set(await fs.readdir(root))
  if (
    actual.size !== expected.size ||
    [...expected].some((filename) => !actual.has(filename))
  ) {
    throw new Error('The private screenshot/trace inventory is incomplete or unexpected')
  }
  const records = []
  for (const filename of [...expected].sort()) {
    records.push(await privateArtifactRecord(root, filename))
  }
  return records
}

function expectedArtifactFilenames() {
  return new Set([
    ...viewports.flatMap(([viewport]) =>
      [...topLevelRoutes.map(([slug]) => slug), 'largest-conversation'].map(
        (slug) => `${viewport}-${slug}.png`,
      ),
    ),
    ...viewports.map(([viewport]) => `${viewport}-trace.zip`),
  ])
}

export async function artifactInventoryComplete(root) {
  const expected = expectedArtifactFilenames()
  const actual = new Set(await fs.readdir(root))
  return !(
    actual.size !== expected.size ||
    [...expected].some((filename) => !actual.has(filename))
  )
}

export async function writePrivateJson(finalPath, payload) {
  await ensurePrivateDirectory(path.dirname(finalPath))
  if (await lstatOrNull(finalPath)) throw new Error('Refusing to overwrite prior UI evidence')
  const temporary = path.join(
    path.dirname(finalPath),
    `.${path.basename(finalPath)}.${process.pid}.${Date.now()}.tmp`,
  )
  let handle
  try {
    handle = await fs.open(
      temporary,
      fsConstants.O_CREAT | fsConstants.O_EXCL | fsConstants.O_WRONLY | fsConstants.O_NOFOLLOW,
      OWNER_FILE_MODE,
    )
    await handle.writeFile(`${JSON.stringify(payload, null, 2)}\n`, 'utf8')
    await handle.sync()
    await handle.close()
    handle = undefined
    // Publishing with link(), instead of rename(), is atomic and cannot replace earlier evidence.
    await fs.link(temporary, finalPath)
    await fs.unlink(temporary)
    const directoryHandle = await fs.open(path.dirname(finalPath), fsConstants.O_RDONLY)
    await directoryHandle.sync()
    await directoryHandle.close()
  } catch (error) {
    if (handle) await handle.close().catch(() => {})
    await fs.unlink(temporary).catch(() => {})
    throw error
  }
}

function percentile95(values) {
  if (!values.length) return null
  const ordered = [...values].sort((left, right) => left - right)
  const position = (ordered.length - 1) * 0.95
  const lower = Math.floor(position)
  const upper = Math.ceil(position)
  if (lower === upper) return ordered[lower]
  return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
}

async function waitForInitialContent(page) {
  await page.locator('#main-content').waitFor({ state: 'visible', timeout: 120_000 })
  const state = await page.waitForFunction(
    () => {
      const main = document.querySelector('#main-content')
      if (!main) return false
      if (main.querySelector('.state-panel--error')) return 'error'
      if (main.querySelector('.state-panel--empty')) return 'empty'
      return main.querySelector('h1') ? 'ready' : false
    },
    undefined,
    { timeout: 120_000 },
  )
  const contentState = await state.jsonValue()
  if (contentState === 'error') throw new Error('A browser route rendered an application error')
  return contentState
}

export function routeSettleTimeout(route) {
  return new URL(route, 'http://127.0.0.1').pathname === '/analytics'
    ? ANALYTICS_ROUTE_SETTLE_LIMIT_MS
    : DEFAULT_ROUTE_SETTLE_LIMIT_MS
}

async function waitForRouteSettled(page, route) {
  const timeout = routeSettleTimeout(route)
  await page.waitForLoadState('networkidle', { timeout })
  const loading = page.locator('.state-panel--loading')
  if (await loading.count()) await loading.waitFor({ state: 'hidden', timeout })
  if (await page.locator('.state-panel--error').count()) {
    throw new Error('A browser route rendered an application error')
  }
}

function requireCondition(condition, message) {
  if (!condition) throw new Error(message)
}

export async function ensureAnalyticsDrilldownVisible(page, locator, sectionId, label) {
  if (await locator.isVisible()) return false
  const drilldown = page.locator(
    `details.analytics-drilldown[data-analytics-section="${sectionId}"]`,
  )
  requireCondition(
    (await drilldown.count()) === 1,
    `The compact Analytics ${label} section is unavailable`,
  )
  await drilldown.locator('summary').click()
  await locator.waitFor({ state: 'visible' })
  return true
}

export async function waitForAnalyticsRouteSettled(page) {
  const route = page.locator('.analytics-page')
  await route.waitFor({ state: 'visible', timeout: ANALYTICS_ROUTE_SETTLE_LIMIT_MS })
  await page.waitForFunction(
    () => document.querySelector('.analytics-page')?.getAttribute('data-analytics-route-state') === 'settled',
    undefined,
    { timeout: ANALYTICS_ROUTE_SETTLE_LIMIT_MS },
  )
  await page.waitForLoadState('networkidle', { timeout: ANALYTICS_ROUTE_SETTLE_LIMIT_MS })
  await page.waitForTimeout(100)
  if ((await route.getAttribute('data-analytics-route-state')) !== 'settled') {
    throw new Error('Analytics started another read after reporting a settled route')
  }
}

export async function assertRouteIdentity(page, baseUrl, route, expectedHeading) {
  const expectedUrl = new URL(route, baseUrl)
  const actualUrl = new URL(page.url())
  requireCondition(
    actualUrl.origin === expectedUrl.origin && actualUrl.pathname === expectedUrl.pathname,
    'A browser route redirected to an unexpected page',
  )
  if (expectedHeading === null) return
  requireCondition(
    typeof expectedHeading === 'string' && expectedHeading.length > 0,
    'A browser route is missing its expected level-one heading contract',
  )
  const heading = page.getByRole('heading', {
    level: 1,
    name: expectedHeading,
    exact: true,
  })
  requireCondition(
    (await heading.count()) === 1 && (await heading.isVisible()),
    'A browser route rendered the wrong level-one heading',
  )
}

export function interactionProof(overrides = {}) {
  return Object.fromEntries(
    interactionProofKeys.map((key) => [key, overrides[key] ?? 0]),
  )
}

export function validateInteractionProof(name, candidate) {
  requireCondition(
    Object.hasOwn(interactionProofMinimums, name),
    'The interaction proof name is not catalogued',
  )
  requireCondition(
    candidate &&
      typeof candidate === 'object' &&
      !Array.isArray(candidate) &&
      Object.keys(candidate).length === interactionProofKeys.length &&
      interactionProofKeys.every((key) => Object.hasOwn(candidate, key)),
    'The interaction proof schema is invalid',
  )
  for (const key of interactionProofKeys) {
    requireCondition(
      Number.isSafeInteger(candidate[key]) && candidate[key] >= 0,
      'An interaction proof metric is invalid',
    )
    requireCondition(
      candidate[key] <= INTERACTION_PROOF_METRIC_MAX,
      'An interaction proof metric exceeds its aggregate-only bound',
    )
  }
  for (const [key, minimum] of Object.entries(interactionProofMinimums[name])) {
    requireCondition(
      candidate[key] >= minimum,
      'An interaction proof minimum was not met',
    )
  }
  if (name === 'events_filter_sort_and_page') {
    requireCondition(
      candidate.nonempty_dataset_count + candidate.source_bound_empty_dataset_count >= 1,
      'Event evidence is neither nonempty nor source-bound unavailable',
    )
  }
  return candidate
}

export function minimumInteractionProof(name) {
  const proof = interactionProof(interactionProofMinimums[name])
  if (name === 'events_filter_sort_and_page') proof.source_bound_empty_dataset_count = 1
  return validateInteractionProof(name, proof)
}

export function validateSummaryProofTotals(candidate) {
  if (
    !candidate ||
    typeof candidate !== 'object' ||
    Array.isArray(candidate) ||
    Object.keys(candidate).length !== interactionProofKeys.length ||
    !interactionProofKeys.every(
      (key) =>
        Object.hasOwn(candidate, key) &&
        Number.isSafeInteger(candidate[key]) &&
        candidate[key] >= 0 &&
        candidate[key] <= interactionNames.length * viewports.length * INTERACTION_PROOF_METRIC_MAX,
    )
  ) {
    return false
  }
  return [
    'assertion_count',
    'dom_api_reconciliation_count',
    'nonempty_dataset_count',
    'pagination_bound_count',
    'pagination_exercised_count',
    'source_jump_count',
    'representative_kind_count',
    'conversation_type_count',
  ].every((key) => candidate[key] > 0)
}

export function reconcileProjectedRows(apiProjection, domRows) {
  requireCondition(Array.isArray(apiProjection), 'The API projection is invalid')
  requireCondition(Array.isArray(domRows) && domRows.length > 0, 'No rendered rows were projected')
  let priorIndex = -1
  for (const row of domRows) {
    requireCondition(
      row &&
        typeof row === 'object' &&
        Object.keys(row).length === 2 &&
        Number.isSafeInteger(row.index) &&
        row.index >= 0 &&
        row.index < apiProjection.length &&
        row.index > priorIndex,
      'A rendered row projection has invalid order or identity',
    )
    requireCondition(
      JSON.stringify(row.value) === JSON.stringify(apiProjection[row.index]),
      'A rendered row does not match its API identity, order, or value',
    )
    priorIndex = row.index
  }
  return domRows.length
}

export function reconcileCompleteProjection(apiProjection, domProjection) {
  requireCondition(
    Array.isArray(apiProjection) &&
      Array.isArray(domProjection) &&
      JSON.stringify(domProjection) === JSON.stringify(apiProjection),
    'The rendered collection does not match the complete API identity, order, or value projection',
  )
  return apiProjection.length
}

export function sourceBoundAvailability(
  payload,
  { coverageKey = 'coverage', expectedStatus = 'unavailable' } = {},
) {
  const items = apiItems(payload)
  if (items.length > 0) {
    return { nonempty_dataset_count: 1, source_bound_empty_dataset_count: 0 }
  }
  const coverage = payload?.[coverageKey]
  requireCondition(
    coverage?.status === expectedStatus && coverage.absence_is_not_zero_history === true,
    'An empty interaction dataset was not source-bound to the expected explicit unavailable coverage state',
  )
  return { nonempty_dataset_count: 0, source_bound_empty_dataset_count: 1 }
}

async function openInteractivePage(page, baseUrl, route) {
  const response = await page.goto(`${baseUrl}${route}`, {
    waitUntil: 'domcontentloaded',
    timeout: 120_000,
  })
  if (!response || !response.ok()) throw new Error('An interaction route did not return success')
  const expected = new URL(route, baseUrl)
  const actual = new URL(page.url())
  if (actual.origin !== expected.origin || actual.pathname !== expected.pathname) {
    throw new Error('An interaction route redirected unexpectedly')
  }
  await waitForInitialContent(page)
  await waitForRouteSettled(page, route)
  await page.waitForTimeout(150)
}

async function requestJson(context, baseUrl, route, parameters = {}) {
  const response = await context.request.get(`${baseUrl}${route}`, {
    params: parameters,
    maxRedirects: 0,
    timeout: 120_000,
  })
  if (!response.ok() || new URL(response.url()).origin !== new URL(baseUrl).origin) {
    throw new Error('Private interaction discovery failed')
  }
  const payload = await response.json()
  if (!payload || typeof payload !== 'object' || Array.isArray(payload)) {
    throw new Error('Private interaction discovery returned an invalid response')
  }
  return payload
}

export function activityMediaDiscoveryParameters(cursor) {
  return {
    view: 'occurrences',
    sort: 'date',
    direction: 'desc',
    limit: 250,
    cursor,
  }
}

function privateKeyword(items) {
  const candidates = []
  for (const item of items ?? []) {
    for (const value of [item?.content, item?.transcription]) {
      if (typeof value !== 'string') continue
      for (const match of value.matchAll(/[\p{L}\p{N}]{5,}/gu)) candidates.push(match[0])
    }
  }
  candidates.sort((left, right) => right.length - left.length || left.localeCompare(right))
  return candidates[0] ?? null
}

async function discoverInteractionTargets(context, baseUrl, largestId) {
  const messages = await requestJson(
    context,
    baseUrl,
    `/api/v1/conversations/${encodeURIComponent(largestId)}/messages`,
    { start: 'latest', limit: 250 },
  )
  const query = privateKeyword(messages.items)

  const conversationTypes = {}
  for (const conversationType of ['direct', 'group']) {
    const payload = await requestJson(context, baseUrl, '/api/v1/conversations', {
      conversation_type: conversationType,
      sort: 'messages',
      direction: 'desc',
      limit: 1,
    })
    const item = apiItems(payload)[0]
    requireCondition(
      item?.id && item?.type === conversationType && item?.message_counts?.total > 0,
      'A required source-backed conversation type is unavailable',
    )
    conversationTypes[conversationType] = item
  }

  const representativeMedia = {}
  for (const mediaType of ['photo', 'video', 'audio', 'document']) {
    let mediaCursor
    for (let pageNumber = 0; pageNumber < 4 && !representativeMedia[mediaType]; pageNumber += 1) {
      const media = await requestJson(context, baseUrl, '/api/v1/media', {
        view: 'occurrences',
        type: mediaType,
        sort: 'date',
        direction: 'desc',
        limit: 250,
        cursor: mediaCursor,
      })
      const item = apiItems(media).find(
        (candidate) => candidate?.id && candidate?.type === mediaType && candidate?.jump?.href,
      )
      if (item) representativeMedia[mediaType] = item
      mediaCursor = media?.page?.next_cursor
      if (!mediaCursor) break
    }
    requireCondition(
      representativeMedia[mediaType],
      'A required source-linked representative media kind is unavailable',
    )
  }

  let messageMedia = null
  let activityMedia = null
  let cursor
  for (let pageNumber = 0; pageNumber < 12 && (!messageMedia || !activityMedia); pageNumber += 1) {
    const media = await requestJson(
      context,
      baseUrl,
      '/api/v1/media',
      activityMediaDiscoveryParameters(cursor),
    )
    for (const item of media.items ?? []) {
      if (!messageMedia && item?.jump?.message_id && item?.id) {
        messageMedia = item
      }
      if (!activityMedia && item?.jump?.activity_id && item?.id) {
        activityMedia = item
      }
    }
    cursor = media?.page?.next_cursor
    if (!cursor) break
  }
  return { query, messageMedia, activityMedia, conversationTypes, representativeMedia }
}

async function followExactJump(page, baseUrl, link) {
  const href = await link.getAttribute('href')
  requireCondition(typeof href === 'string' && href.startsWith('/'), 'A source jump is missing')
  const expected = new URL(href, baseUrl)
  requireCondition(expected.origin === new URL(baseUrl).origin, 'A source jump left the local app')
  await Promise.all([
    page.waitForURL(
      (value) =>
        value.origin === expected.origin &&
        value.pathname === expected.pathname &&
        value.search === expected.search &&
        value.hash === expected.hash,
      { timeout: 120_000 },
    ),
    link.click(),
  ])
  await waitForInitialContent(page)
  if (expected.hash.length > 1) {
    await page.waitForFunction(
      (targetId) => document.getElementById(targetId) !== null,
      expected.hash.slice(1),
      { timeout: 120_000 },
    )
  }
}

async function cursorPageRoundTrip(page, label, options = {}) {
  const pager = page.getByRole('navigation', { name: `${label} pages` })
  await pager.waitFor({ state: 'visible', timeout: 120_000 })
  const currentPage = Number(await pager.getAttribute('data-page-number'))
  requireCondition(Number.isSafeInteger(currentPage) && currentPage >= 1, 'Pager state is invalid')
  const next = pager.getByRole('button', { name: 'Next page' })
  let nextPayload = null
  let exercised = 0
  if ((await next.getAttribute('aria-disabled')) !== 'true') {
    nextPayload = options.pathname
      ? await waitForApiAction(
          page,
          options.pathname,
          (query) => Boolean(query.get('cursor')) && (!options.predicate || options.predicate(query)),
          () => next.click(),
        )
      : await next.click().then(() => null)
    await page.waitForFunction(
      ({ accessibleName, landmarkSelector, prior }) => {
        const navigation = [...document.querySelectorAll(landmarkSelector)].find(
          (element) => element.getAttribute('aria-label') === accessibleName,
        )
        return Number(navigation?.getAttribute('data-page-number')) === prior + 1
      },
      {
        accessibleName: `${label} pages`,
        landmarkSelector: PAGINATION_LANDMARK_SELECTOR,
        prior: currentPage,
      },
      { timeout: 120_000 },
    )
    if (options.assertNextPayload) await options.assertNextPayload(nextPayload)
    exercised = 1
    const nextPager = page.getByRole('navigation', { name: `${label} pages` })
    await nextPager.getByRole('button', { name: 'Previous page' }).click()
    await page.waitForFunction(
      ({ accessibleName, landmarkSelector, prior }) => {
        const navigation = [...document.querySelectorAll(landmarkSelector)].find(
          (element) => element.getAttribute('aria-label') === accessibleName,
        )
        return Number(navigation?.getAttribute('data-page-number')) === prior
      },
      {
        accessibleName: `${label} pages`,
        landmarkSelector: PAGINATION_LANDMARK_SELECTOR,
        prior: currentPage,
      },
      { timeout: 120_000 },
    )
    if (options.assertPreviousPage) {
      const deadline = Date.now() + 120_000
      let lastError
      while (Date.now() < deadline) {
        try {
          await options.assertPreviousPage()
          lastError = null
          break
        } catch (error) {
          lastError = error
          await page.waitForTimeout(50)
        }
      }
      if (lastError) throw lastError
    }
  }
  return { bound: 1, exercised, nextPayload }
}

function recordRouteDiagnostic(
  counters,
  viewport,
  route,
  failureCategoryCounts,
  { performanceOverBudget = false, axeRuleIds = [] } = {},
) {
  if (
    !failureCategoryCounts.some((item) => item.count > 0) &&
    !performanceOverBudget &&
    axeRuleIds.length === 0
  ) {
    return
  }
  counters.routeDiagnostics.push({
    viewport,
    route,
    failure_category_counts: failureCategoryCounts,
    performance_over_budget: performanceOverBudget,
    axe_rule_ids: axeRuleIds,
  })
}

async function auditInteraction(counters, viewportName, name, callback) {
  const startedAt = performance.now()
  const pageErrorsBefore = counters.pageErrorCount
  const externalRequestsBefore = counters.externalRequestCount
  const nonGetHeadRequestsBefore = counters.nonGetHeadRequestCount
  const webSocketRequestsBefore = counters.webSocketRequestCount
  const serviceWorkerRegistrationsBefore = counters.serviceWorkerRegistrationCount
  process.stderr.write(
    `${JSON.stringify({ viewport: viewportName, interaction: name, status: 'started' })}\n`,
  )
  let passed = false
  let proof = interactionProof()
  try {
    proof = validateInteractionProof(name, await callback())
    passed = true
  } catch {
    passed = false
  }
  const evidence = {
    viewport: viewportName,
    interaction: name,
    passed,
    duration_ms: Math.round((performance.now() - startedAt) * 10) / 10,
    proof,
  }
  counters.interactionEvidence.push(evidence)
  const browserErrorCount = counters.pageErrorCount - pageErrorsBefore
  const networkReadOnlyCount =
    counters.externalRequestCount -
    externalRequestsBefore +
    (counters.nonGetHeadRequestCount - nonGetHeadRequestsBefore) +
    (counters.webSocketRequestCount - webSocketRequestsBefore) +
    (counters.serviceWorkerRegistrationCount - serviceWorkerRegistrationsBefore)
  recordRouteDiagnostic(
    counters,
    viewportName,
    interactionRouteSlugs[name],
    [
      { category: 'browser_errors', count: browserErrorCount },
      { category: 'interaction_evidence', count: passed ? 0 : 1 },
      { category: 'network_read_only', count: networkReadOnlyCount },
    ],
  )
  process.stderr.write(`${JSON.stringify(evidence)}\n`)
}

export function readerEdgeRequestRequired(edge, beginningVisible, endVisible) {
  if (edge === 'earliest') return !beginningVisible
  if (edge === 'latest') return !endVisible
  throw new Error('Unsupported conversation-reader edge')
}

export async function reachReaderEdge(page, pathname, edge) {
  const beginning = page.getByText('Beginning of conversation', { exact: true })
  const end = page.getByText('End of conversation', { exact: true })
  const beginningVisible = await beginning.isVisible().catch(() => false)
  const endVisible = await end.isVisible().catch(() => false)
  if (!readerEdgeRequestRequired(edge, beginningVisible, endVisible)) {
    return { exercised: false, payload: null }
  }
  const marker = edge === 'earliest' ? beginning : end
  let resolveResponse
  const matchingResponse = new Promise((resolve) => {
    resolveResponse = resolve
  })
  const onResponse = (response) => {
    const url = new URL(response.url())
    if (
      url.pathname === pathname &&
      response.status() < 400 &&
      url.searchParams.get('start') === edge
    ) {
      resolveResponse(response)
    }
  }
  page.on('response', onResponse)
  let response = null
  try {
    await page
      .getByRole('button', { name: edge === 'earliest' ? 'Beginning' : 'Latest' })
      .click()
    await marker.waitFor({ state: 'visible', timeout: 120_000 })
    response = await Promise.race([
      matchingResponse,
      page.waitForTimeout(250).then(() => null),
    ])
  } finally {
    page.off('response', onResponse)
  }
  const payload = response
    ? await response.json()
    : await requestJson(page.context(), new URL(page.url()).origin, pathname, {
        start: edge,
        limit: 100,
      })
  requireCondition(
    payload && typeof payload === 'object' && !Array.isArray(payload),
    'A reader-edge API response was not a JSON object',
  )
  return { exercised: true, payload }
}

async function auditPage(
  page,
  baseUrl,
  slug,
  route,
  expectedHeading,
  screenshotPath,
  viewportName,
  counters,
) {
  const startedAt = performance.now()
  const pageErrorsBefore = counters.pageErrorCount
  const externalRequestsBefore = counters.externalRequestCount
  const nonGetHeadRequestsBefore = counters.nonGetHeadRequestCount
  const webSocketRequestsBefore = counters.webSocketRequestCount
  const serviceWorkerRegistrationsBefore = counters.serviceWorkerRegistrationCount
  const response = await page.goto(`${baseUrl}${route}`, {
    waitUntil: 'domcontentloaded',
    timeout: 120_000,
  })
  if (!response || !response.ok()) throw new Error('A browser route did not return success')
  const contentState = await waitForInitialContent(page)
  await assertRouteIdentity(page, baseUrl, route, expectedHeading)
  const initialContentMs = Math.round((performance.now() - startedAt) * 10) / 10
  counters.initialContentMs.push(initialContentMs)
  await waitForRouteSettled(page, route)
  await page.waitForTimeout(250)

  const accessibility = await new AxeBuilder({ page }).withTags(WCAG_TAGS).analyze()
  counters.axeViolationCount += accessibility.violations.length
  counters.axeIncompleteCount += accessibility.incomplete.length
  const rootOverflow = await page.evaluate(() => {
    const documentWidth = Math.max(
      document.documentElement.scrollWidth,
      document.body?.scrollWidth ?? 0,
    )
    return documentWidth > window.innerWidth + 1
  })
  if (rootOverflow) counters.horizontalOverflowCount += 1
  const reducedMotionPassed = await page.evaluate(() => {
    if (!window.matchMedia('(prefers-reduced-motion: reduce)').matches) return false
    return !document.getAnimations().some((animation) => {
      const timing = animation.effect?.getComputedTiming()
      return animation.playState === 'running' && Number(timing?.duration ?? 0) > 0
    })
  })
  if (!reducedMotionPassed) counters.reducedMotionFailureCount += 1

  await page.evaluate(() => {
    if (document.activeElement instanceof HTMLElement) document.activeElement.blur()
  })
  await page.keyboard.press('Tab')
  const skipLinkPassed = await page.locator('.skip-link').evaluate(
    (element) => element === document.activeElement,
  )
  if (!skipLinkPassed) throw new Error('The keyboard skip link was not the first tab stop')
  await page.keyboard.press('Enter')
  await page.waitForFunction(() => document.activeElement?.id === 'main-content')
  const skipTargetVisible = await page.evaluate(() => {
    const main = document.querySelector('#main-content')
    const topbar = document.querySelector('.topbar')
    if (!main || !topbar) return false
    return main.getBoundingClientRect().top >= topbar.getBoundingClientRect().bottom - 1
  })
  if (!skipTargetVisible) throw new Error('The keyboard skip target was obscured by the sticky header')
  await page.evaluate(() => {
    if (document.activeElement instanceof HTMLElement) document.activeElement.blur()
    window.scrollTo(0, 0)
  })
  await page.waitForTimeout(20)

  await page.screenshot({ path: screenshotPath, fullPage: true, animations: 'disabled' })
  await fs.chmod(screenshotPath, OWNER_FILE_MODE)
  const screenshotStat = await fs.lstat(screenshotPath)
  if (!screenshotStat.isFile() || screenshotStat.size === 0) {
    throw new Error('A browser screenshot was not published')
  }
  counters.screenshotCount += 1
  counters.pageRuns += 1
  counters.pageEvidence.push({
    viewport: viewportName,
    route: slug,
    content_state: contentState,
    initial_content_ms: initialContentMs,
    axe_violation_count: accessibility.violations.length,
    axe_incomplete_count: accessibility.incomplete.length,
    horizontal_overflow: rootOverflow,
    page_error_count: counters.pageErrorCount - pageErrorsBefore,
    external_request_count: counters.externalRequestCount - externalRequestsBefore,
    non_get_head_request_count:
      counters.nonGetHeadRequestCount - nonGetHeadRequestsBefore,
    reduced_motion_passed: reducedMotionPassed,
    skip_link_passed: skipLinkPassed,
    screenshot: `artifacts/${path.basename(screenshotPath)}`,
  })
  const performanceOverBudget = initialContentMs > INITIAL_CONTENT_LIMIT_MS
  recordRouteDiagnostic(
    counters,
    viewportName,
    slug,
    [
      {
        category: 'accessibility',
        count:
          accessibility.violations.length +
          accessibility.incomplete.length +
          Number(rootOverflow) +
          Number(!reducedMotionPassed),
      },
      {
        category: 'browser_errors',
        count: counters.pageErrorCount - pageErrorsBefore,
      },
      {
        category: 'network_read_only',
        count:
          counters.externalRequestCount -
          externalRequestsBefore +
          (counters.nonGetHeadRequestCount - nonGetHeadRequestsBefore) +
          (counters.webSocketRequestCount - webSocketRequestsBefore) +
          (counters.serviceWorkerRegistrationCount - serviceWorkerRegistrationsBefore),
      },
      { category: 'performance', count: performanceOverBudget ? 1 : 0 },
    ],
    {
      performanceOverBudget,
      axeRuleIds: [...accessibility.violations, ...accessibility.incomplete].map(
        (result) => result.id,
      ),
    },
  )
}

function positiveIntegerAttribute(value) {
  if (value === null || !/^\d+$/.test(value)) return null
  const parsed = Number(value)
  return Number.isSafeInteger(parsed) && parsed > 0 ? parsed : null
}

export function classifyGuardedRequest(baseUrl, rawUrl, method) {
  const allowedOrigin = new URL(baseUrl).origin
  const requestUrl = new URL(rawUrl)
  const isHttp = ['http:', 'https:'].includes(requestUrl.protocol)
  const webSocket = ['ws:', 'wss:'].includes(requestUrl.protocol)
  const comparableUrl = new URL(requestUrl)
  if (webSocket) comparableUrl.protocol = requestUrl.protocol === 'ws:' ? 'http:' : 'https:'
  return {
    external: (isHttp || webSocket) && comparableUrl.origin !== allowedOrigin,
    disallowedMethod: isHttp && !['GET', 'HEAD'].includes(method.toUpperCase()),
    webSocket,
  }
}

export function acceptanceContextOptions(options = {}) {
  return { ...options, serviceWorkers: 'block' }
}

export async function installSameOriginNetworkGuard(
  context,
  baseUrl,
  onBlocked,
  onDisallowedMethod = () => {},
  onWebSocket = () => {},
) {
  await context.route('**/*', async (route) => {
    const request = route.request()
    const { external, disallowedMethod } = classifyGuardedRequest(
      baseUrl,
      request.url(),
      request.method(),
    )
    if (external) onBlocked()
    if (disallowedMethod) onDisallowedMethod()
    if (external || disallowedMethod) {
      await route.abort('blockedbyclient')
      return
    }
    await route.continue()
  })
  await context.routeWebSocket('**/*', async (webSocket) => {
    const classification = classifyGuardedRequest(baseUrl, webSocket.url(), 'GET')
    if (classification.external) onBlocked()
    onWebSocket()
    await webSocket.close({ code: 1008, reason: 'Blocked by read-only acceptance' })
  })
}

export function validateAcceptanceIdentity(payload) {
  requireCondition(
    payload && typeof payload === 'object' && !Array.isArray(payload),
    'Acceptance identity is not an object',
  )
  const keys = Object.keys(payload).sort()
  requireCondition(
    JSON.stringify(keys) === JSON.stringify(ACCEPTANCE_IDENTITY_KEYS),
    'Acceptance identity schema is invalid',
  )
  requireCondition(
    typeof payload.backend_version === 'string' && payload.backend_version.trim().length > 0,
    'Acceptance backend version is invalid',
  )
  for (const key of [
    'catalog_source_revision',
    'backend_source_sha256',
    'frontend_manifest_sha256',
    'server_nonce',
  ]) {
    requireCondition(SHA256_PATTERN.test(payload[key]), 'Acceptance identity digest is invalid')
  }
  return Object.fromEntries(ACCEPTANCE_IDENTITY_KEYS.map((key) => [key, payload[key]]))
}

export async function acceptanceIdentity(context, baseUrl) {
  const response = await context.request.get(`${baseUrl}/api/v1/acceptance/identity`, {
    maxRedirects: 0,
    timeout: 600_000,
  })
  if (!response.ok() || new URL(response.url()).origin !== new URL(baseUrl).origin) {
    throw new Error('The acceptance-read-only identity request failed')
  }
  return validateAcceptanceIdentity(await response.json())
}

async function largestConversationId(context, baseUrl) {
  const response = await context.request.get(`${baseUrl}/api/v1/conversations`, {
    params: { sort: 'messages', direction: 'desc', limit: 1 },
    maxRedirects: 0,
  })
  if (!response.ok() || new URL(response.url()).origin !== new URL(baseUrl).origin) {
    throw new Error('The largest-conversation request failed')
  }
  const payload = await response.json()
  const item = payload?.items?.[0]
  const identifier = item?.id
  const messageCount = item?.message_counts?.total
  if (
    typeof identifier !== 'string' ||
    !identifier ||
    !Number.isSafeInteger(messageCount) ||
    messageCount <= 0
  ) {
    throw new Error('The catalog has no conversation available for reader acceptance')
  }
  return { id: identifier, id_sha256: sha256Text(identifier).slice(0, 16), message_count: messageCount }
}

async function catalogFingerprint(context, baseUrl) {
  const response = await context.request.get(`${baseUrl}/api/v1/overview`, { maxRedirects: 0 })
  if (!response.ok() || new URL(response.url()).origin !== new URL(baseUrl).origin) {
    throw new Error('The catalog overview request failed')
  }
  const payload = await response.json()
  const rootName = payload?.archive?.root_name
  const content = payload?.content
  if (
    typeof rootName !== 'string' ||
    !content ||
    !['conversation_count', 'message_count', 'attachment_count', 'activity_count'].every(
      (key) => Number.isSafeInteger(content[key]) && content[key] >= 0,
    )
  ) {
    throw new Error('The catalog overview response is incomplete')
  }
  return {
    archive_identity_sha256: sha256Text(rootName).slice(0, 16),
    totals: {
      conversations: content.conversation_count,
      messages: content.message_count,
      attachments: content.attachment_count,
      activities: content.activity_count,
    },
  }
}

async function authenticate(browser, baseUrl, token) {
  const context = await browser.newContext(acceptanceContextOptions())
  let blockedRequests = 0
  let nonGetHeadRequests = 0
  let webSocketRequests = 0
  let serviceWorkerRegistrations = 0
  try {
    context.on('serviceworker', () => {
      serviceWorkerRegistrations += 1
    })
    await installSameOriginNetworkGuard(context, baseUrl, () => {
      blockedRequests += 1
    }, () => {
      nonGetHeadRequests += 1
    }, () => {
      webSocketRequests += 1
    })
    const page = await context.newPage()
    const response = await page.goto(`${baseUrl}/auth?token=${encodeURIComponent(token)}`, {
      waitUntil: 'networkidle',
      timeout: 120_000,
    })
    const finalUrl = new URL(page.url())
    if (
      !response ||
      !response.ok() ||
      finalUrl.origin !== new URL(baseUrl).origin ||
      finalUrl.pathname !== '/' ||
      blockedRequests !== 0 ||
      nonGetHeadRequests !== 0 ||
      webSocketRequests !== 0 ||
      serviceWorkerRegistrations !== 0 ||
      context.serviceWorkers().length !== 0
    ) {
      throw new Error('The dedicated startup token was not accepted')
    }
    const state = await context.storageState()
    if (!state.cookies.some((cookie) => cookie.name === SESSION_COOKIE_NAME)) {
      throw new Error('The authenticated browser session cookie is missing')
    }
    return state
  } finally {
    await context.close()
  }
}

async function waitForApiAction(page, pathname, predicate, action) {
  const [response] = await Promise.all([
    page.waitForResponse(
      (response) => {
        const url = new URL(response.url())
        return (
          url.pathname === pathname &&
          response.status() < 400 &&
          (!predicate || predicate(url.searchParams))
        )
      },
      { timeout: 120_000 },
    ),
    action(),
  ])
  const payload = await response.json()
  requireCondition(
    payload && typeof payload === 'object' && !Array.isArray(payload),
    'An interaction API response was not a JSON object',
  )
  return payload
}

function apiItems(payload) {
  requireCondition(Array.isArray(payload?.items), 'An interaction API response has no item list')
  return payload.items
}

async function assertLoadedApiItems(locator, attribute, payload) {
  await locator.waitFor({ state: 'visible', timeout: 120_000 })
  const deadline = Date.now() + 120_000
  let loaded = Number(await locator.getAttribute(attribute))
  while (loaded !== apiItems(payload).length && Date.now() < deadline) {
    await new Promise((resolve) => setTimeout(resolve, 50))
    loaded = Number(await locator.getAttribute(attribute))
  }
  requireCondition(
    Number.isSafeInteger(loaded) && loaded === apiItems(payload).length,
    'Rendered item count does not match the interaction API response',
  )
  // The count and rendered children commit together. Yield once more so same-sized page
  // replacements cannot be compared against the preceding response's children.
  await new Promise((resolve) => setTimeout(resolve, 100))
}

async function jsonProjectionAttribute(locator, attribute) {
  const raw = await locator.getAttribute(attribute)
  let projection
  try {
    projection = JSON.parse(raw ?? '')
  } catch {
    projection = null
  }
  requireCondition(Array.isArray(projection), 'A rendered acceptance projection is invalid')
  return projection
}

function sentenceCaseProjection(value) {
  if (!value) return 'Unknown'
  return String(value)
    .replace(/[._-]+/g, ' ')
    .replace(/\b\w/g, (letter) => letter.toUpperCase())
}

function compactCountProjection(value) {
  return new Intl.NumberFormat('en-US', {
    notation: 'compact',
    maximumFractionDigits: 1,
  }).format(value)
}

function searchScoreProjection(item) {
  if (item.score_kind === 'exact') return 'Exact match'
  if (!Number.isFinite(item.score)) return sentenceCaseProjection(item.score_kind)
  return `${sentenceCaseProjection(item.score_kind)} ${item.score.toFixed(item.score_kind === 'rrf' ? 4 : 3)}`
}

async function assertSearchApiItems(page, payload) {
  const items = apiItems(payload)
  requireCondition(items.length > 0, 'The search interaction returned no source-backed result')
  const results = page.locator('.search-result')
  await results.first().waitFor({ state: 'visible', timeout: 120_000 })
  const domProjection = await results.evaluateAll((elements) =>
    elements.map((element) => ({
      id: element.id.slice('search-result-'.length),
      rank: Number(element.querySelector('.search-result__rank')?.textContent?.replace(/[^0-9]/g, '')),
      score_label: element.querySelector('.search-result__meta span')?.textContent?.trim() ?? null,
      jump_href: element.querySelector('footer a[href]')?.getAttribute('href') ?? null,
    })),
  )
  reconcileCompleteProjection(
    items.map((item) => ({
      id: String(item.id),
      rank: item.rank,
      score_label: searchScoreProjection(item),
      jump_href: item.jump?.href ?? null,
    })),
    domProjection,
  )
  return 1
}

async function searchSourceJump(page, payload) {
  const index = apiItems(payload).findIndex((item) => typeof item?.jump?.href === 'string')
  requireCondition(index >= 0, 'The search interaction returned no source jump')
  const link = page
    .locator('.search-result')
    .nth(index)
    .getByRole('link', { name: /Jump to source/ })
  requireCondition(
    (await link.getAttribute('href')) === apiItems(payload)[index].jump.href,
    'A search source jump does not match the API source',
  )
  return link
}

async function assertConversationApiRows(page, payload) {
  const items = apiItems(payload)
  requireCondition(items.length > 0, 'The conversation interaction returned no source-backed rows')
  const table = page.getByRole('table', { name: 'Conversations' })
  const expected = items.map((item) => ({
    id: String(item.id),
    href: item.href,
    message_total: item.message_counts.total,
    attachment_occurrences: item.attachment_counts.occurrences,
  }))
  reconcileCompleteProjection(
    expected,
    await jsonProjectionAttribute(table, 'data-acceptance-row-projection'),
  )
  const renderedRows = await page
    .locator('[role="table"][aria-label="Conversations"] .conversation-table__row[data-index]')
    .evaluateAll((rows) =>
      rows.map((row) => {
        const cells = [...row.querySelectorAll(':scope > [role="cell"]')]
        const href = cells[0]?.querySelector('a[href]')?.getAttribute('href') ?? null
        const count = (value) => Number(String(value ?? '').replace(/[^0-9]/g, ''))
        const attachmentText = cells[6]?.querySelector('strong')?.textContent?.split('/')[0]
        return {
          index: Number(row.getAttribute('data-index')),
          value: {
            id: href ? decodeURIComponent(href.split('/').at(-1) ?? '') : null,
            href,
            message_total: count(cells[4]?.textContent),
            attachment_occurrences: count(attachmentText),
          },
        }
      }),
    )
  reconcileProjectedRows(expected, renderedRows)
  return 2
}

async function assertAppendedConversationApiRows(page, firstPayload, nextPayload) {
  const first = apiItems(firstPayload).map((item) => ({
    id: String(item.id),
    href: item.href,
    message_total: item.message_counts.total,
    attachment_occurrences: item.attachment_counts.occurrences,
  }))
  const next = apiItems(nextPayload).map((item) => ({
    id: String(item.id),
    href: item.href,
    message_total: item.message_counts.total,
    attachment_occurrences: item.attachment_counts.occurrences,
  }))
  requireCondition(next.length > 0, 'The conversation cursor response was empty')
  const loaded = await jsonProjectionAttribute(
    page.getByRole('table', { name: 'Conversations' }),
    'data-acceptance-row-projection',
  )
  reconcileCompleteProjection([...first, ...next], loaded)
  reconcileCompleteProjection(next, loaded.slice(first.length))
  return 2
}

async function assertMediaApiRows(page, payload) {
  const items = apiItems(payload)
  requireCondition(items.length > 0, 'The media interaction returned no source-backed rows')
  const domRows = await page
    .locator('[role="table"][aria-label="Media files"] .media-table__row[data-index]')
    .evaluateAll((rows, sourceJumpSelector) =>
      rows.map((row) => {
        const cells = [...row.querySelectorAll(':scope > [role="cell"]')]
        return {
          index: Number(row.getAttribute('data-index')),
          value: {
            id: row.id.slice('media-occurrence-'.length),
            occurrence_count: Number(
              cells[6]?.textContent?.replace(/^Occurrences/, '').replace(/[^0-9].*$/, ''),
            ),
            jump_href:
              cells[0]?.querySelector(sourceJumpSelector)?.getAttribute('href') ?? null,
          },
        }
      }),
      MEDIA_SOURCE_JUMP_SELECTOR,
    )
  reconcileProjectedRows(
    items.map((item) => ({
      id: String(item.id),
      occurrence_count: item.occurrence_count ?? 1,
      jump_href: item.jump?.href ?? null,
    })),
    domRows,
  )
  return 1
}

async function assertPeopleApiItems(page, payload) {
  const items = apiItems(payload)
  const domProjection = await page.locator('.person-card').evaluateAll((cards) =>
    cards.map((card) => ({
      id: card.id.slice('relationship-'.length),
      exchange_values: [...card.querySelectorAll('.person-card__metrics dd')].map((value) =>
        value.textContent?.trim(),
      ),
      conversation_hrefs: [...card.querySelectorAll('.person-conversations a[href]')].map((link) =>
        link.getAttribute('href'),
      ),
    })),
  )
  reconcileCompleteProjection(
    items.map((item) => ({
      id: String(item.id),
      exchange_values: item.direct_counts || item.counts
        ? [
            item.direct_counts?.messages ?? item.counts?.messages,
            item.direct_counts?.photos ?? item.counts?.photos,
            item.direct_counts?.videos ?? item.counts?.videos,
            item.direct_counts?.audio ?? item.counts?.audio,
          ].map(compactCountProjection)
        : [],
      conversation_hrefs: (item.conversations ?? []).map((conversation) => conversation.href),
    })),
    domProjection,
  )
  return 1
}

async function assertEventApiItems(page, payload) {
  const items = apiItems(payload)
  const domProjection = await page.locator('.event-card').evaluateAll((cards) =>
    cards.map((card) => {
      const hostLabel = card.querySelector('footer span')?.textContent?.trim() ?? ''
      return {
        id: card.id.slice('event-'.length),
        hosted: hostLabel === 'Hosted by you' ? true : hostLabel === 'Not hosted by you' ? false : null,
      }
    }),
  )
  reconcileCompleteProjection(
    items.map((item) => ({ id: String(item.id), hosted: item.hosted ?? null })),
    domProjection,
  )
  return 1
}

async function assertActivityApiRows(page, payload) {
  const items = apiItems(payload)
  if (items.length === 0) {
    requireCondition(
      (await page.locator('.activity-table__viewport tr[data-index]').count()) === 0,
      'An empty activity API response rendered source rows',
    )
    return 1
  }
  const domRows = await page
    .locator('.activity-table__viewport tr[data-index]')
    .evaluateAll((rows) =>
      rows.map((row) => {
        const cells = [...row.querySelectorAll(':scope > td')]
        return {
          index: Number(row.getAttribute('data-index')),
          value: {
            id: row.id.slice('activity-'.length),
            attachment_count: Number(cells[5]?.textContent?.replace(/[^0-9]/g, '') || 0),
            jump_href: cells[7]?.querySelector('a[href]')?.getAttribute('href') ?? null,
          },
        }
      }),
    )
  reconcileProjectedRows(
    items.map((item) => ({
      id: String(item.id),
      attachment_count: item.attachment_count,
      jump_href: item.jump?.href ?? null,
    })),
    domRows,
  )
  return 1
}

async function assertTimelineApiRows(page, payload) {
  const items = apiItems(payload)
  if (items.length === 0) {
    requireCondition(
      (await page.locator('.timeline-entry').count()) === 0,
      'An empty timeline API response rendered source rows',
    )
    return 1
  }
  const domRows = await page.locator('.timeline-list__row[data-index]').evaluateAll((rows) =>
    rows.map((row) => {
      const entry = row.querySelector('.timeline-entry')
      return {
        index: Number(row.getAttribute('data-index')),
        value: {
          id: entry?.id ?? null,
          jump_href: entry?.querySelector('a[href]')?.getAttribute('href') ?? null,
        },
      }
    }),
  )
  reconcileProjectedRows(
    items.map((item) => ({
      id: `${item.entity_type}-${item.entity_id}`,
      jump_href: item.jump?.href ?? null,
    })),
    domRows,
  )
  return 1
}

async function waitForPanelData(page, selector) {
  await page.waitForFunction(
    (panelSelector) => {
      const panel = document.querySelector(panelSelector)
      return panel && !panel.querySelector('.state-panel--loading')
    },
    selector,
    { timeout: 120_000 },
  )
}

export async function ensureDetailedFiltersVisible(form, toggleSelector) {
  const sortControl = form.getByLabel('Sort by')
  if (await sortControl.isVisible()) return false
  await form.locator(toggleSelector).click()
  await sortControl.waitFor({ state: 'visible', timeout: 120_000 })
  return true
}

async function timelineBounds(page) {
  const timeline = page.locator('.message-timeline')
  await timeline.waitFor({ state: 'visible', timeout: 120_000 })
  const rendered = positiveIntegerAttribute(await timeline.getAttribute('data-rendered-message-count'))
  const loaded = positiveIntegerAttribute(await timeline.getAttribute('data-loaded-message-count'))
  requireCondition(
    Number.isSafeInteger(rendered) &&
      Number.isSafeInteger(loaded) &&
      rendered > 0 &&
      loaded > 0 &&
      rendered <= loaded &&
      rendered <= MAX_RENDERED_MESSAGES &&
      loaded <= MAX_LOADED_MESSAGES,
    'Conversation reader bounds are invalid',
  )
  return { rendered, loaded }
}

function messageProjection(items) {
  return items.map((item) => ({
    id: String(item.id),
    timestamp_ms: item.timestamp_ms,
    kind: item.kind,
    attachment_count: item.attachments.length,
    sender_is_owner: Boolean(item.sender_is_owner),
  }))
}

async function assertMessageTimelineItems(page, items) {
  requireCondition(Array.isArray(items) && items.length > 0, 'The reader API edge is empty')
  const timeline = page.locator('.message-timeline')
  await timeline.waitFor({ state: 'visible', timeout: 120_000 })
  const expected = messageProjection(items)
  reconcileCompleteProjection(
    expected,
    await jsonProjectionAttribute(timeline, 'data-acceptance-message-projection'),
  )
  requireCondition(
    Number(await timeline.getAttribute('data-loaded-message-count')) === expected.length,
    'The reader loaded-message identities do not reconcile with the API edge',
  )
  const renderedRows = await timeline.locator('.message-timeline__item[data-index]').evaluateAll(
    (rows) =>
      rows.map((row) => {
        const message = row.querySelector('.message-row')
        const attachmentLabel = message
          ?.querySelector('.message-attachments')
          ?.getAttribute('aria-label')
        return {
          index: Number(row.getAttribute('data-index')),
          value: {
            id: message?.id.slice('message-'.length) ?? null,
            timestamp_ms: Date.parse(message?.querySelector('time')?.getAttribute('datetime') ?? ''),
            attachment_count: Number(attachmentLabel?.match(/[0-9]+/)?.[0] ?? 0),
            sender_is_owner: message?.classList.contains('message-row--owner') ?? false,
          },
        }
      }),
  )
  reconcileProjectedRows(
    expected.map((item) => ({
      id: item.id,
      timestamp_ms: item.timestamp_ms,
      attachment_count: item.attachment_count,
      sender_is_owner: item.sender_is_owner,
    })),
    renderedRows,
  )
  return 2
}

async function runReadOnlyInteractions(
  page,
  baseUrl,
  largestId,
  targets,
  counters,
  viewportName,
) {
  await auditInteraction(counters, viewportName, 'coverage_status_filter', async () => {
    await openInteractivePage(page, baseUrl, '/coverage')
    const group = page.getByRole('group', { name: 'Filter by coverage status' })
    const unavailable = group.getByRole('button', { name: 'Unavailable' })
    await unavailable.click()
    requireCondition((await unavailable.getAttribute('aria-pressed')) === 'true', 'Filter state failed')
    await page.getByRole('heading', { name: 'Facebook categories' }).waitFor()
    const rows = page.locator('.data-table--coverage tbody tr')
    requireCondition((await rows.count()) > 0, 'The unavailable coverage filter was vacuous')
    for (let index = 0; index < (await rows.count()); index += 1) {
      requireCondition(
        (await rows.nth(index).innerText()).toLowerCase().includes('unavailable'),
        'Coverage status filter left a mismatched rendered row',
      )
    }
    return interactionProof({ assertion_count: 1, nonempty_dataset_count: 1 })
  })

  await auditInteraction(counters, viewportName, 'conversation_sort_and_page', async () => {
    await openInteractivePage(page, baseUrl, '/conversations')
    const filters = page.getByRole('form', { name: 'Conversation filters' })
    await ensureDetailedFiltersVisible(filters, '.conversation-filter-toggle')
    await filters.getByLabel('Direction').selectOption('desc')
    const table = page.getByRole('table', { name: 'Conversations' })
    let reconciled = 0
    let payload
    for (const sort of ['photos', 'videos', 'audio', 'messages']) {
      await ensureDetailedFiltersVisible(filters, '.conversation-filter-toggle')
      await filters.getByLabel('Sort by').selectOption(sort)
      payload = await waitForApiAction(
        page,
        '/api/v1/conversations',
        (query) => query.get('sort') === sort && query.get('direction') === 'desc',
        () => filters.getByRole('button', { name: 'Apply filters' }).click(),
      )
      await assertLoadedApiItems(table, 'data-loaded-row-count', payload)
      reconciled += await assertConversationApiRows(page, payload)
    }
    const loadMore = page.getByRole('button', { name: 'Load more conversations' })
    let paginationExercised = 0
    if (await loadMore.isVisible().catch(() => false)) {
      const before = Number(await table.getAttribute('data-loaded-row-count'))
      const nextPayload = await waitForApiAction(
        page,
        '/api/v1/conversations',
        (query) => Boolean(query.get('cursor')),
        () => loadMore.click(),
      )
      await page.waitForFunction(
        (prior) =>
          Number(document.querySelector('[role="table"][aria-label="Conversations"]')?.getAttribute('data-loaded-row-count')) > prior,
        before,
        { timeout: 120_000 },
      )
      requireCondition(
        Number(await table.getAttribute('data-loaded-row-count')) ===
          before + apiItems(nextPayload).length,
        'Loaded conversation rows do not reconcile with the cursor response',
      )
      reconciled += await assertAppendedConversationApiRows(page, payload, nextPayload)
      paginationExercised = 1
    }
    return interactionProof({
      assertion_count: 8,
      dom_api_reconciliation_count: reconciled,
      nonempty_dataset_count: 4,
      pagination_bound_count: 1,
      pagination_exercised_count: paginationExercised,
    })
  })

  await auditInteraction(counters, viewportName, 'conversation_reader_edges_bounded', async () => {
    await openInteractivePage(
      page,
      baseUrl,
      `/conversations/${encodeURIComponent(largestId)}`,
    )
    const observed = [await timelineBounds(page)]
    const messagePath = `/api/v1/conversations/${encodeURIComponent(largestId)}/messages`
    const initialPayload = await requestJson(page.context(), baseUrl, messagePath, {
      start: 'latest',
      limit: 100,
    })
    let reconciled = await assertMessageTimelineItems(page, apiItems(initialPayload))
    const earliest = await reachReaderEdge(page, messagePath, 'earliest')
    requireCondition(
      earliest.exercised && earliest.payload,
      'The largest conversation did not exercise its earliest API edge',
    )
    observed.push(await timelineBounds(page))
    reconciled += await assertMessageTimelineItems(page, apiItems(earliest.payload))
    const loadNewer = page.getByRole('button', { name: 'Load newer now' })
    requireCondition(
      await loadNewer.isVisible().catch(() => false),
      'The largest conversation did not expose a newer cursor page',
    )
    const before = observed.at(-1).loaded
    const newerPayload = await waitForApiAction(
      page,
      messagePath,
      (query) => Boolean(query.get('cursor')),
      () => loadNewer.click(),
    )
    await page.waitForFunction(
      (prior) =>
        Number(document.querySelector('.message-timeline')?.getAttribute('data-loaded-message-count')) > prior,
      before,
      { timeout: 120_000 },
    )
    const earliestItems = apiItems(earliest.payload)
    const newerItems = apiItems(newerPayload)
    requireCondition(newerItems.length > 0, 'The newer conversation cursor response was empty')
    const earliestIds = new Set(earliestItems.map((item) => String(item.id)))
    requireCondition(
      newerItems.every((item) => !earliestIds.has(String(item.id))),
      'The newer conversation cursor response repeated an earlier API identity',
    )
    const combinedItems = [...earliestItems, ...newerItems].sort(
      (left, right) =>
        left.timestamp_ms - right.timestamp_ms || String(left.id).localeCompare(String(right.id)),
    )
    observed.push(await timelineBounds(page))
    reconciled += await assertMessageTimelineItems(page, combinedItems)
    const latest = await reachReaderEdge(page, messagePath, 'latest')
    requireCondition(
      latest.exercised && latest.payload,
      'The largest conversation did not exercise its latest API edge',
    )
    observed.push(await timelineBounds(page))
    reconciled += await assertMessageTimelineItems(page, apiItems(latest.payload))
    counters.largestRenderedMessageCount = Math.max(
      counters.largestRenderedMessageCount ?? 0,
      ...observed.map((item) => item.rendered),
    )
    counters.largestLoadedMessageCount = Math.max(
      counters.largestLoadedMessageCount ?? 0,
      ...observed.map((item) => item.loaded),
    )
    return interactionProof({
      assertion_count: 8,
      dom_api_reconciliation_count: reconciled,
      nonempty_dataset_count: 1,
      pagination_bound_count: 2,
      pagination_exercised_count: Number(earliest.exercised) + Number(latest.exercised),
    })
  })

  await auditInteraction(counters, viewportName, 'conversation_types_source_bound', async () => {
    let reconciled = 0
    let nonempty = 0
    for (const conversationType of ['direct', 'group']) {
      const target = targets.conversationTypes?.[conversationType]
      requireCondition(target?.id && target?.type === conversationType, 'A conversation type target is invalid')
      await openInteractivePage(
        page,
        baseUrl,
        `/conversations/${encodeURIComponent(target.id)}`,
      )
      requireCondition(
        (await page.getByRole('heading', { level: 1 }).innerText()) ===
          (target.title || 'Untitled conversation'),
        'The conversation reader title does not match its API source',
      )
      requireCondition(
        (await page.locator('.reader-heading__title .eyebrow').innerText()).toLowerCase() ===
          `${conversationType} conversation`,
        'The conversation reader type does not match its API source',
      )
      await timelineBounds(page)
      reconciled += 1
      nonempty += 1
    }
    return interactionProof({
      assertion_count: 4,
      dom_api_reconciliation_count: reconciled,
      nonempty_dataset_count: nonempty,
      conversation_type_count: 2,
    })
  })

  await auditInteraction(counters, viewportName, 'media_modes_and_sort', async () => {
    await openInteractivePage(page, baseUrl, '/media')
    const countMode = page.getByRole('group', { name: 'Count mode' })
    await page.getByRole('group', { name: 'Display mode' }).getByRole('button', { name: /Table/ }).click()
    const form = page.getByRole('form', { name: 'Media filters' })
    await ensureDetailedFiltersVisible(form, '.media-filter-toggle')
    await form.getByLabel('Direction').selectOption('asc')
    let reconciled = 0
    for (const sort of ['conversation', 'date', 'size']) {
      await form.getByLabel('Sort by').selectOption(sort)
      const payload = await waitForApiAction(
        page,
        '/api/v1/media',
        (query) =>
          query.get('view') === 'occurrences' &&
          query.get('sort') === sort &&
          query.get('direction') === 'asc',
        () => form.getByRole('button', { name: 'Apply filters' }).click(),
      )
      await assertLoadedApiItems(
        page.getByRole('table', { name: 'Media files' }),
        'data-loaded-row-count',
        payload,
      )
      reconciled += await assertMediaApiRows(page, payload)
    }
    const uniquePayload = await waitForApiAction(
      page,
      '/api/v1/media',
      (query) =>
        query.get('view') === 'unique' &&
        query.get('sort') === 'size' &&
        query.get('direction') === 'asc',
      () => countMode.getByRole('button', { name: 'Unique files' }).click(),
    )
    await assertLoadedApiItems(
      page.getByRole('table', { name: 'Media files' }),
      'data-loaded-row-count',
      uniquePayload,
    )
    reconciled += await assertMediaApiRows(page, uniquePayload)
    return interactionProof({
      assertion_count: 6,
      dom_api_reconciliation_count: reconciled,
      nonempty_dataset_count: 4,
    })
  })

  for (const [name, target, linkName] of [
    ['media_message_exact_jump', targets.messageMedia, /Jump to message|Open conversation/],
    ['media_activity_exact_jump', targets.activityMedia, 'Jump to activity'],
  ]) {
    await auditInteraction(counters, viewportName, name, async () => {
      requireCondition(target?.id, 'A required media source target is unavailable')
      await openInteractivePage(
        page,
        baseUrl,
        `/media?occurrence=${encodeURIComponent(target.id)}`,
      )
      const focused = page.locator(`#media-occurrence-${target.id}`)
      await focused.waitFor({ state: 'visible', timeout: 120_000 })
      const jump = focused.getByRole('link', { name: linkName })
      requireCondition(
        (await jump.getAttribute('href')) === target.jump?.href,
        'A focused media jump does not match its API source',
      )
      await followExactJump(page, baseUrl, jump)
      return interactionProof({
        assertion_count: 2,
        dom_api_reconciliation_count: 1,
        nonempty_dataset_count: 1,
        source_jump_count: 1,
      })
    })
  }

  await auditInteraction(
    counters,
    viewportName,
    'media_representative_kinds_and_jumps',
    async () => {
      let reconciled = 0
      let jumps = 0
      for (const mediaType of ['photo', 'video', 'audio', 'document']) {
        const target = targets.representativeMedia?.[mediaType]
        requireCondition(
          target?.id && target?.type === mediaType && target?.jump?.href,
          'A representative media target is invalid',
        )
        await openInteractivePage(
          page,
          baseUrl,
          `/media?occurrence=${encodeURIComponent(target.id)}`,
        )
        const focused = page.locator(`#media-occurrence-${target.id}`)
        await focused.waitFor({ state: 'visible', timeout: 120_000 })
        const jump = focused.getByRole('link', {
          name: /Jump to message|Open conversation|Jump to activity/,
        })
        requireCondition(
          (await jump.getAttribute('href')) === target.jump.href,
          'A representative media jump does not match its API source',
        )
        reconciled += 1
        await followExactJump(page, baseUrl, jump)
        jumps += 1
      }
      return interactionProof({
        assertion_count: 8,
        dom_api_reconciliation_count: reconciled,
        nonempty_dataset_count: 4,
        source_jump_count: jumps,
        representative_kind_count: 4,
      })
    },
  )

  await auditInteraction(counters, viewportName, 'keyword_search_exact_jump', async () => {
    requireCondition(targets.query, 'A private keyword-search probe could not be derived')
    await openInteractivePage(page, baseUrl, '/search')
    await page.getByLabel('What do you want to find?').fill(targets.query)
    const payload = await waitForApiAction(
      page,
      '/api/v1/search',
      (query) => query.get('mode') === 'keyword',
      () => page.getByRole('button', { name: 'Search archive' }).click(),
    )
    const reconciled = await assertSearchApiItems(page, payload)
    await followExactJump(
      page,
      baseUrl,
      await searchSourceJump(page, payload),
    )
    return interactionProof({
      assertion_count: 2,
      dom_api_reconciliation_count: reconciled,
      nonempty_dataset_count: 1,
      source_jump_count: 1,
    })
  })

  for (const [interaction, mode] of [
    ['exact_search_exact_jump', 'exact'],
    ['semantic_search_exact_jump', 'semantic'],
    ['hybrid_search_exact_jump', 'hybrid'],
  ]) {
    await auditInteraction(counters, viewportName, interaction, async () => {
      requireCondition(targets.query, 'A private search probe could not be derived')
      await openInteractivePage(page, baseUrl, '/search')
      const modeControl = page.getByRole('radio', {
        name:
          mode === 'exact'
            ? /Exact literal/
            : mode === 'semantic'
              ? /Semantic/
              : /Hybrid/,
      })
      requireCondition(
        await modeControl.isEnabled(),
        mode === 'exact'
          ? 'Exact literal search is unavailable'
          : 'The active semantic index is unavailable',
      )
      await modeControl.locator('..').click()
      requireCondition(await modeControl.isChecked(), 'The requested search mode was not selected')
      await page.getByLabel('What do you want to find?').fill(targets.query)
      const payload = await waitForApiAction(
        page,
        '/api/v1/search',
        (query) => query.get('mode') === mode,
        () => page.getByRole('button', { name: 'Search archive' }).click(),
      )
      const reconciled = await assertSearchApiItems(page, payload)
      await followExactJump(
        page,
        baseUrl,
        await searchSourceJump(page, payload),
      )
      return interactionProof({
        assertion_count: 2,
        dom_api_reconciliation_count: reconciled,
        nonempty_dataset_count: 1,
        source_jump_count: 1,
      })
    })
  }

  await auditInteraction(
    counters,
    viewportName,
    'semantic_argument_search_exact_jump',
    async () => {
      await openInteractivePage(page, baseUrl, '/search')
      const semantic = page.getByRole('radio', { name: /Semantic/ })
      requireCondition(await semantic.isEnabled(), 'The active semantic index is unavailable')
      await semantic.locator('..').click()
      requireCondition(await semantic.isChecked(), 'Semantic search was not selected')
      await page.getByLabel('What do you want to find?').fill('argument')
      const payload = await waitForApiAction(
        page,
        '/api/v1/search',
        (query) => query.get('mode') === 'semantic' && query.get('q') === 'argument',
        () => page.getByRole('button', { name: 'Search archive' }).click(),
      )
      const reconciled = await assertSearchApiItems(page, payload)
      await followExactJump(
        page,
        baseUrl,
        await searchSourceJump(page, payload),
      )
      return interactionProof({
        assertion_count: 2,
        dom_api_reconciliation_count: reconciled,
        nonempty_dataset_count: 1,
        source_jump_count: 1,
      })
    },
  )

  await auditInteraction(counters, viewportName, 'enrichment_exact_source_jump', async () => {
    await openInteractivePage(page, baseUrl, '/enrichment')
    await page.getByRole('tab', { name: 'Audio transcripts' }).click()
    const transcriptPanel = page.locator('#enrichment-panel-transcripts')
    await transcriptPanel.waitFor({ state: 'visible', timeout: 120_000 })
    await waitForPanelData(page, '#enrichment-panel-transcripts')
    const segmentDetails = transcriptPanel.getByText('Timestamped segments', { exact: true }).first()
    if (await segmentDetails.isVisible().catch(() => false)) await segmentDetails.click()
    let jump = transcriptPanel.getByRole('link', { name: /^Jump to source/ }).first()
    if (!(await jump.isVisible().catch(() => false))) {
      await page.getByRole('tab', { name: 'Extracted text' }).click()
      const documentPanel = page.locator('#enrichment-panel-documents')
      await documentPanel.waitFor({ state: 'visible', timeout: 120_000 })
      await waitForPanelData(page, '#enrichment-panel-documents')
      jump = documentPanel.getByRole('link', { name: /^Jump to source/ }).first()
    }
    await jump.waitFor({ state: 'visible', timeout: 120_000 })
    await followExactJump(page, baseUrl, jump)
    return interactionProof({ assertion_count: 1, source_jump_count: 1 })
  })

  await auditInteraction(counters, viewportName, 'people_views_sort_and_page', async () => {
    await openInteractivePage(page, baseUrl, '/people')
    const views = page.getByRole('navigation', { name: 'People view' })
    let reconciled = 0
    let nonempty = 0
    let sourceBoundEmpty = 0
    for (const [view, buttonName] of [
      ['friends', 'Friends'],
      ['requests', 'Requests'],
      ['followers', 'Followers'],
      ['facebook-blocks', 'Facebook blocks'],
      ['messenger-blocks', 'Messenger blocks'],
      ['contacts', 'Messenger contacts'],
    ]) {
      const viewPayload = await waitForApiAction(
        page,
        '/api/v1/people',
        (query) => query.get('view') === view,
        () => views.getByRole('button', { name: buttonName }).click(),
      )
      await assertLoadedApiItems(
        page.getByRole('navigation', { name: 'People results pages' }),
        'data-loaded-item-count',
        viewPayload,
      )
      reconciled += await assertPeopleApiItems(page, viewPayload)
      const datasetProof = sourceBoundAvailability(viewPayload)
      nonempty += datasetProof.nonempty_dataset_count
      sourceBoundEmpty += datasetProof.source_bound_empty_dataset_count
    }
    const payload = await waitForApiAction(
      page,
      '/api/v1/people',
      (query) => query.get('view') === 'contacts' && query.get('sort') === 'photos',
      () => page.getByLabel('Sort by').selectOption('photos'),
    )
    await assertLoadedApiItems(
      page.getByRole('navigation', { name: 'People results pages' }),
      'data-loaded-item-count',
      payload,
    )
    reconciled += await assertPeopleApiItems(page, payload)
    nonempty += Number(apiItems(payload).length > 0)
    const pagination = await cursorPageRoundTrip(page, 'People results', {
      pathname: '/api/v1/people',
      predicate: (query) => query.get('view') === 'contacts' && query.get('sort') === 'photos',
      assertNextPayload: async (nextPayload) => {
        await assertLoadedApiItems(
          page.getByRole('navigation', { name: 'People results pages' }),
          'data-loaded-item-count',
          nextPayload,
        )
        await assertPeopleApiItems(page, nextPayload)
      },
    })
    return interactionProof({
      assertion_count: 8,
      dom_api_reconciliation_count: reconciled,
      nonempty_dataset_count: nonempty,
      source_bound_empty_dataset_count: sourceBoundEmpty,
      pagination_bound_count: pagination.bound,
      pagination_exercised_count: pagination.exercised,
    })
  })

  await auditInteraction(counters, viewportName, 'events_filter_sort_and_page', async () => {
    await openInteractivePage(page, baseUrl, '/events')
    const form = page.locator('form.event-filters')
    await form.getByLabel('Hosted by you').selectOption('yes')
    await form.getByLabel('Sort').selectOption('title')
    await form.getByLabel('Direction').selectOption('asc')
    const payload = await waitForApiAction(
      page,
      '/api/v1/events',
      (query) =>
        query.get('hosted') === 'true' &&
        query.get('sort') === 'title' &&
        query.get('direction') === 'asc',
      () => form.getByRole('button', { name: 'Apply filters' }).click(),
    )
    await assertLoadedApiItems(
      page.getByRole('navigation', { name: 'Event results pages' }),
      'data-loaded-item-count',
      payload,
    )
    const reconciled = await assertEventApiItems(page, payload)
    const datasetProof = sourceBoundAvailability(payload)
    const pagination = await cursorPageRoundTrip(page, 'Event results', {
      pathname: '/api/v1/events',
      predicate: (query) =>
        query.get('hosted') === 'true' &&
        query.get('sort') === 'title' &&
        query.get('direction') === 'asc',
      assertNextPayload: async (nextPayload) => {
        await assertLoadedApiItems(
          page.getByRole('navigation', { name: 'Event results pages' }),
          'data-loaded-item-count',
          nextPayload,
        )
        await assertEventApiItems(page, nextPayload)
      },
    })
    return interactionProof({
      assertion_count: 2,
      dom_api_reconciliation_count: reconciled,
      ...datasetProof,
      pagination_bound_count: pagination.bound,
      pagination_exercised_count: pagination.exercised,
    })
  })

  await auditInteraction(counters, viewportName, 'activity_filter_sort_and_page', async () => {
    await openInteractivePage(page, baseUrl, '/activity')
    const postsPayload = await waitForApiAction(
      page,
      '/api/v1/activity/views/posts',
      null,
      () => page.getByRole('navigation', { name: 'Activity category' }).getByRole('button', { name: 'Posts' }).click(),
    )
    await assertLoadedApiItems(
      page.getByRole('navigation', { name: 'Activity results pages' }),
      'data-loaded-item-count',
      postsPayload,
    )
    let reconciled = await assertActivityApiRows(page, postsPayload)
    requireCondition(apiItems(postsPayload).length > 0, 'The source-backed posts view is empty')
    const form = page.locator('form.filter-panel')
    await form.getByLabel('Attachments').selectOption('yes')
    await form.getByLabel('Post/comment context').selectOption('others_wall')
    await form.getByLabel('Sort').selectOption('title')
    const payload = await waitForApiAction(
      page,
      '/api/v1/activity/views/posts',
      (query) =>
        query.get('has_attachments') === 'true' &&
        query.get('post_context') === 'others_wall' &&
        query.get('sort') === 'title',
      () => form.getByRole('button', { name: 'Apply filters' }).click(),
    )
    await assertLoadedApiItems(
      page.getByRole('navigation', { name: 'Activity results pages' }),
      'data-loaded-item-count',
      payload,
    )
    reconciled += await assertActivityApiRows(page, payload)
    const filteredDatasetProof = sourceBoundAvailability(payload, {
      coverageKey: 'post_context_coverage',
      expectedStatus: 'unknown',
    })
    const pagination = await cursorPageRoundTrip(page, 'Activity results', {
      pathname: '/api/v1/activity/views/posts',
      predicate: (query) =>
        query.get('has_attachments') === 'true' &&
        query.get('post_context') === 'others_wall' &&
        query.get('sort') === 'title',
      assertNextPayload: async (nextPayload) => {
        await assertLoadedApiItems(
          page.getByRole('navigation', { name: 'Activity results pages' }),
          'data-loaded-item-count',
          nextPayload,
        )
        await assertActivityApiRows(page, nextPayload)
      },
    })
    return interactionProof({
      assertion_count: 3,
      dom_api_reconciliation_count: reconciled,
      nonempty_dataset_count: 1 + filteredDatasetProof.nonempty_dataset_count,
      source_bound_empty_dataset_count: filteredDatasetProof.source_bound_empty_dataset_count,
      pagination_bound_count: pagination.bound,
      pagination_exercised_count: pagination.exercised,
    })
  })

  await auditInteraction(
    counters,
    viewportName,
    'activity_post_comment_source_jumps',
    async () => {
      let reconciled = 0
      let jumps = 0
      let commentsPagination
      for (const [view, buttonName] of [
        ['posts', 'Posts'],
        ['comments', 'Comments'],
      ]) {
        await openInteractivePage(page, baseUrl, '/activity')
        const payload = await waitForApiAction(
          page,
          `/api/v1/activity/views/${view}`,
          null,
          () =>
            page
              .getByRole('navigation', { name: 'Activity category' })
              .getByRole('button', { name: buttonName })
              .click(),
        )
        await assertLoadedApiItems(
          page.getByRole('navigation', { name: 'Activity results pages' }),
          'data-loaded-item-count',
          payload,
        )
        requireCondition(apiItems(payload).length > 0, 'A required activity source view is empty')
        reconciled += await assertActivityApiRows(page, payload)
        if (view === 'comments') {
          commentsPagination = await cursorPageRoundTrip(page, 'Activity results', {
            pathname: '/api/v1/activity/views/comments',
            assertNextPayload: async (nextPayload) => {
              await assertLoadedApiItems(
                page.getByRole('navigation', { name: 'Activity results pages' }),
                'data-loaded-item-count',
                nextPayload,
              )
              await assertActivityApiRows(page, nextPayload)
            },
            assertPreviousPage: async () => {
              await assertLoadedApiItems(
                page.getByRole('navigation', { name: 'Activity results pages' }),
                'data-loaded-item-count',
                payload,
              )
              await assertActivityApiRows(page, payload)
            },
          })
        }
        const firstRow = page.locator('.activity-table__viewport tr[data-index="0"]')
        const jump = firstRow.getByRole('link', { name: 'Open' })
        requireCondition(
          (await jump.getAttribute('href')) === apiItems(payload)[0].jump?.href,
          'An activity source jump does not match the API source',
        )
        await followExactJump(page, baseUrl, jump)
        jumps += 1
      }
      return interactionProof({
        assertion_count: 4,
        dom_api_reconciliation_count: reconciled,
        nonempty_dataset_count: 2,
        pagination_bound_count: commentsPagination?.bound ?? 0,
        pagination_exercised_count: commentsPagination?.exercised ?? 0,
        source_jump_count: jumps,
        representative_kind_count: 2,
      })
    },
  )

  await auditInteraction(counters, viewportName, 'timeline_mode_filter_and_page', async () => {
    await openInteractivePage(page, baseUrl, '/timeline')
    const modes = page.getByRole('navigation', { name: 'Timeline mode' })
    await waitForApiAction(
      page,
      '/api/v1/timeline/on-this-day',
      null,
      () => modes.getByRole('button', { name: 'On this day' }).click(),
    )
    const onThisDayPayload = await waitForApiAction(
      page,
      '/api/v1/timeline/on-this-day',
      (query) => query.get('entity_type') === 'message',
      () => page.getByLabel('Record type').selectOption('message'),
    )
    await assertLoadedApiItems(
      page.getByRole('navigation', { name: 'On this day results pages' }),
      'data-loaded-item-count',
      onThisDayPayload,
    )
    let reconciled = await assertTimelineApiRows(page, onThisDayPayload)
    const onThisDayPagination = await cursorPageRoundTrip(page, 'On this day results', {
      pathname: '/api/v1/timeline/on-this-day',
      predicate: (query) => query.get('entity_type') === 'message',
      assertNextPayload: async (nextPayload) => {
        await assertLoadedApiItems(
          page.getByRole('navigation', { name: 'On this day results pages' }),
          'data-loaded-item-count',
          nextPayload,
        )
        await assertTimelineApiRows(page, nextPayload)
      },
    })
    const timelinePayload = await waitForApiAction(
      page,
      '/api/v1/timeline',
      null,
      () => modes.getByRole('button', { name: 'Full timeline' }).click(),
    )
    await assertLoadedApiItems(
      page.getByRole('navigation', { name: 'Timeline results pages' }),
      'data-loaded-item-count',
      timelinePayload,
    )
    requireCondition(apiItems(timelinePayload).length > 0, 'The full timeline is empty')
    reconciled += await assertTimelineApiRows(page, timelinePayload)
    const timelinePagination = await cursorPageRoundTrip(page, 'Timeline results', {
      pathname: '/api/v1/timeline',
      assertNextPayload: async (nextPayload) => {
        await assertLoadedApiItems(
          page.getByRole('navigation', { name: 'Timeline results pages' }),
          'data-loaded-item-count',
          nextPayload,
        )
        await assertTimelineApiRows(page, nextPayload)
      },
    })
    return interactionProof({
      assertion_count: 4,
      dom_api_reconciliation_count: reconciled,
      nonempty_dataset_count: 1 + Number(apiItems(onThisDayPayload).length > 0),
      pagination_bound_count: onThisDayPagination.bound + timelinePagination.bound,
      pagination_exercised_count:
        onThisDayPagination.exercised + timelinePagination.exercised,
    })
  })

  await auditInteraction(counters, viewportName, 'analytics_bucket_queries', async () => {
    await openInteractivePage(page, baseUrl, '/analytics')
    const messageBucket = page.locator('.analytics-controls').getByLabel('Bucket')
    const messagePayload = await waitForApiAction(
      page,
      '/api/v1/analytics/messages',
      (query) => query.get('bucket') === 'year',
      () => messageBucket.selectOption('year'),
    )
    requireCondition(
      messagePayload.bucket === 'year' && (await messageBucket.inputValue()) === 'year',
      'Rendered message bucket does not match the Analytics API response',
    )
    const messagePager = page.getByRole('navigation', { name: 'Message volume periods pages' })
    await ensureAnalyticsDrilldownVisible(page, messagePager, 'messages', 'message activity')
    await page.waitForFunction(
      (expected) => {
        const heading = [...document.querySelectorAll('h2')].find(
          (element) => element.textContent?.trim() === 'Message volume over time',
        )
        const text = heading?.parentElement?.querySelector('p')?.textContent?.replaceAll(',', '') ?? ''
        const values = (text.match(/[0-9]+/g) ?? []).map(Number)
        return JSON.stringify(values.slice(0, 3)) === JSON.stringify(expected)
      },
      [
        messagePayload.summary.messages,
        messagePayload.summary.observed_periods,
        Math.round(messagePayload.summary.average_per_observed_period),
      ],
      { timeout: 120_000 },
    )
    await waitForApiAction(
      page,
      '/api/v1/analytics/messages',
      (query) => query.get('bucket') === 'day' && !query.get('cursor'),
      () => messageBucket.selectOption('day'),
    )
    const messagePagination = await cursorPageRoundTrip(page, 'Message volume periods', {
      pathname: '/api/v1/analytics/messages',
      predicate: (query) => query.get('bucket') === 'day',
      assertNextPayload: async (nextPayload) => {
        requireCondition(
          Number(nextPayload?.page?.returned ?? 0) <= Number(nextPayload?.page?.limit ?? 0),
          'Analytics message page exceeds its declared period limit',
        )
      },
    })
    const nlpFilters = page.getByLabel('Language-model trend filters')
    await ensureAnalyticsDrilldownVisible(
      page,
      nlpFilters,
      'language',
      'language-model signals',
    )
    const nlpBucket = nlpFilters.getByLabel('Time bucket')
    const nlpPayload = await waitForApiAction(
      page,
      '/api/v1/analytics/nlp-trends',
      (query) => query.get('bucket') === 'year',
      () => nlpBucket.selectOption('year'),
    )
    requireCondition(
      nlpPayload.bucket === 'year' &&
        (await nlpBucket.inputValue()) === 'year',
      'Rendered NLP bucket does not match the Analytics API response',
    )
    await page.waitForFunction(
      (expected) => {
        const text = document
          .querySelector('.analytics-nlp-trends .section-header p')
          ?.textContent?.replaceAll(',', '')
        return Number(text?.match(/[0-9]+/)?.[0]) === expected
      },
      nlpPayload.source_sample_count,
      { timeout: 120_000 },
    )
    // The bucket changes above refetch more panels than the two projections this
    // story reconciles directly. Do not navigate into the deletion guard while
    // any of those truthfully advertised queued/active reads are still settling.
    await waitForAnalyticsRouteSettled(page)
    await page.getByRole('heading', { name: 'Archive analytics' }).waitFor()
    return interactionProof({
      assertion_count: 5,
      dom_api_reconciliation_count: 2,
      pagination_bound_count: messagePagination.bound,
      pagination_exercised_count: messagePagination.exercised,
    })
  })

  await auditInteraction(counters, viewportName, 'deletion_preview_guard', async () => {
    await openInteractivePage(page, baseUrl, '/deletion')
    const preview = page.getByRole('button', { name: 'Preview selection' })
    requireCondition(await preview.isDisabled(), 'Unbounded deletion preview was enabled')
    const text = page.getByLabel('Text to match')
    await text.fill('codex-acceptance-nonmatching-sentinel')
    requireCondition(await preview.isEnabled(), 'A narrowed deletion preview was not enabled')
    await text.fill('')
    requireCondition(await preview.isDisabled(), 'Deletion preview guard did not reset')
    return interactionProof({ assertion_count: 3 })
  })
}

async function runViewport(
  browser,
  storageState,
  baseUrl,
  largestId,
  interactionTargets,
  artifacts,
  viewportEntry,
) {
  const [viewportName, viewport] = viewportEntry
  const counters = {
    pageRuns: 0,
    initialContentMs: [],
    axeViolationCount: 0,
    axeIncompleteCount: 0,
    horizontalOverflowCount: 0,
    pageErrorCount: 0,
    externalRequestCount: 0,
    nonGetHeadRequestCount: 0,
    webSocketRequestCount: 0,
    serviceWorkerRegistrationCount: 0,
    reducedMotionFailureCount: 0,
    screenshotCount: 0,
    largestRenderedMessageCount: null,
    largestLoadedMessageCount: null,
    traceReady: false,
    pageEvidence: [],
    interactionEvidence: [],
    routeDiagnostics: [],
  }
  const context = await browser.newContext(
    acceptanceContextOptions({ storageState, viewport, reducedMotion: 'reduce' }),
  )
  context.on('serviceworker', () => {
    counters.serviceWorkerRegistrationCount += 1
  })
  await installSameOriginNetworkGuard(context, baseUrl, () => {
    counters.externalRequestCount += 1
  }, () => {
    counters.nonGetHeadRequestCount += 1
  }, () => {
    counters.webSocketRequestCount += 1
  })
  context.on('requestfailed', (request) => {
    const expectedBrowserAbort = isExpectedBrowserAbort(baseUrl, {
      url: request.url(),
      method: request.method(),
      resourceType: request.resourceType(),
      errorText: request.failure()?.errorText,
    })
    if (!expectedBrowserAbort) counters.pageErrorCount += 1
  })
  const page = await context.newPage()
  page.on('console', (message) => {
    if (message.type() === 'error') counters.pageErrorCount += 1
  })
  page.on('pageerror', () => {
    counters.pageErrorCount += 1
  })
  page.on('response', (response) => {
    if (response.url().includes('/api/') && response.status() >= 400) {
      counters.pageErrorCount += 1
    }
  })
  const tracePath = path.join(artifacts, `${viewportName}-trace.zip`)
  await context.tracing.start({ screenshots: true, snapshots: true, sources: true })
  try {
    for (const [slug, route, expectedHeading] of topLevelRoutes) {
      await auditPage(
        page,
        baseUrl,
        slug,
        route,
        expectedHeading,
        path.join(artifacts, `${viewportName}-${slug}.png`),
        viewportName,
        counters,
      )
    }
    await auditPage(
      page,
      baseUrl,
      'largest-conversation',
      `/conversations/${encodeURIComponent(largestId)}`,
      null,
      path.join(artifacts, `${viewportName}-largest-conversation.png`),
      viewportName,
      counters,
    )
    const timeline = page.locator('.message-timeline')
    await timeline.waitFor({ state: 'visible', timeout: 120_000 })
    counters.largestRenderedMessageCount = positiveIntegerAttribute(
      await timeline.getAttribute('data-rendered-message-count'),
    )
    counters.largestLoadedMessageCount = positiveIntegerAttribute(
      await timeline.getAttribute('data-loaded-message-count'),
    )
    await runReadOnlyInteractions(
      page,
      baseUrl,
      largestId,
      interactionTargets,
      counters,
      viewportName,
    )
  } finally {
    let traceStopped = true
    try {
      await context.tracing.stop({ path: tracePath })
    } catch {
      traceStopped = false
    }
    counters.serviceWorkerRegistrationCount = Math.max(
      counters.serviceWorkerRegistrationCount,
      context.serviceWorkers().length,
    )
    await context.close()
    await securePrivateTree(artifacts)
    const traceStat = await lstatOrNull(tracePath)
    counters.traceReady = Boolean(
      traceStopped &&
        traceStat &&
        !traceStat.isSymbolicLink() &&
        traceStat.isFile() &&
        traceStat.size > 0,
    )
  }
  return counters
}

export function summarizeViewportResults(results) {
  const renderedCounts = results.map((item) => item.largestRenderedMessageCount)
  const loadedCounts = results.map((item) => item.largestLoadedMessageCount)
  const validReaderCounts =
    results.length === viewports.length &&
    results.every(
      (item) =>
        Number.isSafeInteger(item.largestRenderedMessageCount) &&
        item.largestRenderedMessageCount > 0 &&
        Number.isSafeInteger(item.largestLoadedMessageCount) &&
        item.largestLoadedMessageCount > 0 &&
        item.largestRenderedMessageCount <= item.largestLoadedMessageCount,
    )
  const allInitialContent = results.flatMap((item) => item.initialContentMs)
  const pageEvidence = results.flatMap((item) => item.pageEvidence ?? [])
  const expectedPageKeys = new Set(
    viewports.flatMap(([viewport]) =>
      [...topLevelRoutes.map(([slug]) => slug), 'largest-conversation'].map(
        (route) => `${viewport}:${route}`,
      ),
    ),
  )
  const actualPageKeys = new Set(
    pageEvidence.map((item) => `${item.viewport}:${item.route}`),
  )
  const pageEvidenceComplete =
    pageEvidence.length === expectedPageKeys.size &&
    actualPageKeys.size === expectedPageKeys.size &&
    [...expectedPageKeys].every((key) => actualPageKeys.has(key))
  const interactionEvidence = results.flatMap((item) => item.interactionEvidence ?? [])
  const expectedInteractionKeys = new Set(
    viewports.flatMap(([viewport]) =>
      interactionNames.map((name) => `${viewport}:${name}`),
    ),
  )
  const actualInteractionKeys = new Set(
    interactionEvidence.map((item) => `${item.viewport}:${item.interaction}`),
  )
  const interactionProofComplete = interactionEvidence.every((item) => {
    try {
      validateInteractionProof(item.interaction, item.proof)
      return true
    } catch {
      return false
    }
  })
  const interactionProofTotals = interactionProof(
    Object.fromEntries(
      interactionProofKeys.map((key) => [
        key,
        interactionEvidence.reduce(
          (total, item) =>
            total +
            (Number.isSafeInteger(item?.proof?.[key]) && item.proof[key] >= 0
              ? item.proof[key]
              : 0),
          0,
        ),
      ]),
    ),
  )
  const interactionEvidenceComplete =
    interactionEvidence.length === expectedInteractionKeys.size &&
    actualInteractionKeys.size === expectedInteractionKeys.size &&
    [...expectedInteractionKeys].every((key) => actualInteractionKeys.has(key)) &&
    interactionProofComplete &&
    interactionEvidence.every(
      (item) =>
        Object.keys(item).length === 5 &&
        ['viewport', 'interaction', 'passed', 'duration_ms', 'proof'].every((key) =>
          Object.hasOwn(item, key),
        ) &&
        item.passed === true &&
        Number.isFinite(item.duration_ms) &&
        item.duration_ms >= 0,
    )
  return {
    page_runs: results.reduce((total, item) => total + item.pageRuns, 0),
    screenshot_count: results.reduce((total, item) => total + item.screenshotCount, 0),
    initial_content_p95_ms: percentile95(allInitialContent),
    axe_violation_count: results.reduce((total, item) => total + item.axeViolationCount, 0),
    axe_incomplete_count: results.reduce((total, item) => total + item.axeIncompleteCount, 0),
    horizontal_overflow_count: results.reduce(
      (total, item) => total + item.horizontalOverflowCount,
      0,
    ),
    page_error_count: results.reduce((total, item) => total + item.pageErrorCount, 0),
    external_request_count: results.reduce(
      (total, item) => total + item.externalRequestCount,
      0,
    ),
    non_get_head_request_count: results.reduce(
      (total, item) => total + item.nonGetHeadRequestCount,
      0,
    ),
    websocket_request_count: results.reduce(
      (total, item) => total + item.webSocketRequestCount,
      0,
    ),
    service_worker_registration_count: results.reduce(
      (total, item) => total + item.serviceWorkerRegistrationCount,
      0,
    ),
    reduced_motion_failure_count: results.reduce(
      (total, item) => total + item.reducedMotionFailureCount,
      0,
    ),
    largest_rendered_message_count: validReaderCounts ? Math.max(...renderedCounts) : null,
    largest_loaded_message_count: validReaderCounts ? Math.max(...loadedCounts) : null,
    reader_counts_valid: validReaderCounts,
    screenshots_complete: results.every(
      (item) => item.screenshotCount === topLevelRoutes.length + 1,
    ),
    traces_complete: results.every((item) => item.traceReady === true),
    page_evidence_complete: pageEvidenceComplete,
    interaction_runs: interactionEvidence.length,
    interaction_pass_count: interactionEvidence.filter((item) => item.passed === true).length,
    interaction_evidence_complete: interactionEvidenceComplete,
    interaction_proof_complete: interactionProofComplete,
    interaction_proof_totals: interactionProofTotals,
    desktop_and_mobile:
      results.length === viewports.length &&
      results.every((item) => item.pageRuns === topLevelRoutes.length + 1),
  }
}

export function summaryPasses(summary) {
  return (
    summary.page_runs === (topLevelRoutes.length + 1) * viewports.length &&
    summary.screenshot_count === (topLevelRoutes.length + 1) * viewports.length &&
    summary.initial_content_p95_ms !== null &&
    summary.initial_content_p95_ms <= INITIAL_CONTENT_LIMIT_MS &&
    summary.axe_violation_count === 0 &&
    summary.axe_incomplete_count === 0 &&
    summary.horizontal_overflow_count === 0 &&
    summary.page_error_count === 0 &&
    summary.external_request_count === 0 &&
    summary.non_get_head_request_count === 0 &&
    summary.websocket_request_count === 0 &&
    summary.service_worker_registration_count === 0 &&
    summary.reduced_motion_failure_count === 0 &&
    Number.isInteger(summary.largest_rendered_message_count) &&
    summary.largest_rendered_message_count > 0 &&
    summary.largest_rendered_message_count <= MAX_RENDERED_MESSAGES &&
    Number.isInteger(summary.largest_loaded_message_count) &&
    summary.largest_loaded_message_count > 0 &&
    summary.largest_loaded_message_count <= MAX_LOADED_MESSAGES &&
    summary.largest_rendered_message_count <= summary.largest_loaded_message_count &&
    summary.reader_counts_valid === true &&
    summary.screenshots_complete === true &&
    summary.traces_complete === true &&
    summary.page_evidence_complete === true &&
    summary.interaction_runs === interactionNames.length * viewports.length &&
    summary.interaction_pass_count === interactionNames.length * viewports.length &&
    summary.interaction_evidence_complete === true &&
    summary.interaction_proof_complete === true &&
    validateSummaryProofTotals(summary.interaction_proof_totals) &&
    summary.desktop_and_mobile === true
  )
}

function acceptanceSafetyStatement() {
  return {
    aggregate_report_only: true,
    screenshots_and_traces_separate: true,
    owner_only_artifacts: true,
    source_text_emitted_in_report: false,
    token_emitted: false,
    external_network_blocked: true,
    websockets_blocked: true,
    service_workers_blocked: true,
    acceptance_read_only_server: true,
    only_get_head_requests: true,
    analytics_cache_reads_writes_disabled: true,
    mutating_user_actions_performed: false,
  }
}

function completedGateFailureSafetyStatement(summary) {
  return {
    aggregate_report_only: true,
    screenshots_and_traces_separate: true,
    owner_only_artifacts: true,
    source_text_emitted_in_report: false,
    token_emitted: false,
    external_network_guard_installed: true,
    websocket_guard_installed: true,
    service_worker_guard_installed: true,
    external_network_attempts_absent: summary.external_request_count === 0,
    websocket_attempts_absent: summary.websocket_request_count === 0,
    service_worker_registrations_absent:
      summary.service_worker_registration_count === 0,
    acceptance_read_only_server: true,
    only_get_head_requests: summary.non_get_head_request_count === 0,
    analytics_cache_reads_writes_disabled: true,
    mutating_user_actions_performed: false,
  }
}

function nonnegativeInteger(value) {
  return Number.isSafeInteger(value) && value >= 0 ? value : 0
}

function aggregateSummaryRecord(summary) {
  const scalarKeys = [
    'page_runs',
    'screenshot_count',
    'axe_violation_count',
    'axe_incomplete_count',
    'horizontal_overflow_count',
    'page_error_count',
    'external_request_count',
    'non_get_head_request_count',
    'websocket_request_count',
    'service_worker_registration_count',
    'reduced_motion_failure_count',
    'largest_rendered_message_count',
    'largest_loaded_message_count',
    'reader_counts_valid',
    'screenshots_complete',
    'traces_complete',
    'page_evidence_complete',
    'interaction_runs',
    'interaction_pass_count',
    'interaction_evidence_complete',
    'interaction_proof_complete',
    'desktop_and_mobile',
  ]
  return {
    ...Object.fromEntries(scalarKeys.map((key) => [key, summary[key]])),
    interaction_proof_totals: Object.fromEntries(
      interactionProofKeys.map((key) => [key, summary.interaction_proof_totals?.[key]]),
    ),
  }
}

const AXE_RULE_ID_PATTERN = /^[a-z][a-z0-9-]{0,99}$/

export function completedGateRouteDiagnostics(rawDiagnostics) {
  const allowedViewports = viewports.map(([viewport]) => viewport)
  const allowedRoutes = [...topLevelRoutes.map(([slug]) => slug), 'largest-conversation']
  const allowedViewportSet = new Set(allowedViewports)
  const allowedRouteSet = new Set(allowedRoutes)
  const allowedCategorySet = new Set(COMPLETED_GATE_FAILURE_CATEGORIES)
  const aggregates = new Map()

  for (const raw of Array.isArray(rawDiagnostics) ? rawDiagnostics : []) {
    if (
      !raw ||
      typeof raw !== 'object' ||
      !allowedViewportSet.has(raw.viewport) ||
      !allowedRouteSet.has(raw.route)
    ) {
      continue
    }
    const key = `${raw.viewport}:${raw.route}`
    if (!aggregates.has(key)) {
      aggregates.set(key, {
        viewport: raw.viewport,
        route: raw.route,
        categoryCounts: Object.fromEntries(
          COMPLETED_GATE_FAILURE_CATEGORIES.map((category) => [category, 0]),
        ),
        performanceOverBudget: false,
        axeRuleIds: new Set(),
      })
    }
    const aggregate = aggregates.get(key)
    for (const item of Array.isArray(raw.failure_category_counts)
      ? raw.failure_category_counts
      : []) {
      if (
        item &&
        typeof item === 'object' &&
        allowedCategorySet.has(item.category) &&
        Number.isSafeInteger(item.count) &&
        item.count > 0
      ) {
        aggregate.categoryCounts[item.category] += item.count
      }
    }
    aggregate.performanceOverBudget ||= raw.performance_over_budget === true
    for (const ruleId of Array.isArray(raw.axe_rule_ids) ? raw.axe_rule_ids : []) {
      if (typeof ruleId === 'string' && AXE_RULE_ID_PATTERN.test(ruleId)) {
        aggregate.axeRuleIds.add(ruleId)
      }
    }
  }

  const viewportOrder = new Map(allowedViewports.map((viewport, index) => [viewport, index]))
  const routeOrder = new Map(allowedRoutes.map((route, index) => [route, index]))
  const routes = [...aggregates.values()]
    .map((aggregate) => ({
      viewport: aggregate.viewport,
      route: aggregate.route,
      failure_category_counts: COMPLETED_GATE_FAILURE_CATEGORIES.flatMap((category) =>
        aggregate.categoryCounts[category] > 0
          ? [{ category, count: aggregate.categoryCounts[category] }]
          : [],
      ),
      performance_over_budget: aggregate.performanceOverBudget,
      axe_rule_ids: [...aggregate.axeRuleIds].sort(),
    }))
    .filter(
      (route) =>
        route.failure_category_counts.length > 0 ||
        route.performance_over_budget ||
        route.axe_rule_ids.length > 0,
    )
    .sort(
      (left, right) =>
        viewportOrder.get(left.viewport) - viewportOrder.get(right.viewport) ||
        routeOrder.get(left.route) - routeOrder.get(right.route),
    )
  const performanceOverBudgetCount = routes.filter(
    (route) => route.performance_over_budget,
  ).length
  return {
    performance_over_budget: performanceOverBudgetCount > 0,
    performance_over_budget_count: performanceOverBudgetCount,
    routes,
  }
}

export function completedGateFailureCategoryCounts(summary, artifactComplete) {
  const counts = Object.fromEntries(
    COMPLETED_GATE_FAILURE_CATEGORIES.map((category) => [category, 0]),
  )
  counts.accessibility =
    nonnegativeInteger(summary.axe_violation_count) +
    nonnegativeInteger(summary.axe_incomplete_count) +
    nonnegativeInteger(summary.horizontal_overflow_count) +
    nonnegativeInteger(summary.reduced_motion_failure_count)
  counts.browser_errors = nonnegativeInteger(summary.page_error_count)
  counts.network_read_only =
    nonnegativeInteger(summary.external_request_count) +
    nonnegativeInteger(summary.non_get_head_request_count) +
    nonnegativeInteger(summary.websocket_request_count) +
    nonnegativeInteger(summary.service_worker_registration_count)
  counts.performance =
    Number.isFinite(summary.initial_content_p95_ms) &&
    summary.initial_content_p95_ms <= INITIAL_CONTENT_LIMIT_MS
      ? 0
      : 1

  if (
    !Number.isSafeInteger(summary.largest_rendered_message_count) ||
    summary.largest_rendered_message_count <= 0 ||
    summary.largest_rendered_message_count > MAX_RENDERED_MESSAGES
  ) {
    counts.reader_bounds += 1
  }
  if (
    !Number.isSafeInteger(summary.largest_loaded_message_count) ||
    summary.largest_loaded_message_count <= 0 ||
    summary.largest_loaded_message_count > MAX_LOADED_MESSAGES
  ) {
    counts.reader_bounds += 1
  }
  if (
    Number.isSafeInteger(summary.largest_rendered_message_count) &&
    Number.isSafeInteger(summary.largest_loaded_message_count) &&
    summary.largest_rendered_message_count > summary.largest_loaded_message_count
  ) {
    counts.reader_bounds += 1
  }
  if (summary.reader_counts_valid !== true) counts.reader_bounds += 1

  const expectedPageRuns = (topLevelRoutes.length + 1) * viewports.length
  counts.evidence_completeness = [
    summary.page_runs === expectedPageRuns,
    summary.screenshot_count === expectedPageRuns,
    summary.screenshots_complete === true,
    summary.traces_complete === true,
    summary.page_evidence_complete === true,
    summary.desktop_and_mobile === true,
    artifactComplete === true,
  ].filter((complete) => !complete).length

  const expectedInteractionRuns = interactionNames.length * viewports.length
  counts.interaction_evidence = Math.max(
    0,
    expectedInteractionRuns - nonnegativeInteger(summary.interaction_pass_count),
  )
  if (
    counts.interaction_evidence === 0 &&
    (summary.interaction_runs !== expectedInteractionRuns ||
      summary.interaction_evidence_complete !== true ||
      summary.interaction_proof_complete !== true ||
      !validateSummaryProofTotals(summary.interaction_proof_totals))
  ) {
    counts.interaction_evidence = 1
  }

  return COMPLETED_GATE_FAILURE_CATEGORIES.flatMap((category) =>
    counts[category] > 0 ? [{ category, count: counts[category] }] : [],
  )
}

export function completedGateFailureReport(
  summary,
  artifactComplete,
  generatedAt,
  rawRouteDiagnostics = [],
) {
  const failureCategoryCounts = completedGateFailureCategoryCounts(
    summary,
    artifactComplete,
  )
  requireCondition(
    failureCategoryCounts.length > 0,
    'A failed completed gate must have an aggregate failure category',
  )
  return {
    schema_version: REPORT_SCHEMA_VERSION,
    report_kind: 'completed_gate_failure',
    generated_at: generatedAt,
    summary: aggregateSummaryRecord(summary),
    failure_category_counts: failureCategoryCounts,
    route_diagnostics: completedGateRouteDiagnostics(rawRouteDiagnostics),
    gate: { passed: false },
    safety: completedGateFailureSafetyStatement(summary),
  }
}

async function main() {
  const options = parseArguments(process.argv.slice(2))
  const baseUrl = validateLoopbackBaseUrl(options.baseUrl)
  let token = await readStartupToken(options.tokenFile)
  const reportPath = outputPath(options.output)
  await ensurePrivateDirectory(path.dirname(reportPath))
  if (await lstatOrNull(reportPath)) throw new Error('Refusing to overwrite prior UI evidence')
  const artifacts = path.join(path.dirname(reportPath), 'artifacts')
  if (await lstatOrNull(artifacts)) throw new Error('Refusing to overwrite prior UI artifacts')
  await fs.mkdir(artifacts, { mode: OWNER_DIRECTORY_MODE })

  const browser = await chromium.launch({ headless: true })
  let results
  let catalog
  let identity
  let largest
  let interactionTargets
  try {
    const storageState = await authenticate(browser, baseUrl, token)
    token = ''
    const discoveryContext = await browser.newContext(acceptanceContextOptions({ storageState }))
    let discoveryExternalRequests = 0
    let discoveryNonGetHeadRequests = 0
    let discoveryWebSocketRequests = 0
    let discoveryServiceWorkerRegistrations = 0
    try {
      discoveryContext.on('serviceworker', () => {
        discoveryServiceWorkerRegistrations += 1
      })
      await installSameOriginNetworkGuard(discoveryContext, baseUrl, () => {
        discoveryExternalRequests += 1
      }, () => {
        discoveryNonGetHeadRequests += 1
      }, () => {
        discoveryWebSocketRequests += 1
      })
      identity = await acceptanceIdentity(discoveryContext, baseUrl)
      catalog = await catalogFingerprint(discoveryContext, baseUrl)
      largest = await largestConversationId(discoveryContext, baseUrl)
      interactionTargets = await discoverInteractionTargets(
        discoveryContext,
        baseUrl,
        largest.id,
      )
      if (
        discoveryExternalRequests !== 0 ||
        discoveryNonGetHeadRequests !== 0 ||
        discoveryWebSocketRequests !== 0 ||
        discoveryServiceWorkerRegistrations !== 0 ||
        discoveryContext.serviceWorkers().length !== 0
      ) {
        throw new Error('Acceptance discovery attempted a blocked request')
      }
    } finally {
      await discoveryContext.close()
    }
    results = []
    for (const viewportEntry of viewports) {
      results.push(
        await runViewport(
          browser,
          storageState,
          baseUrl,
          largest.id,
          interactionTargets,
          artifacts,
          viewportEntry,
        ),
      )
    }
    const finalDiscoveryContext = await browser.newContext(
      acceptanceContextOptions({ storageState }),
    )
    let finalExternalRequests = 0
    let finalNonGetHeadRequests = 0
    let finalWebSocketRequests = 0
    let finalServiceWorkerRegistrations = 0
    try {
      finalDiscoveryContext.on('serviceworker', () => {
        finalServiceWorkerRegistrations += 1
      })
      await installSameOriginNetworkGuard(finalDiscoveryContext, baseUrl, () => {
        finalExternalRequests += 1
      }, () => {
        finalNonGetHeadRequests += 1
      }, () => {
        finalWebSocketRequests += 1
      })
      const finalCatalog = await catalogFingerprint(finalDiscoveryContext, baseUrl)
      const finalLargest = await largestConversationId(finalDiscoveryContext, baseUrl)
      const finalIdentity = await acceptanceIdentity(finalDiscoveryContext, baseUrl)
      if (
        JSON.stringify(finalCatalog) !== JSON.stringify(catalog) ||
        JSON.stringify(finalLargest) !== JSON.stringify(largest) ||
        JSON.stringify(finalIdentity) !== JSON.stringify(identity) ||
        finalExternalRequests !== 0 ||
        finalNonGetHeadRequests !== 0 ||
        finalWebSocketRequests !== 0 ||
        finalServiceWorkerRegistrations !== 0 ||
        finalDiscoveryContext.serviceWorkers().length !== 0
      ) {
        throw new Error('The catalog or acceptance-server identity changed during browser acceptance')
      }
    } finally {
      await finalDiscoveryContext.close()
    }
  } finally {
    token = ''
    await browser.close()
    await securePrivateTree(artifacts)
  }

  const summary = summarizeViewportResults(results)
  const artifactsComplete = await artifactInventoryComplete(artifacts)
  const summaryPassed = summaryPasses(summary)
  if (!summaryPassed || !artifactsComplete) {
    const routeDiagnostics = results.flatMap((item) => item.routeDiagnostics ?? [])
    await writePrivateJson(
      reportPath,
      completedGateFailureReport(
        summary,
        artifactsComplete,
        new Date().toISOString(),
        routeDiagnostics,
      ),
    )
    await securePrivateTree(path.dirname(reportPath))
    process.stdout.write(
      `${JSON.stringify({ report: reportPath, passed: false, page_runs: summary.page_runs })}\n`,
    )
    return 1
  }

  const artifactManifest = await buildArtifactManifest(artifacts)
  const pageEvidence = results.flatMap((item) => item.pageEvidence)
  const interactionEvidence = results.flatMap((item) => item.interactionEvidence)
  const runnerSha256 = createHash('sha256')
    .update(await fs.readFile(fileURLToPath(import.meta.url)))
    .digest('hex')

  await writePrivateJson(reportPath, {
    schema_version: REPORT_SCHEMA_VERSION,
    generated_at: new Date().toISOString(),
    runner_sha256: runnerSha256,
    catalog: {
      ...catalog,
      acceptance_identity: identity,
      largest_conversation: {
        id_sha256: largest.id_sha256,
        message_count: largest.message_count,
      },
    },
    summary,
    pages: pageEvidence,
    interactions: interactionEvidence,
    artifacts: artifactManifest,
    gate: { passed: true },
    safety: acceptanceSafetyStatement(),
  })
  await securePrivateTree(path.dirname(reportPath))
  process.stdout.write(
    `${JSON.stringify({ report: reportPath, passed: true, page_runs: summary.page_runs })}\n`,
  )
  return 0
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main()
    .then((exitCode) => {
      process.exitCode = exitCode
    })
    .catch(() => {
      process.stderr.write('Real UI acceptance failed before safe report publication.\n')
      process.exitCode = 2
    })
}
