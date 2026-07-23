import assert from 'node:assert/strict'
import { spawnSync } from 'node:child_process'
import fs from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import test from 'node:test'
import { fileURLToPath } from 'node:url'

import {
  activityMediaDiscoveryParameters,
  acceptanceContextOptions,
  acceptanceIdentity,
  artifactInventoryComplete,
  assertRouteIdentity,
  buildArtifactManifest,
  classifyGuardedRequest,
  COMPLETED_GATE_FAILURE_CATEGORIES,
  completedGateFailureCategoryCounts,
  completedGateFailureReport,
  completedGateRouteDiagnostics,
  ensureAnalyticsDrilldownVisible,
  waitForAnalyticsRouteSettled,
  ensureDetailedFiltersVisible,
  ensurePrivateDirectory,
  interactionProof,
  interactionProofKeys,
  interactionNames,
  installSameOriginNetworkGuard,
  isExpectedBrowserAbort,
  outputPath,
  parseArguments,
  minimumInteractionProof,
  MEDIA_SOURCE_JUMP_SELECTOR,
  PAGINATION_LANDMARK_SELECTOR,
  readerEdgeRequestRequired,
  reachReaderEdge,
  reconcileCompleteProjection,
  reconcileProjectedRows,
  routeSettleTimeout,
  readStartupToken,
  reviewRoot,
  sourceBoundAvailability,
  summarizeViewportResults,
  summaryPasses,
  topLevelRoutes,
  validateInteractionProof,
  validateSummaryProofTotals,
  validateAcceptanceIdentity,
  validateLoopbackBaseUrl,
  writePrivateJson,
} from './real-ui-acceptance.mjs'

test('analytics aggregate settlement has a separate bounded production-scale timeout', () => {
  assert.equal(routeSettleTimeout('/analytics'), 300_000)
  assert.equal(routeSettleTimeout('/analytics?bucket=month'), 300_000)
  assert.equal(routeSettleTimeout('/conversations'), 120_000)
})

test('compact analytics interaction controls are revealed before runner actions', async () => {
  let visible = false
  let summaryClicks = 0
  const control = {
    isVisible: async () => visible,
    waitFor: async (options) => {
      assert.deepEqual(options, { state: 'visible' })
      assert.equal(visible, true)
    },
  }
  const drilldown = {
    count: async () => 1,
    locator: (selector) => {
      assert.equal(selector, 'summary')
      return { click: async () => { summaryClicks += 1; visible = true } }
    },
  }
  const page = {
    locator: (selector) => {
      assert.equal(
        selector,
        'details.analytics-drilldown[data-analytics-section="fixture-section"]',
      )
      return drilldown
    },
  }

  assert.equal(
    await ensureAnalyticsDrilldownVisible(
      page,
      control,
      'fixture-section',
      'fixture',
    ),
    true,
  )
  assert.equal(summaryClicks, 1)
  assert.equal(
    await ensureAnalyticsDrilldownVisible(
      page,
      control,
      'fixture-section',
      'fixture',
    ),
    false,
  )
  assert.equal(summaryClicks, 1)
})

test('analytics interaction settlement drains route reads before navigation', async () => {
  const calls = []
  const route = {
    waitFor: async (options) => {
      calls.push(['route-visible', options])
    },
    getAttribute: async (name) => {
      assert.equal(name, 'data-analytics-route-state')
      calls.push(['route-state'])
      return 'settled'
    },
  }
  const page = {
    locator: (selector) => {
      assert.equal(selector, '.analytics-page')
      return route
    },
    waitForFunction: async (_callback, argument, options) => {
      assert.equal(argument, undefined)
      calls.push(['state-settled', options])
    },
    waitForLoadState: async (state, options) => {
      calls.push(['network', state, options])
    },
    waitForTimeout: async (milliseconds) => {
      calls.push(['stability', milliseconds])
    },
  }

  await waitForAnalyticsRouteSettled(page)
  assert.deepEqual(calls, [
    ['route-visible', { state: 'visible', timeout: 300_000 }],
    ['state-settled', { timeout: 300_000 }],
    ['network', 'networkidle', { timeout: 300_000 }],
    ['stability', 100],
    ['route-state'],
  ])
})

test('top-level route identity rejects the correct pathname with the wrong H1', async () => {
  assert.deepEqual(topLevelRoutes, [
    ['home', '/', 'Archive control center'],
    ['coverage', '/coverage', 'Data coverage'],
    ['integrity', '/integrity', 'Integrity'],
    ['jobs', '/jobs', 'Jobs'],
    ['settings', '/settings', 'Settings'],
    ['conversations', '/conversations', 'Conversations'],
    ['media', '/media', 'Media'],
    ['starred', '/starred', 'Starred'],
    ['search', '/search', 'Search the archive'],
    ['enrichment', '/enrichment', 'Transcripts and language signals'],
    ['people', '/people', 'People'],
    ['events', '/events', 'Events'],
    ['activity', '/activity', 'Posts, comments & Facebook activity'],
    ['timeline', '/timeline', 'Timeline'],
    ['support', '/support', 'Support conversations'],
    ['analytics', '/analytics', 'Archive analytics'],
    ['deletion', '/deletion', 'Deletion and quarantine'],
  ])

  const pageWithHeading = (headingText) => ({
    url: () => 'http://127.0.0.1:8765/settings',
    getByRole: (role, options) => {
      assert.equal(role, 'heading')
      assert.deepEqual(options, { level: 1, name: 'Settings', exact: true })
      const matches = headingText === options.name
      return {
        count: async () => Number(matches),
        isVisible: async () => matches,
      }
    },
  })

  await assert.doesNotReject(
    assertRouteIdentity(
      pageWithHeading('Settings'),
      'http://127.0.0.1:8765',
      '/settings',
      'Settings',
    ),
  )
  await assert.rejects(
    assertRouteIdentity(
      pageWithHeading('Archive control center'),
      'http://127.0.0.1:8765',
      '/settings',
      'Settings',
    ),
    /wrong level-one heading/,
  )
})

test('only narrowly scoped browser-cancelled local requests are expected failures', () => {
  const baseUrl = 'http://127.0.0.1:8765'
  const digest = 'a'.repeat(64)
  const expected = {
    url: `${baseUrl}/api/v1/blobs/${digest}`,
    method: 'GET',
    resourceType: 'image',
    errorText: 'net::ERR_ABORTED',
  }
  assert.equal(isExpectedBrowserAbort(baseUrl, expected), true)
  assert.equal(
    isExpectedBrowserAbort(baseUrl, {
      ...expected,
      url: `${baseUrl}/api/v1/derivatives/${digest}`,
      resourceType: 'media',
    }),
    true,
  )
  assert.equal(
    isExpectedBrowserAbort(baseUrl, {
      ...expected,
      url: `${baseUrl}/api/v1/health`,
      resourceType: 'fetch',
    }),
    true,
  )

  for (const invalid of [
    { ...expected, url: `https://example.com/api/v1/blobs/${digest}` },
    { ...expected, method: 'POST' },
    { ...expected, resourceType: 'fetch' },
    { ...expected, errorText: 'net::ERR_FAILED' },
    { ...expected, url: `${baseUrl}/api/v1/blobs/not-a-digest` },
    { ...expected, url: `${baseUrl}/api/v1/search/${digest}` },
    { ...expected, url: `${baseUrl}/api/v1/health`, resourceType: 'image' },
    { ...expected, url: `${baseUrl}/api/v1/health`, resourceType: 'fetch', errorText: 'net::ERR_FAILED' },
    { ...expected, url: `${baseUrl}/api/v1/overview`, resourceType: 'fetch' },
  ]) {
    assert.equal(isExpectedBrowserAbort(baseUrl, invalid), false)
  }
})

test('network guard permits only same-origin GET and HEAD requests', () => {
  const baseUrl = 'http://127.0.0.1:8765'
  assert.deepEqual(
    classifyGuardedRequest(baseUrl, `${baseUrl}/api/v1/overview`, 'GET'),
    { external: false, disallowedMethod: false, webSocket: false },
  )
  assert.deepEqual(
    classifyGuardedRequest(baseUrl, `${baseUrl}/assets/app.js`, 'HEAD'),
    { external: false, disallowedMethod: false, webSocket: false },
  )
  assert.deepEqual(
    classifyGuardedRequest(baseUrl, `${baseUrl}/api/v1/deletions/preview`, 'POST'),
    { external: false, disallowedMethod: true, webSocket: false },
  )
  assert.deepEqual(
    classifyGuardedRequest(baseUrl, 'https://example.com/private', 'GET'),
    { external: true, disallowedMethod: false, webSocket: false },
  )
  assert.deepEqual(
    classifyGuardedRequest(baseUrl, 'ws://127.0.0.1:8765/private', 'GET'),
    { external: false, disallowedMethod: false, webSocket: true },
  )
  assert.deepEqual(
    classifyGuardedRequest(baseUrl, 'wss://example.com/private', 'GET'),
    { external: true, disallowedMethod: false, webSocket: true },
  )
})

test('acceptance contexts always block service workers', () => {
  assert.deepEqual(
    acceptanceContextOptions({ serviceWorkers: 'allow', viewport: { width: 390, height: 844 } }),
    { serviceWorkers: 'block', viewport: { width: 390, height: 844 } },
  )
})

test('network guard closes and records every WebSocket without connecting upstream', async () => {
  let routePattern
  let webSocketPattern
  let webSocketHandler
  let blocked = 0
  let webSockets = 0
  const context = {
    route: async (pattern) => {
      routePattern = pattern
    },
    routeWebSocket: async (pattern, handler) => {
      webSocketPattern = pattern
      webSocketHandler = handler
    },
  }
  await installSameOriginNetworkGuard(
    context,
    'http://127.0.0.1:8765',
    () => {
      blocked += 1
    },
    () => {},
    () => {
      webSockets += 1
    },
  )
  assert.equal(routePattern, '**/*')
  assert.equal(webSocketPattern, '**/*')
  let closeOptions
  await webSocketHandler({
    url: () => 'wss://example.com/private',
    close: async (options) => {
      closeOptions = options
    },
  })
  assert.equal(blocked, 1)
  assert.equal(webSockets, 1)
  assert.deepEqual(closeOptions, {
    code: 1008,
    reason: 'Blocked by read-only acceptance',
  })
})

test('source-bound empty proof requires the expected status and explicit nonzero-history warning', () => {
  assert.deepEqual(
    sourceBoundAvailability({
      items: [],
      coverage: { status: 'unavailable', absence_is_not_zero_history: true },
    }),
    { nonempty_dataset_count: 0, source_bound_empty_dataset_count: 1 },
  )
  assert.deepEqual(
    sourceBoundAvailability(
      {
        items: [],
        post_context_coverage: { status: 'unknown', absence_is_not_zero_history: true },
      },
      { coverageKey: 'post_context_coverage', expectedStatus: 'unknown' },
    ),
    { nonempty_dataset_count: 0, source_bound_empty_dataset_count: 1 },
  )
  assert.deepEqual(sourceBoundAvailability({ items: [{ id: 'source-row' }] }), {
    nonempty_dataset_count: 1,
    source_bound_empty_dataset_count: 0,
  })
  for (const coverage of [
    { status: 'unavailable', absence_is_not_zero_history: false },
    { status: 'available', absence_is_not_zero_history: true },
    { status: 'unavailable', absence_is_not_zero_history: 1 },
  ]) {
    assert.throws(
      () => sourceBoundAvailability({ items: [], coverage }),
      /expected explicit unavailable coverage state/,
    )
  }
})

test('acceptance identity requires the exact server-bound schema', () => {
  const digest = 'a'.repeat(64)
  const identity = {
    catalog_source_revision: digest,
    backend_version: '0.1.0',
    backend_source_sha256: digest,
    frontend_manifest_sha256: digest,
    server_nonce: digest,
  }
  assert.deepEqual(validateAcceptanceIdentity(identity), {
    backend_source_sha256: digest,
    backend_version: '0.1.0',
    catalog_source_revision: digest,
    frontend_manifest_sha256: digest,
    server_nonce: digest,
  })
  assert.throws(
    () => validateAcceptanceIdentity({ ...identity, unexpected: true }),
    /schema is invalid/,
  )
  assert.throws(
    () => validateAcceptanceIdentity({ ...identity, server_nonce: 'not-a-digest' }),
    /digest is invalid/,
  )
  const missing = { ...identity }
  delete missing.frontend_manifest_sha256
  assert.throws(() => validateAcceptanceIdentity(missing), /schema is invalid/)
})

test('acceptance identity alone receives the production-scale request timeout', async () => {
  const digest = 'b'.repeat(64)
  let requestOptions
  const payload = {
    catalog_source_revision: digest,
    backend_version: '0.1.0',
    backend_source_sha256: digest,
    frontend_manifest_sha256: digest,
    server_nonce: digest,
  }
  const result = await acceptanceIdentity(
    {
      request: {
        get: async (_url, options) => {
          requestOptions = options
          return {
            ok: () => true,
            url: () => 'http://127.0.0.1:8765/api/v1/acceptance/identity',
            json: async () => payload,
          }
        },
      },
    },
    'http://127.0.0.1:8765',
  )
  assert.equal(requestOptions.timeout, 600_000)
  assert.equal(requestOptions.maxRedirects, 0)
  assert.deepEqual(result, validateAcceptanceIdentity(payload))
})

test('detailed filters preserve desktop-visible controls and expand hidden mobile controls', async () => {
  let desktopToggleClicks = 0
  const desktopForm = {
    getByLabel: () => ({ isVisible: async () => true }),
    locator: () => ({ click: async () => { desktopToggleClicks += 1 } }),
  }
  assert.equal(
    await ensureDetailedFiltersVisible(desktopForm, '.conversation-filter-toggle'),
    false,
  )
  assert.equal(desktopToggleClicks, 0)

  let mobileToggleClicks = 0
  let mobileWaits = 0
  const mobileForm = {
    getByLabel: () => ({
      isVisible: async () => false,
      waitFor: async ({ state }) => {
        assert.equal(state, 'visible')
        mobileWaits += 1
      },
    }),
    locator: (selector) => ({
      click: async () => {
        assert.equal(selector, '.media-filter-toggle')
        mobileToggleClicks += 1
      },
    }),
  }
  assert.equal(await ensureDetailedFiltersVisible(mobileForm, '.media-filter-toggle'), true)
  assert.equal(await ensureDetailedFiltersVisible(mobileForm, '.media-filter-toggle'), true)
  assert.equal(mobileToggleClicks, 2)
  assert.equal(mobileWaits, 2)
})

test('conversation reader requests only edges not already represented in the DOM', () => {
  assert.equal(readerEdgeRequestRequired('earliest', true, false), false)
  assert.equal(readerEdgeRequestRequired('earliest', false, true), true)
  assert.equal(readerEdgeRequestRequired('latest', false, true), false)
  assert.equal(readerEdgeRequestRequired('latest', true, false), true)
  assert.throws(
    () => readerEdgeRequestRequired('middle', false, false),
    /Unsupported conversation-reader edge/,
  )
})

test('conversation reader accepts a cached edge transition and fetches authoritative proof', async () => {
  let endVisible = false
  let fallbackRequests = 0
  const page = {
    getByText: (text) => ({
      isVisible: async () => text === 'End of conversation' && endVisible,
      waitFor: async ({ state }) => {
        assert.equal(state, 'visible')
        assert.equal(text, 'End of conversation')
        assert.equal(endVisible, true)
      },
    }),
    getByRole: (role, options) => ({
      click: async () => {
        assert.equal(role, 'button')
        assert.deepEqual(options, { name: 'Latest' })
        endVisible = true
      },
    }),
    on: () => undefined,
    off: () => undefined,
    waitForTimeout: async () => undefined,
    url: () => 'http://127.0.0.1:8765/conversations/example',
    context: () => ({
      request: {
        get: async (url, options) => {
          fallbackRequests += 1
          assert.equal(url, 'http://127.0.0.1:8765/api/v1/conversations/example/messages')
          assert.deepEqual(options.params, { start: 'latest', limit: 100 })
          return {
            ok: () => true,
            url: () => url,
            json: async () => ({ items: [{ id: 'authoritative-message' }] }),
          }
        },
      },
    }),
  }

  const edge = await reachReaderEdge(
    page,
    '/api/v1/conversations/example/messages',
    'latest',
  )
  assert.equal(edge.exercised, true)
  assert.deepEqual(edge.payload, { items: [{ id: 'authoritative-message' }] })
  assert.equal(fallbackRequests, 1)
})

test('production interaction probes use source-specific links and role-aware pagination', () => {
  assert.deepEqual(activityMediaDiscoveryParameters(undefined), {
    view: 'occurrences',
    sort: 'date',
    direction: 'desc',
    limit: 250,
    cursor: undefined,
  })
  assert.deepEqual(activityMediaDiscoveryParameters('next-page'), {
    view: 'occurrences',
    sort: 'date',
    direction: 'desc',
    limit: 250,
    cursor: 'next-page',
  })
  assert.equal(MEDIA_SOURCE_JUMP_SELECTOR, 'a.media-jump')
  assert.match(PAGINATION_LANDMARK_SELECTOR, /\[role="navigation"\]/)
})

test('interaction proof schemas and per-story minima fail closed', () => {
  assert.deepEqual(Object.keys(interactionProof()), interactionProofKeys)
  for (const name of interactionNames) {
    assert.deepEqual(validateInteractionProof(name, minimumInteractionProof(name)), minimumInteractionProof(name))
  }

  const conversationProof = minimumInteractionProof('conversation_sort_and_page')
  assert.throws(
    () =>
      validateInteractionProof('conversation_sort_and_page', {
        ...conversationProof,
        pagination_exercised_count: 0,
      }),
    /minimum was not met/,
  )
  assert.throws(
    () =>
      validateInteractionProof('conversation_sort_and_page', {
        ...conversationProof,
        unexpected_private_detail: 1,
      }),
    /schema is invalid/,
  )
  assert.throws(
    () =>
      validateInteractionProof('conversation_sort_and_page', {
        ...conversationProof,
        assertion_count: 101,
      }),
    /aggregate-only bound/,
  )
  const missingMetric = { ...conversationProof }
  delete missingMetric.source_jump_count
  assert.throws(
    () => validateInteractionProof('conversation_sort_and_page', missingMetric),
    /schema is invalid/,
  )
  assert.throws(
    () =>
      validateInteractionProof('events_filter_sort_and_page',
        interactionProof({ assertion_count: 2, dom_api_reconciliation_count: 1, pagination_bound_count: 1 }),
      ),
    /neither nonempty nor source-bound/,
  )
})

test('rendered projections reconcile exact API identity, order, and values', () => {
  const apiProjection = [
    { id: 'first', total: 20 },
    { id: 'second', total: 10 },
    { id: 'third', total: 5 },
  ]
  assert.equal(
    reconcileProjectedRows(apiProjection, [
      { index: 0, value: { id: 'first', total: 20 } },
      { index: 2, value: { id: 'third', total: 5 } },
    ]),
    2,
  )
  assert.equal(reconcileCompleteProjection(apiProjection, structuredClone(apiProjection)), 3)
  assert.equal(reconcileCompleteProjection([], []), 0)
  assert.throws(
    () =>
      reconcileProjectedRows(apiProjection, [
        { index: 1, value: { id: 'second', total: 10 } },
        { index: 0, value: { id: 'first', total: 20 } },
      ]),
    /invalid order or identity/,
  )
  assert.throws(
    () => reconcileProjectedRows(apiProjection, [{ index: 0, value: { id: 'first', total: 19 } }]),
    /does not match/,
  )
  assert.throws(
    () => reconcileCompleteProjection(apiProjection, [...apiProjection].reverse()),
    /complete API identity/,
  )
})

test('arguments and startup tokens fail closed without accepting plaintext token arguments', async () => {
  const fixtureRoot = await fs.mkdtemp(path.join(os.tmpdir(), 'meta-real-ui-token-'))
  try {
    const tokenFile = path.join(fixtureRoot, 'token')
    const token = 'owner-only-dedicated-token-value'
    await fs.writeFile(tokenFile, `${token}\n`, { mode: 0o600 })
    assert.equal(await readStartupToken(tokenFile), token)

    const tokenLink = path.join(fixtureRoot, 'token-link')
    await fs.symlink(tokenFile, tokenLink)
    await assert.rejects(readStartupToken(tokenLink), /non-symlink regular file/)

    await fs.chmod(tokenFile, 0o644)
    await assert.rejects(readStartupToken(tokenFile), /owner-only/)

    const parsed = parseArguments([
      '--',
      '--base-url',
      'http://127.0.0.1:8765',
      '--token-file',
      tokenFile,
      '--output',
      '.private-review/unit/report.json',
    ])
    assert.equal(parsed.tokenFile, tokenFile)
    assert.equal(validateLoopbackBaseUrl(parsed.baseUrl), 'http://127.0.0.1:8765')
    assert.throws(() => parseArguments(['--token', token]), /Unsupported/)
    assert.throws(() => validateLoopbackBaseUrl('https://example.com'), /loopback/)
    assert.throws(() => validateLoopbackBaseUrl('http://127.0.0.1:8765/?q=private'), /query/)
    assert.throws(() => validateLoopbackBaseUrl('http://127.0.0.1:8765/private'), /without a path/)
    assert.throws(() => outputPath('README.md'), /below .private-review/)
  } finally {
    await fs.rm(fixtureRoot, { recursive: true, force: true })
  }
})

test('fatal startup-token validation is not converted into a completed gate report', async () => {
  await ensurePrivateDirectory(reviewRoot)
  const fixtureRoot = await fs.mkdtemp(path.join(reviewRoot, 'unit-real-ui-fatal-token-'))
  const tokenFile = path.join(fixtureRoot, 'token')
  const report = path.join(fixtureRoot, 'report.json')
  try {
    await fs.writeFile(tokenFile, 'malformed\n', { mode: 0o600 })
    const result = spawnSync(
      process.execPath,
      [
        fileURLToPath(new URL('./real-ui-acceptance.mjs', import.meta.url)),
        '--base-url',
        'http://127.0.0.1:8765',
        '--token-file',
        tokenFile,
        '--output',
        path.relative(path.dirname(reviewRoot), report),
      ],
      { cwd: process.cwd(), encoding: 'utf8' },
    )
    assert.equal(result.status, 2)
    assert.equal(result.stdout, '')
    assert.equal(result.stderr, 'Real UI acceptance failed before safe report publication.\n')
    await assert.rejects(fs.lstat(report), { code: 'ENOENT' })
  } finally {
    await fs.rm(fixtureRoot, { recursive: true, force: true })
  }
})

test('private JSON publication is owner-only, atomic, and no-overwrite', async () => {
  await ensurePrivateDirectory(reviewRoot)
  const fixtureRoot = await fs.mkdtemp(path.join(reviewRoot, 'unit-real-ui-output-'))
  const report = path.join(fixtureRoot, 'report.json')
  try {
    await writePrivateJson(report, { sequence: 1 })
    assert.equal((await fs.lstat(fixtureRoot)).mode & 0o777, 0o700)
    assert.equal((await fs.lstat(report)).mode & 0o777, 0o600)
    await assert.rejects(writePrivateJson(report, { sequence: 2 }), /overwrite/)
    assert.deepEqual(JSON.parse(await fs.readFile(report, 'utf8')), { sequence: 1 })
  } finally {
    await fs.rm(fixtureRoot, { recursive: true, force: true })
  }
})

function passingViewportResult() {
  const viewport = passingViewportResult.calls++ === 0 ? 'desktop' : 'mobile'
  const routes = [
    'home',
    'coverage',
    'integrity',
    'jobs',
    'settings',
    'conversations',
    'media',
    'starred',
    'search',
    'enrichment',
    'people',
    'events',
    'activity',
    'timeline',
    'support',
    'analytics',
    'deletion',
    'largest-conversation',
  ]
  return {
    pageRuns: 18,
    screenshotCount: 18,
    initialContentMs: [120, 180],
    axeViolationCount: 0,
    axeIncompleteCount: 0,
    horizontalOverflowCount: 0,
    pageErrorCount: 0,
    externalRequestCount: 0,
    nonGetHeadRequestCount: 0,
    webSocketRequestCount: 0,
    serviceWorkerRegistrationCount: 0,
    reducedMotionFailureCount: 0,
    largestRenderedMessageCount: 24,
    largestLoadedMessageCount: 250,
    traceReady: true,
    pageEvidence: routes.map((route) => ({ viewport, route })),
    interactionEvidence: interactionNames.map((interaction) => ({
      viewport,
      interaction,
      passed: true,
      duration_ms: 10,
      proof: minimumInteractionProof(interaction),
    })),
  }
}
passingViewportResult.calls = 0

test('summary gate rejects missing or zero DOM counts and missing traces', () => {
  assert.equal(interactionNames.length, 21)
  assert.ok(interactionNames.includes('exact_search_exact_jump'))
  assert.ok(interactionNames.includes('semantic_search_exact_jump'))
  assert.ok(interactionNames.includes('semantic_argument_search_exact_jump'))
  assert.ok(interactionNames.includes('hybrid_search_exact_jump'))
  assert.ok(interactionNames.includes('conversation_types_source_bound'))
  assert.ok(interactionNames.includes('media_representative_kinds_and_jumps'))
  assert.ok(interactionNames.includes('activity_post_comment_source_jumps'))
  passingViewportResult.calls = 0
  const passing = summarizeViewportResults([passingViewportResult(), passingViewportResult()])
  assert.equal(summaryPasses(passing), true)
  assert.equal(passing.interaction_proof_complete, true)
  assert.equal(validateSummaryProofTotals(passing.interaction_proof_totals), true)

  for (const invalidValue of [null, 0]) {
    passingViewportResult.calls = 0
    const invalid = passingViewportResult()
    invalid.largestRenderedMessageCount = invalidValue
    assert.equal(
      summaryPasses(summarizeViewportResults([passingViewportResult(), invalid])),
      false,
    )
  }

  passingViewportResult.calls = 0
  const normalBeforeInverted = passingViewportResult()
  const inverted = passingViewportResult()
  inverted.largestRenderedMessageCount = 251
  assert.equal(
    summaryPasses(summarizeViewportResults([normalBeforeInverted, inverted])),
    false,
  )

  passingViewportResult.calls = 0
  const normalBeforeMissingTrace = passingViewportResult()
  const missingTrace = passingViewportResult()
  missingTrace.traceReady = false
  assert.equal(
    summaryPasses(summarizeViewportResults([normalBeforeMissingTrace, missingTrace])),
    false,
  )

  passingViewportResult.calls = 0
  const incompleteAxe = passingViewportResult()
  incompleteAxe.axeIncompleteCount = 1
  assert.equal(
    summaryPasses(summarizeViewportResults([incompleteAxe, passingViewportResult()])),
    false,
  )

  passingViewportResult.calls = 0
  const disallowedMethod = passingViewportResult()
  disallowedMethod.nonGetHeadRequestCount = 1
  assert.equal(
    summaryPasses(summarizeViewportResults([disallowedMethod, passingViewportResult()])),
    false,
  )

  passingViewportResult.calls = 0
  const webSocketAttempt = passingViewportResult()
  webSocketAttempt.webSocketRequestCount = 1
  assert.equal(
    summaryPasses(summarizeViewportResults([webSocketAttempt, passingViewportResult()])),
    false,
  )

  passingViewportResult.calls = 0
  const serviceWorkerAttempt = passingViewportResult()
  serviceWorkerAttempt.serviceWorkerRegistrationCount = 1
  assert.equal(
    summaryPasses(summarizeViewportResults([serviceWorkerAttempt, passingViewportResult()])),
    false,
  )

  passingViewportResult.calls = 0
  const completeInteractions = passingViewportResult()
  const missingInteraction = passingViewportResult()
  missingInteraction.interactionEvidence.pop()
  assert.equal(
    summaryPasses(summarizeViewportResults([completeInteractions, missingInteraction])),
    false,
  )

  passingViewportResult.calls = 0
  const completeBeforeFailedInteraction = passingViewportResult()
  const failedInteraction = passingViewportResult()
  failedInteraction.interactionEvidence[0].passed = false
  assert.equal(
    summaryPasses(summarizeViewportResults([completeBeforeFailedInteraction, failedInteraction])),
    false,
  )

  passingViewportResult.calls = 0
  const completeBeforeVacuousPagination = passingViewportResult()
  const vacuousPagination = passingViewportResult()
  vacuousPagination.interactionEvidence.find(
    (item) => item.interaction === 'conversation_sort_and_page',
  ).proof.pagination_exercised_count = 0
  const vacuousSummary = summarizeViewportResults([
    completeBeforeVacuousPagination,
    vacuousPagination,
  ])
  assert.equal(vacuousSummary.interaction_proof_complete, false)
  assert.equal(summaryPasses(vacuousSummary), false)

  passingViewportResult.calls = 0
  const completeBeforeExtraProofField = passingViewportResult()
  const extraProofField = passingViewportResult()
  extraProofField.interactionEvidence[0].proof.private_title = 1
  assert.equal(
    summaryPasses(summarizeViewportResults([completeBeforeExtraProofField, extraProofField])),
    false,
  )
})

test('completed gate route diagnostics retain only static aggregate metadata', () => {
  const privateMarker = '<article data-private="conversation">private text</article>'
  const diagnostics = completedGateRouteDiagnostics([
    {
      viewport: 'desktop',
      route: 'analytics',
      failure_category_counts: [
        { category: 'accessibility', count: 2 },
        { category: 'performance', count: 1 },
        { category: 'not-allowlisted', count: 99 },
      ],
      performance_over_budget: true,
      axe_rule_ids: [
        'label',
        'color-contrast',
        'label',
        privateMarker,
        'a'.repeat(101),
      ],
      selector: privateMarker,
      node_html: privateMarker,
      message: privateMarker,
      url: 'http://127.0.0.1/private-conversation',
      exact_timing_ms: 9_999.123,
      private_identifier: 'private-conversation-id',
      source_path: '/private/archive/path',
    },
    {
      viewport: 'desktop',
      route: 'analytics',
      failure_category_counts: [{ category: 'interaction_evidence', count: 1 }],
      performance_over_budget: false,
      axe_rule_ids: [],
    },
    {
      viewport: 'mobile',
      route: 'search',
      failure_category_counts: [{ category: 'network_read_only', count: 2 }],
      performance_over_budget: false,
      axe_rule_ids: [],
    },
    {
      viewport: 'private-viewport',
      route: 'analytics',
      failure_category_counts: [{ category: 'accessibility', count: 1 }],
      axe_rule_ids: ['private-rule'],
    },
    {
      viewport: 'desktop',
      route: 'private-route',
      failure_category_counts: [{ category: 'accessibility', count: 1 }],
      axe_rule_ids: ['private-rule'],
    },
  ])

  assert.deepEqual(diagnostics, {
    performance_over_budget: true,
    performance_over_budget_count: 1,
    routes: [
      {
        viewport: 'desktop',
        route: 'analytics',
        failure_category_counts: [
          { category: 'accessibility', count: 2 },
          { category: 'interaction_evidence', count: 1 },
          { category: 'performance', count: 1 },
        ],
        performance_over_budget: true,
        axe_rule_ids: ['color-contrast', 'label'],
      },
      {
        viewport: 'mobile',
        route: 'search',
        failure_category_counts: [{ category: 'network_read_only', count: 2 }],
        performance_over_budget: false,
        axe_rule_ids: [],
      },
    ],
  })
  const serialized = JSON.stringify(diagnostics)
  for (const forbidden of [
    privateMarker,
    'private-conversation-id',
    '/private/archive/path',
    'private-route',
    'private-viewport',
    '9999.123',
    'selector',
    'node_html',
    'message',
    'url',
  ]) {
    assert.equal(serialized.includes(forbidden), false)
  }
})

test('completed gate failures publish only aggregate allowlisted categories', async () => {
  passingViewportResult.calls = 0
  const summary = summarizeViewportResults([
    passingViewportResult(),
    passingViewportResult(),
  ])
  summary.initial_content_p95_ms = 2_500.1
  summary.axe_violation_count = 2
  summary.page_error_count = 1
  summary.external_request_count = 1
  summary.non_get_head_request_count = 2
  summary.websocket_request_count = 3
  summary.service_worker_registration_count = 4
  summary.largest_loaded_message_count = 251
  summary.reader_counts_valid = false
  summary.screenshots_complete = false
  summary.interaction_pass_count = 29
  summary.interaction_evidence_complete = false
  summary.interaction_proof_complete = false
  summary.private_conversation_name = 'private-conversation-marker'
  summary.interaction_proof_totals.private_identifier = 'private-conversation-marker'

  assert.deepEqual(COMPLETED_GATE_FAILURE_CATEGORIES, [
    'accessibility',
    'browser_errors',
    'evidence_completeness',
    'interaction_evidence',
    'network_read_only',
    'performance',
    'reader_bounds',
  ])
  assert.deepEqual(completedGateFailureCategoryCounts(summary, false), [
    { category: 'accessibility', count: 2 },
    { category: 'browser_errors', count: 1 },
    { category: 'evidence_completeness', count: 2 },
    { category: 'interaction_evidence', count: 13 },
    { category: 'network_read_only', count: 10 },
    { category: 'performance', count: 1 },
    { category: 'reader_bounds', count: 2 },
  ])

  const payload = completedGateFailureReport(
    summary,
    false,
    '2026-07-21T00:00:00.000Z',
    [
      {
        viewport: 'mobile',
        route: 'analytics',
        failure_category_counts: [
          { category: 'accessibility', count: 2 },
          { category: 'performance', count: 1 },
        ],
        performance_over_budget: true,
        axe_rule_ids: ['color-contrast'],
        private_text: 'private-conversation-marker',
      },
    ],
  )
  assert.deepEqual(payload.gate, { passed: false })
  assert.equal(payload.report_kind, 'completed_gate_failure')
  assert.equal(JSON.stringify(payload).includes('private-conversation-marker'), false)
  assert.equal(Object.hasOwn(payload.summary, 'initial_content_p95_ms'), false)
  assert.deepEqual(payload.route_diagnostics, {
    performance_over_budget: true,
    performance_over_budget_count: 1,
    routes: [
      {
        viewport: 'mobile',
        route: 'analytics',
        failure_category_counts: [
          { category: 'accessibility', count: 2 },
          { category: 'performance', count: 1 },
        ],
        performance_over_budget: true,
        axe_rule_ids: ['color-contrast'],
      },
    ],
  })
  assert.deepEqual(payload.failure_category_counts, [
    { category: 'accessibility', count: 2 },
    { category: 'browser_errors', count: 1 },
    { category: 'evidence_completeness', count: 2 },
    { category: 'interaction_evidence', count: 13 },
    { category: 'network_read_only', count: 10 },
    { category: 'performance', count: 1 },
    { category: 'reader_bounds', count: 2 },
  ])
  assert.deepEqual(payload.safety, {
    aggregate_report_only: true,
    screenshots_and_traces_separate: true,
    owner_only_artifacts: true,
    source_text_emitted_in_report: false,
    token_emitted: false,
    external_network_guard_installed: true,
    websocket_guard_installed: true,
    service_worker_guard_installed: true,
    external_network_attempts_absent: false,
    websocket_attempts_absent: false,
    service_worker_registrations_absent: false,
    acceptance_read_only_server: true,
    only_get_head_requests: false,
    analytics_cache_reads_writes_disabled: true,
    mutating_user_actions_performed: false,
  })
  assert.equal(Object.hasOwn(payload.safety, 'external_network_blocked'), false)
  assert.equal(Object.hasOwn(payload.safety, 'websockets_blocked'), false)
  assert.equal(Object.hasOwn(payload.safety, 'service_workers_blocked'), false)
  for (const forbiddenKey of [
    'catalog',
    'pages',
    'interactions',
    'artifacts',
    'runner_sha256',
    'exception',
    'error',
    'path',
  ]) {
    assert.equal(Object.hasOwn(payload, forbiddenKey), false)
  }

  await ensurePrivateDirectory(reviewRoot)
  const fixtureRoot = await fs.mkdtemp(path.join(reviewRoot, 'unit-real-ui-failure-'))
  const report = path.join(fixtureRoot, 'report.json')
  try {
    await writePrivateJson(report, payload)
    assert.equal((await fs.lstat(fixtureRoot)).mode & 0o777, 0o700)
    assert.equal((await fs.lstat(report)).mode & 0o777, 0o600)
    const persisted = JSON.parse(await fs.readFile(report, 'utf8'))
    assert.deepEqual(persisted, payload)
    assert.equal(JSON.stringify(persisted).includes('private-conversation-marker'), false)
  } finally {
    await fs.rm(fixtureRoot, { recursive: true, force: true })
  }
})

test('artifact manifest requires the exact owner-only screenshot and trace inventory', async () => {
  await ensurePrivateDirectory(reviewRoot)
  const fixtureRoot = await fs.mkdtemp(path.join(reviewRoot, 'unit-real-ui-artifacts-'))
  const routes = [
    'home',
    'coverage',
    'integrity',
    'jobs',
    'settings',
    'conversations',
    'media',
    'starred',
    'search',
    'enrichment',
    'people',
    'events',
    'activity',
    'timeline',
    'support',
    'analytics',
    'deletion',
    'largest-conversation',
  ]
  try {
    for (const viewport of ['desktop', 'mobile']) {
      for (const route of routes) {
        await fs.writeFile(path.join(fixtureRoot, `${viewport}-${route}.png`), 'image', {
          mode: 0o600,
        })
      }
      await fs.writeFile(path.join(fixtureRoot, `${viewport}-trace.zip`), 'trace', {
        mode: 0o600,
      })
    }
    assert.equal(await artifactInventoryComplete(fixtureRoot), true)
    const manifest = await buildArtifactManifest(fixtureRoot)
    assert.equal(manifest.length, 38)
    assert.ok(manifest.every((item) => /^[a-f0-9]{64}$/.test(item.sha256)))

    const insecureArtifact = path.join(fixtureRoot, 'desktop-home.png')
    await fs.chmod(insecureArtifact, 0o644)
    await assert.rejects(buildArtifactManifest(fixtureRoot), /not owner-only/)
    await fs.chmod(insecureArtifact, 0o600)

    await fs.writeFile(path.join(fixtureRoot, 'unexpected.txt'), 'private', { mode: 0o600 })
    assert.equal(await artifactInventoryComplete(fixtureRoot), false)
    await assert.rejects(buildArtifactManifest(fixtureRoot), /incomplete or unexpected/)
  } finally {
    await fs.rm(fixtureRoot, { recursive: true, force: true })
  }
})
