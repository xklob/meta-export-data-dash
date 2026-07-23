import { expect, type Page, test } from '@playwright/test'
import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process'
import { once } from 'node:events'
import { existsSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { resolve } from 'node:path'

const READY_PREFIX = 'META_ARCHIVE_DISPOSABLE_READY='
const TARGET_TEXT = 'Older chunk'
const DATE_BOUNDARY_TEXT = {
  before: 'Date boundary millisecond before retained',
  start: 'Date boundary first millisecond included',
  end: 'Date boundary final millisecond included',
  after: 'Date boundary exclusive end retained',
} as const
const PROJECT_ROOT = fileURLToPath(new URL('../../../', import.meta.url))

interface DisposableServer {
  process: ChildProcessWithoutNullStreams
  archiveRoot: string
  temporaryRoot: string
  baseURL: string
  accessToken: string
  stderr: () => string
}

interface SearchProjection {
  id: string
  source_id: string
  excerpt: string
  jump: { href: string } | null
}

interface DeletionPreview {
  operation_id: string
  impact: { selected: { message: number } }
}

async function terminateProcess(child: ChildProcessWithoutNullStreams): Promise<void> {
  if (child.exitCode != null || child.signalCode != null) return
  child.kill('SIGTERM')
  const exited = Promise.race([
    once(child, 'exit').then(() => true),
    new Promise<false>((resolveTimeout) => setTimeout(() => resolveTimeout(false), 15_000)),
  ])
  if (!(await exited)) {
    child.kill('SIGKILL')
    await once(child, 'exit')
  }
}

async function launchDisposableServer(): Promise<DisposableServer> {
  const python = resolve(PROJECT_ROOT, '.venv/bin/python3')
  const helper = resolve(PROJECT_ROOT, 'web/tests/helpers/disposable_archive_server.py')
  const child = spawn(python, [helper], {
    cwd: PROJECT_ROOT,
    env: { ...process.env, PYTHONUNBUFFERED: '1' },
    stdio: ['ignore', 'pipe', 'pipe'],
  })
  let stderr = ''
  child.stderr.setEncoding('utf8')
  child.stderr.on('data', (chunk: string) => { stderr += chunk })

  try {
    const ready = await new Promise<Omit<DisposableServer, 'process' | 'stderr'>>((resolveReady, reject) => {
      let stdout = ''
      const timeout = setTimeout(() => reject(new Error(`Disposable backend did not start.\n${stderr}`)), 45_000)
      const fail = (error: Error) => { clearTimeout(timeout); reject(error) }
      child.once('error', fail)
      child.once('exit', (code, signal) => fail(new Error(`Disposable backend exited before readiness (${code ?? signal}).\n${stderr}`)))
      child.stdout.setEncoding('utf8')
      child.stdout.on('data', (chunk: string) => {
        stdout += chunk
        for (const line of stdout.split(/\r?\n/)) {
          if (!line.startsWith(READY_PREFIX)) continue
          clearTimeout(timeout)
          const payload = JSON.parse(line.slice(READY_PREFIX.length)) as {
            archive_root: string
            temporary_root: string
            base_url: string
            access_token: string
          }
          resolveReady({
            archiveRoot: payload.archive_root,
            temporaryRoot: payload.temporary_root,
            baseURL: payload.base_url,
            accessToken: payload.access_token,
          })
          return
        }
      })
    })

    const server = { ...ready, process: child, stderr: () => stderr }
    await expect.poll(async () => {
      try {
        return (await fetch(`${server.baseURL}/api/v1/health`)).status
      } catch {
        return 0
      }
    }, { timeout: 15_000, message: server.stderr() }).toBe(200)
    return server
  } catch (error) {
    await terminateProcess(child)
    throw error
  }
}

async function stopDisposableServer(server: DisposableServer): Promise<void> {
  await terminateProcess(server.process)
  expect(existsSync(server.archiveRoot), `archive cleanup failed; backend stderr:\n${server.stderr()}`).toBe(false)
  expect(existsSync(server.temporaryRoot), `temporary-root cleanup failed; backend stderr:\n${server.stderr()}`).toBe(false)
}

async function keywordSearch(page: Page, baseURL: string): Promise<void> {
  await page.goto(`${baseURL}/search`)
  await page.getByLabel('What do you want to find?').fill(TARGET_TEXT)
  await page.getByRole('button', { name: 'Search archive' }).click()
  await expect(page.getByRole('heading', { name: 'Search results' })).toBeVisible()
}

async function searchProjection(page: Page): Promise<SearchProjection[]> {
  return page.evaluate(async (targetText) => {
    const query = new URLSearchParams({ q: targetText, mode: 'keyword', entity_type: 'message', limit: '50' })
    const response = await fetch(`/api/v1/search?${query}`)
    if (!response.ok) throw new Error(`Search API failed with ${response.status}`)
    const payload = await response.json() as { items: SearchProjection[] }
    return payload.items.map(({ id, source_id, excerpt, jump }) => ({ id, source_id, excerpt, jump }))
  }, TARGET_TEXT)
}

async function previewTargetMessage(page: Page, baseURL: string): Promise<DeletionPreview> {
  await page.goto(`${baseURL}/deletion`)
  await page.getByLabel('Posts and activity').uncheck()
  await page.getByLabel('Text to match').fill(TARGET_TEXT)
  await page.getByLabel('Whole field only').check()
  const responsePromise = page.waitForResponse((response) =>
    response.url().endsWith('/api/v1/deletions/preview') && response.request().method() === 'POST')
  await page.getByRole('button', { name: 'Preview selection' }).click()
  const response = await responsePromise
  expect(response.status()).toBe(201)
  const preview = await response.json() as DeletionPreview
  expect(preview.impact.selected.message).toBe(1)
  await expect(page.getByRole('heading', { name: '2. Review quarantine impact' })).toBeVisible()
  return preview
}

async function commitPreview(page: Page, preview: DeletionPreview, expectedMessages = 1): Promise<void> {
  await page.getByLabel('Type the operation ID to commit logical quarantine').fill(preview.operation_id)
  const requestPromise = page.waitForRequest((request) =>
    request.url().endsWith(`/api/v1/deletions/${preview.operation_id}/commit`) && request.method() === 'POST')
  const responsePromise = page.waitForResponse((response) =>
    response.url().endsWith(`/api/v1/deletions/${preview.operation_id}/commit`) && response.request().method() === 'POST')
  await page.getByRole('button', { name: 'Commit to quarantine' }).click()
  expect((await requestPromise).headers()['x-csrf-token']).toBeTruthy()
  const response = await responsePromise
  expect(response.status()).toBe(200)
  expect(await response.json()).toMatchObject({
    operation_id: preview.operation_id,
    status: 'committed',
    affected: { message: expectedMessages },
  })
  await expect(page.locator('.deletion-detail__header .status-badge')).toHaveText('Committed')
}

async function restorePreview(
  page: Page,
  baseURL: string,
  preview: DeletionPreview,
  expectedMessages = 1,
): Promise<void> {
  await page.goto(`${baseURL}/deletion`)
  const operation = page.getByRole('list', { name: 'Deletion operations' })
    .getByRole('button')
    .filter({ hasText: preview.operation_id })
  await operation.click()
  await page.getByLabel('Type the operation ID to authorize an action').fill(preview.operation_id)
  const restoreRequestPromise = page.waitForRequest((request) =>
    request.url().endsWith(`/api/v1/deletions/${preview.operation_id}/restore`) && request.method() === 'POST')
  const restoreResponsePromise = page.waitForResponse((response) =>
    response.url().endsWith(`/api/v1/deletions/${preview.operation_id}/restore`) && response.request().method() === 'POST')
  await page.getByRole('button', { name: 'Restore quarantine' }).click()
  const restoreRequest = await restoreRequestPromise
  expect(restoreRequest.headers()['x-csrf-token']).toBeTruthy()
  expect(restoreRequest.postDataJSON()).toEqual({ confirmation: preview.operation_id })
  const restoreResponse = await restoreResponsePromise
  expect(restoreResponse.status()).toBe(200)
  expect(await restoreResponse.json()).toMatchObject({
    operation_id: preview.operation_id,
    status: 'restored',
    affected: { message: expectedMessages },
  })
  await expect(page.locator('.deletion-detail__header .status-badge')).toHaveText('Restored')
}

async function proveDateBoundaryDeletion(page: Page, server: DisposableServer): Promise<void> {
  await page.goto(`${server.baseURL}/search`)
  await page.getByLabel('What do you want to find?').fill(DATE_BOUNDARY_TEXT.start)
  await page.getByRole('button', { name: 'Search archive' }).click()
  const startResult = page.locator('.search-result').filter({ hasText: DATE_BOUNDARY_TEXT.start })
  await expect(startResult).toHaveCount(1)
  const sourceHref = await startResult.getByRole('link', { name: 'Jump to source' }).getAttribute('href')
  expect(sourceHref).toBeTruthy()
  const conversationPath = new URL(sourceHref!, server.baseURL).pathname

  await startResult.getByRole('link', { name: 'Jump to source' }).click()
  for (const content of Object.values(DATE_BOUNDARY_TEXT)) {
    await expect(page.getByText(content, { exact: true })).toBeVisible()
  }

  await page.goto(`${server.baseURL}/deletion`)
  await page.getByLabel('Posts and activity').uncheck()
  await page.getByLabel('Inclusive start date').fill('2024-03-10')
  await page.getByLabel('Inclusive end date').fill('2024-03-10')
  const previewRequestPromise = page.waitForRequest((request) =>
    request.url().endsWith('/api/v1/deletions/preview') && request.method() === 'POST')
  const previewResponsePromise = page.waitForResponse((response) =>
    response.url().endsWith('/api/v1/deletions/preview') && response.request().method() === 'POST')
  await page.getByRole('button', { name: 'Preview selection' }).click()
  const previewRequest = await previewRequestPromise
  expect(previewRequest.postDataJSON()).toMatchObject({
    entity_types: ['message'],
    from_date: '2024-03-10',
    to_date: '2024-03-10',
    timezone: 'America/New_York',
  })
  expect(previewRequest.postDataJSON()).not.toHaveProperty('text')
  const previewResponse = await previewResponsePromise
  expect(previewResponse.status()).toBe(201)
  const preview = await previewResponse.json() as DeletionPreview
  expect(preview.impact.selected.message).toBe(2)
  await expect(page.getByRole('heading', { name: '2. Review quarantine impact' })).toBeVisible()

  const previewProofPage = await page.context().newPage()
  try {
    await previewProofPage.goto(`${server.baseURL}${conversationPath}`)
    for (const content of Object.values(DATE_BOUNDARY_TEXT)) {
      await expect(previewProofPage.getByText(content, { exact: true })).toBeVisible()
    }
  } finally {
    await previewProofPage.close()
  }

  await commitPreview(page, preview, 2)
  await page.goto(`${server.baseURL}${conversationPath}`)
  await expect(page.getByText(DATE_BOUNDARY_TEXT.before, { exact: true })).toBeVisible()
  await expect(page.getByText(DATE_BOUNDARY_TEXT.after, { exact: true })).toBeVisible()
  await expect(page.getByText(DATE_BOUNDARY_TEXT.start, { exact: true })).toHaveCount(0)
  await expect(page.getByText(DATE_BOUNDARY_TEXT.end, { exact: true })).toHaveCount(0)

  await restorePreview(page, server.baseURL, preview, 2)
  await page.goto(`${server.baseURL}${conversationPath}`)
  for (const content of Object.values(DATE_BOUNDARY_TEXT)) {
    await expect(page.getByText(content, { exact: true })).toBeVisible()
  }
}

let disposable: DisposableServer | undefined

test.afterAll(async () => {
  if (disposable) await stopDisposableServer(disposable)
})

test('authenticated UI preserves date boundaries, restores exactly, then workspace-purges', async ({ page }) => {
  disposable = await launchDisposableServer()
  await page.goto(`${disposable.baseURL}/auth?token=${encodeURIComponent(disposable.accessToken)}`)
  await expect(page).toHaveURL(`${disposable.baseURL}/`)

  await proveDateBoundaryDeletion(page, disposable)

  await keywordSearch(page, disposable.baseURL)
  const initialResult = page.locator('.search-result').filter({ hasText: TARGET_TEXT })
  await expect(initialResult).toHaveCount(1)
  const initialProjection = await searchProjection(page)
  expect(initialProjection).toHaveLength(1)
  expect(initialProjection[0].jump?.href).toBeTruthy()
  const sourceHref = initialProjection[0].jump!.href

  await initialResult.getByRole('link', { name: 'Jump to source' }).click()
  await expect(page.getByText(TARGET_TEXT, { exact: true })).toBeVisible()

  const preview = await previewTargetMessage(page, disposable.baseURL)

  // Preview must be observational only. Use a second authenticated browser page so the original
  // page retains its one-time in-memory commit token while the real reader and search UI are
  // independently reloaded against the backing catalog.
  const previewProofPage = await page.context().newPage()
  try {
    await previewProofPage.goto(`${disposable.baseURL}${sourceHref}`)
    await expect(previewProofPage.getByText(TARGET_TEXT, { exact: true })).toBeVisible()
    await keywordSearch(previewProofPage, disposable.baseURL)
    await expect(previewProofPage.locator('.search-result').filter({ hasText: TARGET_TEXT })).toHaveCount(1)
    expect(await searchProjection(previewProofPage)).toEqual(initialProjection)
  } finally {
    await previewProofPage.close()
  }
  await commitPreview(page, preview)

  const conversationPath = new URL(sourceHref, disposable.baseURL).pathname
  await page.goto(`${disposable.baseURL}${conversationPath}`)
  await expect(page.getByText(TARGET_TEXT, { exact: true })).toHaveCount(0)
  await expect(page.getByText('A second source segment', { exact: true })).toBeVisible()

  await keywordSearch(page, disposable.baseURL)
  await expect(page.getByText('No matching archive records', { exact: true })).toBeVisible()
  expect(await searchProjection(page)).toEqual([])

  await restorePreview(page, disposable.baseURL, preview)

  await page.goto(`${disposable.baseURL}${sourceHref}`)
  await expect(page.getByText(TARGET_TEXT, { exact: true })).toBeVisible()
  await keywordSearch(page, disposable.baseURL)
  await expect(page.locator('.search-result').filter({ hasText: TARGET_TEXT })).toHaveCount(1)
  expect(await searchProjection(page)).toEqual(initialProjection)

  const purgeOperation = await previewTargetMessage(page, disposable.baseURL)
  await commitPreview(page, purgeOperation)
  await page.getByLabel('Type the operation ID to authorize an action').fill(purgeOperation.operation_id)
  await page.getByLabel('Purge scope').selectOption('workspace')
  const purgePreviewRequestPromise = page.waitForRequest((request) =>
    request.url().endsWith(`/api/v1/deletions/${purgeOperation.operation_id}/purge/preview`)
      && request.method() === 'POST')
  const purgePreviewResponsePromise = page.waitForResponse((response) =>
    response.url().endsWith(`/api/v1/deletions/${purgeOperation.operation_id}/purge/preview`)
      && response.request().method() === 'POST')
  await page.getByRole('button', { name: 'Preview hard purge' }).click()
  const purgePreviewRequest = await purgePreviewRequestPromise
  expect(purgePreviewRequest.headers()['x-csrf-token']).toBeTruthy()
  expect(purgePreviewRequest.postDataJSON()).toEqual({ scope: 'workspace' })
  const purgePreviewResponse = await purgePreviewResponsePromise
  expect(purgePreviewResponse.status()).toBe(200)
  const purgePlan = await purgePreviewResponse.json() as {
    operation_id: string
    scope: string
    executable: boolean
    purge_token: string | null
    impact: { raw_source_files_rewritten: number }
  }
  expect(purgePlan).toMatchObject({
    operation_id: purgeOperation.operation_id,
    scope: 'workspace',
    executable: true,
  })
  expect(purgePlan.purge_token).toBeTruthy()
  expect(purgePlan.impact.raw_source_files_rewritten).toBe(1)
  await expect(page.getByRole('heading', { name: 'Workspace hard purge' })).toBeVisible()

  await page.getByLabel('Type the operation ID to authorize an action').fill(purgeOperation.operation_id)
  const purgeRequestPromise = page.waitForRequest((request) =>
    request.url().endsWith(`/api/v1/deletions/${purgeOperation.operation_id}/purge`)
      && request.method() === 'POST')
  const purgeResponsePromise = page.waitForResponse((response) =>
    response.url().endsWith(`/api/v1/deletions/${purgeOperation.operation_id}/purge`)
      && response.request().method() === 'POST')
  await page.getByRole('button', { name: 'Permanently execute hard purge' }).click()
  const purgeRequest = await purgeRequestPromise
  expect(purgeRequest.headers()['x-csrf-token']).toBeTruthy()
  expect(purgeRequest.postDataJSON()).toMatchObject({
    scope: 'workspace',
    confirmation: purgeOperation.operation_id,
  })
  const purgeResponse = await purgeResponsePromise
  expect(purgeResponse.status()).toBe(200)
  expect(await purgeResponse.json()).toMatchObject({
    operation_id: purgeOperation.operation_id,
    status: 'purged',
    affected: { message: 1 },
  })
  await expect(page.locator('.deletion-detail__header .status-badge')).toHaveText('Purged')

  await page.goto(`${disposable.baseURL}${conversationPath}`)
  await expect(page.getByText(TARGET_TEXT, { exact: true })).toHaveCount(0)
  await expect(page.getByText('A second source segment', { exact: true })).toBeVisible()
  await keywordSearch(page, disposable.baseURL)
  await expect(page.getByText('No matching archive records', { exact: true })).toBeVisible()
  expect(await searchProjection(page)).toEqual([])

  await page.goto(`${disposable.baseURL}/deletion`)
  await page.getByRole('list', { name: 'Deletion operations' })
    .getByRole('button')
    .filter({ hasText: purgeOperation.operation_id })
    .click()
  await expect(page.locator('.deletion-detail__header .status-badge')).toHaveText('Purged')
})
