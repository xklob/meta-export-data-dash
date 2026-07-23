import { defineConfig } from '@playwright/test'

// The dedicated destructive suite owns a single disposable backend and must never share it
// across workers. Browser artifacts may contain fixture content, so keep them owner-readable.
process.umask(0o077)

export default defineConfig({
  testDir: './tests/destructive',
  fullyParallel: false,
  forbidOnly: true,
  retries: 0,
  workers: 1,
  reporter: [['list']],
  outputDir: './test-results/destructive',
  timeout: 90_000,
  use: {
    browserName: 'chromium',
    headless: true,
    trace: 'on',
    screenshot: 'only-on-failure',
  },
})
