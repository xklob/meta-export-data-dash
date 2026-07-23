import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    host: '127.0.0.1',
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: false,
      },
      '/auth': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: false,
      },
    },
  },
  preview: {
    host: '127.0.0.1',
    port: 4173,
  },
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    // Browser/component tests live under src. The scripts directory contains Node's
    // built-in test runner suites, which are exercised by their dedicated package script.
    exclude: ['tests/e2e/**', 'tests/destructive/**', 'scripts/**', 'node_modules/**', 'dist/**'],
    css: true,
    restoreMocks: true,
    coverage: {
      reporter: ['text', 'html'],
    },
  },
})
