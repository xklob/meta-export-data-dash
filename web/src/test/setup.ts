import '@testing-library/jest-dom/vitest'
import { cleanup } from '@testing-library/react'
import { afterEach } from 'vitest'

// Node 25 exposes an incomplete process-level localStorage when no backing
// file is configured. Pin tests to a deterministic browser-compatible store
// instead of allowing that experimental global to shadow JSDOM's storage.
const localValues = new Map<string, string>()
const testLocalStorage: Storage = {
  get length() {
    return localValues.size
  },
  clear: () => localValues.clear(),
  getItem: (key) => localValues.get(String(key)) ?? null,
  key: (index) => [...localValues.keys()][index] ?? null,
  removeItem: (key) => {
    localValues.delete(String(key))
  },
  setItem: (key, value) => {
    localValues.set(String(key), String(value))
  },
}
Object.defineProperty(globalThis, 'localStorage', { configurable: true, value: testLocalStorage })

afterEach(() => {
  cleanup()
  testLocalStorage.clear()
})

Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: (query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: () => undefined,
    removeListener: () => undefined,
    addEventListener: () => undefined,
    removeEventListener: () => undefined,
    dispatchEvent: () => false,
  }),
})

window.scrollTo = () => undefined
