// Global test setup for Phase 4.5 (card t_4a9f172e).
//
// Registers the @testing-library/jest-dom matchers (`toBeInTheDocument`,
// `toHaveTextContent`, ...) so the ClozePage tests can write assertions
// that read like prose instead of `expect(node.textContent).toBe(...)`.
// Loaded once per test file by vitest.config.ts.
//
// We intentionally do not pull in `intersection-observer` or any
// component-specific polyfill here — the cloze page renders plain
// text + buttons, no observers, no async media.

import '@testing-library/jest-dom/vitest'
