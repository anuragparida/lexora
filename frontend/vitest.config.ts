import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'

// Vitest config for Phase 4.5 (card t_4a9f172e).
//
// We reuse the same Vite plugins as the dev server so JSX + (when
// present) Tailwind classes resolve the same way in tests as in the
// browser. jsdom gives us a real DOM; `globals: true` lets test files
// import `describe/it/expect` from the global namespace the way
// @testing-library/jest-dom assumes.
//
// The `/src` test setup file registers the @testing-library/jest-dom
// matchers (`toBeInTheDocument`, `toHaveTextContent`, ...). Keep it
// tiny — global side-effects on every test make debugging harder.
export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./src/test-setup.ts'],
    css: false,
  },
})
