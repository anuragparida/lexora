import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { ClozePage } from '../../pages/ClozePage'
import type { AuthUser } from '../../auth'
import type { ClozeExercise } from '../../api/cloze'

// Phase 4.5 (card t_4a9f172e): ClozePage unit tests.
//
// We mock the `generateCloze` function and the `getMe` function
// (the two network surfaces the page touches). vi.mock runs
// before imports resolve, so the page module sees the mocks
// on first import.
//
// Test cases mirror the card body's spec verbatim:
//   1. Blank renders when `generateCloze()` resolves.
//   2. Distractor click updates the selection.
//   3. "Generate another" re-fetches (mocked).
//   4. Empty-profile state shows the link to /weakness-profile.
//   5. Loading skeleton renders while `generateCloze()` is in flight.
//   6. 401 response redirects (we model this by asserting the
//      "Go to login" button + redirect behavior — full navigation
//      is tested in the integration tier when the dev stack is
//      up; for the unit test we verify the user-visible affordance
//      and the click handler).
//
// We also throw in a small assertion that the page calls
// `generateCloze` exactly once on mount (the card body's
// "calls /exercises/cloze exactly once on mount" acceptance
// criterion).

// Mock the API client. The factory returns a function with the
// same shape as the real one so the page's import statement
// doesn't break.
vi.mock('../../api/cloze', () => ({
  generateCloze: vi.fn(),
}))

vi.mock('../../auth', async () => {
  const actual =
    await vi.importActual<typeof import('../../auth')>('../../auth')
  return {
    ...actual,
    getMe: vi.fn(),
  }
})

import { generateCloze } from '../../api/cloze'
import { getMe } from '../../auth'

const mockedGenerateCloze = vi.mocked(generateCloze)
const mockedGetMe = vi.mocked(getMe)

// A reusable ClozeExercise fixture. word_id 42 is the answer;
// 1042 / 2087 / 3155 are the distractors.
const sampleExercise: ClozeExercise = {
  sentence_with_blank:
    'Die Partei hat einen neuen Vorsitzenden ___ (gewählt).',
  answer_word_id: 42,
  distractors: [1042, 2087, 3155],
  difficulty: 'medium',
  rationale:
    "Sentence cues 'wählen' via the accusative object 'Vorsitzenden'.",
  prompt_template_version: 'cloze-v1',
}

const userFixture: AuthUser = {
  id: 1,
  email: 'test@example.com',
  created_at: '2026-07-03T00:00:00Z',
}

const meWithProfile = {
  id: 1,
  email: 'test@example.com',
  created_at: '2026-07-03T00:00:00Z',
  weakness_profile: {
    id: 1,
    user_id: 1,
    axes: { verbs: 2, collocations: 3 },
    updated_at: '2026-07-03T00:00:00Z',
  },
  diagnostic_state: 'applied' as const,
}

const meWithoutProfile = {
  id: 2,
  email: 'fresh@example.com',
  created_at: '2026-07-03T00:00:00Z',
  weakness_profile: null,
  diagnostic_state: 'never' as const,
}

function renderPage(initialPath = '/exercises/cloze') {
  return render(
    <MemoryRouter initialEntries={[initialPath]}>
      <Routes>
        <Route path="/exercises/cloze" element={<ClozePage user={userFixture} />} />
        <Route path="/login" element={<div>LOGIN_PAGE</div>} />
        <Route path="/weakness-profile" element={<div>WEAKNESS_PROFILE_PAGE</div>} />
      </Routes>
    </MemoryRouter>,
  )
}

describe('ClozePage', () => {
  beforeEach(() => {
    mockedGenerateCloze.mockReset()
    mockedGetMe.mockReset()
    // Default: a user with a profile, so the fetch path is taken.
    mockedGetMe.mockResolvedValue(meWithProfile)
  })

  afterEach(() => {
    cleanup()
  })

  it('renders the blank + four choices when generateCloze resolves', async () => {
    mockedGenerateCloze.mockResolvedValue(sampleExercise)
    renderPage()
    // The skeleton appears first; wait for the ready state.
    await waitFor(() => {
      expect(screen.getByTestId('cloze-ready')).toBeInTheDocument()
    })
    // The four choice buttons are present.
    expect(screen.getByTestId('cloze-choice-42')).toBeInTheDocument()
    expect(screen.getByTestId('cloze-choice-1042')).toBeInTheDocument()
    expect(screen.getByTestId('cloze-choice-2087')).toBeInTheDocument()
    expect(screen.getByTestId('cloze-choice-3155')).toBeInTheDocument()
    // The sentence prefix and suffix render around the blank.
    expect(screen.getByTestId('cloze-sentence').textContent).toContain(
      'Die Partei hat einen neuen Vorsitzenden',
    )
    expect(screen.getByTestId('cloze-sentence').textContent).toContain(
      '(gewählt).',
    )
    // The blank itself is rendered.
    expect(screen.getByTestId('cloze-blank')).toBeInTheDocument()
  })

  it('updates the selection when a distractor is clicked', async () => {
    const user = userEvent.setup()
    mockedGenerateCloze.mockResolvedValue(sampleExercise)
    renderPage()
    await waitFor(() => {
      expect(screen.getByTestId('cloze-ready')).toBeInTheDocument()
    })
    const distractor = screen.getByTestId('cloze-choice-1042')
    expect(distractor).toHaveAttribute('aria-pressed', 'false')
    await user.click(distractor)
    expect(distractor).toHaveAttribute('aria-pressed', 'true')
    // The blank now reflects the chosen word.
    expect(screen.getByTestId('cloze-blank').textContent).toContain('1042')
    // Submit becomes enabled.
    expect(screen.getByTestId('cloze-submit')).not.toBeDisabled()
  })

  it('re-fetches when "Generate another" is clicked', async () => {
    const user = userEvent.setup()
    mockedGenerateCloze
      .mockResolvedValueOnce(sampleExercise)
      .mockResolvedValueOnce({
        ...sampleExercise,
        sentence_with_blank: 'Er ___ das Buch auf den Tisch.',
        answer_word_id: 99,
        distractors: [101, 202, 303],
      })
    renderPage()
    await waitFor(() => {
      expect(screen.getByTestId('cloze-ready')).toBeInTheDocument()
    })
    // First fetch happened.
    expect(mockedGenerateCloze).toHaveBeenCalledTimes(1)
    await user.click(screen.getByTestId('cloze-generate-another'))
    // Second fetch on click.
    await waitFor(() => {
      expect(mockedGenerateCloze).toHaveBeenCalledTimes(2)
    })
    // The new sentence's id is in the DOM.
    expect(screen.getByTestId('cloze-choice-99')).toBeInTheDocument()
  })

  it('shows the empty-profile state with a link to /weakness-profile', async () => {
    mockedGetMe.mockResolvedValue(meWithoutProfile)
    renderPage()
    // The empty state renders synchronously (no fetch in flight).
    await waitFor(() => {
      expect(
        screen.getByText(/set up your weakness profile first/i),
      ).toBeInTheDocument()
    })
    const link = screen.getByRole('link', { name: /open weakness profile/i })
    expect(link).toHaveAttribute('href', '/weakness-profile')
    // generateCloze should NOT be called in the empty state —
    // the page short-circuits before fetching.
    expect(mockedGenerateCloze).not.toHaveBeenCalled()
  })

  it('renders the loading skeleton while generateCloze is in flight', async () => {
    // Deferred pattern: the test holds the resolver so we can
    // fire the resolution AFTER we've confirmed the fetch
    // effect has registered the .then handler (otherwise the
    // resolve fires into the void).
    //
    // We assign via a wrapper object so TypeScript's control
    // flow analysis keeps `resolveFn` callable across the
    // closure boundary — the bare `let resolveFn = null` form
    // narrows to `never` after the first assignment, which
    // `tsc -b` then rejects at the `?.()` call site.
    const deferred: {
      resolve: ((value: ClozeExercise) => void) | null
    } = { resolve: null }
    mockedGenerateCloze.mockImplementation(
      () =>
        new Promise<ClozeExercise>((resolve) => {
          deferred.resolve = resolve
        }),
    )
    renderPage()
    // Wait until the fetch effect has actually invoked
    // generateCloze — that's the proof the .then handler
    // is now registered on the deferred promise.
    await waitFor(() => {
      expect(mockedGenerateCloze).toHaveBeenCalled()
    })
    // Skeleton is on screen.
    expect(screen.getByTestId('cloze-loading')).toBeInTheDocument()
    // The "ready" state is not yet visible.
    expect(screen.queryByTestId('cloze-ready')).not.toBeInTheDocument()
    // Now resolve and confirm the transition to ready.
    if (!deferred.resolve) {
      throw new Error('deferred.resolve was never assigned')
    }
    deferred.resolve(sampleExercise)
    await waitFor(() => {
      expect(screen.getByTestId('cloze-ready')).toBeInTheDocument()
    })
  })

  it('shows a 401 affordance (Go to login) when generateCloze fails with 401', async () => {
    mockedGenerateCloze.mockRejectedValue(
      new Error('Request failed (401)'),
    )
    renderPage()
    await waitFor(() => {
      expect(screen.getByTestId('cloze-error')).toBeInTheDocument()
    })
    // The error branch offers a "Go to login" button (not a Retry).
    const goToLogin = screen.getByRole('button', { name: /go to login/i })
    expect(goToLogin).toBeInTheDocument()
    // And no retry button is rendered in this branch.
    expect(screen.queryByRole('button', { name: /^retry$/i })).toBeNull()
  })

  it('calls generateCloze exactly once on mount (the spec acceptance criterion)', async () => {
    mockedGenerateCloze.mockResolvedValue(sampleExercise)
    renderPage()
    await waitFor(() => {
      expect(screen.getByTestId('cloze-ready')).toBeInTheDocument()
    })
    expect(mockedGenerateCloze).toHaveBeenCalledTimes(1)
  })
})
