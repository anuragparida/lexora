import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { ClozePage } from '../../pages/ClozePage'
import type { AuthUser } from '../../auth'
import type { ClozeExercise, ClozeExerciseOut, GradeResponse } from '../../api/cloze'

// Phase 4.5 (card t_4a9f172e): ClozePage unit tests.
// Phase 5.5 (card t_f253456b): extends with grade-flow cases.
//
// We mock every API surface the page touches:
//   - generateCloze (Phase 4.5)
//   - getMe        (Phase 4.5)
//   - gradeCloze   (Phase 5.5: /exercises/grade POST)
//   - getDueCloze  (Phase 5.5: /exercises/due GET)
//
// vi.mock runs before imports resolve, so the page module sees
// the mocks on first import.

// Mock the API client. The factory returns a function with the
// same shape as the real one so the page's import statement
// doesn't break. We spread the actual module so non-mocked
// exports (`ClozeApiError`, type re-exports) remain accessible
// to the test file.
vi.mock('../../api/cloze', async () => {
  const actual =
    await vi.importActual<typeof import('../../api/cloze')>(
      '../../api/cloze',
    )
  return {
    ...actual,
    generateCloze: vi.fn(),
    gradeCloze: vi.fn(),
    getDueCloze: vi.fn(),
  }
})

vi.mock('../../auth', async () => {
  const actual =
    await vi.importActual<typeof import('../../auth')>('../../auth')
  return {
    ...actual,
    getMe: vi.fn(),
  }
})

// Mock sonner so the toast calls are observable in tests. We
// keep a reference to the spy so each case can assert the call
// shape (Gotcha #4: 422 toast should show the validation
// message from the response body, not a generic string).
vi.mock('sonner', () => ({
  toast: {
    success: vi.fn(),
    error: vi.fn(),
    info: vi.fn(),
    warning: vi.fn(),
  },
}))

import { generateCloze, gradeCloze, getDueCloze, ClozeApiError } from '../../api/cloze'
import { getMe } from '../../auth'
import { toast } from 'sonner'

const mockedGenerateCloze = vi.mocked(generateCloze)
const mockedGradeCloze = vi.mocked(gradeCloze)
const mockedGetDueCloze = vi.mocked(getDueCloze)
const mockedGetMe = vi.mocked(getMe)
const mockedToast = vi.mocked(toast)

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

// A `ClozeExerciseOut` is the response shape for /exercises/due
// (Phase 5.4). It extends `ClozeExercise` with `word_id` and
// `due_from_fsrs`. The first-due variant isn't used by the
// current cases (every happy-path test sets up `secondDueCloze`
// as the next card); keep the type-side sample for the second
// card only and delete the first.
const secondDueCloze: ClozeExerciseOut = {
  ...sampleExercise,
  sentence_with_blank: 'Er ___ das Buch auf den Tisch.',
  answer_word_id: 99,
  distractors: [101, 202, 303],
  word_id: 99,
  due_from_fsrs: true,
}

// A successful GradeResponse shape. Mirrors the backend Pydantic
// GradeResponse (Phase 5.2). The `next_due_at` is in the
// future so the humanizeDelta helper renders "in Xm".
const sampleGradeResponse: GradeResponse = {
  graded: true,
  exercise_id: 42,
  exercise_type: 'cloze',
  next_due_at: new Date(Date.now() + 10 * 60_000).toISOString(),
  card_state: 2,
  stability: 4.2,
  difficulty: 5.1,
  trace_id: null,
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
    mockedGradeCloze.mockReset()
    mockedGetDueCloze.mockReset()
    mockedToast.success.mockReset()
    mockedToast.error.mockReset()
    mockedToast.info.mockReset()
    mockedToast.warning.mockReset()
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
    // Phase 5.5: the four grade buttons render too.
    expect(screen.getByTestId('grade-button-1')).toBeInTheDocument()
    expect(screen.getByTestId('grade-button-2')).toBeInTheDocument()
    expect(screen.getByTestId('grade-button-3')).toBeInTheDocument()
    expect(screen.getByTestId('grade-button-4')).toBeInTheDocument()
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
  })

  it('re-fetches when "Generate another" is clicked', async () => {
    const user = userEvent.setup()
    mockedGenerateCloze
      .mockResolvedValueOnce(sampleExercise)
      .mockResolvedValueOnce(secondDueCloze)
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
      new ClozeApiError(401, 'Not authenticated'),
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

  // -----------------------------------------------------------------
  // Phase 5.5 grade-flow cases.
  // -----------------------------------------------------------------

  it('clicking Good fires gradeCloze with grade=3 and replaces the card from getDueCloze', async () => {
    const user = userEvent.setup()
    mockedGenerateCloze.mockResolvedValue(sampleExercise)
    mockedGradeCloze.mockResolvedValue(sampleGradeResponse)
    mockedGetDueCloze.mockResolvedValue(secondDueCloze)

    renderPage()
    await waitFor(() => {
      expect(screen.getByTestId('cloze-ready')).toBeInTheDocument()
    })
    // The first card is on screen; the answer is word #42.
    expect(screen.getByTestId('cloze-choice-42')).toBeInTheDocument()

    await user.click(screen.getByTestId('grade-button-3'))

    // gradeCloze is called with the answer word_id (which is
    // what 5.3's POST body carries as `exercise_id`) and grade 3.
    await waitFor(() => {
      expect(mockedGradeCloze).toHaveBeenCalledTimes(1)
    })
    expect(mockedGradeCloze).toHaveBeenCalledWith(42, 3)

    // The page then fires getDueCloze and renders the new card.
    await waitFor(() => {
      expect(mockedGetDueCloze).toHaveBeenCalledTimes(1)
    })
    await waitFor(() => {
      // The new card's answer is #99; the previous #42 button
      // should be gone because the choices re-render.
      expect(screen.getByTestId('cloze-choice-99')).toBeInTheDocument()
    })
    expect(screen.queryByTestId('cloze-choice-42')).not.toBeInTheDocument()

    // The success toast fires with the humanized delta.
    expect(mockedToast.success).toHaveBeenCalledTimes(1)
    const toastArg = mockedToast.success.mock.calls[0][0]
    expect(typeof toastArg).toBe('string')
    expect(toastArg).toMatch(/Grade recorded/)
  })

  it('clicking Easy on a hard cloze sends grade=4 with that exercise_id', async () => {
    const user = userEvent.setup()
    mockedGenerateCloze.mockResolvedValue(sampleExercise)
    mockedGradeCloze.mockResolvedValue(sampleGradeResponse)
    mockedGetDueCloze.mockResolvedValue(secondDueCloze)

    renderPage()
    await waitFor(() => {
      expect(screen.getByTestId('cloze-ready')).toBeInTheDocument()
    })

    await user.click(screen.getByTestId('grade-button-4'))

    // The wire payload is { exercise_id: 42, exercise_type: 'cloze',
    // grade: 4 } — verify exercise_id + grade, not the wrapper
    // object literal (the API client owns the body shape).
    await waitFor(() => {
      expect(mockedGradeCloze).toHaveBeenCalledWith(42, 4)
    })
  })

  it('204 from getDueCloze shows the empty-due state ("all caught up")', async () => {
    const user = userEvent.setup()
    mockedGenerateCloze.mockResolvedValue(sampleExercise)
    mockedGradeCloze.mockResolvedValue(sampleGradeResponse)
    mockedGetDueCloze.mockResolvedValue(null) // 204 mapped to null

    renderPage()
    await waitFor(() => {
      expect(screen.getByTestId('cloze-ready')).toBeInTheDocument()
    })

    await user.click(screen.getByTestId('grade-button-3'))

    await waitFor(() => {
      expect(screen.getByTestId('cloze-empty-due')).toBeInTheDocument()
    })
    expect(
      screen.getByText(/all caught up — nothing due right now\./i),
    ).toBeInTheDocument()
    // The grade buttons are NOT in the DOM — there is no card to grade.
    expect(screen.queryByTestId('grade-button-3')).not.toBeInTheDocument()
    // A "Generate another (fresh pick)" CTA is offered so the
    // user can still get a fresh cloze if they want.
    expect(
      screen.getByTestId('cloze-generate-another-empty'),
    ).toBeInTheDocument()
    // The success toast still fires — the grade WAS recorded,
    // there just isn't a next card.
    expect(mockedToast.success).toHaveBeenCalledTimes(1)
  })

  it('422 on gradeCloze shows the validation detail in a toast and keeps the current card', async () => {
    const user = userEvent.setup()
    mockedGenerateCloze.mockResolvedValue(sampleExercise)
    // The Pydantic 422 detail is a free-form string the backend
    // author chooses — we don't care about the literal text,
    // only that it shows up in the toast verbatim. The API
    // client surfaces `first.msg` for array-shaped details;
    // the thrown error carries the raw message.
    mockedGradeCloze.mockRejectedValue(
      new ClozeApiError(422, 'Input should be '),
    )

    renderPage()
    await waitFor(() => {
      expect(screen.getByTestId('cloze-ready')).toBeInTheDocument()
    })

    await user.click(screen.getByTestId('grade-button-3'))

    // Wait for the round-trip to settle.
    await waitFor(() => {
      expect(mockedGradeCloze).toHaveBeenCalledTimes(1)
    })

    // The error toast carries the validation detail verbatim —
    // we DON'T substitute a generic "Grade failed" message on
    // the 422 path (Gotcha #4).
    await waitFor(() => {
      expect(mockedToast.error).toHaveBeenCalledTimes(1)
    })
    const errorArg = mockedToast.error.mock.calls[0][0]
    expect(errorArg).toMatch(/Input should be /)
    // Critical: the current card stays on screen. The user can
    // re-click a grade button without losing their place.
    expect(screen.getByTestId('cloze-ready')).toBeInTheDocument()
    expect(screen.getByTestId('cloze-choice-42')).toBeInTheDocument()
    // getDueCloze is NOT called on failure — the grade never
    // landed, so there's no "next card" to fetch.
    expect(mockedGetDueCloze).not.toHaveBeenCalled()
  })

  it('500 on gradeCloze shows a generic toast and keeps the current card', async () => {
    const user = userEvent.setup()
    mockedGenerateCloze.mockResolvedValue(sampleExercise)
    mockedGradeCloze.mockRejectedValue(
      new ClozeApiError(500, 'Request failed (500)'),
    )

    renderPage()
    await waitFor(() => {
      expect(screen.getByTestId('cloze-ready')).toBeInTheDocument()
    })

    await user.click(screen.getByTestId('grade-button-3'))

    await waitFor(() => {
      expect(mockedGradeCloze).toHaveBeenCalledTimes(1)
    })
    await waitFor(() => {
      expect(mockedToast.error).toHaveBeenCalledTimes(1)
    })
    // The 500 path uses the generic copy — we do NOT surface
    // the "Request failed (500)" string to the user.
    expect(mockedToast.error).toHaveBeenCalledWith('Grade failed — try again')
    // The current card is preserved.
    expect(screen.getByTestId('cloze-ready')).toBeInTheDocument()
    expect(screen.getByTestId('cloze-choice-42')).toBeInTheDocument()
    // The grade buttons re-enable (isGrading flips back to false
    // in the finally block) so the user can retry.
    expect(screen.getByTestId('grade-button-3')).not.toBeDisabled()
  })

  it('does not re-fire gradeCloze on a second click while the first is in flight (isGrading lock)', async () => {
    const user = userEvent.setup()
    mockedGenerateCloze.mockResolvedValue(sampleExercise)
    // Defer the gradeCloze promise so we can click Good twice
    // before the first resolves. Two concurrent gradeCloze
    // calls on the same word would double-schedule the card
    // on the FSRS side.
    const gradeDeferred: {
      resolve: ((value: GradeResponse) => void) | null
    } = { resolve: null }
    mockedGradeCloze.mockImplementation(
      () =>
        new Promise<GradeResponse>((resolve) => {
          gradeDeferred.resolve = resolve
        }),
    )
    mockedGetDueCloze.mockResolvedValue(secondDueCloze)

    renderPage()
    await waitFor(() => {
      expect(screen.getByTestId('cloze-ready')).toBeInTheDocument()
    })

    // First click: gradeCloze is called once.
    await user.click(screen.getByTestId('grade-button-3'))
    await waitFor(() => {
      expect(mockedGradeCloze).toHaveBeenCalledTimes(1)
    })

    // Second click while the first is still in flight. The
    // handler's `if (isGrading) return` guard must drop it.
    await user.click(screen.getByTestId('grade-button-3'))
    // Tiny wait so any (incorrectly-permitted) second call
    // would have flushed.
    await new Promise((resolve) => setTimeout(resolve, 10))
    expect(mockedGradeCloze).toHaveBeenCalledTimes(1)

    // Resolve the deferred promise so the component settles.
    if (!gradeDeferred.resolve) {
      throw new Error('gradeDeferred.resolve was never assigned')
    }
    gradeDeferred.resolve(sampleGradeResponse)
    await waitFor(() => {
      expect(screen.getByTestId('cloze-choice-99')).toBeInTheDocument()
    })
  })
})