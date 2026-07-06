import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { PhraseMatchPage } from '../../pages/PhraseMatchPage'
import {
  ExerciseApiError,
  type PhraseMatchExercise,
} from '../../api/exercises'

// Phase 10.5 (card t_ca1d2da8) — PhraseMatchPage smoke.
//
// What we exercise:
//   1. Mount: the page POSTs /exercises/phrase_match and renders
//      the two phrases + 4-button relation picker.
//   2. Pick a relation: the local state updates (aria-checked
//      flips, the grade bar unlocks).
//   3. Pick a grade: submitPhraseMatchGrade fires with the 5th
//      literal + the relation as the answer + the grade on a
//      single round trip. The page transitions to the
//      post-grade empty state with the "Open session mixer"
//      CTA (Phase 9.6 hand-off).
//
// We mock the api/exercises module — same shape as the
// SessionPage test — so the test runs offline. The shared
// <GradeButtons/> is rendered for real so we exercise the
// relation-picked-locks-grade-bar wiring end-to-end.

// ----- mock shape helpers -----------------------------------------------

const phraseMatchExercise: PhraseMatchExercise = {
  exercise_type: 'phrase_match',
  exercise_id: 700,
  target_word_id: 12,
  word_id: 12,
  prompt_template_version: 'phrase_match-v1',
  enable_rag: false,
  trace_id: 'trace-pm-1',
  latency_ms: 220,
  phrase_a: 'ins Blaue hinein',
  phrase_b: 'ohne klares Ziel',
  relation: 'paraphrase',
  relation_rationale: 'Both phrases describe acting without a clear plan or goal.',
  source_attribution: 'dwds',
}

const gradeResponse = {
  graded: true as const,
  exercise_id: 700,
  exercise_type: 'phrase_match' as const,
  next_due_at: '2026-07-13T10:00:00Z',
  card_state: 2,
  stability: 2.5,
  difficulty: 4.5,
  trace_id: 'trace-pm-grade-1',
}

vi.mock('../../api/exercises', async () => {
  const actual = await vi.importActual<typeof import('../../api/exercises')>(
    '../../api/exercises',
  )
  return {
    ...actual,
    generatePhraseMatch: vi.fn(),
    submitPhraseMatchGrade: vi.fn(),
  }
})

import {
  generatePhraseMatch,
  submitPhraseMatchGrade,
} from '../../api/exercises'

const mockedGeneratePhraseMatch = vi.mocked(generatePhraseMatch)
const mockedSubmitPhraseMatchGrade = vi.mocked(submitPhraseMatchGrade)

// ----- test scaffolding -------------------------------------------------

function renderPhraseMatch(initialPath = '/exercises/phrase_match') {
  return render(
    <MemoryRouter initialEntries={[initialPath]}>
      <Routes>
        <Route
          path="/exercises/phrase_match"
          element={<PhraseMatchPage />}
        />
        <Route
          path="/exercises/session"
          element={<div data-testid="session-mixer">SESSION_MIXER</div>}
        />
        <Route
          path="/login"
          element={<div data-testid="login-page">LOGIN_PAGE</div>}
        />
        <Route path="/" element={<div data-testid="home">HOME</div>} />
      </Routes>
    </MemoryRouter>,
  )
}

describe('PhraseMatchPage (Phase 10.5)', () => {
  beforeEach(() => {
    mockedGeneratePhraseMatch.mockReset()
    mockedSubmitPhraseMatchGrade.mockReset()
  })

  afterEach(() => {
    cleanup()
  })

  it('renders the two-phrase layout + 4-button relation picker on mount', async () => {
    mockedGeneratePhraseMatch.mockResolvedValueOnce(phraseMatchExercise)

    renderPhraseMatch()

    // Phase 10.5 page exposes the load state first; we wait
    // for the ready state where the two phrases + the picker
    // are visible.
    await waitFor(() => {
      expect(
        screen.getByTestId('phrase-match-ready'),
      ).toBeInTheDocument()
    })

    // Both phrases render with the per-type phrase-a/phrase-b
    // testids.
    expect(
      screen.getByTestId('phrase-a-phrase'),
    ).toHaveTextContent('ins Blaue hinein')
    expect(
      screen.getByTestId('phrase-b-phrase'),
    ).toHaveTextContent('ohne klares Ziel')

    // The 4-button relation picker groups under the
    // ``phrase-match-relation-picker`` testid. Each button
    // has the canonical relation literal as ``data-relation``
    // so the wire-shape audit matches the displayed label.
    const picker = screen.getByTestId('phrase-match-relation-picker')
    expect(picker).toBeInTheDocument()
    expect(
      screen.getByTestId('phrase-match-relation-equivalent'),
    ).toHaveAttribute('data-relation', 'equivalent')
    expect(
      screen.getByTestId('phrase-match-relation-paraphrase'),
    ).toHaveAttribute('data-relation', 'paraphrase')
    expect(
      screen.getByTestId('phrase-match-relation-related'),
    ).toHaveAttribute('data-relation', 'related')
    expect(
      screen.getByTestId('phrase-match-relation-unrelated'),
    ).toHaveAttribute('data-relation', 'unrelated')

    // The grade bar should be locked while no relation is
    // picked — the relation gate is the strict discipline.
    const gradeButton = screen.getByTestId('grade-button-3')
    expect(gradeButton).toBeDisabled()

    // The page wired the generator with the Phase 10.5
    // ``word_id=1`` fallback (matches IdiomPage's
    // self-contained surface).
    expect(mockedGeneratePhraseMatch).toHaveBeenCalledWith({
      word_id: 1,
    })
  })

  it('unlocks the grade bar after a relation is picked and fires the grade call', async () => {
    mockedGeneratePhraseMatch.mockResolvedValueOnce(phraseMatchExercise)
    mockedSubmitPhraseMatchGrade.mockResolvedValueOnce(gradeResponse)

    renderPhraseMatch()
    await waitFor(() => {
      expect(
        screen.getByTestId('phrase-match-ready'),
      ).toBeInTheDocument()
    })

    const user = userEvent.setup()

    // Click "paraphrase" — the picker reflects the picked
    // relation via aria-checked.
    const paraphrase = screen.getByTestId(
      'phrase-match-relation-paraphrase',
    )
    await user.click(paraphrase)
    expect(paraphrase).toHaveAttribute('aria-checked', 'true')

    // Grade bar is now enabled (the picked relation unlocks it).
    const goodButton = screen.getByTestId('grade-button-3')
    expect(goodButton).toBeEnabled()

    // Click "Good" (3). The grade call fires with the 5th
    // literal, the relation as the answer, and the grade.
    await user.click(goodButton)

    expect(mockedSubmitPhraseMatchGrade).toHaveBeenCalledWith(
      700, // exercise_id
      'paraphrase', // relation
      3, // the grade
    )

    // Post-grade empty state surfaces with the Open-session
    // mixer CTA (Phase 9.6 hand-off).
    await waitFor(() => {
      expect(
        screen.getByTestId('phrase-match-generate-another-empty'),
      ).toBeInTheDocument()
    })
    expect(
      screen.getByTestId('phrase-match-go-to-session-empty'),
    ).toBeInTheDocument()
  })

  it('shows the notFound state when the route returns 404', async () => {
    mockedGeneratePhraseMatch.mockRejectedValueOnce(
      new ExerciseApiError(404, 'no phrase pair'),
    )

    renderPhraseMatch()

    await waitFor(() => {
      expect(
        screen.getByTestId('phrase-match-not-found'),
      ).toBeInTheDocument()
    })
  })

  it('surfaces the error state and retries via "Generate another"', async () => {
    mockedGeneratePhraseMatch
      .mockRejectedValueOnce(
        new Error('upstream DSPy exploded'),
      )
      .mockResolvedValueOnce(phraseMatchExercise)

    renderPhraseMatch()

    await waitFor(() => {
      expect(screen.getByTestId('phrase-match-error')).toBeInTheDocument()
    })
    expect(screen.getByText(/upstream DSPy exploded/i)).toBeInTheDocument()

    // "Retry" — bumps the fetchAttempt, which re-runs the load
    // effect. The second mock returns the success exercise.
    const user = userEvent.setup()
    await user.click(screen.getByRole('button', { name: /retry/i }))

    await waitFor(() => {
      expect(
        screen.getByTestId('phrase-match-ready'),
      ).toBeInTheDocument()
    })
  })

  it('redirects to /login on 401', async () => {
    mockedGeneratePhraseMatch.mockRejectedValueOnce(
      new ExerciseApiError(401, 'not authenticated'),
    )

    renderPhraseMatch()

    await waitFor(() => {
      expect(screen.getByTestId('phrase-match-error')).toBeInTheDocument()
    })

    const user = userEvent.setup()
    await user.click(screen.getByRole('button', { name: /go to login/i }))

    await waitFor(() => {
      expect(screen.getByTestId('login-page')).toBeInTheDocument()
    })
  })
})
