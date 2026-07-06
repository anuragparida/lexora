import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { SessionPage } from '../../pages/SessionPage'
import type {
  ComprehensionExercise,
  IdiomExercise,
  MatchingExercise,
} from '../../api/exercises'

// Phase 9.6 (card t_f1c63bfc): SessionPage integration test.
//
// What we exercise:
//   1. Mount with a cloze pick: SessionPage renders the
//      shared ExerciseCard for the cloze body.
//   2. Grade the cloze: SessionPage re-fetches the next pick
//      and renders a matching exercise (the per-type endpoint
//      fires).
//   3. End Session: navigate back to home.
//
// We mock the network layer (getNextDuePick + per-type
// generators + gradeExercise) so the test runs offline. The
// shared ``ExerciseCard`` is rendered for real so we exercise
// the card's grade bar + onGraded callback wiring end-to-end.

// ----- mock shape helpers -----------------------------------------------

const clozeExerciseOut = {
  word_id: 42,
  sentence_with_blank: 'Der Kandidat ___ den Vertrag.',
  answer_word_id: 42,
  distractors: [1042, 2087, 3155] as [number, number, number],
  difficulty: 'medium' as const,
  rationale: 'Sentence cues "unterzeichnen" via accusative object.',
  prompt_template_version: 'cloze-v1',
  due_from_fsrs: true,
}

const matchingExercise: MatchingExercise = {
  exercise_type: 'matching',
  exercise_id: 100,
  target_word_id: 17,
  prompt_template_version: 'match-v1',
  enable_rag: false,
  trace_id: 'trace-match-1',
  latency_ms: 240,
  pairs: [
    { left_word_id: 17, right_word_id: 118, right_kind: 'translation' },
    { left_word_id: 22, right_word_id: 119, right_kind: 'translation' },
    { left_word_id: 31, right_word_id: 120, right_kind: 'synonym' },
    { left_word_id: 44, right_word_id: 121, right_kind: 'translation' },
  ],
  partner_translation: null,
}

const comprehensionExercise: ComprehensionExercise = {
  exercise_type: 'comprehension',
  exercise_id: 200,
  target_word_id: 99,
  prompt_template_version: 'comp-v1',
  enable_rag: false,
  trace_id: 'trace-comp-1',
  latency_ms: 312,
  passage: 'Der Architekt plant das neue Gebäude.',
  question: 'Was plant der Architekt?',
  choices: {
    A: 'Eine Straße',
    B: 'Ein Gebäude',
    C: 'Eine Brücke',
    D: 'Einen Park',
  },
  correct_choice: 'B',
  rationale: 'The passage names "das neue Gebäude".',
}

const idiomExercise: IdiomExercise = {
  exercise_type: 'idiom',
  exercise_id: 300,
  word_id: 88,
  target_word_id: 88,
  prompt_template_version: 'idiom-v1',
  enable_rag: false,
  trace_id: 'trace-idiom-1',
  latency_ms: 188,
  phrase: 'Tomaten auf den Augen',
  definition: 'Unable to see what is obvious.',
  example_usage: 'Er hatte Tomaten auf den Augen und übersah den Fehler.',
  source_attribution: 'dwds',
  attested_quote: null,
  attested_source: null,
  frequency_band: 'high',
  cloze_target: null,
}

vi.mock('../../api/session', async () => {
  const actual = await vi.importActual<typeof import('../../api/session')>(
    '../../api/session',
  )
  return {
    ...actual,
    getNextDuePick: vi.fn(),
    gradeSessionExercise: vi.fn(),
  }
})

vi.mock('../../api/exercises', async () => {
  const actual = await vi.importActual<typeof import('../../api/exercises')>(
    '../../api/exercises',
  )
  return {
    ...actual,
    generateMatch: vi.fn(),
    generateComprehension: vi.fn(),
    generateIdiom: vi.fn(),
    generatePhraseMatch: vi.fn(),
    gradeExercise: vi.fn(),
  }
})

import { getNextDuePick, gradeSessionExercise } from '../../api/session'
import {
  generateComprehension,
  generateIdiom,
  generateMatch,
  generatePhraseMatch,
  gradeExercise,
} from '../../api/exercises'

const mockedGetNextDuePick = vi.mocked(getNextDuePick)
const mockedGradeSessionExercise = vi.mocked(gradeSessionExercise)
const mockedGenerateMatch = vi.mocked(generateMatch)
const mockedGenerateComprehension = vi.mocked(generateComprehension)
const mockedGenerateIdiom = vi.mocked(generateIdiom)
const mockedGeneratePhraseMatch = vi.mocked(generatePhraseMatch)
const mockedGradeExercise = vi.mocked(gradeExercise)

// ----- test scaffolding -------------------------------------------------

function renderSession(initialPath = '/exercises/session') {
  return render(
    <MemoryRouter initialEntries={[initialPath]}>
      <Routes>
        <Route
          path="/exercises/session"
          element={<SessionPage />}
        />
        <Route path="/" element={<div>HOME_PAGE</div>} />
        <Route path="/login" element={<div>LOGIN_PAGE</div>} />
      </Routes>
    </MemoryRouter>,
  )
}

describe('SessionPage mixer (Phase 9.6)', () => {
  beforeEach(() => {
    mockedGetNextDuePick.mockReset()
    mockedGradeSessionExercise.mockReset()
    mockedGenerateMatch.mockReset()
    mockedGenerateComprehension.mockReset()
    mockedGenerateIdiom.mockReset()
    mockedGeneratePhraseMatch.mockReset()
    mockedGradeExercise.mockReset()
  })

  afterEach(() => {
    cleanup()
  })

  it('renders the head pick via the shared ExerciseCard and advances on grade', async () => {
    // First pick: cloze (inline body, no per-type call).
    // Second pick: matching (204 + headers → generateMatch).
    // Third pick: empty (end-of-session state).
    mockedGetNextDuePick
      .mockResolvedValueOnce({
        kind: 'pick',
        pick: { kind: 'cloze', exercise: clozeExerciseOut },
      })
      .mockResolvedValueOnce({
        kind: 'pick',
        pick: { kind: 'matching', card_id: 100, word_id: 17 },
      })
      .mockResolvedValueOnce({ kind: 'empty' })

    mockedGenerateMatch.mockResolvedValue(matchingExercise)
    mockedGradeExercise.mockResolvedValue({
      graded: true,
      exercise_id: 42,
      exercise_type: 'cloze',
      next_due_at: '2026-07-12T10:00:00Z',
      card_state: 1,
      stability: 1.0,
      difficulty: 5.0,
      trace_id: 'trace-cloze-1',
    })

    renderSession()

    // First pick: cloze card renders. The shared card exposes
    // a testid like ``exercise-card-cloze`` (Phase 9.5).
    await waitFor(() => {
      expect(
        screen.getByTestId('exercise-card-cloze'),
      ).toBeInTheDocument()
    })

    // Grade the cloze. The shared card's GradeButtons owns the
    // 1/2/3/4 buttons; "Easy" is the 4th.
    const user = userEvent.setup()
    await user.click(screen.getByTestId('grade-button-4'))

    // After grade, the page re-fetches. The matching pick
    // surfaces and the page calls generateMatch({count:4}).
    await waitFor(() => {
      expect(mockedGenerateMatch).toHaveBeenCalledTimes(1)
    })
    await waitFor(() => {
      expect(
        screen.getByTestId('exercise-card-matching'),
      ).toBeInTheDocument()
    })

    // The grading card fired POST /exercises/grade once (for
    // the cloze). gradeExercise is the shared module the card
    // imports.
    expect(mockedGradeExercise).toHaveBeenCalledWith(
      'cloze',
      42, // answer_word_id, used as exercise_id for cloze picks
      4, // the "Easy" grade
    )

    // Grade the matching exercise. This advances again into
    // the empty state.
    mockedGradeExercise.mockResolvedValueOnce({
      graded: true,
      exercise_id: 100,
      exercise_type: 'matching',
      next_due_at: '2026-07-13T10:00:00Z',
      card_state: 2,
      stability: 2.5,
      difficulty: 4.5,
      trace_id: 'trace-match-grade-1',
    })
    await user.click(screen.getByTestId('grade-button-4'))

    // Empty state surfaces with the "Session complete" copy.
    await waitFor(() => {
      expect(screen.getByTestId('session-done')).toBeInTheDocument()
    })
    expect(
      screen.queryByTestId('exercise-card-matching'),
    ).not.toBeInTheDocument()
    // The session fired two /exercises/due fetches (one per
    // grade) and a third that returned empty.
    expect(mockedGetNextDuePick).toHaveBeenCalledTimes(3)
  })

  it('resolves a matching pick to the per-type generator (no word_id passed)', async () => {
    // Phase 9.2's read-side widening — the per-type endpoint
    // for matching doesn't accept ``force_word_id``, so the
    // page just calls generateMatch() with the default
    // ``{count:4}`` body.
    mockedGetNextDuePick.mockResolvedValueOnce({
      kind: 'pick',
      pick: { kind: 'matching', card_id: 100, word_id: 17 },
    })
    mockedGenerateMatch.mockResolvedValue(matchingExercise)

    renderSession()

    await waitFor(() => {
      expect(
        screen.getByTestId('exercise-card-matching'),
      ).toBeInTheDocument()
    })
    expect(mockedGenerateMatch).toHaveBeenCalledWith({ count: 4 })
  })

  it('resolves a comprehension pick to the per-type generator (no word_id)', async () => {
    mockedGetNextDuePick.mockResolvedValueOnce({
      kind: 'pick',
      pick: { kind: 'comprehension', card_id: 200, word_id: 99 },
    })
    mockedGenerateComprehension.mockResolvedValue(comprehensionExercise)

    renderSession()

    await waitFor(() => {
      expect(
        screen.getByTestId('exercise-card-comprehension'),
      ).toBeInTheDocument()
    })
    expect(mockedGenerateComprehension).toHaveBeenCalledWith({})
  })

  it('resolves an idiom pick by passing word_id through (Phase 8.4 requirement)', async () => {
    // Phase 8.4 made word_id required on /exercises/idiom;
    // Phase 9.6 forwards the queue's word_id from the
    // X-Due-Word-Id header so the generator's phrases
    // filter is anchored.
    mockedGetNextDuePick.mockResolvedValueOnce({
      kind: 'pick',
      pick: { kind: 'idiom', card_id: 300, word_id: 88 },
    })
    mockedGenerateIdiom.mockResolvedValue(idiomExercise)

    renderSession()

    await waitFor(() => {
      expect(
        screen.getByTestId('exercise-card-idiom'),
      ).toBeInTheDocument()
    })
    expect(mockedGenerateIdiom).toHaveBeenCalledWith({ word_id: 88 })
  })

  it('shows the empty-state when /exercises/due returns 204 (no headers)', async () => {
    mockedGetNextDuePick.mockResolvedValueOnce({ kind: 'empty' })

    renderSession()

    await waitFor(() => {
      expect(screen.getByTestId('session-done')).toBeInTheDocument()
    })
    expect(
      screen.queryByTestId('exercise-card-cloze'),
    ).not.toBeInTheDocument()
  })

  it('End session navigates to home', async () => {
    mockedGetNextDuePick.mockResolvedValueOnce({
      kind: 'pick',
      pick: { kind: 'cloze', exercise: clozeExerciseOut },
    })

    renderSession()

    await waitFor(() => {
      expect(
        screen.getByTestId('exercise-card-cloze'),
      ).toBeInTheDocument()
    })

    const user = userEvent.setup()
    await user.click(screen.getByTestId('session-end'))

    await waitFor(() => {
      expect(screen.getByText('HOME_PAGE')).toBeInTheDocument()
    })
  })

  it('shows the error surface when /exercises/due returns an error', async () => {
    mockedGetNextDuePick.mockResolvedValueOnce({
      kind: 'error',
      status: 502,
      message: 'upstream LLM down',
    })

    renderSession()

    await waitFor(() => {
      expect(screen.getByTestId('session-error')).toBeInTheDocument()
    })
    // The error body carries the upstream message verbatim.
    expect(screen.getByText(/upstream LLM down/i)).toBeInTheDocument()
  })
})

// ===========================================================================
// Phase 10.6 (card t_da43cc23) — SessionPage mixer widens to
// ``phrase_match`` (the 5th exercise type). The mixer mounts
// ``<PhraseMatchPage />`` directly (instead of the shared
// ``<ExerciseCard />``) because the bespoke 4-button relation
// picker is the page's job; the page receives the queue-supplied
// ``word_id`` and fires ``onGraded`` / ``onGradeError`` callbacks
// back to the mixer so the queue advances uniformly across all
// 5 types.
// ===========================================================================

const phraseMatchExercise = {
  exercise_type: 'phrase_match' as const,
  exercise_id: 700,
  target_word_id: 12,
  word_id: 12,
  prompt_template_version: 'phrase_match-v1',
  enable_rag: false,
  trace_id: 'trace-pm-mixer-1',
  latency_ms: 220,
  phrase_a: 'ins Blaue hinein',
  phrase_b: 'ohne klares Ziel',
  relation: 'paraphrase' as const,
  relation_rationale: 'Both describe acting without a clear plan.',
  source_attribution: 'dwds',
}

describe('SessionPage mixer (Phase 10.6 phrase_match widening)', () => {
  beforeEach(() => {
    mockedGetNextDuePick.mockReset()
    mockedGradeSessionExercise.mockReset()
    mockedGenerateMatch.mockReset()
    mockedGenerateComprehension.mockReset()
    mockedGenerateIdiom.mockReset()
    mockedGeneratePhraseMatch.mockReset()
    mockedGradeExercise.mockReset()
  })

  afterEach(() => {
    cleanup()
  })

  it('resolves a phrase_match pick to the per-type generator (queue-supplied word_id)', async () => {
    // Phase 10.6 (card t_da43cc23) — the mixer's
    // ``resolvePickBody`` calls ``generatePhraseMatch({ word_id
    // : pick.word_id })`` for phrase_match picks. The word_id
    // flows from the queue's X-Due-Word-Id header into the
    // generator's ``select_phrase_pair`` seed (Phase 10.3).
    mockedGetNextDuePick.mockResolvedValueOnce({
      kind: 'pick',
      pick: { kind: 'phrase_match', card_id: 700, word_id: 12 },
    })
    mockedGeneratePhraseMatch.mockResolvedValue(phraseMatchExercise)

    renderSession()

    await waitFor(() => {
      expect(
        screen.getByTestId('phrase-match-ready'),
      ).toBeInTheDocument()
    })
    // The mixer passed the queue-supplied word_id through to
    // the per-type endpoint (not the Phase 10.5 hardcoded
    // ``word_id=1`` fallback).
    expect(mockedGeneratePhraseMatch).toHaveBeenCalledWith({ word_id: 12 })
  })

  it('mounts <PhraseMatchPage /> for a phrase_match pick (not <ExerciseCard />)', async () => {
    // Phase 10.6 (card t_da43cc23) — the mixer's render
    // branch swaps to ``<PhraseMatchPage />`` for
    // phrase_match picks. The bespoke relation picker is
    // rendered; the shared ExerciseCard's wire-only
    // phrase_match branch (which displays a "use the dedicated
    // surface" hint) does NOT appear.
    mockedGetNextDuePick.mockResolvedValueOnce({
      kind: 'pick',
      pick: { kind: 'phrase_match', card_id: 700, word_id: 12 },
    })
    mockedGeneratePhraseMatch.mockResolvedValue(phraseMatchExercise)

    renderSession()

    await waitFor(() => {
      expect(
        screen.getByTestId('phrase-match-ready'),
      ).toBeInTheDocument()
    })
    // The 4-button relation picker mounts — this is the
    // PhraseMatchPage's render signature. ExerciseCard does
    // NOT render this.
    expect(
      screen.getByTestId('phrase-match-relation-picker'),
    ).toBeInTheDocument()
    // The page's two-phrase cards mount with the per-type
    // testids (phrase-a / phrase-b).
    expect(screen.getByTestId('phrase-a-phrase')).toHaveTextContent(
      'ins Blaue hinein',
    )
    expect(screen.getByTestId('phrase-b-phrase')).toHaveTextContent(
      'ohne klares Ziel',
    )
  })
})
