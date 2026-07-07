import { useCallback, useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { toast } from 'sonner'
import {
  ExerciseApiError,
  generatePhraseMatch,
  submitPhraseMatchGrade,
  type PhraseMatchExercise,
  type PhrasePairRelation,
} from '../api/exercises'
import { GradeButtons } from '../components/GradeButtons'
import { humanizeDelta } from '../components/delta'

// Phase 10.5 (card t_ca1d2da8) — thin PhraseMatchPage.
// Phase 10.6 (card t_da43cc23) widens the page to be mountable
// inside the SessionPage mixer via three opt-in props:
// ``word_id``, ``onGraded``, ``onGradeError``. When the mixer
// mounts the page (all three props supplied), the page drops
// its hardcoded ``word_id=1`` fallback for the queue-supplied
// ``word_id``, suppresses the post-grade empty state, and
// fires the callbacks so the mixer can advance. When any of
// the three props is missing, the page falls back to its
// Phase 10.5 standalone surface (hardcoded ``word_id=1`` for
// self-contained testability + the "Generate another / Open
// session mixer" empty-state CTAs). The existing Phase 10.5
// tests exercise the standalone branch and stay unchanged.
//
// Mirrors IdiomPage's state machine (loading / ready / error /
// empty / notFound) so the five per-type pages share the same
// render-tree shape and Phase 9.6's ``SessionPage`` mixer can
// compose them without knowing the per-type differences.
//
// Wire contract: POST /exercises/phrase_match (Phase 10.3,
// ``t_13bb48d2``) with the Phase 10.5 ``PhraseMatchGenerateRequest``
// body. ``word_id`` is REQUIRED (Phase 8.4 idiom discipline —
// empty bodies 422). Response: ``PhraseMatchExercise`` (10.5's
// mirror of Pydantic ``PhraseMatchExerciseOut``, which 10.3 will
// ship).
//
// Wire-flow discipline (Phase 9.6):
//
//   1. On mount: POST /exercises/phrase_match with
//      ``enable_rag`` from the user preference + ``word_id`` from
//      the next-due scheduler. Today, with Phase 9.6 still
//      routing one type per page, we use the same fixed-low
//      ``word_id=1`` fallback IdiomPage uses (Phase 9.5's
//      self-contained surface).
//   2. Display the two phrases + the 4-button relation picker.
//      (Hover-revealed definitions — Phase 8 idiom pattern.)
//   3. User picks a relation → the relation choice is captured
//      locally. The relation is the *answer* (a closed literal
//      from Phase 10.1's ``phrase_pairs.relation`` column).
//   4. User picks a grade → BOTH the relation (the answer) AND
//      the grade (the self-assessment) are submitted to
//      /exercises/grade in one call (Phase 9.6 discipline —
//      ``answer`` field on the body, alongside ``grade``).
//   5. On success: navigate to the session mixer or render a
//      post-grade empty state (the page is standalone-testable:
//      a "Generate another" button + a router-push to
//      /exercises/session is the canonical 10.6 mixer hand-off).
//
// Loading + error states mirror the Phase 9 idiom / matching
// / comprehension surface. 404 (no pair row for the supplied
// word_id) surfaces as ``status === 'notFound'`` with a "Try
// another word" CTA — same discipline as ``IdiomNotFoundError``
// in ``IdiomPage``.

type Status =
  | 'idle'
  | 'loading'
  | 'ready'
  | 'error'
  | 'empty'
  | 'notFound'

// 4-way relation literal → learner-friendly German label.
// The English literal is also rendered on the wire (the data
// attribute) so Phase 10.4's hand-labeled eval set can match
// the displayed-buttons to the recorded relation without
// re-deriving it from the LTR text.
const RELATION_BUTTONS: ReadonlyArray<{
  relation: PhrasePairRelation
  label: string
}> = [
  { relation: 'equivalent', label: 'bedeutungsgleich' },
  { relation: 'paraphrase', label: 'Umschreibung' },
  { relation: 'related', label: 'verwandt' },
  { relation: 'unrelated', label: 'unrelated' },
]

// Phase 10.5 (card t_ca1d2da8): the closed 4-way relation
// literal the learner picks from. Mirrors Phase 10.1's
// ``phrase_pairs.relation`` column. The English literal is
// also rendered on the wire (the data attribute) so Phase 10.4's
// hand-labeled eval set can match the displayed-buttons to the
// recorded relation without re-deriving it from the LTR text.
//
// Phase 10.6 (card t_da43cc23) — re-exported on the props
// interface below so the SessionPage mixer can read the type
// directly when checking which per-type page to mount. Stays
// a closed literal; the mixer's switch on
// ``exercise.exercise_type === 'phrase_match'`` is exhaustive.
export type { PhrasePairRelation }

// Phase 10.6 (card t_da43cc23) — the three opt-in props that
// widen PhraseMatchPage from a standalone page into a
// mixer-mountable component. All three are optional; when any
// is missing the page falls back to its Phase 10.5 standalone
// surface (hardcoded ``word_id=1`` for self-contained testing,
// post-grade empty state with "Generate another / Open session
// mixer" CTAs).
//
// The mixer always supplies all three: ``word_id`` from the
// queue pick, ``onGraded`` for the mixer's ``setFetchAttempt``,
// ``onGradeError`` for the mixer's toast / 401-bounce handler.
// Standalone test users (the Phase 10.5 surface mounted at
// ``/exercises/phrase_match``) supply none of them.
export interface PhraseMatchPageProps {
  /**
   * Phase 10.6 — queue-supplied ``word_id`` from the mixer
   * (replaces the hardcoded ``word_id=1`` fallback used in the
   * Phase 10.5 standalone surface). When omitted, the page
   * falls back to ``word_id=1`` (matches IdiomPage's
   * self-contained testability pattern).
   */
  word_id?: number
  /**
   * Phase 10.6 — fired after the grade round-trip succeeds
   * with the post-grade ``next_due_at``. The mixer uses this
   * to bump ``fetchAttempt`` and fetch the next pick. When
   * omitted, the page renders its standalone post-grade
   * empty-state surface (the "Generate another / Open session
   * mixer" CTA).
   */
  onGraded?: (next_due_at: string) => void
  /**
   * Phase 10.6 — fired when the grade round-trip throws. The
   * mixer uses this to dispatch the 401-bounce / 422 toast
   * (mirrors the SessionPage's own
   * ``handleGradeError``). When omitted, the page renders
   * its standalone error surface (the "Retry" CTA).
   */
  onGradeError?: (err: unknown) => void
}

export function PhraseMatchPage({
  word_id: wordIdProp,
  onGraded,
  onGradeError,
}: PhraseMatchPageProps = {}) {
  const navigate = useNavigate()
  const [status, setStatus] = useState<Status>('loading')
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [exercise, setExercise] =
    useState<PhraseMatchExercise | null>(null)
  const [pickedRelation, setPickedRelation] =
    useState<PhrasePairRelation | null>(null)
  const [isGrading, setIsGrading] = useState(false)
  // ``fetchAttempt`` starts at 1 so the load effect below
  // fires on mount (same pattern as IdiomPage's
  // ``setFetchAttempt(1)`` in the profile-check effect, but
  // without a side-effect-driven kick — Phase 10.5 has no
  // profile gate so the load runs immediately on mount).
  const [fetchAttempt, setFetchAttempt] = useState(1)
  // Distinguishes "post-grade empty" (we just submitted a
  // grade and want the surface that points to the next pair
  // *or* the session mixer hand-off) from "initial empty"
  // (no fetch yet — show the start page). The state
  // disciplines the right empty-state testid below.
  const [hasGraded, setHasGraded] = useState(false)

  useEffect(() => {
    if (fetchAttempt === 0) return
    if (status === 'empty') return
    if (status === 'notFound') return
    if (status !== 'loading') return
    const token = fetchAttempt
    let cancelled = false
    // Phase 10.6 (card t_da43cc23) — the page picks the
    // queue-supplied ``word_id`` from the mixer's prop when
    // mounted inside the session mixer; otherwise it falls back
    // to the Phase 10.5 self-contained ``word_id=1`` (matches
    // IdiomPage's standalone surface for hermetic testing).
    const word_id = wordIdProp ?? 1
    generatePhraseMatch({ word_id })
      .then((m) => {
        if (cancelled) return
        if (token !== fetchAttempt) return
        setPickedRelation(null)
        setExercise(m)
        setStatus('ready')
      })
      .catch((err: unknown) => {
        if (cancelled) return
        if (token !== fetchAttempt) return
        if (err instanceof ExerciseApiError && err.status === 404) {
          setStatus('notFound')
          return
        }
        setErrorMessage(
          err instanceof Error ? err.message : 'Unexpected error',
        )
        setStatus('error')
      })
    return () => {
      cancelled = true
    }
  }, [fetchAttempt, status, wordIdProp])

  const handleGraded = useCallback(
    (next_due_at: string) => {
      setExercise(null)
      setPickedRelation(null)
      setHasGraded(true)
      // Phase 10.6 (card t_da43cc23) — when the mixer mounts
      // the page (via ``onGraded`` callback), we signal the
      // mixer and skip the post-grade empty state. The
      // standalone branch keeps the existing empty-state
      // surface ("Generate another / Open session mixer").
      if (onGraded) {
        onGraded(next_due_at)
        return
      }
      setStatus('empty')
      toast.success(
        `Grade recorded. Next review ${humanizeDelta(next_due_at)}.`,
      )
    },
    [onGraded],
  )

  const handleGradeError = useCallback(
    (err: unknown) => {
      // Phase 10.6 (card t_da43cc23) — when the mixer mounts
      // the page (via ``onGradeError`` callback), we delegate
      // to the mixer and skip the standalone error surface.
      // The mixer's ``handleGradeError`` owns the 401-bounce
      // + 422 toast discipline.
      if (onGradeError) {
        onGradeError(err)
        return
      }
      const errStatus =
        err instanceof ExerciseApiError ? err.status : undefined
      if (errStatus === 401) {
        navigate('/login', {
          replace: true,
          state: { from: '/exercises/phrase_match' },
        })
        return
      }
      if (errStatus === 422) {
        toast.error(
          err instanceof Error ? err.message : 'Grade validation failed',
        )
      } else {
        toast.error('Grade failed — try again')
      }
    },
    [navigate, onGradeError],
  )

  const handlePickRelation = useCallback((relation: PhrasePairRelation) => {
    setPickedRelation(relation)
  }, [])

  const handlePickGrade = useCallback(
    async (grade: 1 | 2 | 3 | 4) => {
      if (!exercise) return
      if (pickedRelation === null) {
        toast.error('Pick a relation first, then grade.')
        return
      }
      if (isGrading) return
      setIsGrading(true)
      try {
        const response = await submitPhraseMatchGrade(
          exercise.exercise_id,
          pickedRelation,
          grade,
        )
        handleGraded(response.next_due_at)
      } catch (err) {
        handleGradeError(err)
      } finally {
        setIsGrading(false)
      }
    },
    [exercise, pickedRelation, isGrading, handleGraded, handleGradeError],
  )

  function handleGenerateAnother() {
    setErrorMessage(null)
    setPickedRelation(null)
    setStatus('loading')
    setFetchAttempt((n) => n + 1)
  }

  function handleRetry() {
    setErrorMessage(null)
    setPickedRelation(null)
    setStatus('loading')
    setFetchAttempt((n) => n + 1)
  }

  function handleGoToSession() {
    navigate('/exercises/session')
  }

  function handleRedirectToLogin() {
    navigate('/login', {
      replace: true,
      state: { from: '/exercises/phrase_match' },
    })
  }

  if (status === 'notFound') {
    return (
      <div
        className="max-w-2xl mx-auto px-6 py-12"
        data-testid="phrase-match-not-found"
      >
        <div
          role="status"
          className="rounded-lg border border-slate-800 bg-slate-900 p-6 space-y-4"
        >
          <h2 className="text-base font-semibold text-slate-100">
            No phrase pair anchored to that word
          </h2>
          <p className="text-sm text-slate-400">
            The chosen word doesn't have a curated{' '}
            <code className="font-mono text-xs">phrase_pairs</code> row yet.
            Generate another or pick a different word — Phase 10.5 keeps the
            page self-contained against the seed corpus.
          </p>
          <div className="pt-1">
            <button
              type="button"
              onClick={handleGenerateAnother}
              className="px-3 py-1.5 text-sm rounded-lg border border-slate-700 text-slate-200 hover:bg-slate-800 transition-colors"
            >
              Generate another
            </button>
          </div>
        </div>
      </div>
    )
  }

  if (status === 'empty') {
    // Post-grade empty — the user just submitted a relation +
    // grade. We DON'T reuse the prior ``exercise`` value to
    // re-render the body (the FSRS scheduler returns ``next_due_at``
    // on /exercises/grade, not the next exercise — that's
    // /exercises/due / Phase 9.6's job). Instead we point the
    // user at the next-due hand-off.
    if (hasGraded) {
      return (
        <div className="max-w-2xl mx-auto px-6 py-12 space-y-6">
          <div
            role="status"
            className="rounded-lg border border-slate-800 bg-slate-900 p-6 space-y-4"
          >
            <h2 className="text-base font-semibold text-slate-100">
              Grade recorded.
            </h2>
            <p className="text-sm text-slate-400">
              The FSRS scheduler will unlock the next phrase pair when it's
              due. Hit Generate another for a fresh pick, or jump into the
              session mixer for the next-due card across types.
            </p>
            <div className="flex flex-wrap items-center gap-2 pt-1">
              <button
                type="button"
                onClick={handleGenerateAnother}
                data-testid="phrase-match-generate-another-empty"
                className="px-3 py-1.5 text-sm rounded-lg border border-slate-700 text-slate-200 hover:bg-slate-800 transition-colors"
              >
                Generate another
              </button>
              <button
                type="button"
                onClick={handleGoToSession}
                data-testid="phrase-match-go-to-session-empty"
                className="px-3 py-1.5 text-sm rounded-lg border border-slate-700 text-slate-200 hover:bg-slate-800 transition-colors"
              >
                Open session mixer
              </button>
            </div>
          </div>
        </div>
      )
    }
    return (
      <div className="max-w-2xl mx-auto px-6 py-12">
        <div
          role="status"
          className="rounded-lg border border-slate-800 bg-slate-900 p-6 space-y-4"
        >
          <h2 className="text-base font-semibold text-slate-100">
            Generate a phrase pair to start
          </h2>
          <p className="text-sm text-slate-400">
            Phrase match exercises draw from the curated{' '}
            <code className="font-mono text-xs">phrase_pairs</code> table (a
            relation between two <code className="font-mono text-xs">
              phrases
            </code>{' '}
            rows). Hit Generate another to surface one.
          </p>
          <div className="pt-1">
            <button
              type="button"
              onClick={handleGenerateAnother}
              data-testid="phrase-match-generate-another-empty-init"
              className="px-3 py-1.5 text-sm rounded-lg border border-slate-700 text-slate-200 hover:bg-slate-800 transition-colors"
            >
              Generate another
            </button>
          </div>
        </div>
      </div>
    )
  }

  if (status === 'loading' || status === 'idle') {
    return (
      <div
        className="max-w-2xl mx-auto px-6 py-12"
        data-testid="phrase-match-loading"
      >
        <div className="rounded-lg border border-slate-800 bg-slate-900 p-6 space-y-4 animate-pulse">
          <div className="h-4 w-1/3 rounded bg-slate-800" />
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 pt-2">
            <div className="h-24 rounded bg-slate-800" />
            <div className="h-24 rounded bg-slate-800" />
          </div>
          <div className="h-4 w-3/4 rounded bg-slate-800" />
        </div>
      </div>
    )
  }

  if (status === 'error') {
    const isAuth =
      !!errorMessage &&
      /401|not authenticated/i.test(errorMessage)
    return (
      <div
        className="max-w-2xl mx-auto px-6 py-12"
        data-testid="phrase-match-error"
      >
        <div
          role="alert"
          className="rounded-lg border border-red-900/60 bg-red-950/40 p-5 space-y-3"
        >
          <p className="text-sm text-red-300">
            {isAuth
              ? "You've been signed out — please log in again."
              : "Couldn't generate a phrase match exercise."}
          </p>
          {errorMessage && (
            <p className="text-xs text-red-400">{errorMessage}</p>
          )}
          <div className="flex items-center gap-2">
            {isAuth ? (
              <button
                type="button"
                onClick={handleRedirectToLogin}
                className="px-3 py-1.5 text-sm rounded-lg bg-blue-600 text-white hover:bg-blue-700 transition-colors"
              >
                Go to login
              </button>
            ) : (
              <button
                type="button"
                onClick={handleRetry}
                className="px-3 py-1.5 text-sm rounded-lg border border-slate-700 text-slate-200 hover:bg-slate-800 transition-colors"
              >
                Retry
              </button>
            )}
          </div>
        </div>
      </div>
    )
  }

  if (!exercise) {
    // Phase 10.6 (card t_da43cc23) — when the mixer mounts the
    // page, the post-grade path (``onGraded`` callback) nulls
    // out ``exercise`` while ``status`` stays at ``'ready'``
    // (the standalone ``setStatus('empty')`` branch is
    // skipped). The mixer's ``fetchAttempt`` bump will mount a
    // fresh PhraseMatchPage shortly, so we render an empty
    // fragment here rather than the standalone "server
    // returned an empty phrase pair" surface (which would
    // confuse the user mid-session).
    if (onGraded) {
      return null
    }
    return (
      <div className="max-w-2xl mx-auto px-6 py-12">
        <div className="rounded-lg border border-slate-800 bg-slate-900 p-6 text-sm text-slate-400">
          The server returned an empty phrase pair. Try again in a moment.
        </div>
        <div className="pt-4">
          <button
            type="button"
            onClick={handleGenerateAnother}
            className="px-3 py-1.5 text-sm rounded-lg border border-slate-700 text-slate-200 hover:bg-slate-800 transition-colors"
          >
            Generate another
          </button>
        </div>
      </div>
    )
  }

  // status === 'ready'
  return (
    <div data-testid="phrase-match-ready">
      <div className="max-w-2xl mx-auto px-6 py-10 space-y-6">
        <div className="rounded-lg border border-slate-800 bg-slate-900 p-6 space-y-5">
          <div className="space-y-1">
            <p className="text-xs uppercase tracking-wide text-slate-500">
              Phrase match
            </p>
            <p className="text-sm text-slate-400">
              How are these two phrases related?
            </p>
          </div>

          <div
            className="grid grid-cols-1 md:grid-cols-2 gap-3"
            data-testid="phrase-match-pair-cards"
          >
            <PhraseCard
              testIdPrefix="phrase-a"
              phrase={exercise.phrase_a}
              definition={exercise.relation_rationale}
            />
            <PhraseCard
              testIdPrefix="phrase-b"
              phrase={exercise.phrase_b}
              definition={exercise.relation_rationale}
            />
          </div>

          <div
            className="grid grid-cols-2 md:grid-cols-4 gap-2"
            role="radiogroup"
            aria-label="Phrase pair relation"
            data-testid="phrase-match-relation-picker"
          >
            {RELATION_BUTTONS.map(({ relation, label }) => {
              const isPicked = pickedRelation === relation
              return (
                <button
                  key={relation}
                  type="button"
                  role="radio"
                  aria-checked={isPicked}
                  onClick={() => handlePickRelation(relation)}
                  data-testid={`phrase-match-relation-${relation}`}
                  data-relation={relation}
                  className={
                    'rounded-lg border px-3 py-2 text-sm transition-colors ' +
                    (isPicked
                      ? 'border-blue-500 bg-blue-950/40 text-slate-100'
                      : 'border-slate-700 bg-slate-950 text-slate-200 hover:bg-slate-800')
                  }
                >
                  {label}
                </button>
              )
            })}
          </div>
          <p className="text-xs text-slate-500">
            {pickedRelation === null
              ? 'Pick a relation, then grade below.'
              : `You picked "${pickedRelation}". Grade your confidence below.`}
          </p>

          <div className="space-y-3 pt-2 border-t border-slate-800">
            <GradeButtons
              onGrade={handlePickGrade}
              disabled={isGrading || pickedRelation === null}
            />
            <p className="text-xs text-slate-500">
              1 = Again · 2 = Hard · 3 = Good · 4 = Easy. The grade is the
              self-assessment (separate from the relation above). You must
              pick a relation before the grade bar unlocks.
            </p>
          </div>

          {exercise.trace_id ? (
            <p className="text-[10px] text-slate-600 pt-1 border-t border-slate-800">
              trace: {exercise.trace_id}
              {exercise.source_attribution ? (
                <span> · {exercise.source_attribution}</span>
              ) : null}
            </p>
          ) : null}
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            onClick={handleGenerateAnother}
            data-testid="phrase-match-generate-another"
            className="px-3 py-1.5 text-sm rounded-lg border border-slate-700 text-slate-200 hover:bg-slate-800 transition-colors"
          >
            Generate another
          </button>
          <button
            type="button"
            onClick={handleGoToSession}
            data-testid="phrase-match-go-to-session"
            className="px-3 py-1.5 text-sm rounded-lg border border-slate-700 text-slate-200 hover:bg-slate-800 transition-colors"
          >
            Open session mixer
          </button>
        </div>
      </div>
    </div>
  )
}

// --- sub-components --------------------------------------------------------

function PhraseCard({
  phrase,
  definition,
  testIdPrefix,
}: {
  phrase: string
  definition: string
  testIdPrefix: 'phrase-a' | 'phrase-b'
}) {
  // The definition is a learner hint (Phase 8 idiom pattern);
  // we reveal it on hover/focus so the relation picker reflects
  // a more-informed judgment rather than a guess from the
  // surface form alone. Keyboard users get the same hint via
  // focus; touch users get it on tap (the button toggles).
  const [revealed, setRevealed] = useState(false)
  return (
    <div
      data-testid={`${testIdPrefix}-card`}
      className="rounded-lg border border-slate-700 bg-slate-950 px-4 py-4 space-y-2"
    >
      <p
        className="text-lg font-semibold text-slate-100"
        data-testid={`${testIdPrefix}-phrase`}
      >
        {phrase}
      </p>
      <button
        type="button"
        onClick={() => setRevealed((r) => !r)}
        onMouseEnter={() => setRevealed(true)}
        onMouseLeave={() => setRevealed(false)}
        onFocus={() => setRevealed(true)}
        onBlur={() => setRevealed(false)}
        aria-pressed={revealed}
        data-testid={`${testIdPrefix}-reveal`}
        className="text-xs text-slate-400 hover:text-slate-200 transition-colors"
      >
        {revealed ? 'Hide hint' : 'Show hint'}
      </button>
      {revealed ? (
        <p
          className="text-sm text-slate-300 italic border-t border-slate-800 pt-2"
          data-testid={`${testIdPrefix}-definition`}
        >
          {definition}
        </p>
      ) : null}
    </div>
  )
}
