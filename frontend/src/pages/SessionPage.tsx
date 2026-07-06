import { useCallback, useEffect, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { toast } from 'sonner'
import type {
  AnyExercise,
  ComprehensionExercise,
  IdiomExercise,
  MatchingExercise,
  PhraseMatchExercise,
} from '../api/exercises'
import {
  ExerciseApiError,
  generateComprehension,
  generateIdiom,
  generateMatch,
  generatePhraseMatch,
} from '../api/exercises'
import { getNextDuePick } from '../api/session'
import type { DuePick, DueQueueResult } from '../api/session'
import { ExerciseCard } from '../components/ExerciseCard'
import { humanizeDelta } from '../components/delta'
import { PhraseMatchPage } from './PhraseMatchPage'

// Phase 9.6 (card t_f1c63bfc) + Phase 10.6 (card t_da43cc23) —
// the study-session mixer.
//
// Phase 10.6 widens the union surface to include
// ``phrase_match`` (the 5th FSRS-graded exercise type). The
// mixer's ``case "phrase_match"`` branch renders the bespoke
// ``<PhraseMatchPage />`` (Phase 10.5) instead of the shared
// ``<ExerciseCard />`` — the 4-button relation picker is the
// page's job, not the card's. The 4 prior branches (cloze /
// matching / comprehension / idiom) are unchanged.
//
// What this page does:
//   1. On mount: fetch the next pick from ``GET /exercises/due``
//      (the union surface widened by Phase 9.2 / 10.6). The
//      pick is either a cloze body (200 + JSON) or a (type,
//      card_id, word_id) header tuple (204 + X-Due-Exercise-
//      Type/-Card-Id/-Word-Id) for matching / comprehension /
//      idiom / phrase_match.
//   2. For cloze / matching / comprehension / idiom: render
//      ``<ExerciseCard exercise={...} />`` for the head pick.
//      The card owns the body, the grade bar, and the
//      ``/exercises/grade`` round-trip — the page just provides
//      the typed ``AnyExercise`` payload.
//   2b. For phrase_match: render ``<PhraseMatchPage />`` with
//      the queue-supplied ``word_id`` and the page's own
//      grade + error callbacks. The bespoke relation picker +
//      locked-until-picked grade bar lives on the page; the
//      mixer only owns the queue lifecycle and the next-pick
//      fetch.
//   3. On grade success: fetch the next pick. The queue is
//      re-fetched on every advance (Phase 9.2's union response
//      carries exactly one pick, not a list) so the backend
//      gets to re-rank the remaining due cards.
//   4. On "End session" / queue-empty: navigate to home so the
//      user sees the master / library view.
//
// Why not hold a local queue of picks and ``.shift()`` on grade:
//   The Phase 9.2 due endpoint returns one pick per call, and
//   each pick's due_date just got bumped by the grade — the
//   next pick must come from a fresh ``/exercises/due`` call
//   to see the updated union. Holding a stale queue would
//   re-issue the same pick after every grade, looping forever.
//   Phase 9.7 (out of scope here) may explore an in-memory
//   queue with explicit de-dup, but for 9.6 the re-fetch-per-
//   grade approach is the honest one and matches what the
//   backend actually returns.

type Status = 'idle' | 'loading' | 'ready' | 'done' | 'error'

export function SessionPage() {
  const navigate = useNavigate()
  const [status, setStatus] = useState<Status>('idle')
  const [exercise, setExercise] = useState<AnyExercise | null>(null)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [gradedCount, setGradedCount] = useState(0)
  const [fetchAttempt, setFetchAttempt] = useState(() => 1) // mount: trigger first fetch
  // Phase 10.6 (card t_da43cc23) — when the picked exercise
  // is a ``phrase_match`` (the bespoke 4-button relation
  // picker), the mixer renders ``<PhraseMatchPage />`` instead
  // of ``<ExerciseCard>``. We track the queue-supplied
  // ``word_id`` so the page mounts with the right pair anchor;
  // for cloze / matching / comprehension / idiom the body
  // already carries its own ``target_word_id`` / ``word_id`` /
  // ``pairs`` and the card renders directly.
  const [phraseMatchWordId, setPhraseMatchWordId] = useState<
    number | null
  >(null)

  // ----- queue lifecycle -------------------------------------------------

  // Step 1+2 (mount): hit ``/exercises/due`` and either populate
  // ``exercise`` with the next renderable body or transition to
  // ``done`` (no picks left). The fetch is gated on a
  // ``fetchAttempt`` counter so a Retry click bumps the counter
  // and triggers a refetch (the same pattern ClozePage uses for
  // its inline flow).
  useEffect(() => {
    if (fetchAttempt === 0) return
    let cancelled = false
    // The setState inside an effect rule wants us to schedule
    // the state transition so it doesn't cascade during commit.
    // ``queueMicrotask`` defers the assignment past the effect's
    // synchronous body but still ahead of the first await —
    // the loading state is visible immediately while the
    // network call kicks off. (A bare setStatus call here
    // would re-render twice in the same tick.)
    queueMicrotask(() => {
      if (!cancelled) setStatus('loading')
    })
    ;(async () => {
      const pick = await getNextDuePick()
      if (cancelled) return
      await advanceToPick(pick)
    })()
    return () => {
      cancelled = true
    }

    async function advanceToPick(pick: DueQueueResult) {
      if (cancelled) return
      if (pick.kind === 'empty') {
        setExercise(null)
        setStatus('done')
        return
      }
      if (pick.kind === 'unauthenticated') {
        navigate('/login', {
          replace: true,
          state: { from: '/exercises/session' },
        })
        return
      }
      if (pick.kind === 'error') {
        setErrorMessage(pick.message)
        setStatus('error')
        return
      }
      // pick.kind === 'pick' — resolve the body inline.
      try {
        const body = await resolvePickBody(pick.pick)
        if (cancelled) return
        // Phase 10.6 (card t_da43cc23) — track the queue-
        // supplied ``word_id`` when the pick is a
        // ``phrase_match``. The render branch reads this state
        // to mount ``<PhraseMatchPage />`` with the right
        // pair anchor. For the 4 prior types the body's own
        // ``target_word_id`` / ``word_id`` / ``pairs`` field
        // carries the wire shape and we set ``null`` to signal
        // "render via ExerciseCard".
        setPhraseMatchWordId(
          pick.pick.kind === 'phrase_match' ? pick.pick.word_id : null,
        )
        setExercise(body)
        setStatus('ready')
      } catch (err) {
        if (cancelled) return
        setErrorMessage(
          err instanceof Error ? err.message : 'Could not build exercise',
        )
        setStatus('error')
      }
    }
  }, [fetchAttempt, navigate])

  // ----- grade-and-advance -----------------------------------------------

  const handleGraded = useCallback(
    (_next_due_at: string) => {
      // The card's ``/exercises/grade`` round-trip succeeded;
      // advance by re-fetching the next pick from the union.
      // The current pick is dropped from the local render so
      // the user sees the loading state until the next body
      // arrives.
      //
      // Phase 10.6 (card t_da43cc23) — also drop the
      // queue-supplied ``word_id`` so the next render branch
      // doesn't accidentally re-mount ``<PhraseMatchPage />``
      // for a non-phrase_match pick. The fetch effect will
      // re-set this on the next ``advanceToPick``.
      setExercise(null)
      setPhraseMatchWordId(null)
      setErrorMessage(null)
      setGradedCount((n) => n + 1)
      setFetchAttempt((n) => n + 1)
      // The toast lives on the page (not the card) so the
      // user gets the FSRS delta even when the next pick is
      // slow to load. The card already invokes ``onGraded``;
      // we fire the toast here off the same callback.
      toast.success(`Grade recorded. Next review ${humanizeDelta(_next_due_at)}.`)
    },
    [],
  )

  const handleGradeError = useCallback(
    (err: unknown) => {
      const status =
        err instanceof ExerciseApiError ? err.status : undefined
      if (status === 401) {
        navigate('/login', {
          replace: true,
          state: { from: '/exercises/session' },
        })
        return
      }
      if (status === 422) {
        toast.error(
          err instanceof Error ? err.message : 'Grade validation failed',
        )
      } else {
        toast.error('Grade failed — try again')
      }
    },
    [navigate],
  )

  // ----- user actions ---------------------------------------------------

  function handleRetry() {
    setErrorMessage(null)
    setFetchAttempt((n) => n + 1)
  }

  function handleEndSession() {
    navigate('/')
  }

  // ----- render branches ------------------------------------------------

  // On mount, kick off the first fetch. The mount effect is
  // gated on ``fetchAttempt === 0`` (no fetch yet) so a Retry
  // click bumps the counter and triggers a refetch via the
  // existing effect — we don't need a separate "initial
  // mount" effect. We initialise the counter inline via a
  // ``useState`` lazy initialiser to avoid the
  // setState-during-render anti-pattern.
  // (See the useState declaration above; this comment is here
  // to mark the architecture so future maintainers don't add
  // a setFetchAttempt during render.)

  if (status === 'loading') {
    return (
      <div
        className="max-w-2xl mx-auto px-6 py-12"
        data-testid="session-loading"
      >
        <div className="rounded-lg border border-slate-800 bg-slate-900 p-6 space-y-4 animate-pulse">
          <div className="h-4 w-3/4 rounded bg-slate-800" />
          <div className="h-24 rounded bg-slate-800" />
          <div className="grid grid-cols-2 gap-3 pt-4">
            <div className="h-10 rounded bg-slate-800" />
            <div className="h-10 rounded bg-slate-800" />
            <div className="h-10 rounded bg-slate-800" />
            <div className="h-10 rounded bg-slate-800" />
          </div>
        </div>
      </div>
    )
  }

  if (status === 'done') {
    const plural = gradedCount === 1 ? 'card' : 'cards'
    return (
      <div
        className="max-w-2xl mx-auto px-6 py-12"
        data-testid="session-done"
      >
        <div
          role="status"
          className="rounded-lg border border-slate-800 bg-slate-900 p-6 space-y-4"
        >
          <h2 className="text-base font-semibold text-slate-100">
            Session complete
          </h2>
          <p className="text-sm text-slate-400">
            {gradedCount === 0
              ? "You don't have any due cards right now. Come back later, or check the home page for a fresh pick."
              : `You graded ${gradedCount} ${plural}. The FSRS scheduler will unlock the next round when they're due.`}
          </p>
          <div className="flex items-center gap-3 pt-1">
            <Link
              to="/"
              className="px-3 py-1.5 text-sm rounded-lg bg-blue-600 text-white hover:bg-blue-700 transition-colors"
            >
              Back to home
            </Link>
            <button
              type="button"
              onClick={handleRetry}
              data-testid="session-recheck"
              className="px-3 py-1.5 text-sm rounded-lg border border-slate-700 text-slate-200 hover:bg-slate-800 transition-colors"
            >
              Re-check for new due cards
            </button>
          </div>
        </div>
      </div>
    )
  }

  if (status === 'error') {
    const isAuth =
      !!errorMessage && /401|not authenticated/i.test(errorMessage)
    return (
      <div
        className="max-w-2xl mx-auto px-6 py-12"
        data-testid="session-error"
      >
        <div
          role="alert"
          className="rounded-lg border border-red-900/60 bg-red-950/40 p-5 space-y-3"
        >
          <p className="text-sm text-red-300">
            {isAuth
              ? "You've been signed out — please log in again."
              : "Couldn't fetch the next exercise."}
          </p>
          {errorMessage && (
            <p className="text-xs text-red-400">{errorMessage}</p>
          )}
          <div className="flex items-center gap-2">
            {isAuth ? (
              <button
                type="button"
                onClick={() =>
                  navigate('/login', {
                    replace: true,
                    state: { from: '/exercises/session' },
                  })
                }
                className="px-3 py-1.5 text-sm rounded-lg bg-blue-600 text-white hover:bg-blue-700 transition-colors"
              >
                Go to login
              </button>
            ) : (
              <>
                <button
                  type="button"
                  onClick={handleRetry}
                  className="px-3 py-1.5 text-sm rounded-lg border border-slate-700 text-slate-200 hover:bg-slate-800 transition-colors"
                >
                  Retry
                </button>
                <button
                  type="button"
                  onClick={handleEndSession}
                  className="px-3 py-1.5 text-sm rounded-lg border border-slate-700 text-slate-200 hover:bg-slate-800 transition-colors"
                >
                  End session
                </button>
              </>
            )}
          </div>
        </div>
      </div>
    )
  }

  // status === 'ready' — render the head pick via the shared
  // ExerciseCard. The card owns the body, the grade bar, and
  // the /exercises/grade round-trip.
  if (!exercise) {
    return (
      <div className="max-w-2xl mx-auto px-6 py-12">
        <div className="rounded-lg border border-slate-800 bg-slate-900 p-6 text-sm text-slate-400">
          The server returned an empty pick. Try again in a moment.
        </div>
        <div className="pt-4">
          <button
            type="button"
            onClick={handleRetry}
            className="px-3 py-1.5 text-sm rounded-lg border border-slate-700 text-slate-200 hover:bg-slate-800 transition-colors"
          >
            Retry
          </button>
        </div>
      </div>
    )
  }

  return (
    <div data-testid="session-ready">
      <div className="max-w-2xl mx-auto px-6 pt-10 pb-2 flex items-center justify-between">
        <p className="text-xs uppercase tracking-wide text-slate-500">
          Study session · graded {gradedCount} {gradedCount === 1 ? 'card' : 'cards'}
        </p>
        <button
          type="button"
          onClick={handleEndSession}
          data-testid="session-end"
          className="text-xs rounded border border-slate-700 px-2 py-1 text-slate-300 hover:bg-slate-800 transition-colors"
        >
          End session
        </button>
      </div>
      {/*
        Phase 10.6 (card t_da43cc23) — the ``phrase_match``
        branch mounts ``<PhraseMatchPage />`` directly. The
        bespoke 4-button relation picker + locked-until-picked
        grade bar is the page's job; the shared
        ``<ExerciseCard />`` only admits the phrase_match wire
        for exhaustiveness without rendering it. The mixer's
        own ``handleGraded`` / ``handleGradeError`` feed back
        via the page's callback props so the queue advances
        uniformly across all 5 exercise types.
      */}
      {exercise?.exercise_type === 'phrase_match' &&
      phraseMatchWordId !== null ? (
        <PhraseMatchPage
          word_id={phraseMatchWordId}
          onGraded={handleGraded}
          onGradeError={handleGradeError}
        />
      ) : (
        <ExerciseCard
          exercise={exercise}
          onGraded={handleGraded}
          onGradeError={handleGradeError}
        />
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// resolvePickBody — turn a ``DuePick`` into the typed ``AnyExercise`` the
// shared ``<ExerciseCard>`` renders, OR — for the Phase 10.6
// phrase_match branch — return the typed pick so the mixer can mount
// ``<PhraseMatchPage />`` directly with the queue-supplied ``word_id``.
//
// Phase 9.2 widened ``/exercises/due`` to the union but only the cloze
// branch returns the body inline (the cloze generator has a
// ``force_word_id`` knob). Non-cloze picks come through as 204 +
// ``X-Due-Exercise-Type``/``X-Due-Card-Id``/``X-Due-Word-Id`` headers,
// and the per-type endpoints are responsible for building the body.
//
// Notes per type:
//   - ``cloze``: body is inline on the pick — return as-is, adapted to
//     ``ClozeExerciseForCard`` (Phase 9.5's shared-card wire shape).
//   - ``matching``: per-type endpoint doesn't take a word_id, so the
//     generated match may target a different word than the queue pick
//     (Phase 9.2 documented this as acceptable read-side widening).
//   - ``comprehension``: same as matching.
//   - ``idiom``: Phase 8.4 made ``word_id`` required; pass it through
//     so the generator's ``phrases WHERE word_id == :word_id`` filter
//     is anchored.
//   - ``phrase_match`` (Phase 10.6): the per-type endpoint requires
//     ``word_id`` (Phase 8.4 idiom discipline, mirrored by 10.3).
//     The generated exercise targets the same word the queue picked,
//     so the relation picker surfaces the right pair.
// ---------------------------------------------------------------------------
async function resolvePickBody(pick: DuePick): Promise<AnyExercise> {
  if (pick.kind === 'cloze') {
    // Adapt ``ClozeExerciseOut`` (Phase 5.5 wire) into the
    // ``ClozeExerciseForCard`` shape the shared card expects.
    // The card already accepts the inline wire; we just stamp
    // the discriminator field the card requires.
    const e = pick.exercise
    return {
      exercise_type: 'cloze',
      target_word_id: e.word_id,
      prompt_template_version: e.prompt_template_version,
      enable_rag: false,
      trace_id: null,
      latency_ms: 0,
      sentence_with_blank: e.sentence_with_blank,
      answer_word_id: e.answer_word_id,
      distractors: [...e.distractors],
      difficulty: e.difficulty,
      rationale: e.rationale,
    }
  }
  if (pick.kind === 'matching') {
    const body: MatchingExercise = await generateMatch({ count: 4 })
    return body
  }
  if (pick.kind === 'comprehension') {
    const body: ComprehensionExercise = await generateComprehension({})
    return body
  }
  if (pick.kind === 'idiom') {
    const body: IdiomExercise = await generateIdiom({
      word_id: pick.word_id,
    })
    return body
  }
  // pick.kind === 'phrase_match' — Phase 10.6 (card t_da43cc23).
  // Mirrors the idiom discipline: the per-type endpoint takes
  // ``word_id`` as a required knob and resolves the curated
  // phrase_pairs row for that seed. We pass the queue-supplied
  // word_id through so the generator's ``select_phrase_pair``
  // is anchored.
  const body: PhraseMatchExercise = await generatePhraseMatch({
    word_id: pick.word_id,
  })
  return body
}
