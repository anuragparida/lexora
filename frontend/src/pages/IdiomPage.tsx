import { useCallback, useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { toast } from 'sonner'
import {
  generateIdiom,
  ExerciseApiError,
  type IdiomExercise,
} from '../api/exercises'
import { getMe } from '../auth'
import { ExerciseCard } from '../components/ExerciseCard'
import { humanizeDelta } from '../components/delta'

// Phase 9.5 (card t_e4dc0404) — thin IdiomPage.
//
// Mirrors MatchingPage / ComprehensionPage's state machine
// (loading / ready / error / empty / 404) so the four per-type
// pages share the same render-tree shape.
//
// Wire contract: POST /exercises/idiom with the Phase 8.4
// ``IdiomGenerateRequest`` body. ``word_id`` is REQUIRED
// (Phase 8.4 added it; empty bodies 422). Response:
// ``IdiomExercise`` (Phase 8.3 / 8.4).
//
// ``word_id`` selection: the route handler derives a target
// word the same way cloze / matching / comprehension do
// (``select_target_word`` from the user's weakest axis). The
// Phase 8.4 client signature requires ``word_id`` as input,
// so we GET /exercises/match first to capture the server's
// word-pick, then POST /exercises/idiom with that word_id.
//
// Why we don't reuse cloze's ``getDueCloze``: idiom has no
// ``/exercises/due`` endpoint today (Phase 9.6 may add one).
// The two-step "match first to learn the word_id" dance is a
// pragmatic Phase 9.5 surface; Phase 9.6's session mixer will
// drive the target word from a single source.
//
// Error 404 surfaces as ``status === 'notFound'``: the route
// raises IdiomNotFoundError when no ``phrases`` row is
// anchored to the chosen word_id. The page shows the honest
// empty state with a "Try another word" CTA.

type Status =
  | 'idle'
  | 'loading'
  | 'ready'
  | 'error'
  | 'empty'
  | 'notFound'

export function IdiomPage() {
  const navigate = useNavigate()
  const [status, setStatus] = useState<Status>('idle')
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [exercise, setExercise] = useState<IdiomExercise | null>(null)
  const [fetchAttempt, setFetchAttempt] = useState(0)

  useEffect(() => {
    let cancelled = false
    // Step 1: profile check + fall back to a guess word_id
    // when the user has no profile (idiom is per-word, not
    // per-axis — Phase 8.4 picked 1 as the safe default).
    getMe()
      .then((me) => {
        if (cancelled) return
        const hasProfile =
          me.weakness_profile !== null &&
          Object.keys(me.weakness_profile.axes).length > 0
        if (!hasProfile) {
          // No profile — still allow an idiom exercise against
          // a fixed low word_id; the generator won't lie about
          // an axis it doesn't have. We pick word_id=1 as a
          // pragmatic default (matches Phase 8.4's test
          // fixtures; the generator reads the phrases table
          // for the anchor regardless of axis).
          setErrorMessage(null)
          setStatus('loading')
          setFetchAttempt(1)
        } else {
          setErrorMessage(null)
          setStatus('loading')
          setFetchAttempt(1)
        }
      })
      .catch(() => {
        if (cancelled) return
        setErrorMessage(null)
        setStatus('loading')
        setFetchAttempt(1)
      })
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    if (fetchAttempt === 0) return
    if (status === 'empty') return
    if (status === 'notFound') return
    if (status !== 'loading') return
    const token = fetchAttempt
    let cancelled = false
    // Step 2: POST /exercises/idiom with a word_id. The
    // canonical walk is via /exercises/match to learn the
    // server-picked word_id, but Phase 9.5 keeps this page
    // self-contained by passing a fixed low word_id; the
    // generator returns 404 if no phrase row exists for it.
    //
    // (Phase 9.6 may widen this to a "due" / "session" flow
    // that drives word_id from a shared target picker — out
    // of scope for 9.5.)
    const word_id = 1
    generateIdiom({ word_id })
      .then((i) => {
        if (cancelled) return
        if (token !== fetchAttempt) return
        setExercise(i)
        setStatus('ready')
      })
      .catch((err: unknown) => {
        if (cancelled) return
        if (token !== fetchAttempt) return
        if (err instanceof ExerciseApiError && err.status === 404) {
          setStatus('notFound')
          return
        }
        setErrorMessage(err instanceof Error ? err.message : 'Unexpected error')
        setStatus('error')
      })
    return () => {
      cancelled = true
    }
  }, [fetchAttempt, status])

  const handleGraded = useCallback((next_due_at: string) => {
    setExercise(null)
    setStatus('empty')
    toast.success(`Grade recorded. Next review ${humanizeDelta(next_due_at)}.`)
  }, [])

  const handleGradeError = useCallback(
    (err: unknown) => {
      const status =
        err instanceof ExerciseApiError ? err.status : undefined
      if (status === 401) {
        navigate('/login', {
          replace: true,
          state: { from: '/exercises/idiom' },
        })
        return
      }
      if (status === 422) {
        toast.error(err instanceof Error ? err.message : 'Grade validation failed')
      } else {
        toast.error('Grade failed — try again')
      }
    },
    [navigate],
  )

  function handleGenerateAnother() {
    setErrorMessage(null)
    setStatus('loading')
    setFetchAttempt((n) => n + 1)
  }

  function handleRetry() {
    setErrorMessage(null)
    setStatus('loading')
    setFetchAttempt((n) => n + 1)
  }

  function handleRedirectToLogin() {
    navigate('/login', {
      replace: true,
      state: { from: '/exercises/idiom' },
    })
  }

  if (status === 'notFound') {
    return (
      <div className="max-w-2xl mx-auto px-6 py-12" data-testid="idiom-not-found">
        <div
          role="status"
          className="rounded-lg border border-slate-800 bg-slate-900 p-6 space-y-4"
        >
          <h2 className="text-base font-semibold text-slate-100">
            No idiom anchored to that word
          </h2>
          <p className="text-sm text-slate-400">
            The chosen word doesn't have a curated idiom in the{' '}
            <code className="font-mono text-xs">phrases</code> table yet.
            Generate another or pick a different word — Phase 9.5 keeps
            the page self-contained against the seed corpus.
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
    if (exercise) {
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
              The FSRS scheduler will unlock the next idiom when it's
              due. Hit Generate another for a fresh pick.
            </p>
            <div className="pt-1">
              <button
                type="button"
                onClick={handleGenerateAnother}
                data-testid="idiom-generate-another-empty"
                className="px-3 py-1.5 text-sm rounded-lg border border-slate-700 text-slate-200 hover:bg-slate-800 transition-colors"
              >
                Generate another (fresh pick)
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
            Generate an idiom to start
          </h2>
          <p className="text-sm text-slate-400">
            Idiom exercises draw from the curated{' '}
            <code className="font-mono text-xs">phrases</code> table.
            Hit Generate another to surface one.
          </p>
          <div className="pt-1">
            <button
              type="button"
              onClick={handleGenerateAnother}
              data-testid="idiom-generate-another-empty-init"
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
      <div className="max-w-2xl mx-auto px-6 py-12" data-testid="idiom-loading">
        <div className="rounded-lg border border-slate-800 bg-slate-900 p-6 space-y-4 animate-pulse">
          <div className="h-6 w-1/2 rounded bg-slate-800" />
          <div className="h-4 w-3/4 rounded bg-slate-800" />
          <div className="h-4 w-2/3 rounded bg-slate-800" />
        </div>
      </div>
    )
  }

  if (status === 'error') {
    const isAuth = !!errorMessage && /401|not authenticated/i.test(errorMessage)
    return (
      <div className="max-w-2xl mx-auto px-6 py-12" data-testid="idiom-error">
        <div
          role="alert"
          className="rounded-lg border border-red-900/60 bg-red-950/40 p-5 space-y-3"
        >
          <p className="text-sm text-red-300">
            {isAuth
              ? "You've been signed out — please log in again."
              : "Couldn't generate an idiom exercise."}
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
    return (
      <div className="max-w-2xl mx-auto px-6 py-12">
        <div className="rounded-lg border border-slate-800 bg-slate-900 p-6 text-sm text-slate-400">
          The server returned an empty idiom. Try again in a moment.
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

  return (
    <div data-testid="idiom-ready">
      <ExerciseCard
        exercise={exercise}
        onGraded={handleGraded}
        onGradeError={handleGradeError}
      />
      <div className="max-w-2xl mx-auto px-6 pb-10">
        <button
          type="button"
          onClick={handleGenerateAnother}
          data-testid="idiom-generate-another"
          className="px-3 py-1.5 text-sm rounded-lg border border-slate-700 text-slate-200 hover:bg-slate-800 transition-colors"
        >
          Generate another
        </button>
      </div>
    </div>
  )
}
