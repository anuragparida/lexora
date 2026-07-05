import { useCallback, useEffect, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { toast } from 'sonner'
import {
  generateMatch,
  ExerciseApiError,
  type MatchingExercise,
} from '../api/exercises'
import { getMe } from '../auth'
import { ExerciseCard } from '../components/ExerciseCard'
import { humanizeDelta } from '../components/delta'

// Phase 9.5 (card t_e4dc0404) — thin MatchingPage.
//
// The page mirrors ClozePage's state machine exactly
// (loading / ready / error / empty) so the four per-type
// pages share the same render-tree shape and the Phase 9.6
// ``SessionPage`` can compose them without knowing the
// per-type differences. The body render lives in
// ``<ExerciseCard />``, which switches on ``exercise_type``
// to pick the matching UI; this page just owns the fetch
// lifecycle and the grade-bar success / error toasts.
//
// Wire contract: POST /exercises/match with the Phase 6.3
// ``MatchGenerateRequest`` body (defaults to ``{}`` for a
// 4-pair match). Response: ``MatchingExercise`` (Phase 6.3).
//
// No ``Props { user: AuthUser }`` for now: the matching route
// does not (yet) branch on user features. Reserved for
// Phase 9.6 where the SessionPage mixer may pass a shared
// ``user`` down.

type Status = 'idle' | 'loading' | 'ready' | 'error' | 'empty'

export function MatchingPage() {
  const navigate = useNavigate()
  const [status, setStatus] = useState<Status>('idle')
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [exercise, setExercise] = useState<MatchingExercise | null>(null)
  const [fetchAttempt, setFetchAttempt] = useState(0)

  // Empty-profile guard — mirrors ClozePage's branch
  // (Phase 4.5). When the user has no weakness profile,
  // /exercises/match would 4xx from ``select_target_word``
  // anyway; the page catches it earlier as an honest
  // empty state with a link to /weakness-profile. Only
  // kicks off the fetch when the profile is non-empty.
  useEffect(() => {
    let cancelled = false
    getMe()
      .then((me) => {
        if (cancelled) return
        const empty =
          me.weakness_profile === null ||
          Object.keys(me.weakness_profile.axes).length === 0
        if (empty) {
          setStatus('empty')
        } else {
          setErrorMessage(null)
          setStatus('loading')
          setFetchAttempt(1)
        }
      })
      .catch(() => {
        // 401 / network — fall through to a fetch attempt
        // (the fetch will 401 too and we'll surface the
        // toast).
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
    if (status !== 'loading') return
    const token = fetchAttempt
    let cancelled = false
    generateMatch({ count: 4 })
      .then((m) => {
        if (cancelled) return
        if (token !== fetchAttempt) return
        setExercise(m)
        setStatus('ready')
      })
      .catch((err: unknown) => {
        if (cancelled) return
        if (token !== fetchAttempt) return
        setErrorMessage(err instanceof Error ? err.message : 'Unexpected error')
        setStatus('error')
      })
    return () => {
      cancelled = true
    }
  }, [fetchAttempt, status])

  const handleGraded = useCallback((next_due_at: string) => {
    // The grade succeeded. We don't auto-fetch the next
    // matching here (matching has no /exercises/due yet —
    // Phase 9.6's session mixer owns fan-out). Drop back
    // to the empty-state with a "Grade recorded" toast
    // so the user can hit Generate another for a fresh
    // pick.
    setExercise(null)
    setStatus('empty')
    toast.success(`Grade recorded. Next review ${humanizeDelta(next_due_at)}.`)
  }, [])

  const handleGradeError = useCallback(
    (err: unknown) => {
      const status =
        err instanceof ExerciseApiError ? err.status : undefined
      if (status === 401) {
        navigate('/login', { replace: true, state: { from: '/exercises/match' } })
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
    navigate('/login', { replace: true, state: { from: '/exercises/match' } })
  }

  if (status === 'empty') {
    if (exercise) {
      // Post-grade empty — we have a body to render via the
      // shared ExerciseCard so the user sees the FSRS-scheduled
      // outcome note above the "Generate another" CTA.
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
              The FSRS scheduler will unlock the next match when it's due.
              Hit Generate another for a fresh pick.
            </p>
            <div className="pt-1">
              <button
                type="button"
                onClick={handleGenerateAnother}
                data-testid="match-generate-another-empty"
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
            Set up your weakness profile first
          </h2>
          <p className="text-sm text-slate-400">
            Matching exercises target a word on your weakest axis. Without a
            weakness profile, we don't know which axis to target — and
            picking at random would just frustrate you. Tell us where you're
            shaky and we'll start there.
          </p>
          <div className="flex items-center gap-3 pt-1">
            <Link
              to="/weakness-profile"
              className="px-3 py-1.5 text-sm rounded-lg bg-blue-600 text-white hover:bg-blue-700 transition-colors"
            >
              Open weakness profile
            </Link>
            <span className="text-xs text-slate-500">
              (You can also <Link to="/diagnostic" className="underline">run the diagnostic</Link> to skip the manual form.)
            </span>
          </div>
        </div>
      </div>
    )
  }

  if (status === 'loading' || status === 'idle') {
    return (
      <div className="max-w-2xl mx-auto px-6 py-12" data-testid="match-loading">
        <div className="rounded-lg border border-slate-800 bg-slate-900 p-6 space-y-4 animate-pulse">
          <div className="h-4 w-3/4 rounded bg-slate-800" />
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

  if (status === 'error') {
    const isAuth = !!errorMessage && /401|not authenticated/i.test(errorMessage)
    return (
      <div className="max-w-2xl mx-auto px-6 py-12" data-testid="match-error">
        <div
          role="alert"
          className="rounded-lg border border-red-900/60 bg-red-950/40 p-5 space-y-3"
        >
          <p className="text-sm text-red-300">
            {isAuth
              ? "You've been signed out — please log in again."
              : "Couldn't generate a matching exercise."}
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

  // status === 'ready'
  if (!exercise) {
    return (
      <div className="max-w-2xl mx-auto px-6 py-12">
        <div className="rounded-lg border border-slate-800 bg-slate-900 p-6 text-sm text-slate-400">
          The server returned an empty match. Try again in a moment.
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
    <div data-testid="match-ready">
      <ExerciseCard
        exercise={exercise}
        onGraded={handleGraded}
        onGradeError={handleGradeError}
      />
      <div className="max-w-2xl mx-auto px-6 pb-10">
        <button
          type="button"
          onClick={handleGenerateAnother}
          data-testid="match-generate-another"
          className="px-3 py-1.5 text-sm rounded-lg border border-slate-700 text-slate-200 hover:bg-slate-800 transition-colors"
        >
          Generate another
        </button>
      </div>
    </div>
  )
}
