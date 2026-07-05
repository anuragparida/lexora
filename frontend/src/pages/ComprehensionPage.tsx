import { useCallback, useEffect, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { toast } from 'sonner'
import {
  generateComprehension,
  ExerciseApiError,
  type ComprehensionExercise,
} from '../api/exercises'
import { getMe } from '../auth'
import { ExerciseCard } from '../components/ExerciseCard'
import { humanizeDelta } from '../components/delta'

// Phase 9.5 (card t_e4dc0404) — thin ComprehensionPage.
//
// Mirrors MatchingPage's state machine (loading / ready /
// error / empty) so the four per-type pages share the same
// render-tree shape. The body lives in ``<ExerciseCard />``
// which switches on ``exercise_type`` to render passage +
// question + 4-option MC.
//
// Wire contract: POST /exercises/comprehension with the
// Phase 6.5 ``ComprehensionGenerateRequest`` body (empty body
// ok; enable_rag defaults to False). Response:
// ``ComprehensionExercise`` (Phase 6.5).

type Status = 'idle' | 'loading' | 'ready' | 'error' | 'empty'

export function ComprehensionPage() {
  const navigate = useNavigate()
  const [status, setStatus] = useState<Status>('idle')
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [exercise, setExercise] = useState<ComprehensionExercise | null>(null)
  const [fetchAttempt, setFetchAttempt] = useState(0)

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
    generateComprehension()
      .then((c) => {
        if (cancelled) return
        if (token !== fetchAttempt) return
        setExercise(c)
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
          state: { from: '/exercises/comprehension' },
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
      state: { from: '/exercises/comprehension' },
    })
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
              The FSRS scheduler will unlock the next comprehension when it's
              due. Hit Generate another for a fresh pick.
            </p>
            <div className="pt-1">
              <button
                type="button"
                onClick={handleGenerateAnother}
                data-testid="comprehension-generate-another-empty"
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
            Comprehension exercises read a passage on a word from your weakest
            axis. Without a weakness profile, we don't know which axis to
            target — picking at random would just frustrate you.
          </p>
          <div className="flex items-center gap-3 pt-1">
            <Link
              to="/weakness-profile"
              className="px-3 py-1.5 text-sm rounded-lg bg-blue-600 text-white hover:bg-blue-700 transition-colors"
            >
              Open weakness profile
            </Link>
            <span className="text-xs text-slate-500">
              (You can also <Link to="/diagnostic" className="underline">run the diagnostic</Link>.)
            </span>
          </div>
        </div>
      </div>
    )
  }

  if (status === 'loading' || status === 'idle') {
    return (
      <div className="max-w-2xl mx-auto px-6 py-12" data-testid="comprehension-loading">
        <div className="rounded-lg border border-slate-800 bg-slate-900 p-6 space-y-4 animate-pulse">
          <div className="h-4 w-3/4 rounded bg-slate-800" />
          <div className="h-4 w-2/3 rounded bg-slate-800" />
          <div className="grid grid-cols-1 gap-3 pt-4">
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
      <div className="max-w-2xl mx-auto px-6 py-12" data-testid="comprehension-error">
        <div
          role="alert"
          className="rounded-lg border border-red-900/60 bg-red-950/40 p-5 space-y-3"
        >
          <p className="text-sm text-red-300">
            {isAuth
              ? "You've been signed out — please log in again."
              : "Couldn't generate a comprehension exercise."}
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
          The server returned an empty comprehension. Try again in a moment.
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
    <div data-testid="comprehension-ready">
      <ExerciseCard
        exercise={exercise}
        onGraded={handleGraded}
        onGradeError={handleGradeError}
      />
      <div className="max-w-2xl mx-auto px-6 pb-10">
        <button
          type="button"
          onClick={handleGenerateAnother}
          data-testid="comprehension-generate-another"
          className="px-3 py-1.5 text-sm rounded-lg border border-slate-700 text-slate-200 hover:bg-slate-800 transition-colors"
        >
          Generate another
        </button>
      </div>
    </div>
  )
}
