import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  answerDiagnostic,
  applyDiagnostic,
  getDiagnosticResult,
  startDiagnostic,
  type DiagnosticQuestion as Question,
  type DiagnosticResult as Result,
} from '../api/diagnostic'
import { DiagnosticQuestion } from '../components/DiagnosticQuestion'
import { DiagnosticResult } from '../components/DiagnosticResult'
import { ProgressBar } from '../components/ProgressBar'

// Phase 3.2 (card t_64055c49): multi-step diagnostic page.
//
// Lifecycle:
//   1. mount -> POST /diagnostic/start -> store session_id + questions
//   2. user selects + Next -> POST /diagnostic/answer (optimistic advance,
//      revert on error). Back just decrements (server already has the answer).
//   3. on the 10th question, the CTA reads "See results" — we fire
//      /diagnostic/result instead of /diagnostic/answer and flip to the
//      result screen.
//   4. result screen: Apply -> POST /diagnostic/apply -> /weakness-profile.
//      Edit manually -> /weakness-profile with no PUT.
//
// Errors are surfaced per-step:
//   - start failure: top-of-page banner with a Retry button.
//   - per-answer failure: inline error under the question; we revert to
//     the previous question and let the user retry by clicking Next again.
//   - result failure: full-screen error with Retry.
//   - apply failure: inline error on the result screen; user can retry.
//
// State is local (useState). No state library, no test runner. The
// selectedLabel for each question is kept in a plain Record — selecting
// a different choice overwrites it (server is idempotent on answer).

type LoadState = 'loading' | 'ready' | 'error'
type ResultState = 'idle' | 'loading' | 'ready' | 'error'

export function DiagnosticPage() {
  const navigate = useNavigate()

  const [loadState, setLoadState] = useState<LoadState>('loading')
  const [loadError, setLoadError] = useState<string | null>(null)
  // A monotonically increasing counter that the Retry handler bumps
  // to re-trigger the start effect. The effect's dep array re-runs on
  // change; the start function lives outside the effect so we never
  // call setState synchronously inside the effect body.
  const [startAttempt, setStartAttempt] = useState(0)

  const [sessionId, setSessionId] = useState<string | null>(null)
  const [questions, setQuestions] = useState<Question[]>([])
  // Selected labels per question id. Persists across Back/Next so the
  // user sees their previous choice when they navigate back.
  const [selected, setSelected] = useState<Record<string, string>>({})
  const [index, setIndex] = useState(0)

  const [answerSubmitting, setAnswerSubmitting] = useState(false)
  const [answerError, setAnswerError] = useState<string | null>(null)

  const [resultState, setResultState] = useState<ResultState>('idle')
  const [resultError, setResultError] = useState<string | null>(null)
  const [result, setResult] = useState<Result | null>(null)

  const [applying, setApplying] = useState(false)
  const [applyError, setApplyError] = useState<string | null>(null)

  // Initial start call. Re-runs when startAttempt bumps (Retry button).
  useEffect(() => {
    let cancelled = false
    startDiagnostic()
      .then((start) => {
        if (cancelled) return
        setSessionId(start.session_id)
        setQuestions(start.questions)
        setSelected({})
        setIndex(0)
        setResultState('idle')
        setResult(null)
        setLoadState('ready')
        setLoadError(null)
      })
      .catch((err: unknown) => {
        if (cancelled) return
        setLoadError(err instanceof Error ? err.message : 'Could not start the probe')
        setLoadState('error')
      })
    return () => {
      cancelled = true
    }
  }, [startAttempt])

  function handleRetryStart() {
    setLoadState('loading')
    setLoadError(null)
    setStartAttempt((n) => n + 1)
  }

  const currentQuestion = questions[index] ?? null
  const isLast = questions.length > 0 && index === questions.length - 1
  const total = questions.length

  function handleSelect(label: string) {
    if (!currentQuestion) return
    setSelected((prev) => ({ ...prev, [currentQuestion.id]: label }))
    // Clear any stale per-answer error when the user picks a new option.
    if (answerError) setAnswerError(null)
  }

  async function handleAdvance() {
    if (!sessionId || !currentQuestion) return
    const label = selected[currentQuestion.id]
    if (label === undefined) return
    if (answerSubmitting) return

    if (!isLast) {
      // Optimistic: advance first, then fire the answer. If the API
      // call fails, revert to the previous question with an inline
      // error so the user can click Next again to retry.
      setAnswerError(null)
      const previousIndex = index
      setIndex((i) => i + 1)
      setAnswerSubmitting(true)
      try {
        await answerDiagnostic(sessionId, currentQuestion.id, label)
        // Index already advanced; nothing to do.
      } catch (err) {
        setIndex(previousIndex)
        setAnswerError(
          err instanceof Error ? err.message : "Couldn't save your answer",
        )
      } finally {
        setAnswerSubmitting(false)
      }
      return
    }

    // Last question: don't fire another /answer — go straight to result.
    setAnswerError(null)
    setResultState('loading')
    setResultError(null)
    try {
      // Optimistically fire the final answer so the server is in sync if
      // the user re-runs the probe. Failures here don't block the result
      // fetch (the server tolerates a missing final answer — the score
      // would just be missing one contributor).
      try {
        await answerDiagnostic(sessionId, currentQuestion.id, label)
      } catch {
        // Swallow: continue to the result endpoint regardless. The
        // scoring function tolerates a missing answer.
      }
      const r = await getDiagnosticResult(sessionId)
      setResult(r)
      setResultState('ready')
    } catch (err) {
      setResultError(
        err instanceof Error ? err.message : "Couldn't load your results",
      )
      setResultState('error')
    }
  }

  function handleBack() {
    if (index === 0) return
    setIndex((i) => i - 1)
    setAnswerError(null)
  }

  async function handleApply() {
    if (!sessionId || applying) return
    setApplying(true)
    setApplyError(null)
    try {
      await applyDiagnostic(sessionId)
      // Force the WeaknessProfilePage to re-fetch on next mount by
      // navigating with a state nudge; the page already does
      // getWeaknessProfile(user.id) on mount with [user.id] dep, so a
      // plain navigate is enough.
      navigate('/weakness-profile', { replace: true })
    } catch (err) {
      setApplyError(
        err instanceof Error ? err.message : "Couldn't apply this profile",
      )
      setApplying(false)
    }
  }

  function handleEdit() {
    navigate('/weakness-profile', { replace: true })
  }

  // -------- render --------

  if (loadState === 'loading') {
    return (
      <div className="max-w-2xl mx-auto px-6 py-12">
        <div className="rounded-lg border border-slate-800 bg-slate-900 p-6 text-sm text-slate-400">
          Starting the diagnostic probe…
        </div>
      </div>
    )
  }

  if (loadState === 'error') {
    return (
      <div className="max-w-2xl mx-auto px-6 py-12 space-y-4">
        <div
          role="alert"
          className="rounded-lg border border-red-900/60 bg-red-950/40 p-5 space-y-3"
        >
          <p className="text-sm text-red-300">
            Couldn't start the diagnostic probe.
          </p>
          {loadError && <p className="text-xs text-red-400">{loadError}</p>}
          <button
            type="button"
            onClick={handleRetryStart}
            className="px-3 py-1.5 text-sm rounded-lg border border-slate-700 text-slate-200 hover:bg-slate-800 transition-colors"
          >
            Retry
          </button>
        </div>
      </div>
    )
  }

  // Ready: show progress + the active question (or the result screen).
  return (
    <div className="max-w-2xl mx-auto px-6 py-10 space-y-8">
      <ProgressBar current={index + 1} total={total} />

      {resultState === 'ready' && result ? (
        <DiagnosticResult
          result={result}
          applying={applying}
          applyError={applyError}
          onApply={handleApply}
          onEdit={handleEdit}
        />
      ) : resultState === 'loading' ? (
        <div className="rounded-lg border border-slate-800 bg-slate-900 p-6 text-sm text-slate-400">
          Computing your results…
        </div>
      ) : resultState === 'error' ? (
        <div className="space-y-3" role="alert">
          <div className="rounded-lg border border-red-900/60 bg-red-950/40 p-5">
            <p className="text-sm text-red-300">Couldn't load your results</p>
            {resultError && (
              <p className="mt-1 text-xs text-red-400">{resultError}</p>
            )}
          </div>
          <button
            type="button"
            onClick={async () => {
              if (!sessionId) return
              setResultState('loading')
              setResultError(null)
              try {
                const r = await getDiagnosticResult(sessionId)
                setResult(r)
                setResultState('ready')
              } catch (err) {
                setResultError(
                  err instanceof Error
                    ? err.message
                    : "Couldn't load your results",
                )
                setResultState('error')
              }
            }}
            className="px-3 py-1.5 text-sm rounded-lg border border-slate-700 text-slate-200 hover:bg-slate-800 transition-colors"
          >
            Retry
          </button>
        </div>
      ) : currentQuestion ? (
        <DiagnosticQuestion
          question={currentQuestion}
          index={index}
          total={total}
          selectedLabel={selected[currentQuestion.id] ?? null}
          onSelect={handleSelect}
          onAdvance={handleAdvance}
          onBack={handleBack}
          canGoBack={index > 0}
          isLast={isLast}
          submitting={answerSubmitting}
          errorMessage={answerError}
        />
      ) : (
        <div className="rounded-lg border border-slate-800 bg-slate-900 p-6 text-sm text-slate-400">
          No questions available.
        </div>
      )}
    </div>
  )
}
