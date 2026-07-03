import { useCallback, useEffect, useMemo, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { toast } from 'sonner'
import {
  generateCloze,
  getDueCloze,
  gradeCloze,
  ClozeApiError,
  type ClozeExercise,
  type ClozeExerciseOut,
  type Grade,
} from '../api/cloze'
import { getMe, type AuthUser } from '../auth'
import { GradeButtons } from '../components/GradeButtons'

// Phase 4.5 (card t_4a9f172e): minimal cloze exercise surface.
// Phase 5.5 (card t_f253456b): wired to /exercises/grade +
// /exercises/due. The placeholder Submit is gone; the inline
// grade-then-next flow IS the page's primary action.
//
// State machine (the only one in the page):
//
//   idle        -> user landed; we haven't fetched yet
//   loading     -> /exercises/cloze POST in flight
//   ready       -> ClozeExercise on hand, options rendered
//   error       -> fetch failed; toast + retry button
//   empty       -> user has no weakness profile; show the honest
//                  empty state with a link to /weakness-profile
//   emptyDue    -> /exercises/due returned 204 ("all caught up").
//                  Reached AFTER a successful grade. The page
//                  drops back to this state and shows an honest
//                  "nothing due" message — we never fake a fresh
//                  pick to keep the user engaged.
//
// The grade click adds a sub-state `isGrading` so the four
// grade buttons disable during the round-trip; double-clicks
// during a pending grade would otherwise double-fire
// `gradeCloze` (and `getDueCloze`) on the same word, which the
// closed loop is explicitly NOT designed to handle (FSRS would
// schedule the same card twice, corrupting `reps` / `state`).
//
// The blank is rendered inline as a styled `<span>` (not an
// `<input>` — the card body offers either, and a clickable blank
// keeps the multiple-choice buttons as the canonical interaction
// surface). The distractor / correct-answer buttons render the
// four word_ids as their labels for Phase 4; a future endpoint
// (Phase 5+) will resolve word_id -> German string.
//
// `getMe()` is used to detect the empty-profile case. The
// backend's `/exercises/cloze` returns a 4xx when the user has no
// profile too, but doing the client-side check first lets us show
// the user a friendly "set your profile first" message instead of
// a generic 4xx toast. The two are equivalent in terms of truth —
// the server is the source of truth for the wire format, but the
// empty-state is a UX concern the server doesn't dictate.

type Status = 'idle' | 'loading' | 'ready' | 'error' | 'empty' | 'emptyDue'

interface Choice {
  word_id: number
  isAnswer: boolean
}

interface Props {
  user: AuthUser
}

// Fisher–Yates shuffle, in place. Used to randomize the order of
// the four multiple-choice buttons. Deterministic per page mount
// is NOT desired — re-renders within the same mount should keep
// the same order so a click "sticks". Re-shuffling happens when
// the user clicks "Generate another" or when a new card arrives
// via the grade-then-next flow.
function shuffle<T>(arr: readonly T[]): T[] {
  const out = arr.slice()
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1))
    ;[out[i], out[j]] = [out[j], out[i]]
  }
  return out
}

// Split a sentence-with-blank into the parts surrounding the
// single `___` marker. Returns three segments: the prefix, the
// blank marker, and the suffix. If `___` is missing (defensive —
// the Pydantic model on the server requires it but we never
// trust the wire blindly) we return the whole sentence as the
// prefix and empty suffix/blank. Used only for rendering.
function splitOnBlank(sentence: string): {
  before: string
  after: string
  hasBlank: boolean
} {
  const marker = '___'
  const idx = sentence.indexOf(marker)
  if (idx === -1) {
    return { before: sentence, after: '', hasBlank: false }
  }
  return {
    before: sentence.slice(0, idx),
    after: sentence.slice(idx + marker.length),
    hasBlank: true,
  }
}

// Project a `next_due_at` ISO timestamp into a human-readable
// delta string ("in 12m", "in 2h", "in 3d", "now"). The grade
// toast shows this so the user sees the FSRS scheduling effect
// without having to inspect a debug panel. Phase 5.5 only
// renders this string in the success toast; the underlying
// timestamp is preserved on `exercise` for any future UI that
// needs it.
function humanizeDelta(nextDueAt: string): string {
  const now = Date.now()
  const target = new Date(nextDueAt).getTime()
  if (!Number.isFinite(target)) return 'soon'
  const deltaMs = target - now
  if (deltaMs <= 0) return 'now'
  const minutes = Math.round(deltaMs / 60_000)
  if (minutes < 1) return 'now'
  if (minutes < 60) return `in ${minutes}m`
  const hours = Math.round(minutes / 60)
  if (hours < 48) return `in ${hours}h`
  const days = Math.round(hours / 24)
  return `in ${days}d`
}

// Extract the user-visible error message from whatever the API
// layer threw. The 422 case carries a Pydantic `detail` array;
// the 401/500/204 paths return plain strings. We surface the
// `detail` field when present so the validation message reaches
// the user (Gotcha #4: 422 toast should show the validation
// message from the response body, not a generic string).
function extractErrorMessage(err: unknown): string {
  if (err instanceof Error) {
    return err.message
  }
  return 'Unexpected error'
}

export function ClozePage({ user }: Props) {
  const navigate = useNavigate()
  const [status, setStatus] = useState<Status>('idle')
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [exercise, setExercise] = useState<ClozeExercise | null>(null)
  const [selectedWordId, setSelectedWordId] = useState<number | null>(null)
  // Bumped on every "Generate another" click. Drives the
  // fetch effect's dep array; isolation prevents stale closures
  // from racing the previous in-flight request.
  const [fetchAttempt, setFetchAttempt] = useState(0)
  // True while a grade round-trip is in flight. Disables the
  // four grade buttons so a double-click can't fire `gradeCloze`
  // twice on the same word — which would double-schedule the
  // card on the FSRS side.
  const [isGrading, setIsGrading] = useState(false)

  // Detect the empty-profile case once on mount. We don't need
  // to re-probe on every "Generate another" — if the user has
  // filled in their profile mid-session, the next "Generate"
  // will just succeed.
  //
  // On success (user has a profile) we set status='loading'
  // synchronously and bump fetchAttempt to 1. The fetch
  // effect then kicks off and resolves into 'ready' or
  // 'error'. We avoid setting status from inside the effect
  // body to keep the React-hooks/exhaustive-deps lint rule
  // happy.
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
          // Reset transient state synchronously here (the
          // .then callback is a microtask, not the effect
          // body, so the setState-in-effect lint rule does
          // not apply — the rule covers synchronous calls
          // in the effect body, not state updates in
          // promise callbacks).
          setErrorMessage(null)
          setSelectedWordId(null)
          setStatus('loading')
          setFetchAttempt(1)
        }
      })
      .catch(() => {
        // 401 -> getMe throws; ProtectedRoute's effect already
        // handles the redirect on its own mount. We still need
        // to do SOMETHING here so the page doesn't sit in
        // 'idle' forever. Fall through to a fetch attempt —
        // if the user is genuinely 401, the fetch will 401 too
        // and we'll surface the error toast.
        if (cancelled) return
        setErrorMessage(null)
        setSelectedWordId(null)
        setStatus('loading')
        setFetchAttempt(1)
      })
    return () => {
      cancelled = true
    }
  }, [])

  // Fetch effect. Re-runs when fetchAttempt bumps.
  // fetchAttempt starts at 0 (no fetch). The getMe effect
  // above sets it to 1 once the user is known to have a
  // profile (or once a 401-class failure of getMe has been
  // swallowed). After that, the user can re-trigger it by
  // clicking "Generate another" (which bumps the counter
  // past 1).
  //
  // The "loading" status is set in the click handlers (which
  // run outside the effect), not in this effect body — the
  // React-hooks/exhaustive-deps lint rule flags synchronous
  // setState calls inside an effect as a cascading-render
  // smell. The click handlers call setStatus('loading') and
  // setFetchAttempt(n + 1) in the same render; the effect
  // then only handles the async resolution.
  useEffect(() => {
    if (fetchAttempt === 0) return
    if (status === 'empty') return
    if (status === 'loading') {
      // The user (or getMe's resolution) just set us into
      // loading — kick off the actual fetch. The transient
      // state (errorMessage, selectedWordId) is reset by the
      // trigger (click handler or getMe's .then), not here,
      // to keep this effect pure of setState calls.
      const token = fetchAttempt
      let cancelled = false
      generateCloze()
        .then((cloze) => {
          if (cancelled) return
          if (token !== fetchAttempt) return
          setExercise(cloze)
          setSelectedWordId(null)
          setStatus('ready')
        })
        .catch((err: unknown) => {
          if (cancelled) return
          if (token !== fetchAttempt) return
          const msg = extractErrorMessage(err)
          setErrorMessage(msg)
          setStatus('error')
        })
      return () => {
        cancelled = true
      }
    }
  }, [fetchAttempt, status])

  // Build the four choices (3 distractors + 1 answer) and
  // shuffle them once per exercise. Recomputed when the
  // exercise changes so "Generate another" gets a fresh order.
  const choices: Choice[] = useMemo(() => {
    if (!exercise) return []
    const all: Choice[] = [
      { word_id: exercise.answer_word_id, isAnswer: true },
      ...exercise.distractors.map((id) => ({
        word_id: id,
        isAnswer: false,
      })),
    ]
    return shuffle(all)
  }, [exercise])

  // The grade-then-next handler. Wires Phase 5.5's primary
  // closed-loop flow:
  //
  //   1. Lock the buttons (isGrading=true).
  //   2. POST /exercises/grade with the literal exercise_type
  //      and the chosen grade.
  //   3. On 200: optimistically GET /exercises/due for the
  //      next card. If the server returns 204, transition to
  //      the "emptyDue" state ("All caught up"). If it
  //      returns a `ClozeExerciseOut`, replace the current
  //      exercise and toast the interval change.
  //   4. On 422: surface the Pydantic validation detail in a
  //      toast. The current card stays on screen — we do NOT
  //      advance on validation failure.
  //   5. On 500/network: surface a generic "Grade failed —
  //      try again" toast. The current card stays on screen
  //      (the user can re-click the same grade button).
  //
  // We use useCallback so the GradeButtons' prop reference is
  // stable across renders that don't change the handler —
  // matters for React.memo and for the Phase-6 split where
  // GradeButtons will be its own memo'd subtree.
  const handleGrade = useCallback(
    async (grade: Grade) => {
      if (!exercise) return
      if (isGrading) return
      const exerciseIdForGrade = exercise.answer_word_id
      setIsGrading(true)
      try {
        const response = await gradeCloze(exerciseIdForGrade, grade)
        // Grade landed on the server. Now optimistically fetch
        // the next due card. If that fetch also errors out,
        // we still tell the user the grade was recorded —
        // they can click "Generate another" to retry the
        // next-card fetch independently.
        let nextExercise: ClozeExerciseOut | null = null
        try {
          nextExercise = await getDueCloze()
        } catch {
          // Swallow: the grade already succeeded; the
          // next-card fetch is best-effort. Fall through to
          // a "generate another" recovery message.
        }
        if (nextExercise === null) {
          // 204 from /exercises/due: nothing due right now.
          // This is an honest state — do NOT fabricate a
          // fresh pick to keep the user engaged.
          setExercise(null)
          setSelectedWordId(null)
          setStatus('emptyDue')
          toast.success(
            `Grade recorded. Next review ${humanizeDelta(response.next_due_at)}.`,
          )
        } else {
          setExercise(nextExercise)
          setSelectedWordId(null)
          setStatus('ready')
          toast.success(
            `Grade recorded. Next review ${humanizeDelta(response.next_due_at)}.`,
          )
        }
      } catch (err: unknown) {
        // We discriminate the failure shape via the HTTP status
        // code on ClozeApiError (Phase 5.5: the API layer throws
        // a typed error that carries `status`). String-matching
        // the message would be fragile — Pydantic's 422 detail
        // is a free-form string the backend author chooses.
        //
        // Status codes:
        //   401 -> auth bounce. Redirect to /login and stop.
        //   422 -> Pydantic validation. Surface the detail
        //          verbatim so the user sees the field-level
        //          reason (Gotcha #4: not a generic string).
        //   500 -> server fault. Generic copy. The user retries.
        //   else (network, parse) -> generic copy.
        const status =
          err instanceof ClozeApiError ? err.status : undefined
        const msg = extractErrorMessage(err)
        if (status === 401) {
          navigate('/login', {
            replace: true,
            state: { from: '/exercises/cloze' },
          })
          return
        }
        if (status === 422) {
          toast.error(msg)
        } else {
          // 500, network errors, unexpected shapes: generic copy.
          toast.error('Grade failed — try again')
        }
        // Critical: keep the current card on screen. The user
        // can re-click any grade button; the closed loop
        // does not advance on failure (the server never
        // acknowledged the grade).
      } finally {
        setIsGrading(false)
      }
    },
    [exercise, isGrading, navigate],
  )

  // ---- handlers ----

  function handleSelect(wordId: number) {
    setSelectedWordId(wordId)
  }

  function handleGenerateAnother() {
    // Reset transient state synchronously here (not in the
    // effect body) so the React-hooks lint rule stays happy.
    setErrorMessage(null)
    setSelectedWordId(null)
    setStatus('loading')
    setFetchAttempt((n) => n + 1)
  }

  function handleRetry() {
    setErrorMessage(null)
    setSelectedWordId(null)
    setStatus('loading')
    setFetchAttempt((n) => n + 1)
  }

  function handleRedirectToLogin() {
    // 401 surfaced: redirect to /login with a replace so the
    // back button doesn't return the user to the cloze page
    // (which would just bounce them again). Mirrors
    // ProtectedRoute's redirect pattern.
    navigate('/login', { replace: true, state: { from: '/exercises/cloze' } })
  }

  // ---- render branches ----

  if (status === 'empty') {
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
            Cloze exercises target a word on your weakest axis. Without a
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
      <div className="max-w-2xl mx-auto px-6 py-12" data-testid="cloze-loading">
        <div className="rounded-lg border border-slate-800 bg-slate-900 p-6 space-y-4 animate-pulse">
          <div className="h-4 w-3/4 rounded bg-slate-800" />
          <div className="h-4 w-2/3 rounded bg-slate-800" />
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
      <div className="max-w-2xl mx-auto px-6 py-12" data-testid="cloze-error">
        <div
          role="alert"
          className="rounded-lg border border-red-900/60 bg-red-950/40 p-5 space-y-3"
        >
          <p className="text-sm text-red-300">
            {isAuth
              ? "You've been signed out — please log in again."
              : "Couldn't generate a cloze."}
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

  // 204 from /exercises/due: nothing due right now. Phase 5.5's
  // honest empty state — do NOT generate a fresh cloze to fill
  // the screen. The user has caught up.
  if (status === 'emptyDue') {
    return (
      <div
        className="max-w-2xl mx-auto px-6 py-12"
        data-testid="cloze-empty-due"
      >
        <div
          role="status"
          className="rounded-lg border border-slate-800 bg-slate-900 p-6 space-y-4"
        >
          <h2 className="text-base font-semibold text-slate-100">
            All caught up — nothing due right now.
          </h2>
          <p className="text-sm text-slate-400">
            You graded the last scheduled review. Your FSRS schedule will
            unlock the next card when it's due. If you want a fresh pick
            anyway, generate one below — it will be tagged as a new card
            on the backend, not a scheduled review.
          </p>
          <div className="pt-1">
            <button
              type="button"
              onClick={handleGenerateAnother}
              data-testid="cloze-generate-another-empty"
              className="px-3 py-1.5 text-sm rounded-lg border border-slate-700 text-slate-200 hover:bg-slate-800 transition-colors"
            >
              Generate another (fresh pick)
            </button>
          </div>
        </div>
      </div>
    )
  }

  // status === 'ready' — exercise is non-null here.
  if (!exercise) {
    return (
      <div className="max-w-2xl mx-auto px-6 py-12">
        <div className="rounded-lg border border-slate-800 bg-slate-900 p-6 text-sm text-slate-400">
          The server returned an empty cloze. Try again in a moment.
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

  const { before, after, hasBlank } = splitOnBlank(exercise.sentence_with_blank)

  return (
    <div className="max-w-2xl mx-auto px-6 py-10 space-y-6" data-testid="cloze-ready">
      <div className="rounded-lg border border-slate-800 bg-slate-900 p-6 space-y-5">
        <div className="text-xs uppercase tracking-wide text-slate-500">
          Cloze · {exercise.difficulty} · prompt {exercise.prompt_template_version}
        </div>

        <p
          className="text-lg leading-relaxed text-slate-100"
          data-testid="cloze-sentence"
        >
          {before}
          <span
            data-testid="cloze-blank"
            className="mx-1 inline-block min-w-[3rem] border-b-2 border-blue-500 px-2 py-0.5 text-center text-slate-300 align-baseline"
          >
            {selectedWordId === null
              ? '___'
              : hasBlank
                ? `(${wordIdLabel(selectedWordId)})`
                : ''}
          </span>
          {after}
        </p>

        <div
          className="grid grid-cols-2 gap-3"
          role="group"
          aria-label="Cloze choices"
        >
          {choices.map((c) => {
            const isSelected = selectedWordId === c.word_id
            return (
              <button
                key={c.word_id}
                type="button"
                onClick={() => handleSelect(c.word_id)}
                aria-pressed={isSelected}
                data-testid={`cloze-choice-${c.word_id}`}
                className={
                  'rounded-lg border px-4 py-3 text-left text-sm transition-colors ' +
                  (isSelected
                    ? 'border-blue-500 bg-blue-950/40 text-slate-100'
                    : 'border-slate-700 bg-slate-950 text-slate-200 hover:bg-slate-800')
                }
              >
                {wordIdLabel(c.word_id)}
              </button>
            )
          })}
        </div>

        <div className="space-y-3 pt-2">
          <GradeButtons onGrade={handleGrade} disabled={isGrading} />
          <div className="flex items-center gap-3">
            <button
              type="button"
              onClick={handleGenerateAnother}
              data-testid="cloze-generate-another"
              className="px-3 py-1.5 text-sm rounded-lg border border-slate-700 text-slate-200 hover:bg-slate-800 transition-colors"
            >
              Generate another
            </button>
            <span className="text-xs text-slate-500">
              Logged in as {user.email}
            </span>
          </div>
        </div>

        {exercise.rationale && (
          <p className="text-xs text-slate-500 pt-2 border-t border-slate-800">
            <span className="font-semibold text-slate-400">Why this cloze: </span>
            {exercise.rationale}
          </p>
        )}
      </div>
    </div>
  )
}

// Render a `word_id` as a human-readable label. The Phase 4
// backend's `ClozeExercise` doesn't carry the resolved German
// word string — the Pydantic model only carries `word_id`. Phase
// 5's grading integration will likely add a `word_text` field or
// a small `/words/{id}` lookup. For Phase 4 we render `#42` as
// a clear placeholder so the user knows what the buttons are.
// The card body says "the distractor words" — the words
// themselves are not on the wire today, only the ids.
function wordIdLabel(id: number): string {
  return `word #${id}`
}