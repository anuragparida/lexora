import { useEffect, useMemo, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { toast } from 'sonner'
import { generateCloze, type ClozeExercise } from '../api/cloze'
import { getMe, type AuthUser } from '../auth'

// Phase 4.5 (card t_4a9f172e): minimal cloze exercise surface.
//
// One component, one route (`/exercises/cloze`), one render path.
// Phase 4 does NOT split `App.tsx` — that's a separate refactor
// card (forbidden by the standing rule: no opportunistic refactors).
//
// State machine (the only one in the page):
//
//   idle        -> user landed; we haven't fetched yet
//   loading     -> /exercises/cloze POST in flight
//   ready       -> ClozeExercise on hand, options rendered
//   error       -> fetch failed; toast + retry button
//   empty       -> user has no weakness profile; show the honest
//                  empty state with a link to /weakness-profile
//
// The blank is rendered inline as a styled `<span>` (not an
// `<input>` — the card body offers either, and a clickable blank
// keeps the multiple-choice buttons as the canonical interaction
// surface). The distractor / correct-answer buttons render the
// four word_ids as their labels for Phase 4; a future endpoint
// (Phase 5+) will resolve word_id -> German string. The card
// body says this is OK ("or however the team prefers" — we
// document the choice in the docstring and the test).
//
// `getMe()` is used to detect the empty-profile case. The
// backend's `/exercises/cloze` returns a 4xx when the user has no
// profile too, but doing the client-side check first lets us show
// the user a friendly "set your profile first" message instead of
// a generic 4xx toast. The two are equivalent in terms of truth —
// the server is the source of truth for the wire format, but the
// empty-state is a UX concern the server doesn't dictate.

type Status = 'idle' | 'loading' | 'ready' | 'error' | 'empty'

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
// the user clicks "Generate another".
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
          setStatus('ready')
        })
        .catch((err: unknown) => {
          if (cancelled) return
          if (token !== fetchAttempt) return
          const msg =
            err instanceof Error ? err.message : 'Could not generate a cloze'
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

  // ---- handlers ----

  function handleSelect(wordId: number) {
    setSelectedWordId(wordId)
  }

  function handleSubmit() {
    if (selectedWordId === null) return
    // Phase 4 has no grading endpoint. Per the card body, this
    // is the honest "coming soon" sonner toast — it explicitly
    // names Phase 5 so the user knows what's coming. We add a
    // tiny correctness hint so the user gets feedback even
    // before Phase 5's grading loop lands (the server's
    // Pydantic-validated answer_word_id is the source of
    // truth).
    const correct =
      exercise && selectedWordId === exercise.answer_word_id
    toast.info(
      correct
        ? 'Phase 5 will grade this — coming soon. (That would have been correct.)'
        : 'Phase 5 will grade this — coming soon.',
    )
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

        <div className="flex items-center gap-3 pt-2">
          <button
            type="button"
            onClick={handleSubmit}
            disabled={selectedWordId === null}
            data-testid="cloze-submit"
            className="px-3 py-1.5 text-sm rounded-lg bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            Submit
          </button>
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
