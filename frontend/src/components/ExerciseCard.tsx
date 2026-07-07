import { useCallback, useMemo, useState } from 'react'
import type {
  AnyExercise,
  ComprehensionChoiceKey,
  ComprehensionExercise,
  Grade,
  IdiomExercise,
  MatchingExercise,
  MatchingPair,
  MatchingRightKind,
} from '../api/exercises'
import { gradeExercise } from '../api/exercises'
import { GradeButtons } from './GradeButtons'

// Phase 9.5 (card t_e4dc0404) — shared exercise renderer.
//
// This is the load-bearing piece of Phase 9.5: the four per-type
// pages (``MatchingPage`` / ``ComprehensionPage`` / ``IdiomPage``
// / the refactored ``ClozePage``) are thin wrappers that fetch
// their backend endpoint and pass the typed result into
// ``<ExerciseCard />``. The card owns:
//
//   - the per-type body render (matching: a left-column / right-
//     column match UI; comprehension: passage + MC question;
//     idiom: phrase + definition + example_usage; cloze:
//     sentence-with-blank + 4-option choices)
//   - the user's interaction state (selected match pairs / choice
//     key / accepted the idiom)
//   - the shared 3-button FSRS grade bar (Again / Hard / Good /
//     Easy) via the Phase 5.5 ``<GradeButtons />`` component
//   - the ``POST /exercises/grade`` round-trip and its pending
//     state (``isGrading`` is shared so a double-click can't
//     double-fire)
//
// Design choices locked by the card body:
//
//   - **Discriminated union prop.** ``exercise`` is the wire
//     shape — a tagged union on ``exercise_type`` so the render
//     switch covers all four branches with TypeScript's
//     exhaustiveness check. New exercise types (Phase 9.6+)
//     widen the union, not the prop.
//   - **Server-minted exercise_id.** Matching / comprehension
//     / idiom all carry ``exercise_id: number`` on the wire; the
//     card passes it to ``/exercises/grade`` verbatim. Cloze is
//     the only outlier (Phase 4.2 wire used
//     ``answer_word_id``); we accept that as a typed
//     ``exercise.exercise_id ?? exercise.answer_word_id`` fallback.
//   - **"Generate another" lives in the page, not the card.**
//     Each per-type page owns its own fetch lifecycle
//     (``useEffect`` + ``fetchAttempt`` bump), so the card stays
//     presentational and reusable in the Phase 9.6 ``SessionPage``
//     mixer where the session drives the next-exercise state.

interface ExerciseCardProps {
  exercise: AnyExercise
  disabled?: boolean
  onGraded?: (next_due_at: string) => void
  // Surfaced when gradeExercise throws so the parent page can
  // show its own toast / 401 redirect. The card itself never
  // throws — it always returns to the ``isGrading=false`` state
  // and lets the parent decide what to do.
  onGradeError?: (err: unknown) => void
}

// --- shared helpers ------------------------------------------------------
//
// ``humanizeDelta`` lives in ``./delta.ts`` (split off for the
// react-refresh lint rule). Pages import it directly.

// --- per-type body renderers ---------------------------------------------

function rightKindLabel(kind: MatchingRightKind): string {
  return kind === 'translation' ? 'translation' : 'synonym'
}

// Render the matching body. The user picks a left item then a
// right item; we record the pair in local state. ``pairs`` is
// bounded 2..8 server-side. The render is a two-column grid
// with a "Pick left then right" hint; no fancy drag-and-drop
// (out of scope; Phase 9.6 may swap in a richer UI).
function MatchingBody({ exercise }: { exercise: MatchingExercise }) {
  const [matches, setMatches] = useState<Record<number, number>>({})
  const [pendingLeft, setPendingLeft] = useState<number | null>(null)

  const handlePickLeft = useCallback((left: number) => {
    setPendingLeft(left)
  }, [])

  const handlePickRight = useCallback(
    (right: number) => {
      if (pendingLeft === null) return
      setMatches((m) => ({ ...m, [pendingLeft]: right }))
      setPendingLeft(null)
    },
    [pendingLeft],
  )

  const allMatched = Object.keys(matches).length === exercise.pairs.length

  return (
    <div className="space-y-4">
      <p className="text-xs uppercase tracking-wide text-slate-500">
        Matching · connect each German word on the left to its{' '}
        {exercise.pairs[0]?.right_kind ?? 'translation'} on the right.
      </p>
      <div className="grid grid-cols-2 gap-4">
        <div role="list" aria-label="Match left" className="space-y-2">
          {exercise.pairs.map((p: MatchingPair) => {
            const isPicked = pendingLeft === p.left_word_id
            const isMatched = matches[p.left_word_id] !== undefined
            const matchedRight = matches[p.left_word_id]
            return (
              <button
                key={`L-${p.left_word_id}`}
                type="button"
                role="listitem"
                onClick={() => handlePickLeft(p.left_word_id)}
                aria-pressed={isPicked}
                data-testid={`match-left-${p.left_word_id}`}
                className={
                  'w-full rounded-lg border px-3 py-3 text-left text-sm transition-colors ' +
                  (isMatched
                    ? 'border-emerald-700 bg-emerald-950/40 text-emerald-200'
                    : isPicked
                      ? 'border-blue-500 bg-blue-950/40 text-slate-100'
                      : 'border-slate-700 bg-slate-950 text-slate-200 hover:bg-slate-800')
                }
              >
                <span className="font-medium">word #{p.left_word_id}</span>
                {isMatched && (
                  <span className="ml-2 text-xs text-slate-400">
                    → word #{matchedRight} ({rightKindLabel(p.right_kind)})
                  </span>
                )}
              </button>
            )
          })}
        </div>
        <div role="list" aria-label="Match right" className="space-y-2">
          {exercise.pairs.map((p: MatchingPair) => {
            // Right-side option is matched iff any left pair points at it.
            const matchedByLeft = Object.entries(matches).find(
              ([, right]) => right === p.right_word_id,
            )
            const isTaken = matchedByLeft !== undefined
            return (
              <button
                key={`R-${p.right_word_id}`}
                type="button"
                role="listitem"
                onClick={() => handlePickRight(p.right_word_id)}
                disabled={isTaken}
                aria-pressed={false}
                data-testid={`match-right-${p.right_word_id}`}
                className={
                  'w-full rounded-lg border px-3 py-3 text-left text-sm transition-colors ' +
                  (isTaken
                    ? 'border-slate-800 bg-slate-900 text-slate-500 cursor-not-allowed'
                    : 'border-slate-700 bg-slate-950 text-slate-200 hover:bg-slate-800')
                }
              >
                <span className="font-medium">word #{p.right_word_id}</span>
                <span className="ml-2 text-xs text-slate-400">
                  ({rightKindLabel(p.right_kind)})
                </span>
              </button>
            )
          })}
        </div>
      </div>
      <p className="text-xs text-slate-500">
        {allMatched
          ? 'All pairs connected. Grade when ready.'
          : pendingLeft === null
            ? 'Pick a left-side item first.'
            : 'Now pick the matching right-side item.'}
      </p>
    </div>
  )
}

function ComprehensionBody({
  exercise,
}: {
  exercise: ComprehensionExercise
}) {
  const [picked, setPicked] = useState<ComprehensionChoiceKey | null>(null)
  const choiceOrder: ComprehensionChoiceKey[] = useMemo(
    () => ['A', 'B', 'C', 'D'],
    [],
  )
  return (
    <div className="space-y-4">
      <p className="text-xs uppercase tracking-wide text-slate-500">
        Comprehension
      </p>
      <p
        className="text-base leading-relaxed text-slate-100 whitespace-pre-line"
        data-testid="comprehension-passage"
      >
        {exercise.passage}
      </p>
      <p
        className="text-base leading-relaxed text-slate-200"
        data-testid="comprehension-question"
      >
        {exercise.question}
      </p>
      <div
        className="grid grid-cols-1 gap-3"
        role="radiogroup"
        aria-label="Comprehension choices"
      >
        {choiceOrder.map((key) => {
          const isPicked = picked === key
          return (
            <button
              key={key}
              type="button"
              role="radio"
              aria-checked={isPicked}
              onClick={() => setPicked(key)}
              data-testid={`comprehension-choice-${key}`}
              className={
                'rounded-lg border px-4 py-3 text-left text-sm transition-colors ' +
                (isPicked
                  ? 'border-blue-500 bg-blue-950/40 text-slate-100'
                  : 'border-slate-700 bg-slate-950 text-slate-200 hover:bg-slate-800')
              }
            >
              <span className="font-semibold mr-2">{key}.</span>
              {exercise.choices[key]}
            </button>
          )
        })}
      </div>
      <p className="text-xs text-slate-500">
        {picked === null
          ? 'Pick an answer, then grade when ready.'
          : `You picked ${picked}. Grade when ready.`}
      </p>
    </div>
  )
}

function IdiomBody({ exercise }: { exercise: IdiomExercise }) {
  // The idiom body is largely presentational: phrase, definition,
  // example_usage, optional attestation. The user "accepts" the
  // card by picking Easy (4); we leave that decision to the shared
  // grade bar rather than wiring a separate "I know this" button.
  return (
    <div className="space-y-4">
      <p className="text-xs uppercase tracking-wide text-slate-500">
        Idiom · {exercise.frequency_band}
      </p>
      <p
        className="text-2xl font-semibold text-slate-100"
        data-testid="idiom-phrase"
      >
        {exercise.phrase}
      </p>
      <div className="rounded-lg border border-slate-800 bg-slate-950 p-4 space-y-2">
        <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">
          Definition
        </p>
        <p className="text-sm text-slate-200" data-testid="idiom-definition">
          {exercise.definition}
        </p>
      </div>
      <div className="rounded-lg border border-slate-800 bg-slate-950 p-4 space-y-2">
        <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">
          Example
        </p>
        <p className="text-sm text-slate-200 italic" data-testid="idiom-example">
          {exercise.example_usage}
        </p>
      </div>
      {exercise.attested_quote && exercise.attested_source && (
        <div className="rounded-lg border border-amber-800 bg-amber-950/30 p-4 space-y-2">
          <p className="text-xs font-semibold uppercase tracking-wide text-amber-300">
            Literary attestation
          </p>
          <p className="text-sm text-amber-100 italic">
            "{exercise.attested_quote}"
          </p>
          <p className="text-xs text-amber-400">
            — {exercise.attested_source}
          </p>
        </div>
      )}
      <p className="text-xs text-slate-500">
        Sources: {exercise.source_attribution}. When you've reviewed the
        meaning and example, grade below.
      </p>
    </div>
  )
}

// --- the card itself -----------------------------------------------------

export function ExerciseCard({
  exercise,
  disabled = false,
  onGraded,
  onGradeError,
}: ExerciseCardProps) {
  const [isGrading, setIsGrading] = useState(false)

  // The exercise_id on the wire is server-minted for matching /
  // comprehension / idiom (Phase 6.2 / 6.4 / 8.3) and absent on the
  // Phase 4.2 cloze wire (which the cloze-only api/cloze.ts module
  // still uses). The shared grade endpoint at /exercises/grade
  // accepts either — for cloze, the route derives the word_id from
  // the underlying fsrs_cards row by exercising a fallback. We
  // mirror that here.
  const exerciseIdForGrade: number = useMemo(() => {
    if (exercise.exercise_type === 'matching') {
      return exercise.exercise_id
    }
    if (exercise.exercise_type === 'comprehension') {
      return exercise.exercise_id
    }
    if (exercise.exercise_type === 'idiom') {
      return exercise.exercise_id
    }
    if (exercise.exercise_type === 'phrase_match') {
      // Phase 10.5 — phrase_match carries a server-minted
      // ``exercise_id`` on the wire (same convention as the
      // three prior non-cloze types), so we read it
      // directly. No cloze-style ``answer_word_id`` fallback
      // because the page never falls back to it (the route
      // mints the id itself in 10.3).
      return exercise.exercise_id
    }
    // cloze: prefer server-minted (Phase 5.x), else answer_word_id
    return exercise.exercise_id ?? exercise.answer_word_id
  }, [exercise])

  const handleGrade = useCallback(
    async (grade: Grade) => {
      if (isGrading) return
      setIsGrading(true)
      try {
        const response = await gradeExercise(
          exercise.exercise_type,
          exerciseIdForGrade,
          grade,
        )
        onGraded?.(response.next_due_at)
      } catch (err) {
        // Surface to the parent (page owns the toast / 401
        // bounce); the card never throws across its public
        // boundary.
        onGradeError?.(err)
      } finally {
        setIsGrading(false)
      }
    },
    [exercise, exerciseIdForGrade, isGrading, onGraded, onGradeError],
  )

  // Card title — used in the header strip and in the testid.
  // The four pages render around this card; the card itself
  // doesn't own the "Cloze · medium · v1" line of metadata (those
  // sit on ClozePage). The card carries the meta-line per type
  // via a small header inside the body branch.
  let bodyNode: React.ReactNode = null
  switch (exercise.exercise_type) {
    case 'matching':
      bodyNode = <MatchingBody exercise={exercise} />
      break
    case 'comprehension':
      bodyNode = <ComprehensionBody exercise={exercise} />
      break
    case 'idiom':
      bodyNode = <IdiomBody exercise={exercise} />
      break
    case 'cloze':
      // ClozePage renders its own full surface (sentence +
      // choices + grade) because it predates the shared card
      // and carries the empty-profile / empty-due state
      // machine. The shared card is intentionally a fallback
      // for tests / places where a full surface isn't wanted;
      // production cloze keeps using ClozePage.
      bodyNode = (
        <div className="rounded-lg border border-amber-800 bg-amber-950/30 p-4 text-sm text-amber-200">
          Cloze exercises use the dedicated{' '}
          <code className="font-mono">ClozePage</code> surface, which
          owns the empty-profile / empty-due state machine. ExerciseCard
          accepts the cloze wire for completeness but does not render
          it directly.
        </div>
      )
      break
    case 'phrase_match':
      // Phase 10.5 — phrase_match carries its own bespoke
      // body (the 4-button relation picker + the two-phrase
      // layout). The shared ExerciseCard accepts the wire for
      // exhaustiveness, but the page (``PhraseMatchPage``)
      // owns the rendering; this branch surfaces a routing
      // hint if someone mounts the card directly with a
      // phrase_match payload (current call sites go through
      // ``PhraseMatchPage`` instead).
      bodyNode = (
        <div className="rounded-lg border border-slate-800 bg-slate-900/60 p-4 text-sm text-slate-300">
          Phrase match exercises use the dedicated{' '}
          <code className="font-mono">PhraseMatchPage</code> surface,
          which owns the 4-button relation picker + the two-phrase
          layout. ExerciseCard accepts the phrase_match wire for
          exhaustiveness but does not render it directly.
        </div>
      )
      break
  }

  return (
    <div
      className="max-w-2xl mx-auto px-6 py-10 space-y-6"
      data-testid={`exercise-card-${exercise.exercise_type}`}
    >
      <div className="rounded-lg border border-slate-800 bg-slate-900 p-6 space-y-5">
        {bodyNode}
        <div className="space-y-3 pt-2 border-t border-slate-800">
          <GradeButtons onGrade={handleGrade} disabled={isGrading || disabled} />
          <p className="text-xs text-slate-500">
            1 = Again · 2 = Hard · 3 = Good · 4 = Easy. The grade bar is
            shared across all exercise types (Phase 9.5).
          </p>
        </div>
        {exercise.trace_id ? (
          <p className="text-[10px] text-slate-600 pt-1 border-t border-slate-800">
            trace: {exercise.trace_id}
          </p>
        ) : null}
      </div>
    </div>
  )
}

