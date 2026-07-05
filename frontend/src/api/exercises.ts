// Phase 9.5 (card t_e4dc0404) — shared exercise types + the
// per-type API clients for matching / comprehension / idiom.
//
// Lives alongside the Phase 4.5 / 5.5 cloze client
// (``api/cloze.ts``); the four modules share ``GradeRequest`` /
// ``GradeResponse`` semantics — same wire, different modules —
// so the shared ``ExerciseCard`` can import one canonical type
// for each exercise surface and call the same ``/exercises/grade``
// endpoint regardless of which page mounted the card.
//
// Source-of-truth references:
//   - ``backend/app/match.py``       — MatchingPair + MatchingExercise
//   - ``backend/app/schemas.py``     — MatchingExerciseOut,
//                                      ComprehensionExerciseOut,
//                                      IdiomExerciseOut (6.2, 6.4, 8.3)
//   - ``backend/app/main.py``        — POST /exercises/match (6.3),
//                                      POST /exercises/comprehension (6.5),
//                                      POST /exercises/idiom (8.4)
//
// Field names mirror the wire format exactly (snake_case). The
// discriminate-on-the-wire ``exercise_type: Literal[...]`` field
// is preserved on each response interface so the ``ExerciseCard``
// can `switch` on it without re-deriving the heuristic from the
// presence of a particular optional field.
//
// Phase 5.3 / 5.4 / 6.6 — the ``gradeCloze`` / ``gradeMatch`` /
// ``gradeComprehension`` / ``gradeIdiom`` thin wrappers below all
// POST to the SAME endpoint (``/exercises/grade``) with the same
// body shape (``{exercise_id, exercise_type, grade}``) — only the
// literal ``exercise_type`` differs. We deliberately do NOT merge
// the four cloze/matching/comprehension/idiom clients into one
// mega-module (would be harder to grep), but each function calls
// into the shared ``gradeExercise(exercise_type, exercise_id,
// grade)`` helper so the wire shape is the single source of truth.

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:18700'

// --- shared wire shapes ----------------------------------------------------

// Phase 5.3 (locked) — the FSRS grade scale, identical on the four
// per-type routes. Phase 6.6's ``ExerciseType`` dispatch widens this
// to ``Literal["cloze", "matching", "comprehension", "idiom"]``.
export type Grade = 1 | 2 | 3 | 4

// Phase 6.6 — the wire-level discriminator that the backend
// (``backend/app/main.py`` ``match payload.exercise_type:``) reads
// to route to ``_grade_cloze`` / ``_grade_match`` / etc. Mirrors
// ``backend/app/schemas.py::ExerciseType = Literal["cloze",
// "matching", "comprehension", "idiom"]``. Phase 8.3 added
// ``"idiom"``.
export type ExerciseType = 'cloze' | 'matching' | 'comprehension' | 'idiom'

// Phase 5.3 — request body for POST /exercises/grade, shared by
// every per-type call (clo­ze / matching / comprehension / idiom).
// Field names match the Pydantic ``GradeRequest`` model exactly.
export interface GradeRequest {
  exercise_id: number
  exercise_type: ExerciseType
  grade: Grade
}

// Phase 5.3 — response body for POST /exercises/grade. Mirrors
// ``backend/app/schemas.py::GradeResponse``.
export interface GradeResponse {
  graded: true
  exercise_id: number
  exercise_type: ExerciseType
  next_due_at: string // ISO-8601 UTC
  card_state: number // 1=Learning, 2=Review, 3=Relearning
  stability: number
  difficulty: number
  trace_id: string | null
}

// --- error handling --------------------------------------------------------

interface ApiErrorBody {
  detail?: string | Array<{ msg?: string; loc?: unknown }>
}

// Typed error that carries the HTTP status code alongside the
// user-visible message. Mirrors ``ClozeApiError`` in
// ``api/cloze.ts`` so the shared ``ExerciseCard`` can
// discriminate 422 / 401 / 500 with the same pattern. We extend
// the existing ``ClozeApiError`` shape (Phase 5.5's typed-error
// pattern) and re-export it under a neutral name so the new
// pages don't need to import from the cloze module.
export class ExerciseApiError extends Error {
  readonly status: number
  constructor(status: number, message: string) {
    super(message)
    this.name = 'ExerciseApiError'
    this.status = status
  }
}

async function toApiError(res: Response): Promise<ExerciseApiError> {
  let detail: string | undefined
  try {
    const body = (await res.json()) as ApiErrorBody
    if (typeof body.detail === 'string') {
      detail = body.detail
    } else if (Array.isArray(body.detail) && body.detail.length > 0) {
      const first = body.detail[0]
      if (first && typeof first.msg === 'string') {
        detail = first.msg
      }
    }
  } catch {
    // body wasn't JSON; fall through to the generic message.
  }
  const message = detail ?? `Request failed (${res.status})`
  return new ExerciseApiError(res.status, message)
}

// --- shared base fields (Phase 6.1 mixin) --------------------------------

// Phase 6.1 — the ``BaseExerciseFields`` mixin carries
// exercise_type (a closed literal), target_word_id (the canonical
// cross-exercise-type FK), prompt_template_version (A/B key),
// enable_rag (echoed), trace_id (Langfuse span id), and
// latency_ms. Each per-type response below extends this.
export interface BaseExerciseFields {
  exercise_type: ExerciseType
  target_word_id: number
  prompt_template_version: string
  enable_rag: boolean
  trace_id: string | null
  latency_ms: number
}

// --- matching (Phase 6.2 / 6.3) -------------------------------------------

// ``MatchingPair`` mirrors ``backend/app/match.py::MatchingPair``:
// three fields — the left + right word_ids and the right_kind
// (translation / synonym). Pydantic enforces the literal at the
// route layer; we mirror it here as a string-literal union so a
// future widening to a 3rd ``right_kind`` is a single edit + a
// coordinated type narrowing.
export type MatchingRightKind = 'translation' | 'synonym'
export interface MatchingPair {
  left_word_id: number
  right_word_id: number
  right_kind: MatchingRightKind
}

// Mirrors ``backend/app/schemas.py::MatchingExerciseOut``. The
// ``pairs`` array is bounded in ``[2, 8]`` server-side; the
// array shape here is open because the wire carries between 2
// and 8 inclusive. ``partner_translation`` is Phase 7.4's
// bilingual read-through (the cloze / matching fields are
// sibling). ``exercise_id`` is server-minted per generation so
// the grade_logs row is deterministic for Ragas join (Phase 6.7).
export interface MatchingExercise extends BaseExerciseFields {
  exercise_type: 'matching'
  exercise_id: number
  pairs: MatchingPair[]
  partner_translation: string | null
}

// --- comprehension (Phase 6.4 / 6.5) -------------------------------------

// The four MC choices the comprehension generator emits. Pydantic
// requires all four keys (A/B/C/D); the closed Literal is the
// type-level guardrail that prevents a future maintainer from
// shipping a 5th option. Mirrors
// ``backend/app/schemas.py::ComprehensionChoice = Literal["A",
// "B", "C", "D"]``.
export type ComprehensionChoiceKey = 'A' | 'B' | 'C' | 'D'

// Mirrors ``backend/app/schemas.py::ComprehensionExerciseOut``:
// passage (3-5 sentences, 20..600 chars), question (5..300 chars),
// choices (dict keyed A/B/C/D, each 1..200 chars), correct_choice
// (the answer key — NOT an index), rationale (1..400 chars).
export interface ComprehensionExercise extends BaseExerciseFields {
  exercise_type: 'comprehension'
  exercise_id: number
  passage: string
  question: string
  choices: Record<ComprehensionChoiceKey, string>
  correct_choice: ComprehensionChoiceKey
  rationale: string
}

// --- idiom (Phase 8.3 / 8.4) ----------------------------------------------

// Closed frequency-band literal; the wire-level guardrail that
// keeps the cloze-within-idiom variant from drifting to a 4th
// band. Mirrors
// ``backend/app/schemas.py::IdiomFrequencyBand``.
export type IdiomFrequencyBand = 'high' | 'mid' | 'low'

// Mirrors ``backend/app/schemas.py::IdiomExerciseOut``. The
// ``word_id`` echo is the request-side name (see IdiomGenerateRequest
// below); ``target_word_id`` is the cross-exercise-type canonical
// name from Phase 6.1's mixin. Both fields are kept for
// forward-compatibility — Phase 9's study-session mixer reads
// the canonical name, while the Phase 8.4 test suite reads the
// request-shape name.
export interface IdiomExercise extends BaseExerciseFields {
  exercise_type: 'idiom'
  exercise_id: number
  word_id: number
  phrase: string
  definition: string
  example_usage: string
  source_attribution: string
  attested_quote: string | null
  attested_source: string | null
  frequency_band: IdiomFrequencyBand
  cloze_target: string | null
}

// --- cloze (re-exported for the shared ExerciseCard) -----------------------

// Mirrors ``api/cloze.ts::ClozeExercise`` — re-declared here so the
// shared ``ExerciseCard`` can switch on ``exercise.exercise_type``
// without importing the cloze-specific module directly. ClozePage
// keeps using its own cloze-only client for the empty-state
// UX (``getDueCloze``); the shared card only needs the response
// shape for the render.
//
// Phase 6.6 plan §"Hard rules" #1 — the ``exercise_type`` field
// is a closed literal on both the wire and in our view, so the
// ``switch`` in ``ExerciseCard`` covers all four branches with
// no fall-through case (TypeScript's exhaustiveness check makes
// a missing branch a compile error).
export interface ClozeExerciseForCard extends BaseExerciseFields {
  exercise_type: 'cloze'
  exercise_id?: number // Phase 4.2 wire omits it; 5.x adds it
  sentence_with_blank: string
  answer_word_id: number
  distractors: number[]
  difficulty: 'easy' | 'medium' | 'hard'
  rationale: string
}

// --- discriminated union used by ExerciseCard -----------------------------

export type AnyExercise =
  | ClozeExerciseForCard
  | MatchingExercise
  | ComprehensionExercise
  | IdiomExercise

// --- per-type request shapes ---------------------------------------------

// Phase 6.2 — ``MatchGenerateRequest``. Empty body parses to the
// defaults (count=4, enable_rag=False, partner_lang="de") — the
// route is thin and the server picks the target word.
export interface MatchGenerateRequest {
  count?: number
  enable_rag?: boolean
  partner_lang?: 'de' | 'en'
}

// Phase 6.4 — ``ComprehensionGenerateRequest``. The no-knob
// shape; RAG-on is opt-in.
export interface ComprehensionGenerateRequest {
  enable_rag?: boolean
}

// Phase 8.4 — ``IdiomGenerateRequest``. ``word_id`` is REQUIRED
// (Phase 8.3 shipped without it; 8.4 made it required so the
// generator's ``phrases WHERE word_id == :word_id`` filter is
// anchored). Empty body fails with HTTP 422.
export interface IdiomGenerateRequest {
  word_id: number
  enable_rag?: boolean
}

// --- per-type API clients (Phase 9.5) ------------------------------------

// POST /exercises/match — Phase 6.3. Cookie auth (Phase 2);
// the route picks the target word.
export async function generateMatch(
  body: MatchGenerateRequest = {},
): Promise<MatchingExercise> {
  const res = await fetch(`${API_URL}/exercises/match`, {
    method: 'POST',
    credentials: 'include',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    throw await toApiError(res)
  }
  return (await res.json()) as MatchingExercise
}

// POST /exercises/comprehension — Phase 6.5.
export async function generateComprehension(
  body: ComprehensionGenerateRequest = {},
): Promise<ComprehensionExercise> {
  const res = await fetch(`${API_URL}/exercises/comprehension`, {
    method: 'POST',
    credentials: 'include',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    throw await toApiError(res)
  }
  return (await res.json()) as ComprehensionExercise
}

// POST /exercises/idiom — Phase 8.4. ``word_id`` is required
// (will 422 on empty body).
export async function generateIdiom(
  body: IdiomGenerateRequest,
): Promise<IdiomExercise> {
  const res = await fetch(`${API_URL}/exercises/idiom`, {
    method: 'POST',
    credentials: 'include',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    throw await toApiError(res)
  }
  return (await res.json()) as IdiomExercise
}

// --- shared grader (Phase 5.3 / 6.6) --------------------------------------

// Phase 5.3 + 6.6 — POST /exercises/grade. The single endpoint
// dispatches on ``exercise_type`` server-side, so the wire
// carries the discriminator explicitly. The shared function
// below is the per-type callers' canonical entry point — the
// four ``gradeCloze`` / ``gradeMatch`` / ``gradeComprehension``
// / ``gradeIdiom`` wrappers in ``api/cloze.ts`` are kept for
// backward compatibility (Phase 5.5 tests) and forward here.
//
// ``exercise_id`` field rules:
//   - cloze (Phase 4.2 wire): the FSRS card backs the cloze, so
//     the route derives word_id from the card row; 5.3 reads
//     ``answer_word_id`` as ``exercise_id``.
//   - matching (Phase 6.2): the server-minted exercise_id is
//     the discriminator on the wire; the route derives word_id
//     from the exercise row.
//   - comprehension (Phase 6.4): same — server-minted
//     exercise_id.
//   - idiom (Phase 8.3): same — server-minted exercise_id.
//
// All four pass through the same ``_grade_one`` path on the
// server; the only per-type bit is the literal discriminator.
export async function gradeExercise(
  exercise_type: ExerciseType,
  exercise_id: number,
  grade: Grade,
): Promise<GradeResponse> {
  const body: GradeRequest = { exercise_type, exercise_id, grade }
  const res = await fetch(`${API_URL}/exercises/grade`, {
    method: 'POST',
    credentials: 'include',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    throw await toApiError(res)
  }
  return (await res.json()) as GradeResponse
}

// Phase 9.5 convenience wrappers — these keep the four
// per-type page components tidy (one line per type) while
// sharing the same wire shape. They are NOT a fan-out: they
// each call ``gradeExercise`` with a different literal so the
// server's match statement dispatches to the right handler.
export function gradeMatch(exercise_id: number, grade: Grade) {
  return gradeExercise('matching', exercise_id, grade)
}
export function gradeComprehension(exercise_id: number, grade: Grade) {
  return gradeExercise('comprehension', exercise_id, grade)
}
export function gradeIdiom(exercise_id: number, grade: Grade) {
  return gradeExercise('idiom', exercise_id, grade)
}
