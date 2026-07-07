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
//
// Phase 10.5 (card t_ca1d2da8) — additive widening to 5. The
// prior 4 stay narrow-compatible (no narrowing); ``"phrase_match"``
// is the 5th wire literal for ``POST /exercises/phrase_match``
// (Phase 10.3, ``t_13bb48d2``) and the 5th route through
// ``_grade_one`` (``backend/app/main.py`` ``@app.post
// "/exercises/grade"``).
export type ExerciseType =
  | 'cloze'
  | 'matching'
  | 'comprehension'
  | 'idiom'
  | 'phrase_match'

// Phase 10.5 — the 4-way relation literal that Phase 10.1's
// ``phrase_pairs.relation`` column encodes and Phase 10.3's
// ``/exercises/phrase_match`` response echoes on the wire.
// Mirrors ``backend/app/schemas.py::PhrasePairRelation``
// (added by 10.1, ``t_18c90a68``). Closed-literal: any other
// string is a wire validation error on both ends.
export type PhrasePairRelation =
  | 'equivalent'
  | 'paraphrase'
  | 'related'
  | 'unrelated'

// Phase 5.3 — request body for POST /exercises/grade, shared by
// every per-type call (clo­ze / matching / comprehension / idiom /
// phrase_match). Field names match the Pydantic ``GradeRequest``
// model exactly.
//
// Phase 10.5 — ``answer`` is an optional extension field used
// by ``phrase_match`` to ship the learner's relation choice
// alongside the FSRS grade in a single round trip (Phase 9.6's
// "answer + grade in one request" discipline). Today's Pydantic
// ``GradeRequest`` uses Pydantic v2's default
// ``extra="ignore"`` so an unknown ``answer`` field is silently
// dropped; when Phase 10.3 widens the backend Pydantic model
// the frontend is already correctly shaped. The other 4 types
// leave ``answer`` unset (Pydantic excludes ``undefined`` from
// the JSON body).
export interface GradeRequest {
  exercise_id: number
  exercise_type: ExerciseType
  grade: Grade
  /**
   * Optional per-type answer payload. Phase 10.5 sets this to
   * the PhrasePairRelation literal for ``exercise_type ===
   * "phrase_match"``; the other 4 exercise types leave it
   * unset. The wire layer accepts any JSON-serialisable value
   * here today (Pydantic's ``extra="ignore"`` drops it), and
   * Phase 10.3's backend widening will tighten the type when
   * it lands.
   */
  answer?: unknown
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

// --- phrase_match (Phase 10.3 / 10.5) -------------------------------------

// Phase 10.5 (card t_ca1d2da8) — outbound shape for
// ``POST /exercises/phrase_match``. The two phrases are the
// surfaced pair; ``relation_rationale`` is the single
// learner-facing hint the page hover-reveals (Phase 8 idiom
// pattern, applied per-pair). ``relation`` is the closed
// 4-way literal the LLM picked (the "ground truth" relation
// the learner is being asked to estimate); we surface it on
// the wire but the page does NOT show it during the answer
// step — only after the grade is recorded (Phase 9.6
// discipline).
//
// Pydantic ``PhraseMatchExerciseOut`` is added in 10.3;
// Phase 10.5 is the frontend's mirror, kept structurally
// identical to the wire (snake_case). ``target_word_id`` is
// the canonical Phase 6.1 cross-exercise-type name; the
// request-side ``word_id`` echoes as ``word_id`` on the
// response too (Phase 10.2 widens phrase-match to match the
// cloze / matching / comprehension / idiom wire shapes, both
// fields are kept for forward compatibility). ``source_attribution``
// mirrors the 8.3 idiom field (closed comma-joined literal).
export interface PhraseMatchExercise extends BaseExerciseFields {
  exercise_type: 'phrase_match'
  exercise_id: number
  phrase_a: string
  phrase_b: string
  // Request-side pair-selector seed echoed from
  // ``phrase_a_id``/``phrase_b_id`` resolution (NOT a
  // ``words.id`` FK for phrase-match — see
  // ``app.phrase_match.select_phrase_pair`` docstring).
  word_id: number
  // Closed 4-way relation literal the LLM picked — the
  // "ground truth" relation. The page does NOT show this
  // during the answer step; the relation the learner picks
  // is what the grade call submits (the answer), and the
  // server is the sole source of truth for the canonical
  // relation.
  relation: PhrasePairRelation
  // 1..400 chars — learner-facing explanation of why the
  // server picked ``relation``. The page uses this as the
  // hover-revealed hint (mirrors the Phase 8 idiom
  // ``definition`` pattern).
  relation_rationale: string
  // Comma-joined subset of
  // ``Literal['dwds','goethe','schiller','bge-m3-cosine']``
  // (same shape as 8.3 ``IdiomExerciseOut.source_attribution``).
  // Surfaced only in the trace footer; not used in the answer
  // step.
  source_attribution: string
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

// Phase 10.5 (card t_ca1d2da8) — additive widening to 5 for
// the discriminated union. Phase 9.5's switch in
// ``ExerciseCard`` doesn't render ``phrase_match`` (the
// bespoke 4-button relation picker is the page's job; the
// shared card's render doesn't know about the relation
// literal). The union widening here keeps the type system
// honest for callers that DO want to consume phrase_match.
export type AnyExercise =
  | ClozeExerciseForCard
  | MatchingExercise
  | ComprehensionExercise
  | IdiomExercise
  | PhraseMatchExercise

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

// Phase 10.5 (card t_ca1d2da8) — ``PhraseMatchGenerateRequest``.
// ``word_id`` mirrors the Phase 8.4 idiom discipline (required
// — the curated ``phrase_pairs`` table is per-word: every pair
// row carries a ``phrase_a_id`` / ``phrase_b_id`` whose phrases
// anchor to a specific ``words.id`` via the
// ``phrases.word_id`` FK from Phase 8.1). Empty body fails
// with HTTP 422. ``enable_rag`` is the opt-in nearest-neighbor
// flag for the bge-m3 retrieve step in the Phase 10.3
// generator; defaults to ``false`` so the wire shape is
// reproducible for the offline A/B eval (Phase 9.4 + 9.7
// discipline).
export interface PhraseMatchGenerateRequest {
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

// --- phrase_match (Phase 10.3 / 10.5) -------------------------------------

// POST /exercises/phrase_match — Phase 10.3 (``t_13bb48d2``).
// ``word_id`` is required (mirrors Phase 8.4's idiom
// discipline). The route mints a fresh ``exercise_id`` per
// generation (same convention as the four prior per-type
// routes — Phase 5.3 / 6.x / 8.3). When no pair row exists
// for the supplied ``word_id`` the server returns 404 and the
// page surfaces ``status === 'notFound'`` (same discipline as
// ``IdiomNotFoundError`` handled in ``IdiomPage``).
export async function generatePhraseMatch(
  body: PhraseMatchGenerateRequest,
): Promise<PhraseMatchExercise> {
  const res = await fetch(`${API_URL}/exercises/phrase_match`, {
    method: 'POST',
    credentials: 'include',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    throw await toApiError(res)
  }
  return (await res.json()) as PhraseMatchExercise
}

// POST /exercises/grade — Phase 10.5 reads it with the 5th
// literal and the relation choice on the same call (Phase 9.6
// discipline: the answer + grade are stored in distinct
// fields on the grade call). Today's Pydantic ``GradeRequest``
// has ``exercise_id`` + ``exercise_type`` + ``grade`` and
// uses Pydantic v2's default ``extra="ignore"`` so the
// additional ``answer`` field is silently dropped; the
// backend's GradeRequest widening (10.3) will tighten the
// type when it lands.
//
// We deliberately use a typed ``body`` here (``{ exercise_id,
// exercise_type, grade, answer }``) instead of inlining a
// generic ``gradeExercise('phrase_match', ...)`` call because
// the relation literal is required and a typed call site
// guards against accidentally sending a typo relation.
export async function submitPhraseMatchGrade(
  exercise_id: number,
  relation: PhrasePairRelation,
  grade: Grade,
): Promise<GradeResponse> {
  const body: GradeRequest = {
    exercise_type: 'phrase_match',
    exercise_id,
    grade,
    answer: relation,
  }
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
