// Phase 4.5 (card t_4a9f172e): cloze exercise API client.
// Phase 5.5 (card t_f253456b): extends with gradeCloze + getDueCloze.
//
// Talks to the Phase 4.2 backend (card t_bdd9ffbe):
//   POST /exercises/cloze   (auth-gated, cookie-based) -> ClozeExercise
// Phase 5.3 (card t_5160eecf) wire shape (locked in this module):
//   POST /exercises/grade   (auth-gated, cookie-based) -> GradeResponse
// Phase 5.4 (card t_e8548d6d) wire shape:
//   GET  /exercises/due     (auth-gated, cookie-based) -> ClozeExerciseOut | 204
//
// Mirrors the API-client shape used in `weakness.ts` and
// `diagnostic.ts`: response interfaces mirror the backend Pydantic
// models exactly (see `backend/app/cloze.py`, `backend/app/schemas.py`),
// the network helpers use `credentials: 'include'` (Phase 2 cookie
// auth), and a `parseError` helper surfaces the server's `detail`
// string.
//
// We intentionally keep the type next to the API function (option
// 1 from the card body) — splitting into `types/cloze.ts` would
// be over-engineering for the small endpoint set.

// The response shape mirrors the `ClozeExercise` Pydantic model in
// `backend/app/cloze.py` (4.2's deliverable). Field names match
// the wire format exactly (snake_case).
export interface ClozeExercise {
  // German sentence with `___` marking the cloze position. The
  // backend Pydantic model says: "The LLM must not mutate the
  // answer word's case, article, or surrounding word forms." The
  // frontend renders this verbatim with the blank replaced by a
  // styled inline element.
  sentence_with_blank: string
  // FK to words.id of the correct answer. We do NOT fetch the
  // word's German string from this; the backend's `distractors`
  // carries word_ids too, and the lookup happens in 4.5's UI via
  // a future `words` endpoint (out of scope). For Phase 4 the
  // button labels fall back to the `word_id` itself when no
  // resolved string is available.
  answer_word_id: number
  // Exactly 3 FKs to words.id of plausible wrong answers. Same
  // word_type as answer_word_id. Pydantic enforces min_length=3
  // max_length=3 on the server; the assertion is repeated here
  // as a type-level guardrail.
  distractors: [number, number, number]
  // Self-rated difficulty. Pydantic Literal["easy", "medium",
  // "hard"] — same string union on both ends.
  difficulty: 'easy' | 'medium' | 'hard'
  // One-sentence explanation of the cloze design. Pydantic
  // enforces min_length=1 max_length=400.
  rationale: string
  // Bumped when the backend prompt template changes. Enables A/B
  // eval in Phase 5. Module-level constant in `app/cloze.py`:
  // `PROMPT_TEMPLATE_VERSION = "cloze-v1"`.
  prompt_template_version: string
}

// `ClozeExerciseOut` is the wire shape for `GET /exercises/due`.
// It carries the same `ClozeExercise` fields PLUS the Phase 5.4
// metadata: the `word_id` (so the frontend can reconcile the
// response against the previously-shown cloze) and `due_from_fsrs`
// (so the frontend can render "fresh word" vs "scheduled review"
// differently if it wants). Phase 5.5 treats both flags as
// observable side-channels — it does not gate UX on them.
export interface ClozeExerciseOut extends ClozeExercise {
  word_id: number
  due_from_fsrs: boolean
}

// The FSRS grade scale. Hard-coded to the four-button set the
// Phase 5 spec defines (Again/Hard/Good/Easy). Mirrors the
// backend's `Literal[1, 2, 3, 4]` guardrail on GradeRequest.grade
// — the wire rejects out-of-range values with 422 before we get
// here, but the literal type keeps the call sites honest.
export type Grade = 1 | 2 | 3 | 4

// Phase 5 is cloze-only (Hard rule #2 of Phase 5: "Cloze-only
// grading in Phase 5"). The literal type, not `string`, is the
// type-level guardrail that prevents a future maintainer from
// silently widening this enum.
export type ExerciseType = 'cloze'

// Request body for `POST /exercises/grade`. Mirrors the backend
// Pydantic `GradeRequest` schema in `backend/app/schemas.py`.
// Field names match the wire format exactly (snake_case).
export interface GradeRequest {
  exercise_id: number
  exercise_type: ExerciseType
  grade: Grade
}

// Response shape for `POST /exercises/grade`. Mirrors the
// backend Pydantic `GradeResponse` schema. We carry the next
// review timestamp and the post-review card state so the
// frontend can render a sonner toast with "Next review in Xm"
// without a second round-trip.
export interface GradeResponse {
  graded: true
  exercise_id: number
  exercise_type: ExerciseType
  next_due_at: string // ISO-8601 UTC; Date on the server side.
  card_state: number // 1=Learning, 2=Review, 3=Relearning
  stability: number
  difficulty: number
  trace_id: string | null
}

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:18700'

interface ApiErrorBody {
  detail?: string | Array<{ msg?: string; loc?: unknown }>
}

// A typed error that carries the HTTP status code alongside
// the user-visible message. The page-level catch handler uses
// `status` to discriminate 422 (Pydantic validation — surface
// the detail verbatim) from 500 (server fault — generic copy)
// from 401 (auth bounce — redirect to /login).
//
// We extend `Error` so `instanceof Error` checks keep working
// in any future generic error-handling code that doesn't know
// about ClozeApiError specifically. The `status` field is the
// discrimination handle; the `message` field is the same string
// the old `parseError()` returned, so anything that read
// `err.message` still gets a useful value.
export class ClozeApiError extends Error {
  readonly status: number
  constructor(status: number, message: string) {
    super(message)
    this.name = 'ClozeApiError'
    this.status = status
  }
}

// Build a ClozeApiError from a non-ok Response. The message is
// the parsed `detail` field when the body is JSON, falling back
// to `Request failed (N)` when the body is missing / not JSON.
// We always include the status code on the message too — some
// callers prefer string-matching over `instanceof`, and the
// string form is harmless to have alongside the typed status.
async function toApiError(res: Response): Promise<ClozeApiError> {
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
  return new ClozeApiError(res.status, message)
}

// Cookie-based auth (Phase 2): the httpOnly `lexora_token` cookie
// travels via `credentials: 'include'`. No Authorization header.
// The body is empty `{}` because word selection is server-driven
// (deterministic from the user's weakness profile — see 4.2's
// `select_target_word`).
export async function generateCloze(): Promise<ClozeExercise> {
  const res = await fetch(`${API_URL}/exercises/cloze`, {
    method: 'POST',
    credentials: 'include',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({}),
  })
  if (!res.ok) {
    throw await toApiError(res)
  }
  return (await res.json()) as ClozeExercise
}

// Phase 5.5 (card t_f253456b): grade a cloze.
//
// POST /exercises/grade with the literal-typed body. The backend
// Pydantic schema enforces:
//   - exercise_type: Literal["cloze"]   → 422 on anything else
//   - grade: Literal[1, 2, 3, 4]        → 422 on anything else
//   - exercise_id: positive int         → 422 on zero/negative
//
// We send the literal type as a string ("cloze") and a number
// grade so the wire matches what the Pydantic validator expects.
// The TypeScript `as const` on the literal would not survive JSON
// serialisation (numbers come through as numbers, strings as
// strings), so we explicitly cast at the call site to keep the
// type-level guardrail visible.
export async function gradeCloze(
  exercise_id: number,
  grade: Grade,
): Promise<GradeResponse> {
  const body: GradeRequest = {
    exercise_id,
    exercise_type: 'cloze',
    grade,
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

// Phase 5.5 (card t_f253456b): fetch the next due cloze.
//
// GET /exercises/due. Returns `null` on 204 (no cards due right
// now — the "all caught up" honest empty state). On 200, returns
// a `ClozeExerciseOut` carrying the FSRS metadata the inline
// grade-then-next flow needs (`word_id` to reconcile state,
// `due_from_fsrs` to distinguish fresh picks from scheduled
// reviews — the latter is observable side-channel only, the
// 5.5 UI does not branch on it).
//
// We deliberately do NOT throw on 204 — 204 is a valid response,
// not an error. The empty state is honest ("All caught up")
// and lives in the page's render branch, not in the catch path.
export async function getDueCloze(): Promise<ClozeExerciseOut | null> {
  const res = await fetch(`${API_URL}/exercises/due`, {
    method: 'GET',
    credentials: 'include',
    headers: { Accept: 'application/json' },
  })
  if (res.status === 204) {
    return null
  }
  if (!res.ok) {
    throw await toApiError(res)
  }
  return (await res.json()) as ClozeExerciseOut
}