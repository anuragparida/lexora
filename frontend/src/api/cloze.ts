// Phase 4.5 (card t_4a9f172e): cloze exercise API client.
//
// Talks to the Phase 4.2 backend (card t_bdd9ffbe):
//   POST /exercises/cloze (auth-gated, cookie-based) -> ClozeExercise
//
// Mirrors the API-client shape used in `weakness.ts` and
// `diagnostic.ts`: a `ClozeExercise` interface that exactly matches
// the backend Pydantic model (see `backend/app/cloze.py`), a
// `generateCloze()` function that fetches with
// `credentials: 'include'` (Phase 2 cookie auth), and a small
// `parseError` helper that surfaces the server's `detail` string.
//
// The Phase 4.2 backend is the source of truth for the wire format.
// Phase 4.5's job is to render whatever the server sends, not to
// re-validate the shape — the backend's Pydantic model does that.
// The TypeScript types here are a thin wire-format mirror.
//
// We intentionally keep the type next to the API function (option
// 1 from the card body) — splitting into `types/cloze.ts` would
// be over-engineering for a single endpoint with one response
// shape. If Phase 5/6 add matching + comprehension endpoints, the
// types can split then.

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

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:18700'

interface ApiErrorBody {
  detail?: string
}

async function parseError(res: Response): Promise<string> {
  try {
    const body = (await res.json()) as ApiErrorBody
    if (typeof body.detail === 'string') return body.detail
  } catch {
    // body wasn't JSON; fall through
  }
  return `Request failed (${res.status})`
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
    throw new Error(await parseError(res))
  }
  return (await res.json()) as ClozeExercise
}
