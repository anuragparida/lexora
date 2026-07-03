// Phase 5.6 (card t_f9375354): the due-queue API client.
//
// Talks to the Phase 5.4 backend (card t_e8548d6d):
//   GET /exercises/due  -> 200 + ClozeExerciseOut + {due_from_fsrs: bool}
//                       | 204 No Content
//                       | 401 Unauthorized
//
// The first-login gate (5.6) only needs to know "is there at least
// one card due?" — it doesn't care which word or which sentence.
// So the shape returned to the gate is a discriminated union:
//
//   { kind: 'due',     exercise: ClozeExercise }   — server returned 200
//   { kind: 'no_cards' }                          — server returned 204
//
// The HTTP layer never throws on a 204 (it's a success). A 401
// collapses into `kind: 'no_cards'` because the gate treats it the
// same way (fall through to the profile-state branches — the
// existing behaviour). Network errors also collapse to
// `kind: 'no_cards'` for the same reason. Anything else (5xx) is
// surfaced as `kind: 'error'` so the gate can decide whether to
// log/display; for the gate's purposes both error and no_cards
// fall through to the existing logic, but exposing `error` keeps
// the surface honest.
//
// Why a discriminated union instead of throwing:
//   - The gate's spec (card body §"Scope", points 3-5) lists 204,
//     401, and network error as "fall through" — so we have to
//     return success-or-fallthrough at the call site, not throw
//     and let the gate catch.
//   - The shape stays tiny and offline-testable: a stub
//     implementation in the test file is a single return.
//
// The body of a 200 response is *not* exposed to the gate; the
// /exercises/due page (Phase 5.5 inline flow on ClozePage) is the
// one that fetches the exercise itself. The gate only branches on
// "due / not-due". We still parse and validate the wire format
// here (the Pydantic schema enforces it server-side; we type-narrow
// so TypeScript keeps the gate honest about not looking inside the
// body) — but the `exercise` payload is preserved so a future
// caller that wants to short-circuit the refetch can opt in.

import type { ClozeExercise } from './cloze'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:18700'

export type DueCheck =
  | { kind: 'due'; exercise: ClozeExercise }
  | { kind: 'no_cards' }
  | { kind: 'error'; status?: number; message: string }

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

// Phase 4.5's `ClozeExercise` interface is the source of truth for
// the wire format — the backend Pydantic model on /exercises/due
// returns the same shape plus a `due_from_fsrs: bool` (which we
// don't surface here; the gate doesn't branch on it). We accept
// the extra field by widening with `& { due_from_fsrs?: boolean }`
// so the JSON.parse result type-checks without lying about the
// surface we expose.
type DueExercise = ClozeExercise & { due_from_fsrs?: boolean }

export async function getDueCloze(): Promise<DueCheck> {
  let res: Response
  try {
    res = await fetch(`${API_URL}/exercises/due`, {
      method: 'GET',
      credentials: 'include',
    })
  } catch (err) {
    // Network-level failure (DNS, offline, CORS, abort, ...).
    // The gate treats this exactly like 204: fall through to the
    // existing profile-state branches. Surface the message for
    // future observability; the gate does not display it.
    return {
      kind: 'error',
      message: err instanceof Error ? err.message : String(err),
    }
  }

  if (res.status === 401) {
    // Auth is invalid. Same fall-through behaviour as 204 (the
    // gate spec, point 4). We could log the user out here but
    // /auth/me already does that — duplicating it would race with
    // the auth state on a transient 401. Just report.
    return { kind: 'no_cards' }
  }

  if (res.status === 204) {
    return { kind: 'no_cards' }
  }

  if (!res.ok) {
    return {
      kind: 'error',
      status: res.status,
      message: await parseError(res),
    }
  }

  // 200 + JSON body — at least one card is due.
  const body = (await res.json()) as DueExercise
  return { kind: 'due', exercise: body }
}