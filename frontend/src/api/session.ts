// Phase 9.6 (card t_f1c63bfc) — the session-mixer API client.
//
// The Phase 9.6 ``SessionPage`` mixer is the union consumer: it
// hits ``GET /exercises/due`` with no type filter (which now
// returns the union across cloze / matching / comprehension /
// idiom per Phase 9.2), advances the local queue on each grade,
// and uses ``/auth/me.due_by_type`` for the first-login gate's
// "is there anything to study?" branch.
//
// Why this lives in its own module rather than alongside
// ``api/due.ts`` (Phase 5.6) or ``api/cloze.ts`` (Phase 4.5):
//
//   - ``api/due.ts`` is the cloze-only gate client that returns
//     a discriminated union ``{kind: 'due' | 'no_cards' |
//     'error'}``. The session mixer needs the *headers* on the
//     204 response (``X-Due-Exercise-Type`` / ``-Card-Id`` /
//     ``-Word-Id``) and the raw body for cloze picks, which
//     would force the existing ``DueCheck`` shape to grow a
//     fifth variant. A separate module is cleaner.
//
//   - ``api/exercises.ts`` is the per-type generator/grade
//     surface (Phase 9.5). The session mixer *grades* via the
//     same shared endpoint (``/exercises/grade``), so it imports
//     ``gradeExercise`` from there. We deliberately don't re-
//     declare the grader here — the wire shape stays single-
//     sourced.
//
// Wire contracts (mirrors ``backend/app/main.py``):
//
//   GET /exercises/due (no filter)
//     200 + body                          -> cloze pick
//                                            (Phase 4.2 / 5.4 wire;
//                                             typed as ClozeOut below)
//     204 + X-Due-Exercise-Type header     -> matching/comprehension/
//                                            idiom pick; the headers
//                                            carry the dispatch info
//                                            the mixer reads
//     204 (no headers)                    -> nothing due
//     401                                 -> unauthenticated; the
//                                            mixer treats it like
//                                            "nothing due" rather
//                                            than throwing
//
//   GET /auth/me.due_by_type
//     Phase 9.2 always returns a 4-key dict (cloze / matching /
//     comprehension / idiom). The frontend gate sums the values
//     to decide whether the session mixer has anything to do.

import type { ClozeExerciseOut } from './cloze'
import type { Grade, GradeResponse } from './exercises'
import { gradeExercise } from './exercises'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:18700'

// --- shared helpers ------------------------------------------------------

interface ApiErrorBody {
  detail?: string | Array<{ msg?: string; loc?: unknown }>
}

// Mirrors the typed-error pattern from ``api/cloze.ts`` and
// ``api/exercises.ts`` — never throws on 204 (a valid "nothing
// due" response); throws on 5xx with a structured error so the
// page can render the alert + Retry CTA.
export class SessionApiError extends Error {
  readonly status: number
  constructor(status: number, message: string) {
    super(message)
    this.name = 'SessionApiError'
    this.status = status
  }
}

async function toApiError(res: Response): Promise<SessionApiError> {
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
    // body wasn't JSON; fall through.
  }
  const message = detail ?? `Request failed (${res.status})`
  return new SessionApiError(res.status, message)
}

// --- /auth/me.due_by_type shape -----------------------------------------

// Phase 9.2 (``backend/app/schemas.py::MeOut.due_by_type``) —
// always-present 4-key dict. The shape is closed at the wire so
// the gate / mixer can ``Object.values(due_by_type).reduce``
// without null-checking the dict.
export type DueByType = {
  cloze: number
  matching: number
  comprehension: number
  idiom: number
}

// Phase 9.2 — extended ``MePayload`` shape. Kept here (rather
// than in ``auth.ts``) because the session mixer is the only
// consumer today; if a future page reads ``due_by_type``
// independently, hoist this into ``auth.ts``.
export interface MePayloadWithDue {
  due_by_type: DueByType
}

// --- /exercises/due union shape -----------------------------------------

// The 4 exercise types the union can pick from. Mirrors the
// ``ExerciseType`` literal in ``api/exercises.ts``; we re-declare
// it as a tighter subset because the mixer's grade-and-advance
// path only ever handles these 4.
export type SessionExerciseType = 'cloze' | 'matching' | 'comprehension' | 'idiom'

// The mixed pick the session mixer dequeues. Cloze picks carry
// the body inline; non-cloze picks carry the (type, card_id,
// word_id) tuple the mixer uses to call the per-type generator
// endpoint. The discriminator is `kind` so the mixer can switch
// exhaustively without TypeScript falling back to `any`.
export type DuePick =
  | {
      kind: 'cloze'
      exercise: ClozeExerciseOut
    }
  | {
      kind: 'matching' | 'comprehension' | 'idiom'
      card_id: number
      word_id: number
    }

export type DueQueueResult =
  | { kind: 'pick'; pick: DuePick }
  | { kind: 'empty' }
  | { kind: 'unauthenticated' }
  | { kind: 'error'; status: number; message: string }

// Phase 9.6 — fetch the next pick from the due-queue union.
//
// We do NOT auto-fetch by ``word_id`` for non-cloze types — the
// matching / comprehension / idiom generators don't take a
// ``force_word_id`` knob yet (per the Phase 9.2 route docstring:
// "non-cloze inline generation is deferred to the per-type
// endpoints"). The mixer uses the (card_id, word_id) tuple to
// call the per-type endpoints itself.
//
// Returns a discriminated union — never throws on 204 or 401
// because both are valid "nothing to render here" surfaces;
// throws on 5xx via ``SessionApiError``.
export async function getNextDuePick(): Promise<DueQueueResult> {
  let res: Response
  try {
    res = await fetch(`${API_URL}/exercises/due`, {
      method: 'GET',
      credentials: 'include',
    })
  } catch (err) {
    return {
      kind: 'error',
      status: 0,
      message: err instanceof Error ? err.message : String(err),
    }
  }

  if (res.status === 401) {
    return { kind: 'unauthenticated' }
  }

  if (res.status === 204) {
    // Distinguish "no due cards at all" from "non-cloze pick —
    // look at the X-Due-Exercise-Type header to fetch the
    // matching per-type body".
    const headerType = res.headers.get('X-Due-Exercise-Type')
    const headerCardId = res.headers.get('X-Due-Card-Id')
    const headerWordId = res.headers.get('X-Due-Word-Id')
    if (
      headerType !== null &&
      headerCardId !== null &&
      headerWordId !== null
    ) {
      // The Phase 9.2 route only emits these headers for the
      // union branch where ``type=any`` (the default). Validate
      // the type is in the closed union before trusting it.
      if (
        headerType === 'matching' ||
        headerType === 'comprehension' ||
        headerType === 'idiom'
      ) {
        const card_id = Number(headerCardId)
        const word_id = Number(headerWordId)
        if (Number.isFinite(card_id) && Number.isFinite(word_id)) {
          return {
            kind: 'pick',
            pick: { kind: headerType, card_id, word_id },
          }
        }
      }
      // Header was malformed (unparseable numbers or unknown
      // type) — surface as an error so the mixer can show a
      // retry CTA rather than silently rendering nothing.
      return {
        kind: 'error',
        status: 204,
        message: `Malformed X-Due-Exercise-Type headers (type=${headerType})`,
      }
    }
    return { kind: 'empty' }
  }

  if (!res.ok) {
    try {
      throw await toApiError(res)
    } catch (err) {
      return {
        kind: 'error',
        status:
          err instanceof SessionApiError ? err.status : res.status,
        message:
          err instanceof Error ? err.message : 'Request failed',
      }
    }
  }

  // 200 + JSON body — Phase 9.2 returns the cloze pick inline
  // (the inline path is gated on the cloze wire; non-cloze
  // picks always go through the 204 + headers path).
  const body = (await res.json()) as ClozeExerciseOut
  return { kind: 'pick', pick: { kind: 'cloze', exercise: body } }
}

// --- session-shared grade (re-export of the per-type helper) ------------

// The mixer doesn't know which per-type endpoint generated the
// next pick (it can be cloze / matching / comprehension /
// idiom), but the grade surface is shared (``/exercises/grade``,
// ``exercise_type`` literal discriminator). We re-export the
// canonical helper here so callers don't need to know whether
// to import from ``api/cloze.ts`` or ``api/exercises.ts``.
export async function gradeSessionExercise(
  exercise_type: SessionExerciseType,
  exercise_id: number,
  grade: Grade,
): Promise<GradeResponse> {
  return gradeExercise(exercise_type, exercise_id, grade)
}