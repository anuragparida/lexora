// Phase 3.2 (card t_64055c49): diagnostic probe API client.
//
// Talks to the Phase 3.1 backend (card t_41d85c32):
//   POST /diagnostic/start                  -> { session_id, questions: [...] }
//   POST /diagnostic/answer                 -> { answered, total }
//   GET  /diagnostic/result?session_id=...  -> { axes, reasons }
//   POST /diagnostic/apply                  -> WeaknessProfile
//
// Every route is gated by `Depends(get_current_user)`, so requests must
// carry the `lexora_token` httpOnly cookie. We use `credentials: 'include'`
// for that — the localStorage mirror in `auth.ts` is not sent. The probe
// is fully deterministic: the same answer set always yields the same
// axes/reasons, so the result endpoint never has to persist anything.

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:18700'

export interface DiagnosticChoice {
  label: string
}

export interface DiagnosticQuestion {
  id: string
  prompt: string
  kind: string
  choices: DiagnosticChoice[]
}

export interface DiagnosticStart {
  session_id: string
  questions: DiagnosticQuestion[]
}

export interface DiagnosticProgress {
  answered: number
  total: number
}

export type Axes = Record<string, number>

export interface DiagnosticResult {
  axes: Axes
  reasons: Record<string, string>
}

export interface AppliedWeaknessProfile {
  id: number
  user_id: number
  axes: Axes
  updated_at: string
}

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

export async function startDiagnostic(): Promise<DiagnosticStart> {
  const res = await fetch(`${API_URL}/diagnostic/start`, {
    method: 'POST',
    credentials: 'include',
  })
  if (!res.ok) {
    throw new Error(await parseError(res))
  }
  return (await res.json()) as DiagnosticStart
}

export async function answerDiagnostic(
  sessionId: string,
  questionId: string,
  choiceLabel: string,
): Promise<DiagnosticProgress> {
  const res = await fetch(`${API_URL}/diagnostic/answer`, {
    method: 'POST',
    credentials: 'include',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      session_id: sessionId,
      question_id: questionId,
      choice_label: choiceLabel,
    }),
  })
  if (!res.ok) {
    throw new Error(await parseError(res))
  }
  return (await res.json()) as DiagnosticProgress
}

export async function getDiagnosticResult(
  sessionId: string,
): Promise<DiagnosticResult> {
  const res = await fetch(
    `${API_URL}/diagnostic/result?session_id=${encodeURIComponent(sessionId)}`,
    {
      credentials: 'include',
    },
  )
  if (!res.ok) {
    throw new Error(await parseError(res))
  }
  return (await res.json()) as DiagnosticResult
}

export async function applyDiagnostic(
  sessionId: string,
): Promise<AppliedWeaknessProfile> {
  const res = await fetch(`${API_URL}/diagnostic/apply`, {
    method: 'POST',
    credentials: 'include',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId }),
  })
  if (!res.ok) {
    throw new Error(await parseError(res))
  }
  return (await res.json()) as AppliedWeaknessProfile
}
