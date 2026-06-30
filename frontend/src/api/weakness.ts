// Phase 2.4 (card t_c9c15278): weakness profile API client.
//
// Talks to the Phase 2.1 + 2.2 backend (cards t_6318d0e1 + t_74c3aa1e):
//   GET  /weakness-profile/{user_id} -> { id, user_id, axes, updated_at }
//   PUT  /weakness-profile/{user_id} -> { id, user_id, axes, updated_at }
//
// Both routes are gated by `Depends(get_current_user)`, so the request
// must carry the `lexora_token` httpOnly cookie. We use
// `credentials: 'include'` for that. The localStorage copy of the token
// (set by `auth.ts`) is not sent — the server verifies the cookie.
//
// The 10 axes (verbs, prepositional_combos, ...) are listed in the page
// component. The backend only validates the values (int in [0, 3]); it
// doesn't care which axes are present, so a partial PUT is a valid
// upsert and an empty `{}` is a valid reset.

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:18700'

export type Axes = Record<string, number>

export interface WeaknessProfile {
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

export async function getWeaknessProfile(
  userId: number,
): Promise<WeaknessProfile> {
  const res = await fetch(`${API_URL}/weakness-profile/${userId}`, {
    credentials: 'include',
  })
  if (!res.ok) {
    throw new Error(await parseError(res))
  }
  return (await res.json()) as WeaknessProfile
}

export async function putWeaknessProfile(
  userId: number,
  axes: Axes,
): Promise<WeaknessProfile> {
  const res = await fetch(`${API_URL}/weakness-profile/${userId}`, {
    method: 'PUT',
    credentials: 'include',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ axes }),
  })
  if (!res.ok) {
    throw new Error(await parseError(res))
  }
  return (await res.json()) as WeaknessProfile
}
