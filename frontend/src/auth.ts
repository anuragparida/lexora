// Auth client for Phase 2.3 (card t_ffe6d6af).
//
// Talks to the Phase 2.2 backend (card t_74c3aa1e):
//   POST /auth/signup  -> { access_token, user }
//   POST /auth/login   -> { access_token, user }
//   POST /auth/logout  -> 204
//   GET  /auth/me      -> { id, email, created_at }
//
// The real auth primitive is the `lexora_token` httpOnly cookie set by the
// backend — every server request goes through it. The localStorage copy
// under `lexora_token` is a *mirror* used only for client-side routing
// decisions (e.g. `ProtectedRoute`'s initial render before the cookie round-trip
// completes). It is XSS-readable, which is acceptable for this single-user
// local portfolio app because no third-party scripts run in the page.
//
// `credentials: 'include'` is what makes the cookie travel; the JSON body is
// the convenient bearer-style token we cache so the UI doesn't have to call
// `/auth/me` just to know if someone is "logged in".

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:18700'
export const TOKEN_KEY='***'
const AUTH_EVENT = 'lexora:auth-change'

export interface AuthUser {
  id: number
  email: string
  created_at: string
}

export interface AuthResponse {
  access_token: string
  user: AuthUser
}

// Emit a custom event so the global Header can re-probe /auth/me without
// us having to plumb user state through the route tree. Listeners are
// notified whenever the local auth state changes (login, signup, logout,
// or a /auth/me 401 that drops the local copy).
function notifyAuthChange(): void {
  window.dispatchEvent(new Event(AUTH_EVENT))
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

function storeToken(token: string): void {
  localStorage.setItem(TOKEN_KEY, token)
}

function clearStoredToken(): void {
  localStorage.removeItem(TOKEN_KEY)
}

export function getStoredToken(): string | null {
  return localStorage.getItem(TOKEN_KEY)
}

// Exposed so React components can subscribe to auth changes without us
// having to lift user state up to App.
export const AUTH_CHANGE_EVENT = AUTH_EVENT

export async function signup(
  email: string,
  password: string,
): Promise<AuthResponse> {
  const res = await fetch(`${API_URL}/auth/signup`, {
    method: 'POST',
    credentials: 'include',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password }),
  })
  if (!res.ok) {
    throw new Error(await parseError(res))
  }
  const data = (await res.json()) as AuthResponse
  storeToken(data.access_token)
  notifyAuthChange()
  return data
}

export async function login(
  email: string,
  password: string,
): Promise<AuthResponse> {
  const res = await fetch(`${API_URL}/auth/login`, {
    method: 'POST',
    credentials: 'include',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password }),
  })
  if (!res.ok) {
    throw new Error(await parseError(res))
  }
  const data = (await res.json()) as AuthResponse
  storeToken(data.access_token)
  notifyAuthChange()
  return data
}

export async function logout(): Promise<void> {
  // Even if the server call fails we still clear the local copy — the user
  // clicked "logout" and we shouldn't leave a stale token in localStorage
  // because of a transient 5xx. The cookie clear is best-effort.
  try {
    await fetch(`${API_URL}/auth/logout`, {
      method: 'POST',
      credentials: 'include',
    })
  } finally {
    clearStoredToken()
    notifyAuthChange()
  }
}

export async function getMe(): Promise<AuthUser> {
  const res = await fetch(`${API_URL}/auth/me`, {
    credentials: 'include',
  })
  if (res.status === 401) {
    // Token (cookie or localStorage) didn't survive validation — drop the
    // local copy so the next render starts clean.
    clearStoredToken()
    notifyAuthChange()
    throw new Error('Not authenticated')
  }
  if (!res.ok) {
    throw new Error(await parseError(res))
  }
  return (await res.json()) as AuthUser
}