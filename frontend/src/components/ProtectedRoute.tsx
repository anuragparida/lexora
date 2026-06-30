import { useEffect, useState, type ReactNode } from 'react'
import { Navigate, useLocation } from 'react-router-dom'
import { getMe, type AuthUser } from '../auth'

// Phase 2.3 (card t_ffe6d6af): route gate.
//
// Probes /auth/me on mount. Three states:
//   - loading: render a small placeholder so we don't flash a redirect.
//   - ok: render the children with the user in context (we use a callback
//     prop so a parent page can read the user if it wants).
//   - 401: redirect to /login, remembering where the user came from so the
//     post-login flow could send them back (Phase 2.4+ may wire that up).

interface Props {
  children: (user: AuthUser) => ReactNode
}

export function ProtectedRoute({ children }: Props) {
  const location = useLocation()
  const [user, setUser] = useState<AuthUser | null | undefined>(undefined)

  useEffect(() => {
    let cancelled = false
    getMe()
      .then((u) => {
        if (!cancelled) setUser(u)
      })
      .catch(() => {
        if (!cancelled) setUser(null)
      })
    return () => {
      cancelled = true
    }
  }, [])

  if (user === undefined) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-slate-950">
        <div className="text-slate-400 text-sm">Loading…</div>
      </div>
    )
  }

  if (user === null) {
    return <Navigate to="/login" replace state={{ from: location.pathname }} />
  }

  return <>{children(user)}</>
}