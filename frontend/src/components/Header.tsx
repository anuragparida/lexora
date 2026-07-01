import { useEffect, useState } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { logout, getMe, AUTH_CHANGE_EVENT, type AuthUser } from '../auth'

// Phase 2.3 (card t_ffe6d6af) + Phase 3.2 (card t_64055c49): app header bar.
//
// Three modes:
//   - unauthenticated: shows "Log in" / "Sign up" links.
//   - authenticated: shows email + nav links ("Weakness profile",
//     "Run diagnostic") + "Log out" button.
//   - unknown (during the /auth/me probe): renders nothing on the right side
//     to avoid a flash of "Log in" before the cookie probe resolves.
//
// Logout clears the cookie + localStorage and redirects to /login.

export function Header() {
  const navigate = useNavigate()
  const [user, setUser] = useState<AuthUser | null | undefined>(undefined)
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    let cancelled = false
    const probe = () => {
      getMe()
        .then((u) => {
          if (!cancelled) setUser(u)
        })
        .catch(() => {
          if (!cancelled) setUser(null)
        })
    }
    probe()
    // Re-probe whenever auth.ts signals a state change (login/signup/
    // logout/401). The Header is mounted once at the App layout level so
    // this is the only way it learns about state changes from anywhere.
    window.addEventListener(AUTH_CHANGE_EVENT, probe)
    return () => {
      cancelled = true
      window.removeEventListener(AUTH_CHANGE_EVENT, probe)
    }
  }, [])

  async function handleLogout() {
    if (busy) return
    setBusy(true)
    try {
      await logout()
    } finally {
      setBusy(false)
      setUser(null)
      navigate('/login', { replace: true })
    }
  }

  return (
    <header className="bg-slate-900 shadow-sm border-b border-slate-800">
      <div className="px-6 py-3 flex items-center justify-between">
        <Link to="/" className="flex items-center gap-2">
          <span className="text-lg font-bold text-slate-100">Lexora</span>
          <span className="text-xs text-slate-500 hidden sm:inline">
            German vocabulary
          </span>
        </Link>
        <nav className="flex items-center gap-3">
          {user === undefined ? null : user === null ? (
            <>
              <Link
                to="/login"
                className="px-3 py-1.5 text-sm text-slate-300 hover:text-slate-100 transition-colors"
              >
                Log in
              </Link>
              <Link
                to="/signup"
                className="px-3 py-1.5 text-sm bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
              >
                Sign up
              </Link>
            </>
          ) : (
            <>
              <span className="text-sm text-slate-400 hidden sm:inline">
                {user.email}
              </span>
              <Link
                to="/weakness-profile"
                className="px-3 py-1.5 text-sm text-slate-300 hover:text-slate-100 transition-colors"
              >
                Weakness profile
              </Link>
              <Link
                to="/diagnostic"
                className="px-3 py-1.5 text-sm text-slate-300 hover:text-slate-100 transition-colors"
              >
                Run diagnostic
              </Link>
              <button
                type="button"
                onClick={handleLogout}
                disabled={busy}
                className="px-3 py-1.5 text-sm border border-slate-700 text-slate-300 rounded-lg hover:bg-slate-800 disabled:opacity-50 transition-colors"
              >
                {busy ? '…' : 'Log out'}
              </button>
            </>
          )}
        </nav>
      </div>
    </header>
  )
}