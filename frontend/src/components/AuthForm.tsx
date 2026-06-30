import { useState, type FormEvent } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { login, signup, type AuthUser } from '../auth'

// Phase 2.3 (card t_ffe6d6af): combined signup/login form.
// `mode` chooses the initial panel; the toggle is a plain link switch so
// the form stays a single component (no router gymnastics).
//
// On success we store the token (already done inside `login` / `signup`)
// and navigate to `/`. The Header in the new layout will pick up the
// logged-in state from `getMe` on mount.

interface Props {
  mode: 'login' | 'signup'
}

export function AuthForm({ mode }: Props) {
  const navigate = useNavigate()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (submitting) return
    setSubmitting(true)
    setError(null)
    try {
      const fn = mode === 'login' ? login : signup
      const res: AuthResponseLike = await fn(email, password)
      // AuthResponse shape lives in `auth.ts`; we type-narrow it here
      // without re-importing just to keep this component self-contained.
      void (res as { access_token: string; user: AuthUser })
      navigate('/', { replace: true })
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Something went wrong')
    } finally {
      setSubmitting(false)
    }
  }

  const isLogin = mode === 'login'
  const heading = isLogin ? 'Log in' : 'Sign up'
  const cta = isLogin ? 'Log in' : 'Create account'
  const switchLabel = isLogin ? "Don't have an account?" : 'Already have an account?'
  const switchTarget = isLogin ? '/signup' : '/login'
  const switchCta = isLogin ? 'Sign up' : 'Log in'

  return (
    <div className="min-h-screen flex items-center justify-center px-4 bg-slate-950">
      <div className="w-full max-w-sm rounded-lg shadow-sm border p-8 bg-slate-900 border-slate-800">
        <h1 className="text-2xl font-bold text-slate-100 mb-1">Lexora</h1>
        <p className="text-sm text-slate-400 mb-6">{heading}</p>
        <form onSubmit={handleSubmit} className="space-y-4" noValidate>
          <div>
            <label htmlFor="email" className="block text-sm font-medium text-slate-300 mb-1">
              Email
            </label>
            <input
              id="email"
              type="email"
              autoComplete="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="w-full px-3 py-2 rounded-lg border bg-slate-800 border-slate-700 text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder="you@example.com"
            />
          </div>
          <div>
            <label htmlFor="password" className="block text-sm font-medium text-slate-300 mb-1">
              Password
            </label>
            <input
              id="password"
              type="password"
              autoComplete={isLogin ? 'current-password' : 'new-password'}
              required
              minLength={8}
              maxLength={128}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full px-3 py-2 rounded-lg border bg-slate-800 border-slate-700 text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder="At least 8 characters"
            />
          </div>
          {error && (
            <p className="text-sm text-red-400" role="alert">
              {error}
            </p>
          )}
          <button
            type="submit"
            disabled={submitting}
            className="w-full px-4 py-2 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {submitting ? '…' : cta}
          </button>
        </form>
        <p className="mt-6 text-sm text-slate-400 text-center">
          {switchLabel}{' '}
          <Link to={switchTarget} className="text-blue-400 hover:text-blue-300 font-medium">
            {switchCta}
          </Link>
        </p>
      </div>
    </div>
  )
}

// Local shape just to satisfy the void-cast above without re-importing.
interface AuthResponseLike {
  access_token: string
  user: AuthUser
}