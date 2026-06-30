import type { AuthUser } from '../auth'

// Phase 2.3 placeholder for /weakness-profile.
// Phase 2.4 (card t_c9c15278) replaces this body with the 10-axis slider
// UI. The route + auth gate are wired in this card so 2.4 hangs off them.

interface Props {
  user: AuthUser
}

export function WeaknessProfilePage({ user }: Props) {
  return (
    <div className="max-w-2xl mx-auto px-6 py-12">
      <h1 className="text-2xl font-bold text-slate-100 mb-2">
        Weakness profile
      </h1>
      <p className="text-sm text-slate-400 mb-6">
        Signed in as <span className="text-slate-300">{user.email}</span>
      </p>
      <div className="rounded-lg border border-slate-800 bg-slate-900 p-6">
        <p className="text-slate-300">
          Phase 2.4 lands the sliders here.
        </p>
        <p className="text-slate-500 text-sm mt-2">
          The 10-axis self-assessment UI ships on card{' '}
          <code className="text-slate-400">t_c9c15278</code>. This route and
          the auth gate are wired in 2.3 so 2.4 only has to render the form.
        </p>
      </div>
    </div>
  )
}