import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  getMe,
  type AuthUser,
  type DiagnosticState,
} from '../auth'
import { getWeaknessProfile, putWeaknessProfile } from '../api/weakness'
import { AxisSlider } from '../components/AxisSlider'

// Phase 2.4 (card t_c9c15278) + Phase 3.3 (card t_ff6fa637):
// 10-axis weakness profile form.
//
// Phase 2.4: loads the saved axes on mount via GET
// /weakness-profile/{user_id}; the backend auto-creates an empty
// profile on first read so we never see a 404 in practice.
// Renders one slider per axis in a vertical stack. The Save button
// PUTs the current values; on 200 a brief "Saved" toast confirms,
// on error the message renders inline. No undo and no confirmation
// prompt — the user re-declares freely.
//
// Phase 3.3 added a small "Run the diagnostic?" banner at the top
// of the page. It only appears when the user's saved axes are
// empty AND they've never run the probe (``diagnostic_state`` is
// ``never``) or ran it in the past (``diagnostic_state`` is
// ``applied``). The banner is hidden when ``axes`` is non-empty
// (the user has declared a profile and the diagnostic suggestion
// is moot) and when the probe is currently in progress
// (``in_progress`` — the user is mid-flow, don't interrupt) or
// completed-but-not-yet-applied (``completed`` — the user is on
// the result page; they don't need a banner here).
//
// The diagnostic_state is fetched on mount via ``getMe()`` — the
// auth probe the rest of the app already runs. If the fetch
// fails (transient 5xx, network blip), the banner is simply
// hidden. State stays local (useState + useEffect). No
// Redux/Zustand. The authoritative copy of the JWT lives in the
// httpOnly cookie; the ProtectedRoute gate above has already
// verified the user before we render.

interface AxisSpec {
  key: string
  label: string
  hint: string
}

const AXES: ReadonlyArray<AxisSpec> = [
  { key: 'verbs', label: 'Verbs', hint: 'conjugation patterns' },
  {
    key: 'prepositional_combos',
    label: 'Prepositional combos',
    hint: 'verb + preposition + case pairings',
  },
  {
    key: 'collocations',
    label: 'Collocations',
    hint: 'adjective + noun, verb + noun, etc.',
  },
  {
    key: 'idioms',
    label: 'Idioms',
    hint: 'fixed idiomatic expressions',
  },
  {
    key: 'abstract_nouns',
    label: 'Abstract nouns',
    hint: 'Gefühl, Freiheit, etc.',
  },
  { key: 'adjectives', label: 'Adjectives', hint: 'declension + comparison' },
  { key: 'adverbs', label: 'Adverbs', hint: 'temporal, modal, etc.' },
  { key: 'prepositions', label: 'Prepositions', hint: 'case governance' },
  {
    key: 'pronouns',
    label: 'Pronouns',
    hint: 'personal, reflexive, relative',
  },
  {
    key: 'conjunctions',
    label: 'Conjunctions',
    hint: 'coordinating + subordinating',
  },
] as const

type Status = 'idle' | 'loading' | 'saving' | 'saved' | 'error'

interface Props {
  user: AuthUser
}

export function WeaknessProfilePage({ user }: Props) {
  const [values, setValues] = useState<Record<string, number>>({})
  const [status, setStatus] = useState<Status>('loading')
  const [error, setError] = useState<string | null>(null)
  // ``null`` means "we don't know yet" (the /auth/me probe hasn't
  // resolved, or it failed). ``DiagnosticState`` is the resolved
  // value. The banner only renders for ``never`` and ``applied``
  // once the user has loaded their empty profile.
  const [diagnosticState, setDiagnosticState] = useState<
    DiagnosticState | null
  >(null)

  // Load on mount. The backend auto-creates an empty profile on first
  // read, so a 404 should be treated as "start at 0" — we keep the
  // request resilient anyway. The initial `status` is already 'loading',
  // so the effect only needs to flip to 'idle' or 'error' on resolution
  // (no synchronous setState in the effect body).
  useEffect(() => {
    let cancelled = false
    getWeaknessProfile(user.id)
      .then((profile) => {
        if (cancelled) return
        // Only carry over known axis keys; the backend might have stale
        // entries from a future schema bump and we don't want to render
        // phantom rows.
        const filtered: Record<string, number> = {}
        for (const spec of AXES) {
          const v = profile.axes[spec.key]
          if (typeof v === 'number' && v >= 0 && v <= 3) {
            filtered[spec.key] = Math.floor(v)
          }
        }
        setValues(filtered)
        setStatus('idle')
      })
      .catch((err: unknown) => {
        if (cancelled) return
        // 404 still lands here in practice (defence in depth — the
        // route says auto-create, but if the user_id was deleted we
        // fall back to a clean 0-state so the form is usable).
        setValues({})
        setError(err instanceof Error ? err.message : 'Could not load profile')
        setStatus('error')
      })
    return () => {
      cancelled = true
    }
  }, [user.id])

  // Phase 3.3 (card t_ff6fa637): probe /auth/me once on mount to
  // learn the user's diagnostic_state. The banner only appears
  // when (a) the profile is empty and (b) the state is "never" /
  // "applied". A failure here is non-fatal — the page renders
  // without a banner. We deliberately do NOT depend on
  // ``status`` / ``values`` here: the banner decision is
  // orthogonal to the loading flow, and re-running this effect
  // on every state change would re-fetch /auth/me needlessly.
  useEffect(() => {
    let cancelled = false
    getMe()
      .then((me) => {
        if (cancelled) return
        setDiagnosticState(me.diagnostic_state)
      })
      .catch(() => {
        // Transient 5xx or 401 — leave diagnosticState as null
        // and the banner stays hidden. The header re-probes
        // /auth/me on every auth-change event so the next mount
        // will pick up the real state.
      })
    return () => {
      cancelled = true
    }
  }, [user.id])

  // Auto-clear the "Saved" toast after 2s.
  useEffect(() => {
    if (status !== 'saved') return
    const t = window.setTimeout(() => setStatus('idle'), 2000)
    return () => window.clearTimeout(t)
  }, [status])

  function handleChange(key: string, next: number) {
    setValues((prev) => ({ ...prev, [key]: next }))
    if (status === 'saved' || status === 'error') {
      // Editing after a save/error invalidates the previous state.
      setStatus('idle')
      setError(null)
    }
  }

  async function handleSave() {
    if (status === 'saving') return
    setStatus('saving')
    setError(null)
    try {
      // Only send keys we have a value for — backend treats missing as
      // "no opinion", but we mirror the current state to keep the
      // round-trip exact.
      const payload: Record<string, number> = {}
      for (const spec of AXES) {
        const v = values[spec.key]
        if (typeof v === 'number') payload[spec.key] = v
      }
      await putWeaknessProfile(user.id, payload)
      setStatus('saved')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not save')
      setStatus('error')
    }
  }

  return (
    <div className="max-w-2xl mx-auto px-6 py-10">
      <header className="mb-8">
        <h1 className="text-2xl font-bold text-slate-100">Weakness profile</h1>
        <p className="text-sm text-slate-400 mt-1">
          Signed in as{' '}
          <span className="text-slate-300">{user.email}</span>. Declare how
          shaky each grammar axis feels — these are self-assessments, not
          measurements. The scale runs 0 (unknown) to 3 (critical).
        </p>
      </header>

      {status !== 'loading' &&
      Object.keys(values).length === 0 &&
      (diagnosticState === 'never' || diagnosticState === 'applied') ? (
        <div className="mb-6 rounded-lg border border-blue-900/60 bg-blue-950/40 p-4 flex items-center justify-between gap-4">
          <p className="text-sm text-slate-300">
            Want us to suggest axes based on a quick probe?{' '}
            <span className="text-slate-400">
              Ten multiple-choice questions, no LLM, takes about a minute.
            </span>
          </p>
          <Link
            to="/diagnostic"
            className="px-3 py-1.5 text-sm bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 transition-colors shrink-0"
          >
            Run the diagnostic →
          </Link>
        </div>
      ) : null}

      {status === 'loading' ? (
        <div className="rounded-lg border border-slate-800 bg-slate-900 p-6 text-sm text-slate-400">
          Loading your profile…
        </div>
      ) : (
        <div className="space-y-7">
          {AXES.map((spec) => (
            <AxisSlider
              key={spec.key}
              axisKey={spec.key}
              label={spec.label}
              hint={spec.hint}
              value={values[spec.key] ?? 0}
              onChange={(next) => handleChange(spec.key, next)}
              disabled={status === 'saving'}
            />
          ))}

          <div className="pt-2 flex items-center gap-4">
            <button
              type="button"
              onClick={handleSave}
              disabled={status === 'saving'}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {status === 'saving' ? 'Saving…' : 'Save'}
            </button>
            {status === 'saved' && (
              <span
                role="status"
                className="text-sm text-emerald-400"
              >
                Saved
              </span>
            )}
            {status === 'error' && error && (
              <span role="alert" className="text-sm text-red-400">
                {error}
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
