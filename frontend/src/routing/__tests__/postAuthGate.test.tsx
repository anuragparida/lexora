import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { AuthForm } from '../../components/AuthForm'
import type { AuthUser, MePayload } from '../../auth'

// Phase 5.6 (card t_f9375354) + Phase 9.6 (card t_f1c63bfc):
// gate-integration tests.
//
// What we test (Phase 5.6 baseline preserved; Phase 9.6 widens):
//   1. User with nonzero due_by_type
//        -> gate navigates to /exercises/session (Phase 9.6 widens
//           the Phase 5.6 cloze-only /exercises/due branch)
//   2. User with all-zero due_by_type + no weakness profile
//        -> gate falls through to /diagnostic (Phase 3.3 behavior)
//   3. User with all-zero due_by_type + weakness profile
//        -> gate falls through to /weakness-profile (Phase 3.3)
//   4. User with NO due_by_type field (pre-9.2 payload) +
//      axes empty + state=never -> /diagnostic (graceful fallback)
//   5. Nonzero due_by_type wins over an existing weakness profile
//      (gate priority order)
//
// Why we no longer mock the cloze-only ``getDueCloze`` client:
//   Phase 9.6 replaces the async ``/exercises/due`` round-trip
//   with a synchronous read of the ``due_by_type`` dict that
//   already arrives on ``MePayload``. The gate's network path
//   is now zero-cost; the only failure mode is "field absent",
//   which we cover by feeding ``me`` without the dict.
//
// We test through AuthForm (the gate's only real caller) rather
// than `postAuthGate` directly so we also exercise the wiring
// (the navigate call, the navigate-replace semantics). Pure
// `postAuthRoute` behaviour is unchanged from Phase 3.3 — we
// don't retest that here.

vi.mock('../../auth', async () => {
  const actual = await vi.importActual<typeof import('../../auth')>('../../auth')
  return {
    ...actual,
    login: vi.fn(),
    signup: vi.fn(),
    getMe: vi.fn(),
  }
})

import { login, signup, getMe } from '../../auth'

const mockedLogin = vi.mocked(login)
const mockedSignup = vi.mocked(signup)
const mockedGetMe = vi.mocked(getMe)

// A user object the login/signup responses can carry. The form
// doesn't read it post-submit — it's just the contract surface.
const userFixture: AuthUser = {
  id: 1,
  email: 'test@example.com',
  created_at: '2026-07-03T00:00:00Z',
}

// A signed-in fixture with an empty weakness profile and a fresh
// diagnostic state — the classic "first login" profile.
const meFreshUser: MePayload = {
  id: 1,
  email: 'test@example.com',
  created_at: '2026-07-03T00:00:00Z',
  weakness_profile: null,
  diagnostic_state: 'never',
  due_by_type: { cloze: 0, matching: 0, comprehension: 0, idiom: 0 },
}

// A signed-in fixture with a non-empty weakness profile — the
// "completed diagnostic" profile.
const meProfileUser: MePayload = {
  id: 2,
  email: 'returning@example.com',
  created_at: '2026-07-03T00:00:00Z',
  weakness_profile: {
    id: 5,
    user_id: 2,
    axes: { verbs: 2, collocations: 3 },
    updated_at: '2026-07-03T00:00:00Z',
  },
  diagnostic_state: 'applied',
  due_by_type: { cloze: 0, matching: 0, comprehension: 0, idiom: 0 },
}

// AuthResponse shape — minimal, the form doesn't introspect it.
const authResponse = {
  access_token: 'fake-token',
  user: userFixture,
}

// A fixture with nonzero due_by_type — the Phase 9.6 widening.
// Note: ``matching > 0`` is the meaningful case here — a
// matching-only due count would have been a cloze-only stranding
// bug under Phase 5.6's gate.
const meSessionUser: MePayload = {
  id: 3,
  email: 'session@example.com',
  created_at: '2026-07-04T00:00:00Z',
  weakness_profile: null,
  diagnostic_state: 'never',
  due_by_type: { cloze: 0, matching: 2, comprehension: 1, idiom: 0 },
}

// A pre-Phase-9.2 payload where ``due_by_type`` is absent.
// The gate must fall through to the legacy branches rather
// than throw — defence in depth against stale cached logins.
const meLegacyUser: MePayload = {
  id: 4,
  email: 'legacy@example.com',
  created_at: '2026-07-01T00:00:00Z',
  weakness_profile: null,
  diagnostic_state: 'never',
}

function renderForm(initialPath = '/login') {
  return render(
    <MemoryRouter initialEntries={[initialPath]}>
      <Routes>
        <Route path="/login" element={<AuthForm mode="login" />} />
        <Route path="/signup" element={<AuthForm mode="signup" />} />
        <Route
          path="/exercises/session"
          element={<div>EXERCISES_SESSION_PAGE</div>}
        />
        <Route path="/exercises/cloze" element={<div>EXERCISES_CLOZE_PAGE</div>} />
        <Route path="/diagnostic" element={<div>DIAGNOSTIC_PAGE</div>} />
        <Route path="/weakness-profile" element={<div>WEAKNESS_PROFILE_PAGE</div>} />
        <Route path="/" element={<div>HOME_PAGE</div>} />
      </Routes>
    </MemoryRouter>,
  )
}

async function submitLoginForm() {
  const user = userEvent.setup()
  // Fill the email and password fields.
  await user.type(screen.getByLabelText(/email/i), 'test@example.com')
  await user.type(screen.getByLabelText(/password/i), 'correcthorse123')
  // Submit. The mocked `login` resolves immediately so the gate
  // fires inside the same tick.
  await user.click(screen.getByRole('button', { name: /^log in$/i }))
}

describe('AuthForm gate (Phase 9.6 widening)', () => {
  beforeEach(() => {
    mockedLogin.mockReset()
    mockedSignup.mockReset()
    mockedGetMe.mockReset()
    // Default: login succeeds, getMe returns the fresh-user
    // payload. Tests override per-case.
    mockedLogin.mockResolvedValue(authResponse)
    mockedSignup.mockResolvedValue(authResponse)
  })

  afterEach(() => {
    cleanup()
  })

  it('routes to /exercises/session when due_by_type has a nonzero sum (Phase 9.6 widening)', async () => {
    mockedGetMe.mockResolvedValue(meSessionUser)
    renderForm()
    await submitLoginForm()
    await waitFor(() => {
      expect(screen.getByText('EXERCISES_SESSION_PAGE')).toBeInTheDocument()
    })
    // Sanity: the gate did NOT route to /diagnostic or
    // /weakness-profile even though ``meSessionUser`` would
    // normally send it there (profile empty + state=never).
    expect(screen.queryByText('DIAGNOSTIC_PAGE')).not.toBeInTheDocument()
    expect(screen.queryByText('WEAKNESS_PROFILE_PAGE')).not.toBeInTheDocument()
  })

  it('routes to /exercises/session when only the matching bucket is nonzero (the cloze-only stranding bug)', async () => {
    // Under Phase 5.6, a matching-only due count would have
    // fallen through to the profile branches because the gate
    // only checked cloze. Phase 9.6 widens this.
    const meMatchingOnly: MePayload = {
      ...meFreshUser,
      due_by_type: { cloze: 0, matching: 3, comprehension: 0, idiom: 0 },
    }
    mockedGetMe.mockResolvedValue(meMatchingOnly)
    renderForm()
    await submitLoginForm()
    await waitFor(() => {
      expect(screen.getByText('EXERCISES_SESSION_PAGE')).toBeInTheDocument()
    })
    expect(screen.queryByText('DIAGNOSTIC_PAGE')).not.toBeInTheDocument()
  })

  it('falls through to /diagnostic when due_by_type is all-zero + no profile', async () => {
    mockedGetMe.mockResolvedValue(meFreshUser)
    renderForm()
    await submitLoginForm()
    await waitFor(() => {
      expect(screen.getByText('DIAGNOSTIC_PAGE')).toBeInTheDocument()
    })
    expect(screen.queryByText('EXERCISES_SESSION_PAGE')).not.toBeInTheDocument()
    expect(screen.queryByText('WEAKNESS_PROFILE_PAGE')).not.toBeInTheDocument()
  })

  it('falls through to /weakness-profile when due_by_type is all-zero + profile exists', async () => {
    mockedGetMe.mockResolvedValue(meProfileUser)
    renderForm()
    await submitLoginForm()
    await waitFor(() => {
      expect(screen.getByText('WEAKNESS_PROFILE_PAGE')).toBeInTheDocument()
    })
    expect(screen.queryByText('EXERCISES_SESSION_PAGE')).not.toBeInTheDocument()
    expect(screen.queryByText('DIAGNOSTIC_PAGE')).not.toBeInTheDocument()
  })

  it('falls through to /diagnostic when due_by_type is absent (pre-9.2 payload graceful fallback)', async () => {
    // A pre-Phase-9.2 backend payload omits due_by_type. The
    // gate must not throw — it sums to zero and falls through
    // to the legacy profile branches.
    mockedGetMe.mockResolvedValue(meLegacyUser)
    renderForm()
    await submitLoginForm()
    await waitFor(() => {
      expect(screen.getByText('DIAGNOSTIC_PAGE')).toBeInTheDocument()
    })
    expect(screen.queryByText('EXERCISES_SESSION_PAGE')).not.toBeInTheDocument()
  })

  it('nonzero due_by_type wins over an existing weakness profile (gate priority order)', async () => {
    // Even with axes non-empty, a nonzero due_by_type routes
    // the user to /exercises/session. The Phase 3.3 priority
    // was "axes non-empty -> /weakness-profile"; Phase 9.6's
    // widening inserts BEFORE that one, so a learner with both
    // a filled profile AND outstanding cards lands on the
    // study-session mixer first.
    const meReturningWithDue: MePayload = {
      ...meProfileUser,
      due_by_type: { cloze: 1, matching: 0, comprehension: 0, idiom: 0 },
    }
    mockedGetMe.mockResolvedValue(meReturningWithDue)
    renderForm()
    await submitLoginForm()
    await waitFor(() => {
      expect(screen.getByText('EXERCISES_SESSION_PAGE')).toBeInTheDocument()
    })
    expect(screen.queryByText('WEAKNESS_PROFILE_PAGE')).not.toBeInTheDocument()
  })

  it('signup + fresh profile + all-zero due_by_type still routes to /diagnostic (Phase 3.3 regression)', async () => {
    // Defensive: the new branch only fires AFTER the user is
    // authenticated. Signup -> login-shaped token -> getMe ->
    // gate. A fresh signup must still land on /diagnostic when
    // no due cards exist.
    mockedSignup.mockResolvedValue(authResponse)
    mockedGetMe.mockResolvedValue(meFreshUser)
    const user = userEvent.setup()
    renderForm('/signup')
    await user.type(screen.getByLabelText(/email/i), 'new@example.com')
    await user.type(screen.getByLabelText(/password/i), 'correcthorse123')
    await user.click(screen.getByRole('button', { name: /create account/i }))
    await waitFor(() => {
      expect(screen.getByText('DIAGNOSTIC_PAGE')).toBeInTheDocument()
    })
    expect(screen.queryByText('EXERCISES_SESSION_PAGE')).not.toBeInTheDocument()
  })
})
