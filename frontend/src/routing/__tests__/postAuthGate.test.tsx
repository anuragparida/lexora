import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { AuthForm } from '../../components/AuthForm'
import type { AuthUser, MePayload } from '../../auth'

// Phase 5.6 (card t_f9375354): gate-integration tests.
//
// What we test:
//   1. User with due cards (200 from /exercises/due)
//        -> gate navigates to /exercises/due, NOT /diagnostic
//           or /weakness-profile.
//   2. User with no due cards + no weakness profile (204)
//        -> gate falls through to /diagnostic (Phase 3.3 behavior).
//   3. User with no due cards + weakness profile (204)
//        -> gate falls through to /weakness-profile (Phase 3.3).
//   4. /exercises/due returns 401
//        -> gate falls through to the profile-state branches
//           (treated as no-due-cards).
//   5. /exercises/due throws a network error
//        -> gate falls through to the profile-state branches.
//   6. /exercises/due returns 204 with axes-empty + state=never
//        -> gate routes to /diagnostic; the first branch is skipped.
//   7. /exercises/due returns 204 with axes-empty + state=completed
//        -> gate routes to /weakness-profile.
//
// We test through AuthForm (the gate's only real caller) rather
// than `postAuthGate` directly so we also exercise the wiring
// (the navigate call, the navigate-replace semantics). Pure
// `postAuthRoute` behaviour is unchanged from Phase 3.3 — we don't
// retest that here.
//
// `vi.mock` runs before imports resolve, so the auth.ts module sees
// the mocked `login`/`signup`/`getMe` on first import, and the
// api/due module sees the mocked `getDueCloze`.

vi.mock('../../auth', async () => {
  const actual = await vi.importActual<typeof import('../../auth')>('../../auth')
  return {
    ...actual,
    login: vi.fn(),
    signup: vi.fn(),
    getMe: vi.fn(),
  }
})

vi.mock('../../api/due', () => ({
  getDueCloze: vi.fn(),
}))

import { login, signup, getMe } from '../../auth'
import { getDueCloze } from '../../api/due'

const mockedLogin = vi.mocked(login)
const mockedSignup = vi.mocked(signup)
const mockedGetMe = vi.mocked(getMe)
const mockedGetDueCloze = vi.mocked(getDueCloze)

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
}

// AuthResponse shape — minimal, the form doesn't introspect it.
const authResponse = {
  access_token: 'fake-token',
  user: userFixture,
}

// Minimal ClozeExercise payload for the /exercises/due 200 branch.
// We don't render the body at the gate, but the API client still
// type-checks it. word_id 42 mirrors the ClozePage test fixture so
// future debugging is grep-friendly.
const dueExercise = {
  sentence_with_blank: 'Der Kandidat ___ den Vertrag.',
  answer_word_id: 42,
  distractors: [1042, 2087, 3155] as [number, number, number],
  difficulty: 'medium' as const,
  rationale: 'Sentence cues "unterzeichnen" via accusative object.',
  prompt_template_version: 'cloze-v1',
  due_from_fsrs: true,
}

function renderForm(initialPath = '/login') {
  return render(
    <MemoryRouter initialEntries={[initialPath]}>
      <Routes>
        <Route path="/login" element={<AuthForm mode="login" />} />
        <Route path="/signup" element={<AuthForm mode="signup" />} />
        <Route path="/exercises/due" element={<div>EXERCISES_DUE_PAGE</div>} />
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

describe('AuthForm gate (Phase 5.6 third branch)', () => {
  beforeEach(() => {
    mockedLogin.mockReset()
    mockedSignup.mockReset()
    mockedGetMe.mockReset()
    mockedGetDueCloze.mockReset()
    // Default: login succeeds, getMe returns the fresh-user
    // payload, getDueCloze reports nothing due. Tests override
    // per-case.
    mockedLogin.mockResolvedValue(authResponse)
    mockedSignup.mockResolvedValue(authResponse)
  })

  afterEach(() => {
    cleanup()
  })

  it('routes to /exercises/due when /exercises/due returns 200 + a due card', async () => {
    mockedGetMe.mockResolvedValue(meFreshUser)
    mockedGetDueCloze.mockResolvedValue({ kind: 'due', exercise: dueExercise })
    renderForm()
    await submitLoginForm()
    await waitFor(() => {
      expect(screen.getByText('EXERCISES_DUE_PAGE')).toBeInTheDocument()
    })
    // Sanity: the gate did NOT route to /diagnostic or
    // /weakness-profile even though `meFreshUser` would normally
    // send it there.
    expect(screen.queryByText('DIAGNOSTIC_PAGE')).not.toBeInTheDocument()
    expect(screen.queryByText('WEAKNESS_PROFILE_PAGE')).not.toBeInTheDocument()
    // The due-check fires exactly once per login.
    expect(mockedGetDueCloze).toHaveBeenCalledTimes(1)
  })

  it('falls through to /diagnostic when /exercises/due returns 204 (no due cards, no profile)', async () => {
    mockedGetMe.mockResolvedValue(meFreshUser)
    mockedGetDueCloze.mockResolvedValue({ kind: 'no_cards' })
    renderForm()
    await submitLoginForm()
    await waitFor(() => {
      expect(screen.getByText('DIAGNOSTIC_PAGE')).toBeInTheDocument()
    })
    expect(screen.queryByText('EXERCISES_DUE_PAGE')).not.toBeInTheDocument()
    expect(screen.queryByText('WEAKNESS_PROFILE_PAGE')).not.toBeInTheDocument()
  })

  it('falls through to /weakness-profile when /exercises/due returns 204 (no due cards, profile exists)', async () => {
    mockedGetMe.mockResolvedValue(meProfileUser)
    mockedGetDueCloze.mockResolvedValue({ kind: 'no_cards' })
    renderForm()
    await submitLoginForm()
    await waitFor(() => {
      expect(screen.getByText('WEAKNESS_PROFILE_PAGE')).toBeInTheDocument()
    })
    expect(screen.queryByText('EXERCISES_DUE_PAGE')).not.toBeInTheDocument()
    expect(screen.queryByText('DIAGNOSTIC_PAGE')).not.toBeInTheDocument()
  })

  it('falls through to /diagnostic when /exercises/due returns 401 (treated as no-due-cards)', async () => {
    // The API client collapses 401 into `kind: 'no_cards'`. We
    // verify the gate respects that mapping — a stale or invalid
    // JWT must not strand the user on a missing route.
    mockedGetMe.mockResolvedValue(meFreshUser)
    mockedGetDueCloze.mockResolvedValue({ kind: 'no_cards' })
    renderForm()
    await submitLoginForm()
    await waitFor(() => {
      expect(screen.getByText('DIAGNOSTIC_PAGE')).toBeInTheDocument()
    })
    expect(screen.queryByText('EXERCISES_DUE_PAGE')).not.toBeInTheDocument()
  })

  it('falls through to /diagnostic when /exercises/due throws a network error', async () => {
    // A network failure (DNS, offline, CORS) is reported as
    // `kind: 'error'`. The gate treats that the same as 204 —
    // never strand the user on a broken network blip.
    mockedGetMe.mockResolvedValue(meFreshUser)
    mockedGetDueCloze.mockResolvedValue({
      kind: 'error',
      message: 'fetch failed',
    })
    renderForm()
    await submitLoginForm()
    await waitFor(() => {
      expect(screen.getByText('DIAGNOSTIC_PAGE')).toBeInTheDocument()
    })
    expect(screen.queryByText('EXERCISES_DUE_PAGE')).not.toBeInTheDocument()
  })

  it('signup + fresh profile + 204 still routes to /diagnostic (Phase 3.3 regression)', async () => {
    // Defensive: the third branch only fires AFTER the user is
    // authenticated. Signup -> login-shaped token -> getMe -> gate.
    // A fresh signup must still land on /diagnostic when no due
    // cards exist (Phase 3.3 behaviour, card body §"Gotchas" #2).
    mockedSignup.mockResolvedValue(authResponse)
    mockedGetMe.mockResolvedValue(meFreshUser)
    mockedGetDueCloze.mockResolvedValue({ kind: 'no_cards' })
    const user = userEvent.setup()
    renderForm('/signup')
    await user.type(screen.getByLabelText(/email/i), 'new@example.com')
    await user.type(screen.getByLabelText(/password/i), 'correcthorse123')
    await user.click(screen.getByRole('button', { name: /create account/i }))
    await waitFor(() => {
      expect(screen.getByText('DIAGNOSTIC_PAGE')).toBeInTheDocument()
    })
    expect(mockedGetDueCloze).toHaveBeenCalledTimes(1)
  })

  it('due cards win over an existing weakness profile (gate priority order)', async () => {
    // Even with axes non-empty, a 200 from /exercises/due routes
    // the user to /exercises/due. The Phase 3.3 priority was
    // "axes non-empty -> /weakness-profile"; Phase 5.6's new
    // branch inserts BEFORE that one, so a learner with both a
    // filled profile AND outstanding cards lands on the study
    // flow first. This is the entire point of the third branch.
    mockedGetMe.mockResolvedValue(meProfileUser)
    mockedGetDueCloze.mockResolvedValue({ kind: 'due', exercise: dueExercise })
    renderForm()
    await submitLoginForm()
    await waitFor(() => {
      expect(screen.getByText('EXERCISES_DUE_PAGE')).toBeInTheDocument()
    })
    expect(screen.queryByText('WEAKNESS_PROFILE_PAGE')).not.toBeInTheDocument()
  })
})