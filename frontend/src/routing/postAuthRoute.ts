// Phase 3.3 (card t_ff6fa637) + Phase 5.6 (card t_f9375354):
// the post-signup first-login gate.
//
// Pure-function branch (Phase 3.3, unchanged): given the response
// from `GET /auth/me`, decide where to land the user when there are
// NO due cards. This stays in its own file so:
//   - the routing logic is unit-testable without rendering anything
//   - the AuthForm / Header / ProtectedRoute can share one source of
//     truth (today only AuthForm uses it, but a future "load on
//     mount" header badge could re-use it).
//
// Routing rules (from the card body — the spec pseudocode):
//
//   axes non-empty                                -> /weakness-profile
//   axes empty AND state in {never, in_progress}   -> /diagnostic
//   axes empty AND state in {completed, applied}  -> /weakness-profile
//     (user has been through the probe or has set
//      axes manually; respect that decision)
//
// The "axes empty AND state = applied" branch is the trickiest: a
// user can apply a probe, then zero the sliders (PUT empty axes).
// The naive rule "no axes -> run probe" would re-route them back to
// /diagnostic on the next sign-in, but they've already made a
// deliberate choice to set axes manually — the spec is explicit
// that we must respect that.
//
// "null" weakness_profile is treated as empty (pre-Phase-2.1 user
// who never loaded the profile page; the server never auto-creates
// on /auth/me).
//
// Phase 5.6 (card t_f9375354) layers a third branch on TOP of this
// pure-function gate. The new branch fires BEFORE the profile-state
// branches and is a network call: GET /exercises/due. If the server
// returns 200 (any card is due), we route to /exercises/due. If it
// returns 204 / 401 / errors, we fall through to the existing logic.
//
// Why an async wrapper instead of mutating the pure function:
//   - Phase 3.3's `postAuthRoute(me)` is imported by tests that
//     pass a hand-built MePayload. Keeping that signature synchronous
//     preserves those tests and the offline-test guarantee.
//   - The new wrapper `postAuthGate(me)` is the one AuthForm calls;
//     it does the due-check then delegates to the pure function.
//   - The pure function is now an implementation detail of the
//     wrapper, but it's still exported for tests.
import type { DiagnosticState, MePayload, WeaknessProfileSummary } from '../auth'
import { getDueCloze } from '../api/due'

export type PostAuthRoute = '/exercises/due' | '/weakness-profile' | '/diagnostic'

// Phase 3.3's original union — kept for back-compat in the pure
// function's return type and for tests that import it directly. The
// new wrapper widens to include '/exercises/due'.
export type PostAuthRouteLegacy = '/weakness-profile' | '/diagnostic'

function isEmptyProfile(
  profile: WeaknessProfileSummary | null,
): boolean {
  if (profile === null) return true
  // ``Object.keys`` is the cheap path — works for the 10-axis
  // profile the backend ships. A future ``null``/missing-axes
  // would still be treated as empty (the server never produces
  // that shape today, so this is defence in depth).
  return Object.keys(profile.axes).length === 0
}

// Phase 3.3 (card t_ff6fa637): pure-function gate. Decides between
// /weakness-profile and /diagnostic given only the /auth/me payload.
// Synchronous so tests can pass a hand-built MePayload without
// touching the network. The Phase 5.6 async wrapper calls this on
// the fall-through path (no due cards).
export function postAuthRoute(me: MePayload): PostAuthRouteLegacy {
  if (!isEmptyProfile(me.weakness_profile)) {
    return '/weakness-profile'
  }
  if (
    me.diagnostic_state === 'never' ||
    me.diagnostic_state === 'in_progress'
  ) {
    return '/diagnostic'
  }
  // ``completed`` / ``applied`` / null (defensive — server never
  // returns null for diagnostic_state today; the Pydantic default
  // is ``"never"``) all fall through to the profile page.
  return '/weakness-profile'
}

// Phase 5.6 (card t_f9375354): async gate with the third branch.
//
// Order of operations (per the card body §"Scope"):
//   1. GET /exercises/due
//   2. 200                 -> /exercises/due  (early return)
//   3. 204 / 401 / error   -> fall through to the legacy pure gate
//                              using the MePayload we already have.
//
// We deliberately do NOT change the legacy `postAuthRoute` signature
// (still synchronous, still takes MePayload only). The async gate
// is a new symbol — AuthForm is the only caller that should use it.
//
// The `getDueCloze` client swallows network errors into the
// discriminated `kind: 'error'` branch; the gate treats that the
// same as `kind: 'no_cards'` (fall through) per the card body's
// gotcha #3 ("The gate fires on EVERY login, not just first-login"
// — we want graceful degradation when the due-endpoint is down,
// not a stuck login form).
export async function postAuthGate(me: MePayload): Promise<PostAuthRoute> {
  const due = await getDueCloze()
  if (due.kind === 'due') {
    return '/exercises/due'
  }
  // 'no_cards' | 'error' both fall through to the legacy gate.
  return postAuthRoute(me)
}

// Re-export the diagnostic-state union so callers don't have to
// import from ``../auth`` just to type-narrow a branch.
export type { DiagnosticState }