// Phase 3.3 (card t_ff6fa637): the post-signup first-login gate.
//
// Pure function: given the response from `GET /auth/me`, decide where
// to land the user. Lives in its own file so:
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
import type { DiagnosticState, MePayload, WeaknessProfileSummary } from '../auth'

export type PostAuthRoute = '/weakness-profile' | '/diagnostic'

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

export function postAuthRoute(me: MePayload): PostAuthRoute {
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

// Re-export the diagnostic-state union so callers don't have to
// import from ``../auth`` just to type-narrow a branch.
export type { DiagnosticState }
