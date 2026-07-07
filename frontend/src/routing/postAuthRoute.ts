// Phase 3.3 (card t_ff6fa637) + Phase 5.6 (card t_f9375354) +
// Phase 9.6 (card t_f1c63bfc): the post-signup first-login gate.
//
// Pure-function branch (Phase 3.3, unchanged): given the response
// from `GET /auth/me`, decide where to land the user when there
// are NO due cards. This stays in its own file so:
//   - the routing logic is unit-testable without rendering anything
//   - the AuthForm / Header / ProtectedRoute can share one source of
//     truth (today only AuthForm uses it, but a future "load on
//     mount" header badge could re-use it).
//
// Routing rules (from the card body — the spec pseudocode):
//
//   sum(due_by_type) > 0                    -> /exercises/session
//   sum(due_by_type) == 0 AND profile empty -> /diagnostic
//                                              (per Phase 5.6)
//   sum(due_by_type) == 0 AND profile set   -> /weakness-profile
//                                              (per Phase 3.3)
//
// "null" weakness_profile is treated as empty (pre-Phase-2.1 user
// who never loaded the profile page; the server never auto-creates
// on /auth/me).
//
// "The 'axes empty AND state = applied' branch is the trickiest:
// a user can apply a probe, then zero the sliders (PUT empty
// axes). The naive rule 'no axes -> run probe' would re-route
// them back to /diagnostic on the next sign-in, but they've
// already made a deliberate choice to set axes manually — the
// spec is explicit that we must respect that." — unchanged from
// Phase 3.3.
//
// ------------------------------------------------------------------------
// PHASE 6 HARD RULE #11 — DELIBERATE OFFENSE (Phase 9.6 + Phase 10.6)
// ------------------------------------------------------------------------
// Phase 6 hard rule #11 said: "the first-login gate stays
// cloze-only." Phase 9.6 widens this — the gate now reads the
// full ``due_by_type`` union (cloze + matching + comprehension
// + idiom) and routes to ``/exercises/session`` whenever ANY
// type has outstanding cards. This is intentional: the Phase
// 9 plan delivers a study-session mixer that fuses the 4
// types, and a cloze-only gate would strand users with
// matching-only due cards on the wrong landing page.
//
// Phase 10.6 (card t_da43cc23) widens the union additively to
// include ``phrase_match`` (the 5th FSRS-graded exercise type
// per Phase 10.1 schema + 10.2 Literal widening + 10.3 endpoint
// + 10.5 frontend page). The gate now sums the 5-key
// ``due_by_type`` dict; a learner with only a ``phrase_match``
// card due lands on the study-session mixer.
//
// The offense is called out here so a future maintainer who
// re-asserts the Phase 6 hard rule #11 (or a project-policy
// sweep that assumes gates stay cloze-only) knows it's a
// deliberate Phase 9 / 10 widening, not a regression. See the
// ``PHASE-9.md`` spec for the plan-level rationale.
// ------------------------------------------------------------------------
//
// Phase 5.6 layered a third branch on TOP of the pure gate
// (cloze-only ``/exercises/due`` round-trip). Phase 9.6
// replaces that branch with the union-aware ``due_by_type``
// read; the legacy pure gate is preserved as the fall-through
// path. The new shape is:
//
//   async postAuthGate(me):
//     sum > 0          -> /exercises/session (early return)
//     sum == 0 / undef -> legacy pure gate(me) (fall through)
//
// Why the gate stays async: ``due_by_type`` arrives on the
// ``MePayload`` returned by ``getMe()`` (no extra round-trip),
// so the gate is now effectively sync on the network layer —
// we keep the ``async`` signature because the call site
// (``AuthForm``) is already inside an ``await`` chain, and a
// future widen-the-payload redesign can re-introduce an
// endpoint call without rippling through the gate's callers.

import type {
  DiagnosticState,
  MePayload,
  WeaknessProfileSummary,
} from '../auth'

export type PostAuthRoute =
  | '/exercises/session'
  | '/weakness-profile'
  | '/diagnostic'

// Phase 3.3's original union — kept for back-compat in the pure
// function's return type and for tests that import it directly.
// The widened wrapper returns the broader union above.
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

// Phase 9.6 (card t_f1c63bfc) + Phase 10.6 (card t_da43cc23):
// sum the ``due_by_type`` counts.
//
// Phase 9.6 widened the gate from cloze-only (Phase 5.6) to the
// union of cloze / matching / comprehension / idiom. Phase 10.6
// widens the union additively to include ``phrase_match`` (the
// 5th FSRS-graded exercise type) so a learner with only a
// phrase_match card due lands on ``/exercises/session`` instead
// of falling through to ``/weakness-profile``.
//
// Defensive against a missing / undefined ``due_by_type`` field
// (a pre-Phase-9.2 backend that predates the union widening, or
// a stale cached login payload) — both produce a zero sum and
// fall through to the legacy pure gate.
//
// Note: the backend always emits the dict with all 5 keys at
// zero on a pre-9.1 legacy schema; the optional ``?`` on the
// TypeScript type only exists to keep the gate robust against
// an extremely old cached payload. The runtime path doesn't
// need the optional chain in practice.
function totalDue(me: MePayload): number {
  const d = me.due_by_type
  if (d === undefined || d === null) return 0
  return d.cloze + d.matching + d.comprehension + d.idiom + d.phrase_match
}

// Phase 3.3 (card t_ff6fa637): pure-function gate. Decides between
// /weakness-profile and /diagnostic given only the /auth/me payload.
// Synchronous so tests can pass a hand-built MePayload without
// touching the network. The Phase 9.6 async wrapper calls this on
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

// Phase 9.6 (card t_f1c63bfc): async gate with the union-aware
// branch.
//
// Order of operations (per the card body §"Scope"):
//   1. Read ``due_by_type`` from the already-fetched ``MePayload``.
//   2. sum > 0   -> /exercises/session (early return)
//   3. sum == 0  -> fall through to the legacy pure gate
//                   using the MePayload we already have.
//
// We deliberately do NOT change the legacy `postAuthRoute` signature
// (still synchronous, still takes MePayload only). The async gate
// is the only symbol `AuthForm` calls; it does the due-check then
// delegates to the pure function.
//
// Phase 5.6's note about graceful degradation still holds: a
// missing ``due_by_type`` field (network / parsing failure on
// the wire) is treated as zero sum, which falls through to the
// legacy branches rather than stranding the user.
export async function postAuthGate(me: MePayload): Promise<PostAuthRoute> {
  if (totalDue(me) > 0) {
    return '/exercises/session'
  }
  // sum == 0 (or absent) falls through to the legacy pure gate.
  return postAuthRoute(me)
}

// Re-export the diagnostic-state union so callers don't have to
// import from ``../auth`` just to type-narrow a branch.
export type { DiagnosticState }