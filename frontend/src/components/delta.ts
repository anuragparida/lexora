// Project a `next_due_at` ISO timestamp into a human-readable
// delta string ("in 12m", "in 2h", "in 3d", "now"). The grade
// toast on every per-type page shows this so the user sees the
// FSRS scheduling effect without having to inspect a debug panel.
//
// Lives in its own file (split from ``ExerciseCard.tsx``) so the
// react-refresh lint rule is happy: the rule allows a fast-
// refresh-aware file to export only React components, so the
// pure-function helper has to live somewhere else. Same shape
// as the Phase 5.5 ``humanizeDelta`` that used to be inlined at
// the bottom of ``pages/ClozePage.tsx``.
export function humanizeDelta(nextDueAt: string): string {
  const now = Date.now()
  const target = new Date(nextDueAt).getTime()
  if (!Number.isFinite(target)) return 'soon'
  const deltaMs = target - now
  if (deltaMs <= 0) return 'now'
  const minutes = Math.round(deltaMs / 60_000)
  if (minutes < 1) return 'now'
  if (minutes < 60) return `in ${minutes}m`
  const hours = Math.round(minutes / 60)
  if (hours < 48) return `in ${hours}h`
  const days = Math.round(hours / 24)
  return `in ${days}d`
}
