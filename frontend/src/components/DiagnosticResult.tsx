import type { DiagnosticResult as Result } from '../api/diagnostic'

// Phase 3.2 (card t_64055c49): result-review screen for the diagnostic
// probe. Lists non-zero axes with a 0-3 visual score indicator and the
// one-line reason string. Two CTAs: "Apply this profile" (PUTs the
// result via /diagnostic/apply) and "Edit manually" (skips apply).
//
// Score-to-color mapping matches the spec:
//   0  -> gray-700   (not rendered — zero-score axes are filtered out)
//   1  -> yellow-500
//   2  -> orange-500
//   3  -> red-500
//
// Zero-axis case: shows a "comfortable across the board" message. The
// Apply button is still enabled — an all-zeros UPSERT is a valid
// weakness profile state and the user might still want to anchor it.

const SCORE_COLOR: Record<number, string> = {
  0: 'bg-slate-700',
  1: 'bg-yellow-500',
  2: 'bg-orange-500',
  3: 'bg-red-500',
}

function titleCase(snake: string): string {
  return snake
    .split('_')
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ')
}

interface Props {
  result: Result
  applying: boolean
  applyError: string | null
  onApply: () => void
  onEdit: () => void
}

export function DiagnosticResult({
  result,
  applying,
  applyError,
  onApply,
  onEdit,
}: Props) {
  // Sort axes deterministically by descending score, then by axis name
  // so the user sees the most-shaky axes first. Zero-score axes are
  // dropped from the list (their reasons are empty by spec).
  const entries = Object.entries(result.axes)
    .filter(([, score]) => score > 0)
    .sort(([aKey, aScore], [bKey, bScore]) => {
      if (aScore !== bScore) return bScore - aScore
      return aKey.localeCompare(bKey)
    })

  const allZero = entries.length === 0

  return (
    <div className="space-y-6">
      <header className="space-y-1">
        <h2 className="text-lg sm:text-xl font-semibold text-slate-100">
          Suggested weakness profile
        </h2>
        <p className="text-sm text-slate-400">
          Based on your answers, here's where to focus. Apply this to your
          profile, or edit it manually.
        </p>
      </header>

      {allZero ? (
        <div className="rounded-lg border border-slate-800 bg-slate-900 p-5">
          <p className="text-sm text-slate-300">
            Looks like you're comfortable across the board. You can still
            set axes manually if you want.
          </p>
        </div>
      ) : (
        <ul className="space-y-2.5">
          {entries.map(([axisKey, score]) => {
            const reason = result.reasons[axisKey]
            return (
              <li
                key={axisKey}
                className="rounded-lg border border-slate-800 bg-slate-900 px-4 py-3"
              >
                <div className="flex items-center justify-between gap-4">
                  <span className="text-sm font-medium text-slate-100">
                    {titleCase(axisKey)}
                  </span>
                  <span
                    className="flex items-center gap-1.5"
                    aria-label={`Score ${score} of 3`}
                  >
                    {[0, 1, 2, 3].map((i) => (
                      <span
                        key={i}
                        className={[
                          'inline-block h-2.5 w-2.5 rounded-full',
                          i < score
                            ? (SCORE_COLOR[score] ?? 'bg-slate-700')
                            : 'bg-slate-700/60',
                        ].join(' ')}
                      />
                    ))}
                  </span>
                </div>
                {reason && (
                  <p className="mt-1.5 text-xs text-slate-400 leading-snug">
                    {reason}
                  </p>
                )}
              </li>
            )
          })}
        </ul>
      )}

      {applyError && (
        <p className="text-sm text-red-400" role="alert">
          {applyError}
        </p>
      )}

      <div className="flex flex-wrap items-center gap-3 pt-2">
        <button
          type="button"
          onClick={onApply}
          disabled={applying}
          className="px-4 py-2 text-sm rounded-lg bg-blue-600 text-white font-medium hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {applying ? 'Applying…' : 'Apply this profile'}
        </button>
        <button
          type="button"
          onClick={onEdit}
          disabled={applying}
          className="px-4 py-2 text-sm rounded-lg border border-slate-700 text-slate-300 hover:bg-slate-800 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          Edit manually
        </button>
      </div>
    </div>
  )
}
