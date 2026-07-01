// Phase 3.2 (card t_64055c49): thin progress bar for the multi-step
// diagnostic probe. Renders a filled track + the "N of M" label above.
// Zero state on the bar when current === 0 and the "M" dot fills to
// fully-saturated blue when current === total. No animation — keep
// the snap predictable for QA walks.

interface Props {
  current: number
  total: number
}

export function ProgressBar({ current, total }: Props) {
  const safeTotal = total > 0 ? total : 1
  const safeCurrent = Math.max(0, Math.min(safeTotal, current))
  const pct = (safeCurrent / safeTotal) * 100
  return (
    <div className="space-y-1.5" aria-label={`Question ${safeCurrent} of ${safeTotal}`}>
      <div className="flex items-baseline justify-between">
        <span className="text-xs font-medium uppercase tracking-wider text-slate-400">
          Diagnostic probe
        </span>
        <span className="text-xs tabular-nums text-slate-400">
          {safeCurrent} of {safeTotal}
        </span>
      </div>
      <div
        className="h-1.5 w-full rounded-full bg-slate-800 overflow-hidden"
        role="progressbar"
        aria-valuenow={safeCurrent}
        aria-valuemin={0}
        aria-valuemax={safeTotal}
      >
        <div
          className="h-full bg-blue-500 transition-all duration-200"
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  )
}
