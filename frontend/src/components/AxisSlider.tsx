import * as Slider from '@radix-ui/react-slider'
import { useId } from 'react'

// Phase 2.4 (card t_c9c15278): single-axis slider row.
//
// One row in the weakness profile form. Renders a label, the
// @radix-ui/react-slider primitive, and a tick-label helper that
// shows the current integer score (0-3) and its semantic name
// (unknown / shaky / developing / critical).
//
// The score is clamped on the parent side; this component is dumb
// about domain rules and just emits integer changes via `onChange`.

const TICK_LABELS = ['unknown', 'shaky', 'developing', 'critical'] as const

export interface AxisSliderProps {
  axisKey: string
  label: string
  hint: string
  value: number
  onChange: (next: number) => void
  disabled?: boolean
}

export function AxisSlider({
  axisKey,
  label,
  hint,
  value,
  onChange,
  disabled = false,
}: AxisSliderProps) {
  const id = useId()
  const tickIndex = Math.max(0, Math.min(3, value))
  return (
    <div className="space-y-2">
      <div className="flex items-baseline justify-between gap-3">
        <label
          htmlFor={id}
          className="text-sm font-medium text-slate-200"
        >
          {label}
        </label>
        <span className="text-xs text-slate-400 tabular-nums">
          {value} · {TICK_LABELS[tickIndex]}
        </span>
      </div>
      <Slider.Root
        id={id}
        min={0}
        max={3}
        step={1}
        value={[value]}
        onValueChange={(values) => {
          const next = values[0] ?? 0
          onChange(next)
        }}
        disabled={disabled}
        aria-label={label}
        data-axis={axisKey}
        className="relative flex w-full touch-none select-none items-center h-5"
      >
        <Slider.Track className="relative h-1.5 w-full grow rounded-full bg-slate-800">
          <Slider.Range className="absolute h-full rounded-full bg-blue-500" />
        </Slider.Track>
        <Slider.Thumb
          className="block h-4 w-4 rounded-full border border-blue-300 bg-blue-500 shadow transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-400 disabled:opacity-50"
        />
      </Slider.Root>
      <p className="text-xs text-slate-500">{hint}</p>
    </div>
  )
}
