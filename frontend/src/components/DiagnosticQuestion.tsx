import { useId } from 'react'
import type { DiagnosticQuestion as Question } from '../api/diagnostic'

// Phase 3.2 (card t_64055c49): single-question view for the diagnostic
// probe. Renders the prompt as a heading and the choices as a vertical
// stack of radio-style rows. The selected choice is highlighted in the
// app's blue accent; the rest sit on slate-900.
//
// `onAdvance` fires when the user clicks Next (or "See results" on the
// last question). The parent owns the answer-persistence / page-advance
// lifecycle so this component stays dumb about API state. `canGoBack`
// is wired from the parent's index so we don't render a Back button on
// the first question. `isLast` switches the CTA label and disables the
// spinner-on-answer behaviour (the parent fires a different endpoint
// for the final step).

interface Props {
  question: Question
  index: number
  total: number
  selectedLabel: string | null
  onSelect: (label: string) => void
  onAdvance: () => void
  onBack: () => void
  canGoBack: boolean
  isLast: boolean
  submitting: boolean
  errorMessage: string | null
}

export function DiagnosticQuestion({
  question,
  index,
  total,
  selectedLabel,
  onSelect,
  onAdvance,
  onBack,
  canGoBack,
  isLast,
  submitting,
  errorMessage,
}: Props) {
  const groupId = useId()
  const cta = isLast ? 'See results' : 'Next'
  const disabled = selectedLabel === null || submitting

  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <p className="text-xs font-medium uppercase tracking-wider text-slate-500">
          Question {index + 1} of {total}
        </p>
        <h2 className="text-lg sm:text-xl font-semibold text-slate-100 leading-snug">
          {question.prompt}
        </h2>
      </div>

      <div
        role="radiogroup"
        aria-labelledby={`${groupId}-prompt`}
        className="space-y-2"
      >
        {question.choices.map((choice, choiceIndex) => {
          const id = `${groupId}-c${choiceIndex}`
          const isSelected = selectedLabel === choice.label
          return (
            <label
              key={choice.label}
              htmlFor={id}
              className={[
                'flex items-start gap-3 rounded-lg border px-4 py-3 cursor-pointer transition-colors',
                isSelected
                  ? 'border-blue-500 bg-blue-950/40 ring-1 ring-blue-500/40'
                  : 'border-slate-800 bg-slate-900 hover:border-slate-700 hover:bg-slate-800/60',
              ].join(' ')}
            >
              <input
                id={id}
                type="radio"
                name={groupId}
                value={choice.label}
                checked={isSelected}
                onChange={() => onSelect(choice.label)}
                disabled={submitting}
                className="mt-0.5 h-4 w-4 shrink-0 accent-blue-500 cursor-pointer disabled:cursor-not-allowed"
              />
              <span className="text-sm text-slate-200 leading-snug">
                {choice.label}
              </span>
            </label>
          )
        })}
      </div>

      {errorMessage && (
        <p className="text-sm text-red-400" role="alert">
          {errorMessage}
        </p>
      )}

      <div className="flex items-center gap-3 pt-2">
        <button
          type="button"
          onClick={onBack}
          disabled={!canGoBack || submitting}
          className="px-4 py-2 text-sm rounded-lg border border-slate-700 text-slate-300 hover:bg-slate-800 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        >
          Back
        </button>
        <button
          type="button"
          onClick={onAdvance}
          disabled={disabled}
          className="px-4 py-2 text-sm rounded-lg bg-blue-600 text-white font-medium hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {submitting ? '…' : cta}
        </button>
      </div>
    </div>
  )
}
