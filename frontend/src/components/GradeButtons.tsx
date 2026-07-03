import type { Grade } from '../api/cloze'

// Phase 5.5 (card t_f253456b): the four-button FSRS grade row.
//
// The Phase 5 spec wires the closed-loop middle here: the user
// clicks one of Again / Hard / Good / Easy, the page fires
// `POST /exercises/grade`, and the inline next-due flow re-
// fetches from `GET /exercises/due` to render the next card.
//
// This component is intentionally dumb — it owns no state, fires
// no network calls, and knows nothing about the underlying
// cloze. It just renders four buttons and forwards the chosen
// grade to the parent. That separation keeps it trivially
// testable (`GradeButtons.test.tsx` does not need to mock any
// network layer) and reusable if a future exercise type needs
// the same row.
//
// Visual design: each button uses a semantic color hint that
// maps to the FSRS semantic (red for Again — you forgot;
// amber for Hard — barely; blue for Good — comfortable;
// green for Easy — too easy). Colors carry information beyond
// the label, which matters once keyboard shortcuts are wired
// (Phase 6+, not in scope here).
//
// Accessibility:
//   - The four buttons are inside a role="group" with an
//     aria-label, so screen readers announce them as a single
//     "FSRS grade" widget rather than four disconnected
//     buttons.
//   - Native <button> elements handle keyboard activation
//     (Tab focus + Enter/Space) without any custom handlers —
//     the card body's "Keyboard accessible: tab order,
//     Enter/Space activate" requirement is satisfied by default
//     when we use semantic buttons rather than divs.
//   - The `disabled` flag on the row freezes every button so
//     the user can't double-click during the in-flight grade
//     request (handled by the parent via `disabled={isGrading}`).

export interface GradeButtonsProps {
  onGrade: (grade: Grade) => void
  disabled?: boolean
}

interface ButtonSpec {
  grade: Grade
  label: string
  ariaLabel: string
  // Tailwind classes keyed by the button's semantic meaning.
  // Kept inline (not a classnames helper) — there are only
  // four of them and the variants live in one place.
  className: string
}

// The four-button set, in the spec order. The order is the
// keyboard tab order too (Again → Hard → Good → Easy), which
// matches the conventional left-to-right reading flow for a
// grade row.
const BUTTONS: readonly ButtonSpec[] = [
  {
    grade: 1,
    label: 'Again (1)',
    ariaLabel: 'Grade: Again (1) — I forgot',
    className:
      'border-red-700 bg-red-950/40 text-red-200 hover:bg-red-900/60 focus-visible:ring-red-500',
  },
  {
    grade: 2,
    label: 'Hard (2)',
    ariaLabel: 'Grade: Hard (2) — barely recalled',
    className:
      'border-amber-700 bg-amber-950/40 text-amber-200 hover:bg-amber-900/60 focus-visible:ring-amber-500',
  },
  {
    grade: 3,
    label: 'Good (3)',
    ariaLabel: 'Grade: Good (3) — recalled comfortably',
    className:
      'border-blue-700 bg-blue-950/40 text-blue-200 hover:bg-blue-900/60 focus-visible:ring-blue-500',
  },
  {
    grade: 4,
    label: 'Easy (4)',
    ariaLabel: 'Grade: Easy (4) — too easy',
    className:
      'border-emerald-700 bg-emerald-950/40 text-emerald-200 hover:bg-emerald-900/60 focus-visible:ring-emerald-500',
  },
] as const

export function GradeButtons({ onGrade, disabled = false }: GradeButtonsProps) {
  return (
    <div
      role="group"
      aria-label="FSRS grade"
      data-testid="grade-buttons"
      className="grid grid-cols-2 sm:grid-cols-4 gap-3"
    >
      {BUTTONS.map((spec) => (
        <button
          key={spec.grade}
          type="button"
          onClick={() => onGrade(spec.grade)}
          disabled={disabled}
          aria-label={spec.ariaLabel}
          aria-pressed={false}
          data-testid={`grade-button-${spec.grade}`}
          className={
            'rounded-lg border px-4 py-3 text-sm font-medium transition-colors ' +
            'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 focus-visible:ring-offset-slate-950 ' +
            'disabled:opacity-50 disabled:cursor-not-allowed ' +
            spec.className
          }
        >
          {spec.label}
        </button>
      ))}
    </div>
  )
}