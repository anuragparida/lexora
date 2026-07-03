import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { GradeButtons } from '../../components/GradeButtons'

// Phase 5.5 (card t_f253456b): GradeButtons unit tests.
//
// The component is intentionally dumb — it owns no state, fires
// no network calls, knows nothing about the underlying cloze.
// The test surface is just "render four buttons; click one;
// assert the right onGrade() fires." No network mocks needed,
// which is the whole point of the dumb-component split.
//
// Three cases (per the card body):
//   1. Renders four buttons (the spec's four-grade row).
//   2. Clicking "Good" calls onGrade(3).
//   3. Buttons are disabled when disabled=true.

describe('GradeButtons', () => {
  afterEach(() => {
    cleanup()
  })

  it('renders four grade buttons in spec order (1/2/3/4)', () => {
    render(<GradeButtons onGrade={vi.fn()} />)
    // The four data-testid hooks are the spec's stable contract
    // — they are what the future integration suite hooks into.
    expect(screen.getByTestId('grade-button-1')).toBeInTheDocument()
    expect(screen.getByTestId('grade-button-2')).toBeInTheDocument()
    expect(screen.getByTestId('grade-button-3')).toBeInTheDocument()
    expect(screen.getByTestId('grade-button-4')).toBeInTheDocument()
    // The group wrapper carries the role + label that screen
    // readers announce.
    expect(screen.getByRole('group', { name: /fsrs grade/i })).toBeInTheDocument()
    // Spot-check the visible labels — the user-facing copy is
    // part of the contract (a future maintainer cannot rename
    // "Good" to "Fine" without seeing the test fail).
    expect(screen.getByTestId('grade-button-1')).toHaveTextContent(/Again \(1\)/)
    expect(screen.getByTestId('grade-button-2')).toHaveTextContent(/Hard \(2\)/)
    expect(screen.getByTestId('grade-button-3')).toHaveTextContent(/Good \(3\)/)
    expect(screen.getByTestId('grade-button-4')).toHaveTextContent(/Easy \(4\)/)
  })

  it('calls onGrade(3) when the Good button is clicked', () => {
    const onGrade = vi.fn()
    render(<GradeButtons onGrade={onGrade} />)
    fireEvent.click(screen.getByTestId('grade-button-3'))
    expect(onGrade).toHaveBeenCalledTimes(1)
    expect(onGrade).toHaveBeenCalledWith(3)
  })

  it('calls onGrade with the matching literal for every button', () => {
    const onGrade = vi.fn()
    render(<GradeButtons onGrade={onGrade} />)
    fireEvent.click(screen.getByTestId('grade-button-1'))
    fireEvent.click(screen.getByTestId('grade-button-2'))
    fireEvent.click(screen.getByTestId('grade-button-4'))
    expect(onGrade).toHaveBeenCalledTimes(3)
    expect(onGrade).toHaveBeenNthCalledWith(1, 1)
    expect(onGrade).toHaveBeenNthCalledWith(2, 2)
    expect(onGrade).toHaveBeenNthCalledWith(3, 4)
  })

  it('disables every button when disabled=true', () => {
    const onGrade = vi.fn()
    render(<GradeButtons onGrade={onGrade} disabled={true} />)
    expect(screen.getByTestId('grade-button-1')).toBeDisabled()
    expect(screen.getByTestId('grade-button-2')).toBeDisabled()
    expect(screen.getByTestId('grade-button-3')).toBeDisabled()
    expect(screen.getByTestId('grade-button-4')).toBeDisabled()
    // Defense-in-depth: even if the disabled attribute were
    // bypassed by an a11y extension, the click handler must not
    // fire. fireEvent.click on a disabled button is a no-op in
    // jsdom, so we just confirm onGrade was never called.
    fireEvent.click(screen.getByTestId('grade-button-3'))
    expect(onGrade).not.toHaveBeenCalled()
  })

  it('does not disable any button when disabled is omitted (default false)', () => {
    render(<GradeButtons onGrade={vi.fn()} />)
    expect(screen.getByTestId('grade-button-1')).not.toBeDisabled()
    expect(screen.getByTestId('grade-button-2')).not.toBeDisabled()
    expect(screen.getByTestId('grade-button-3')).not.toBeDisabled()
    expect(screen.getByTestId('grade-button-4')).not.toBeDisabled()
  })
})