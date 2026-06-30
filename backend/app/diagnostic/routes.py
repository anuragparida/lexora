"""Phase 3.1 — diagnostic probe endpoints (card t_41d85c32).

Four auth-gated routes, all under ``Depends(get_current_user)``,
mounted at the ``/diagnostic`` prefix by ``app.main``:

- ``POST /diagnostic/start``  — create / resume a session, return the
  stripped question bank.
- ``POST /diagnostic/answer`` — record one (question, choice) answer.
- ``GET  /diagnostic/result`` — recompute the deterministic score.
- ``POST /diagnostic/apply``  — UPSERT the score into the caller's
  ``WeaknessProfile`` via the existing ``upsert_weakness_profile``.

NO LLM call. NO Langfuse tracing. The scoring is the pure function
in ``app.diagnostic.scoring`` — the same answers always yield the
same result, so ``/result`` never has to persist anything.

The question bank's scoring fields (``delta`` / ``weight`` /
``axis_tags``) are stripped before serialization — see
``_public_questions``.
"""
from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app import crud, models, schemas
from app.auth import get_current_user
from app.database import get_db
from app.diagnostic import questions as qbank
from app.diagnostic.scoring import answers_from_dict, score

router = APIRouter()


def _public_questions() -> list[schemas.DiagnosticQuestionOut]:
    """Map the in-code bank to the client-facing shape.

    Strips ``delta`` (per choice) and ``weight`` / ``axis_tags`` (per
    question) — those are server-side scoring inputs and must never
    cross the wire.
    """
    return [
        schemas.DiagnosticQuestionOut(
            id=q.id,
            prompt=q.prompt,
            kind=q.kind,
            choices=[
                schemas.DiagnosticChoiceOut(label=c.label) for c in q.choices
            ],
        )
        for q in qbank.QUESTIONS
    ]


def _require_owned_session(
    db: Session, session_id: str, user: models.User
) -> models.DiagnosticSession:
    """Load a session and assert it belongs to ``user``.

    Returns 404 for both "no such session" and "belongs to another
    user" — the same opaque shape so a probe can't tell a real
    foreign session id from a non-existent one (no enumeration).
    """
    session = crud.get_diagnostic_session(db, session_id)
    if session is None or session.user_id != user.id:
        raise HTTPException(status_code=404, detail="session not found")
    return session


@router.post("/start", response_model=schemas.DiagnosticStartOut)
def start_diagnostic(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Create or resume a diagnostic session.

    Reuse rules (avoid duplicate open sessions):
    - If the user has an ``in_progress`` session, return it.
    - Else if the most recent session is ``completed`` (not yet
      ``applied``), return that one.
    - Else (no sessions, or the latest is ``applied`` / ``skipped``),
      create a fresh ``in_progress`` session.

    The response always carries the full (stripped) question bank so
    the client can render the probe without a second round-trip.
    """
    sessions = crud.list_diagnostic_sessions(db, current_user.id)
    reuse: models.DiagnosticSession | None = None

    in_progress = next(
        (s for s in sessions if s.status == "in_progress"), None
    )
    if in_progress is not None:
        reuse = in_progress
    elif sessions and sessions[0].status == "completed":
        # ``sessions`` is newest-first; resume the latest completed
        # but not-yet-applied session.
        reuse = sessions[0]

    if reuse is not None:
        session = reuse
    else:
        session = crud.create_diagnostic_session(
            db, str(uuid.uuid4()), current_user.id
        )

    return schemas.DiagnosticStartOut(
        session_id=session.id,
        questions=_public_questions(),
    )


@router.post("/answer", response_model=schemas.DiagnosticAnswerOut)
def answer_diagnostic(
    payload: schemas.DiagnosticAnswerIn,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Record one answer for a session.

    Validation:
    - session must belong to the caller (404 otherwise).
    - ``question_id`` must be a valid bank id (400 otherwise).
    - ``choice_label`` must match one of the question's choices
      (400 otherwise).

    Idempotent per ``(session, question)`` — re-answering overwrites.
    Returns the progress counters ``{answered, total}``.
    """
    session = _require_owned_session(db, payload.session_id, current_user)

    question = qbank.QUESTION_BY_ID.get(payload.question_id)
    if question is None:
        raise HTTPException(
            status_code=400, detail=f"unknown question_id: {payload.question_id}"
        )

    valid_labels = {c.label for c in question.choices}
    if payload.choice_label not in valid_labels:
        raise HTTPException(
            status_code=400,
            detail=(
                f"invalid choice_label for {payload.question_id}: "
                f"{payload.choice_label!r}"
            ),
        )

    crud.record_diagnostic_answer(
        db, session, payload.question_id, payload.choice_label
    )
    answered = len(crud.get_diagnostic_answers(session))
    return schemas.DiagnosticAnswerOut(
        answered=answered, total=qbank.TOTAL_QUESTIONS
    )


@router.get("/result", response_model=schemas.DiagnosticResultOut)
def result_diagnostic(
    session_id: str = Query(..., min_length=1),
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Compute the deterministic score for a session's recorded
    answers. Read-only — no status flip, no persistence.

    The same answer set always yields the same ``{axes, reasons}``.
    """
    session = _require_owned_session(db, session_id, current_user)
    answers = crud.get_diagnostic_answers(session)
    axes, reasons = score(answers_from_dict(answers))
    return schemas.DiagnosticResultOut(axes=axes, reasons=reasons)


@router.post("/apply", response_model=schemas.WeaknessProfileOut)
def apply_diagnostic(
    payload: schemas.DiagnosticApplyIn,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Finalise a session into the caller's weakness profile.

    Validation:
    - session must belong to the caller (404 otherwise).
    - session must be ``in_progress`` or ``completed`` — applying an
      already-``applied`` session returns 409 (idempotency guard).

    Side effects: recompute the score from the recorded answers (never
    trust a cached result — an answer might have changed since
    ``/result``), UPSERT it via the existing
    ``crud.upsert_weakness_profile`` helper, then flip the session to
    ``applied`` and stamp ``completed_at``.

    Returns the updated ``WeaknessProfile`` (same shape as
    ``GET /weakness-profile/{user_id}``).
    """
    session = _require_owned_session(db, payload.session_id, current_user)

    if session.status == "applied":
        raise HTTPException(
            status_code=409, detail="session already applied"
        )

    answers = crud.get_diagnostic_answers(session)
    axes, _reasons = score(answers_from_dict(answers))

    profile = crud.upsert_weakness_profile(db, current_user.id, axes)
    crud.mark_diagnostic_session_status(
        db, session, "applied", set_completed_at=True
    )

    return {
        "id": profile.id,
        "user_id": profile.user_id,
        "axes": crud.serialize_weakness_profile_axes(profile),
        "updated_at": profile.updated_at,
    }
