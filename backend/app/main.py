import os
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from fastapi import FastAPI, Depends, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from typing import Any, List, Optional, Literal
from app import auth, crud, fsrs, models, schemas, anki_builder, bootstrap, retrieval
from app.database import get_db, engine
from app.diagnostic.routes import router as diagnostic_router
from app.embeddings import embed_one, EmbeddingError
from app.observability import _ensure_client, get_langfuse
from app.passwords import hash_password

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("lexora.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: ensure schema (Alembic owns migrations), seed corpus if
    Postgres is empty, and warm the Langfuse client (no tracing yet).
    Phase 4 wires real observability call sites."""
    logger.info("startup: ensuring schema via Base.metadata.create_all")
    models.Base.metadata.create_all(bind=engine)
    logger.info("startup: seeding corpus from SQLite if Postgres is empty")
    bootstrap.seed_corpus()
    logger.info("startup: warming Langfuse client (Phase 4 will use it)")
    _ensure_client()
    yield


app = FastAPI(title="German Vocabulary API", version="0.3.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"message": "German Vocabulary API", "version": "0.1.0"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.get("/words", response_model=schemas.WordListResponse)
def read_words(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    word_types: List[str] = Query(None),
    frequencies: List[str] = Query(None),
    db: Session = Depends(get_db),
):
    return crud.get_words(
        db,
        skip=skip,
        limit=limit,
        word_types=word_types,
        frequencies=frequencies,
    )


@app.get("/words/search", response_model=schemas.WordListResponse)
def search_words(
    q: str = Query(..., min_length=1),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    word_types: List[str] = Query(None),
    frequencies: List[str] = Query(None),
    db: Session = Depends(get_db),
):
    return crud.search_words(
        db,
        query=q,
        skip=skip,
        limit=limit,
        word_types=word_types,
        frequencies=frequencies,
    )


@app.get("/words/filters/options")
def get_filter_options(db: Session = Depends(get_db)):
    return {
        "word_types": crud.get_word_types(db),
        "frequencies": crud.get_frequencies(db),
    }


@app.get("/words/{word_id}", response_model=schemas.WordResponse)
def read_word(word_id: int, db: Session = Depends(get_db)):
    word = crud.get_word(db, word_id=word_id)
    if word is None:
        raise HTTPException(status_code=404, detail="Word not found")
    return word


@app.get("/retrieve")
def retrieve_endpoint(
    query: str = Query(..., min_length=1, description="Text to embed and search for"),
    k: int = Query(10, ge=1, le=100, description="Max items to return per source"),
    source: Literal["words", "examples", "both"] = Query(
        "both", description="Which table(s) to search"
    ),
    db: Session = Depends(get_db),
):
    """Top-k nearest neighbours by cosine distance.

    The query text is embedded on demand via OpenRouter; the result
    is the top-k rows whose precomputed embeddings are closest to
    that vector. Score is 1 - cosine_distance (higher = more similar).

    The endpoint is plumbing: no consumer wires retrieval into a
    prompt yet. Phase 4's exercise generator and Phase 6's RAG
    prompt both depend on this shape.

    On the SQLite dev fallback the endpoint returns 503 — pgvector
    has no analogue on SQLite, and a "no results" response would
    hide the configuration mismatch.
    """
    if not retrieval._is_postgres_target():
        raise HTTPException(
            status_code=503,
            detail=(
                "/retrieve requires Postgres + pgvector. The active "
                "DATABASE_URL points at a non-Postgres backend."
            ),
        )

    started = time.perf_counter()
    try:
        query_vec = embed_one(query)
    except EmbeddingError as exc:
        logger.error("retrieve: embedding failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"embedding provider error: {exc}")
    embed_ms = int((time.perf_counter() - started) * 1000)

    started_q = time.perf_counter()
    items = retrieval.retrieve(db, query_vec, k=k, source=source)
    query_ms = int((time.perf_counter() - started_q) * 1000)
    total_ms = int((time.perf_counter() - started) * 1000)

    # Langfuse trace — first real consumer of the observability
    # wrapper from Phase 0. Skip silently when keys are missing so
    # local dev (no keys) doesn't break the endpoint.
    _trace_retrieval(
        query=query,
        k=k,
        source=source,
        embed_ms=embed_ms,
        query_ms=query_ms,
        total_ms=total_ms,
        result_count=len(items),
    )

    return {
        "query": query,
        "source": source,
        "k": k,
        "result_count": len(items),
        "latency_ms": total_ms,
        "items": items,
    }


def _trace_retrieval(
    *,
    query: str,
    k: int,
    source: str,
    embed_ms: int,
    query_ms: int,
    total_ms: int,
    result_count: int,
) -> None:
    """Emit one Langfuse trace per retrieval call (best-effort).

    Phase 1 fix (t_2e386ba9 / Helena review §7): use the v2 SDK's
    ``client.span(...)`` + ``span.update(...)`` + ``span.end()``
    sequence. The earlier ``client.start_as_current_observation``
    call is a v3-only API; on the v2.60.10 SDK (the floor pinned by
    pyproject.toml to match the v2.95.11 server) the method does
    not exist and the call was silently swallowed by the
    ``except Exception`` below, so every retrieval was untraced.
    The non-context-manager shape is the canonical v2 pattern:
    span() returns a handle, update() merges metadata, end()
    closes the observation, flush() pushes to the ingestion API.
    """
    client = get_langfuse()
    if client is None:
        # Keys missing — already warned at startup. Don't spam per call.
        return
    span = None
    try:
        span = client.span(name="lexora.retrieval")
        span.update(
            metadata={
                "query_text": query,
                "query_len_chars": len(query),
                "k": k,
                "source": source,
                "result_count": result_count,
                # v2 ``span.update`` does not accept a separate
                # ``metrics`` kwarg — latency is encoded as metadata
                # keys with ``_ms`` suffixes so it's filterable in
                # the UI. Phase 4 will wire the v3 generation API
                # for true usage/cost metrics.
                "embed_latency_ms": embed_ms,
                "query_latency_ms": query_ms,
                "total_latency_ms": total_ms,
            },
        )
        span.end()
        # Langfuse buffers traces and flushes asynchronously. Force
        # a flush so the trace is queryable in the UI before the
        # request returns — important for QA validation, less
        # important for steady-state traffic.
        client.flush()
    except Exception as exc:
        # Tracing failures must never break the request. Log and move on.
        logger.warning("retrieve: Langfuse trace failed (non-fatal): %s", exc)
        # Make sure a half-opened span is closed even on failure so
        # we don't leak a background flush task.
        if span is not None:
            try:
                span.end()
            except Exception:
                pass


@app.post("/decks/generate")
def generate_deck(
    word_types: Optional[List[str]] = Query(None),
    frequencies: Optional[List[str]] = Query(None),
    direction: str = Query("both"),
    db: Session = Depends(get_db),
):
    """Generate an Anki deck from filtered words."""
    try:
        filepath = anki_builder.create_anki_deck(
            db,
            word_types=word_types,
            frequencies=frequencies,
            direction=direction,
        )
        filename = filepath.split("/")[-1]
        return {
            "message": "Deck generated successfully",
            "filename": filename,
            "filepath": filepath,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/decks/list")
def list_decks():
    """List all generated decks."""
    decks = []
    if os.path.exists(anki_builder.DECKS_DIR):
        for filename in os.listdir(anki_builder.DECKS_DIR):
            if filename.endswith(".apkg"):
                filepath = os.path.join(anki_builder.DECKS_DIR, filename)
                decks.append(
                    {
                        "filename": filename,
                        "created": os.path.getctime(filepath),
                        "size": os.path.getsize(filepath),
                    }
                )
    return {"decks": sorted(decks, key=lambda x: x["created"], reverse=True)}


# ---------------------------------------------------------------------------
# Phase 2.2 — Auth + gated weakness-profile routes (card t_74c3aa1e)
#
# Replaces the Phase 2.1 routes:
#   - ``POST /users`` (raw ``password_hash``)  →  ``POST /auth/signup`` (bcrypt)
#   - ``GET /users/me`` (501 placeholder)      →  ``GET /auth/me`` (JWT decode)
#   - ``GET/PUT /weakness-profile/{user_id}``  →  same routes, now auth-gated
#
# New surface added in this card:
#   - ``POST /auth/signup`` — bcrypt hash, issue JWT cookie, return
#     ``{access_token, user}``.
#   - ``POST /auth/login``  — verify bcrypt, issue JWT cookie.
#   - ``POST /auth/logout`` — clear the cookie. 204.
#   - ``GET  /auth/me``     — decode cookie or Bearer header, return
#     the ``User`` row.
#
# Existing endpoints (``/words``, ``/decks/generate``, ``/retrieve``,
# etc.) stay untouched and unauthenticated. Only ``/auth/*`` and
# ``/weakness-profile/*`` route through ``get_current_user``.
# ---------------------------------------------------------------------------


def _issue_token_and_respond(
    user: models.User, db: Session
) -> schemas.AuthResponse:
    """Mint a JWT for the user and shape the response body.

    The cookie is set as a side-effect on the FastAPI ``Response``
    by the route — this helper only builds the body so a future
    caller (e.g. a CLI script) can reuse the auth shape without
    needing a Response object.
    """
    token = auth.create_access_token(user.id)
    user_out = schemas.UserOut.model_validate(user)
    return schemas.AuthResponse(access_token=token, user=user_out)


@app.post(
    "/auth/signup",
    response_model=schemas.AuthResponse,
    status_code=201,
)
def signup(
    payload: schemas.SignupRequest,
    response: Response,
    db: Session = Depends(get_db),
):
    """Create a new user, hash the password with bcrypt, issue a
    JWT cookie, return ``{access_token, user}``.

    Duplicate emails return 409 (pre-check + IntegrityError
    fallback, same shape). Email format is validated by
    ``EmailStr`` (Pydantic); password length is validated by the
    field bounds ``[8, 128]``. 422 on either failure.
    """
    # Pre-check the email — fast path. The unique constraint is
    # still the source of truth so two concurrent signups with the
    # same email both surface 409 cleanly.
    if crud.get_user_by_email(db, payload.email) is not None:
        raise HTTPException(status_code=409, detail="email already registered")
    pw_hash = hash_password(payload.password)
    try:
        user = crud.create_user(
            db, email=payload.email, password_hash=pw_hash
        )
    except IntegrityError:
        # Concurrent insert beat us to it. Roll back the in-flight
        # session and surface the same 409 the pre-check would have.
        db.rollback()
        raise HTTPException(status_code=409, detail="email already registered")

    auth.set_auth_cookie(response, auth.create_access_token(user.id))
    return _issue_token_and_respond(user, db)


@app.post(
    "/auth/login",
    response_model=schemas.AuthResponse,
)
def login(
    payload: schemas.LoginRequest,
    response: Response,
    db: Session = Depends(get_db),
):
    """Verify the (email, password) pair and issue a JWT cookie.

    Returns 401 on any failure — the response body does not
    distinguish "no such email" from "wrong password" so a
    username-enumeration probe gets the same shape either way.
    """
    user = crud.authenticate_user(db, payload.email, payload.password)
    if user is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials",
        )
    auth.set_auth_cookie(response, auth.create_access_token(user.id))
    return _issue_token_and_respond(user, db)


@app.post("/auth/logout", status_code=204)
def logout(response: Response):
    """Clear the ``lexora_token`` cookie. Always 204 — logout
    is idempotent (a second logout is a no-op).

    Note: the route MUST return the injected ``response`` itself
    (not a fresh ``Response(status_code=204)``) — FastAPI's
    response-cycle replaces the response object on return, and a
    fresh response would drop the Set-Cookie header that
    ``clear_auth_cookie`` just added.
    """
    auth.clear_auth_cookie(response)
    response.status_code = 204
    return response


@app.get("/auth/me", response_model=schemas.MeOut)
def read_me(
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db),
):
    """Return the authenticated user plus the two fields the
    post-signup first-login gate needs: ``weakness_profile`` and
    ``diagnostic_state``.

    401 if no / invalid / expired token (raised by the dependency,
    same opaque shape for all failure modes — never leaks whether
    the cookie was missing vs. the signature was bad vs. the user
    was deleted).

    Phase 3.3 (card t_ff6fa637) added the two new fields. The
    response shape is non-breaking for clients that only read
    ``id`` / ``email`` — the header, the protected-route gate, and
    the weakness-profile page all keep working without changes.

    Computation:

    - ``weakness_profile``: a single ``GET /weakness-profile``
      lookup. ``None`` if the user has no profile row yet (we
      deliberately do NOT auto-create here — the
      ``/weakness-profile/{user_id}`` route is the only place that
      auto-creates; the ``/auth/me`` payload is a read-only
      projection). The frontend's gate treats ``None`` as empty.
    - ``diagnostic_state``: a single query against
      ``diagnostic_sessions`` for the user's most-recent row.
      ``"never"`` when no row exists, otherwise the literal status
      of the latest row. The status string is already constrained
      to the four expected values by the model layer; we still
      defensively fall back to ``"never"`` on any unexpected value
      so a future bug never reaches the SPA as a 500.
    """
    profile_row = crud.get_weakness_profile(db, current_user.id)
    if profile_row is None:
        weakness_profile_payload = None
    else:
        weakness_profile_payload = {
            "id": profile_row.id,
            "user_id": profile_row.user_id,
            "axes": crud.serialize_weakness_profile_axes(profile_row),
            "updated_at": profile_row.updated_at,
        }

    sessions = crud.list_diagnostic_sessions(db, current_user.id)
    if not sessions:
        diagnostic_state: schemas.DiagnosticState = "never"
    else:
        # ``list_diagnostic_sessions`` returns newest-first; the
        # gate on the client only cares about the most recent
        # session's status, so we read row zero and trust the
        # status string verbatim (``in_progress`` | ``completed``
        # | ``applied``; the model also allows ``skipped`` which
        # the gate treats as "user has decided to set axes
        # manually" — same routing as ``completed`` / ``applied``).
        latest_status = sessions[0].status
        if latest_status in ("in_progress", "completed", "applied"):
            diagnostic_state = latest_status  # type: ignore[assignment]
        else:
            # ``skipped`` (or any future status) → map to
            # ``completed`` so the gate doesn't re-route the user
            # back to the probe they've already opted out of. The
            # literal type is open; the gate only branches on
            # the two "send to diagnostic" values
            # (``never`` / ``in_progress``).
            diagnostic_state = "completed"

    return schemas.MeOut(
        id=current_user.id,
        email=current_user.email,
        created_at=current_user.created_at,
        weakness_profile=weakness_profile_payload,
        diagnostic_state=diagnostic_state,
    )


@app.get(
    "/weakness-profile/{user_id}",
    response_model=schemas.WeaknessProfileOut,
)
def read_weakness_profile(
    user_id: int,
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db),
):
    """Return the weakness profile for a user.

    Phase 2.2 auth: the caller must be authenticated. The JWT
    subject (``current_user.id``) must match ``user_id`` — a
    logged-in user can only read their own profile, and probing
    another user's ``user_id`` returns 403.

    Auto-creates an empty default profile on first read so a
    fresh user always sees a stable response shape. Returns 404
    if the user_id doesn't exist (defence in depth — the
    dependency already verified ``current_user.id`` is a real
    row, but the route guards in case the URL parameter is
    edited by hand).
    """
    if current_user.id != user_id:
        raise HTTPException(
            status_code=403, detail="cannot read another user's profile"
        )
    if crud.get_user_by_id(db, user_id) is None:
        raise HTTPException(status_code=404, detail="user not found")
    profile = crud.get_weakness_profile(db, user_id)
    if profile is None:
        profile = crud.create_empty_weakness_profile(db, user_id)
    return {
        "id": profile.id,
        "user_id": profile.user_id,
        "axes": crud.serialize_weakness_profile_axes(profile),
        "updated_at": profile.updated_at,
    }


@app.put(
    "/weakness-profile/{user_id}",
    response_model=schemas.WeaknessProfileOut,
)
def write_weakness_profile(
    user_id: int,
    payload: schemas.WeaknessProfileUpdate,
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db),
):
    """Upsert the weakness profile for a user.

    Phase 2.2 auth: same as ``GET /weakness-profile/{user_id}``.
    The ``WeaknessProfileUpdate`` Pydantic model already validates
    ``axes`` (each value must be an int in [0, 3]) so a 422
    surfaces on bad input. Returns 404 if the user_id doesn't
    exist (defence in depth — the dependency verified
    ``current_user.id`` is real, but the route guards against
    hand-edited URLs).
    """
    if current_user.id != user_id:
        raise HTTPException(
            status_code=403, detail="cannot write another user's profile"
        )
    if crud.get_user_by_id(db, user_id) is None:
        raise HTTPException(status_code=404, detail="user not found")
    profile = crud.upsert_weakness_profile(db, user_id, payload.axes)
    return {
        "id": profile.id,
        "user_id": profile.user_id,
        "axes": crud.serialize_weakness_profile_axes(profile),
        "updated_at": profile.updated_at,
    }


# ---------------------------------------------------------------------------
# Phase 4.2 — Cloze exercise route (card t_bdd9ffbe)
#
# Single auth-gated endpoint, ``POST /exercises/cloze``. Body: ``{}``
# (word selection is server-driven from the user's weakness profile).
# Response: the ``ClozeExercise`` Pydantic model.
#
# Phase 6.1 widens the request with ``enable_rag``; Phase 7.3 widens
# it with ``collocation`` (card t_bdd6ab24).
#
# Two response branches (Phase 7.3):
#
# - ``collocation=False`` (default) — wraps the standard
#   ``ClozeExerciseOut`` payload as
#   ``{collocation: false, partner_lemma: null, <ClozeExerciseOut
#   fields>...}``. Prompt bytes match Phase 6.1 verbatim (Hard
#   rule H10).
# - ``collocation=True`` — wraps the ``CollocationExerciseOut``
#   payload as
#   ``{collocation: true, partner_lemma: <value>, <CollocationExerciseOut
#   fields>...}``. Prompt template / generator is ``Phase 7.2``'s
#   ``generate_collocation``.
#
# Errors:
# - 401: missing / invalid JWT (raised by ``get_current_user``).
# - 502: LLM transport / provider error or persistent schema
#   violation (``ClozeGenerationError`` /
#   ``CollocationGenerationError``). The body carries the
#   structured fields so an operator can triage without re-running.
# - 500: corpus inconsistency (e.g. ``select_target_word`` raised
#   ``ValueError`` because the mapped ``word_type`` has zero rows).
# ---------------------------------------------------------------------------


@app.post(
    "/exercises/cloze",
    response_model=schemas.ClozeGenerateResponse,
)
def generate_cloze_exercise(
    payload: schemas.ClozeGenerateRequest = schemas.ClozeGenerateRequest(),
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db),
) -> Any:
    """Generate one cloze exercise for the logged-in learner.

    The route is intentionally thin — all logic lives in
    ``app.cloze.generate_cloze`` (standard branch) or
    ``app.collocation.generate_collocation`` (collocation branch).
    We import lazily inside the handler so the module-level
    ``main.py`` stays import-cheap and so the existing test suite
    (which imports ``app.main`` before the cloze module) doesn't
    pay the OpenAI / instructor import cost.

    **Phase 6.1** — the request body is a
    ``ClozeGenerateRequest`` with the two fields
    ``enable_rag: bool = False`` and
    ``collocation: bool = False`` (Phase 7.3). An empty body ``{}``
    parses to the defaults, preserving the Phase 4.5 / 6.1 wire
    contract verbatim — existing callers see no change.

    **Phase 7.3** — when ``collocation=True``, the route picks a
    target word (deterministic via ``select_target_word``) and
    calls ``generate_collocation(db, user_id, target_word_id)``.
    The response payload carries the
    ``CollocationExerciseOut`` shape (Phase 7.2's schema).

    Response (Phase 7.3) — the route returns a
    ``ClozeGenerateResponse`` (single-class wrapper with
    ``extra='allow'`` carrying either the standard cloze data or
    the collocation-cloze data alongside the discriminator
    fields):

    - ``collocation`` — ``True`` on the collocation branch,
      ``False`` on the standard branch
    - ``partner_lemma`` — populated only on the collocation branch
      (``None`` on the standard branch; mirrors
      ``collocations.partner_lemma``)
    - cloze-specific fields (``sentence_with_blank``,
      ``distractors``, etc.) on the standard branch
    - collocation-specific fields (``prompt``, ``partner_register``,
      ``source_corpus``, etc.) on the collocation branch

    Errors:
    - 401: missing / invalid JWT (raised by ``get_current_user``).
    - 422: invalid request body (Pydantic — non-bool
      ``enable_rag`` / ``collocation``, etc.). FastAPI's default
      422 envelope.
    - 502: LLM transport / provider error or persistent schema
      violation (``ClozeGenerationError`` /
      ``CollocationGenerationError``). The body carries the
      structured fields so an operator can triage without
      re-running.
    - 500: corpus inconsistency (e.g. ``select_target_word`` raised
      ``ValueError`` because the mapped ``word_type`` has zero
      rows, or the collocation generator found zero matching
      rows for the picked target word).
    """
    started = time.perf_counter()

    if payload.collocation:
        # Phase 7.3 — collocation branch.
        #
        # The standard cloze flow picks the target word inside
        # ``generate_cloze`` (deterministic seed of ``(user_id, axis,
        # date.today())``). The collocation generator requires an
        # explicit ``target_word_id``, so we replicate the word
        # selection here via ``select_target_word`` — same seed,
        # same outcome as ``generate_cloze`` would have produced.
        #
        # Picked word is then passed to ``generate_collocation``
        # which consumes the collocations table for that target.
        # The collocation table is read-only at this layer (Hard
        # rule #2 — type-level guardrail via omission).
        from app import cloze as _cloze_mod  # lazy, has ``select_target_word``
        from app.collocation import (
            CollocationGenerationError,
            generate_collocation,
            PROMPT_TEMPLATE_VERSION as COLLOCATION_PROMPT_VERSION,
        )
        from app.llm import LLMError

        try:
            target_word = _cloze_mod.select_target_word(
                db, current_user.id
            )
        except ValueError as exc:
            logger.error(
                "cloze(collocation=true): corpus inconsistency on "
                "select_target_word for user_id=%d: %s",
                current_user.id,
                exc,
            )
            raise HTTPException(status_code=500, detail=str(exc))

        try:
            collocation_exercise = generate_collocation(
                db, current_user.id, target_word.id
            )
        except LLMError as exc:
            logger.error(
                "cloze(collocation=true): LLM transport failure for "
                "user_id=%d word_id=%d: %s",
                current_user.id,
                target_word.id,
                exc,
            )
            raise HTTPException(
                status_code=502,
                detail=f"collocation generation failed: {exc}",
            )
        except CollocationGenerationError as exc:
            logger.error(
                "cloze(collocation=true): schema dead-letter for "
                "user_id=%d word_id=%d after %d attempt(s): %s",
                current_user.id,
                target_word.id,
                exc.schema_retry_count,
                exc.last_validation_error,
            )
            raise HTTPException(
                status_code=502,
                detail={
                    "error": "collocation_generation_failed",
                    "schema_retry_count": exc.schema_retry_count,
                    "last_validation_error": exc.last_validation_error,
                },
            )
        except ValueError as exc:
            # Corpus inconsistency on the collocation generator side
            # (no collocation row for the picked target word). 500 —
            # operator needs to extend the seed scripts in 7.1.
            logger.error(
                "cloze(collocation=true): corpus inconsistency for "
                "user_id=%d word_id=%d: %s",
                current_user.id,
                target_word.id,
                exc,
            )
            raise HTTPException(status_code=500, detail=str(exc))

        latency_ms = int((time.perf_counter() - started) * 1000)

        # Lock the prompt_template_version on the way out so a
        # future ``generate_collocation`` that forgets to set it
        # can't desync the contract. Map the generator's
        # ``CollocationExercise`` to the wire-level
        # ``CollocationExerciseOut`` with the Phase 6.1 shared
        # metadata + the Phase 7.3 discriminator fields stamped.
        #
        # The wire output drops the generator-internal
        # ``target_word_id`` (which equals the wire-level
        # ``target_lemma``'s id — we keep the wire field name
        # ``target_word_id`` for cross-exercise-type consumers
        # and stamp it from the picked target word, mirroring
        # the standard branch's behaviour on line 684).
        collocation_payload = schemas.CollocationExerciseOut(
            # Collocation-specific fields
            target_lemma=target_word.word,
            target_translation_en=(
                (target_word.translations or "").split(",")[0].strip()
                if getattr(target_word, "translations", None)
                else ""
            ),
            prompt=collocation_exercise.prompt,
            partner_lemma=collocation_exercise.partner_lemma,
            partner_register=collocation_exercise.partner_register,
            source_corpus=collocation_exercise.source_corpus,
            rationale=collocation_exercise.rationale,
            retrieval_chunks=[],
            # Shared metadata (Phase 6.1)
            exercise_type="cloze",
            target_word_id=target_word.id,
            prompt_template_version=COLLOCATION_PROMPT_VERSION,
            enable_rag=False,  # collocation branch doesn't honour enable_rag
            trace_id=None,  # Phase 4.3 hook returns None
            latency_ms=latency_ms,
        )

        # Merge into the ``ClozeGenerateResponse`` wrapper. We
        # serialise the inner payload via ``model_dump`` and merge
        # with the discriminator keys so the wire is a single
        # flat object — easier for the SPA to consume than nested
        # under a ``payload`` key.
        response_dict = collocation_payload.model_dump()
        response_dict["collocation"] = True
        response_dict["partner_lemma"] = (
            collocation_exercise.partner_lemma
        )
        # ``CollocationExerciseOut`` also has a ``partner_lemma``
        # field already; Pydantic would double-set it. Drop the
        # duplicate (the wrapper's top-level key wins).
        return response_dict

    # ----- Phase 6.1 standard branch (collocation=False default) -----
    from app.cloze import (
        ClozeGenerationError,
        generate_cloze,
        PROMPT_TEMPLATE_VERSION,
    )
    from app.llm import LLMError

    started = time.perf_counter()
    try:
        exercise = generate_cloze(
            db, current_user.id, enable_rag=payload.enable_rag
        )
    except LLMError as exc:
        logger.error(
            "cloze: LLM transport failure for user_id=%d: %s",
            current_user.id,
            exc,
        )
        raise HTTPException(
            status_code=502,
            detail=f"cloze generation failed: {exc}",
        )
    except ClozeGenerationError as exc:
        logger.error(
            "cloze: schema dead-letter for user_id=%d after %d attempt(s): %s",
            current_user.id,
            exc.schema_retry_count,
            exc.last_validation_error,
        )
        raise HTTPException(
            status_code=502,
            detail={
                "error": "cloze_generation_failed",
                "schema_retry_count": exc.schema_retry_count,
                "last_validation_error": exc.last_validation_error,
            },
        )
    except ValueError as exc:
        # Corpus inconsistency (e.g. axis-mapped word_type has zero
        # rows). 500 — operator needs to look at the seed data.
        logger.error("cloze: corpus inconsistency for user_id=%d: %s",
                     current_user.id, exc)
        raise HTTPException(status_code=500, detail=str(exc))

    latency_ms = int((time.perf_counter() - started) * 1000)

    # Lock the prompt_template_version on the way out so a future
    # ``generate_cloze`` that forgets to set it can't desync the
    # contract. Map the generator's ``ClozeExercise`` to the
    # wire-level ``ClozeExerciseOut`` with the Phase 6.1 shared
    # fields stamped + Phase 7.3's discriminator echoes.
    cloze_payload = schemas.ClozeExerciseOut(
        # Generator fields
        sentence_with_blank=exercise.sentence_with_blank,
        answer_word_id=exercise.answer_word_id,
        distractors=exercise.distractors,
        difficulty=exercise.difficulty,
        rationale=exercise.rationale,
        # Shared metadata (Phase 6.1)
        exercise_type="cloze",
        target_word_id=exercise.answer_word_id,
        prompt_template_version=PROMPT_TEMPLATE_VERSION,
        enable_rag=payload.enable_rag,
        trace_id=None,  # Phase 4.3 hook returns None; 6.x widens to a real id
        latency_ms=latency_ms,
    )

    # Phase 7.3 — wrapper-stamp the standard branch payload with
    # the collocation=False discriminator keys. We merge with the
    # inner payload so the SPA sees a single flat object; the new
    # echo fields appear at the top level next to the existing
    # cloze fields.
    response_dict = cloze_payload.model_dump()
    response_dict["collocation"] = False
    response_dict["partner_lemma"] = None
    return response_dict


# ---------------------------------------------------------------------------
# Phase 6.3 — ``POST /exercises/match`` (card t_39d85400)
#
# Wire surface for the matching generator (Phase 6.2's
# ``app.match.generate_match``). The route is intentionally thin:
# body validation, generator call, error translation, response
# stamping. All prompt + RAG + Langfuse logic lives in
# ``app.match``.
#
# Wire contract:
#   - 401: missing / invalid JWT (raised by ``get_current_user``).
#   - 422: Pydantic validation error (FastAPI default for the
#     ``MatchGenerateRequest`` body — ``count`` out of [2, 8] lands
#     here, NOT in our handler).
#   - 502: LLM transport failure (``LLMError``) or persistent schema
#     violation (``MatchingGenerationError``). The body carries the
#     structured fields so an operator can triage without re-running.
#   - 500: corpus inconsistency (e.g. ``select_target_word`` raised
#     ``ValueError`` because the mapped ``word_type`` has zero rows).
#   - 200 + ``MatchingExerciseOut`` otherwise.
#
# Hard rule surface (from card body):
#   - #1 RAG-on is opt-in (read from body, default ``False``).
#   - #2 ``/retrieve`` is consumed as-is (the generator owns the
#     retrieval helper, not the route).
#   - #3 ``exercise_type`` discriminator is ``Literal["matching"]``
#     — stamped at response time by the schema default.
#   - #5 every state-mutating call is traced — ``generate_match``
#     owns the ``_trace_match`` call; the route doesn't add a second
#     wrapper.
# ---------------------------------------------------------------------------


@app.post("/exercises/match", response_model=schemas.MatchingExerciseOut)
def generate_match_exercise(
    payload: schemas.MatchGenerateRequest,
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db),
) -> Any:
    """Generate one matching exercise for the logged-in learner.

    Body: ``MatchGenerateRequest`` — ``count`` (default 4, in
    [2, 8]) and ``enable_rag`` (default ``False``). The server
    picks the target word via ``select_target_word`` and builds the
    rest.

    The route mirrors ``POST /exercises/cloze``: thin handler,
    imports ``app.match`` lazily so the module-level ``main.py``
    stays import-cheap, locks ``prompt_template_version`` on the
    way out, and stamps a server-minted ``exercise_id`` so the same
    id re-appears on the future ``grade_logs`` row for Ragas join
    determinism.
    """
    import os
    import time

    from app.match import (
        MatchingGenerationError,
        PROMPT_TEMPLATE_VERSION,
        generate_match,
    )
    from app.llm import LLMError

    _match_started = time.perf_counter()
    try:
        exercise = generate_match(
            db,
            current_user.id,
            count=payload.count,
            enable_rag=payload.enable_rag,
        )
    except LLMError as exc:
        logger.error(
            "match: LLM transport failure for user_id=%d "
            "count=%d enable_rag=%s: %s",
            current_user.id,
            payload.count,
            payload.enable_rag,
            exc,
        )
        raise HTTPException(
            status_code=502,
            detail=f"match generation failed: {exc}",
        )
    except MatchingGenerationError as exc:
        # The trace is already recorded by ``generate_match`` before
        # it raises; we don't call ``_trace_match`` a second time.
        # ``exercise`` is unbound in this branch (the exception fired
        # before the assignment); log the trace metadata the dead-letter
        # carries instead of guessing the target.
        logger.error(
            "match: schema dead-letter for user_id=%d count=%d "
            "enable_rag=%s after %d attempt(s): %s",
            current_user.id,
            payload.count,
            payload.enable_rag,
            exc.schema_retry_count,
            exc.last_validation_error,
        )
        raise HTTPException(
            status_code=502,
            detail={
                "error": "match_generation_failed",
                "schema_retry_count": exc.schema_retry_count,
                "last_validation_error": exc.last_validation_error,
            },
        )
    except ValueError as exc:
        # Corpus inconsistency (e.g. axis-mapped word_type has zero
        # rows, or ``force_word_id`` doesn't resolve). 500 — operator
        # needs to look at the seed data.
        logger.error(
            "match: corpus inconsistency for user_id=%d: %s",
            current_user.id,
            exc,
        )
        raise HTTPException(status_code=500, detail=str(exc))

    # Stamp the wire-only fields: ``exercise_id`` (server-minted per
    # generation, mirrors Phase 6.6's plan to re-use it on the
    # ``grade_logs`` row) and ``exercise_type`` (the
    # ``BaseExerciseFields`` default already sets this to
    # ``"matching"``). Lock ``prompt_template_version`` on the way
    # out so a future ``generate_match`` that forgets to set it
    # can't desync the contract.
    #
    # Round-trip through ``model_dump`` / ``model_validate`` so a
    # generator-side ``app.match.MatchingPair`` (the nested-model
    # class the instructor layer validates against) and a
    # wire-side ``app.schemas.MatchingPair`` (the one FastAPI's
    # ``response_model`` advertises) survive a ``del sys.modules``
    # test that re-imports ``app.match`` and creates a fresh class
    # object. Without the round-trip, the nested-model revalidation
    # would reject the wire build with "Input should be a valid
    # dictionary or instance of MatchingPair" when the cached
    # ``app.schemas.MatchingPair`` class identity drifts.
    exercise_id = int.from_bytes(os.urandom(8), "big", signed=True)
    match_latency_ms = int((time.perf_counter() - _match_started) * 1000)
    return schemas.MatchingExerciseOut.model_validate(
        {
            "exercise_id": exercise_id,
            "target_word_id": exercise.target_word_id,
            "prompt_template_version": PROMPT_TEMPLATE_VERSION,
            "enable_rag": payload.enable_rag,
            "trace_id": None,
            "latency_ms": match_latency_ms,
            "pairs": [p.model_dump() for p in exercise.pairs],
        }
    )


# ---------------------------------------------------------------------------
# Phase 6.5 — ``POST /exercises/comprehension`` (card t_dba4a40c)
#
# Wire surface for the comprehension generator (Phase 6.4's
# ``app.comprehension.generate_comprehension``). The route is
# intentionally thin: body validation, generator call, error
# translation, response stamping. All prompt + RAG + Langfuse
# logic lives in ``app.comprehension``.
#
# Wire contract:
#   - 401: missing / invalid JWT (raised by ``get_current_user``).
#   - 422: Pydantic validation error (FastAPI default for the
#     ``ComprehensionGenerateRequest`` body — malformed
#     ``enable_rag`` types land here, NOT in our handler). The
#     comprehension request is a single optional ``enable_rag``
#     bool, so 422 is rare; the matching route's count-bounds
#     422s do not apply here.
#   - 502: LLM transport failure (``LLMError``) or persistent schema
#     violation (``ComprehensionGenerationError``). The body
#     carries the structured fields so an operator can triage
#     without re-running.
#   - 500: corpus inconsistency (e.g. ``select_target_word`` raised
#     ``ValueError`` because the mapped ``word_type`` has zero
#     rows).
#   - 200 + ``ComprehensionExerciseOut`` otherwise.
#
# Hard rule surface (from card body):
#   - #1 RAG-on is opt-in (read from body, default ``False``).
#   - #2 ``/retrieve`` is consumed as-is (the generator owns the
#     retrieval helper, not the route).
#   - #3 ``exercise_type`` discriminator is ``Literal["comprehension"]``
#     — stamped at response time by the schema default.
#   - #5 every state-mutating call is traced — ``generate_comprehension``
#     owns the ``_trace_comprehension`` call; the route doesn't add
#     a second wrapper.
#   - #12 Existing callers stay byte-for-byte unchanged. The
#     generator module is imported lazily inside the handler so the
#     module-level ``main.py`` stays import-cheap; we never
#     ``from app.comprehension import ...`` at the top of the file.
# ---------------------------------------------------------------------------


@app.post(
    "/exercises/comprehension",
    response_model=schemas.ComprehensionExerciseOut,
)
def generate_comprehension_exercise(
    payload: schemas.ComprehensionGenerateRequest,
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db),
) -> Any:
    """Generate one comprehension exercise for the logged-in learner.

    Body: ``ComprehensionGenerateRequest`` — ``enable_rag`` (default
    ``False``). The server picks the target word via
    ``select_target_word`` and builds the rest. There is no
    ``count`` knob — comprehension generates one passage + one
    question per call (mirrors cloze, not matching).

    The route mirrors ``POST /exercises/match`` and
    ``POST /exercises/cloze``: thin handler, imports
    ``app.comprehension`` lazily so the module-level ``main.py``
    stays import-cheap, locks ``prompt_template_version`` on the
    way out, and stamps a server-minted ``exercise_id`` so the same
    id re-appears on the future ``grade_logs`` row for Ragas join
    determinism (Phase 6.6 + 6.7 follow-up).
    """
    import os
    import time

    from app.comprehension import (
        ComprehensionGenerationError,
        PROMPT_TEMPLATE_VERSION,
        generate_comprehension,
    )
    from app.llm import LLMError

    _compr_started = time.perf_counter()
    try:
        exercise = generate_comprehension(
            db,
            current_user.id,
            enable_rag=payload.enable_rag,
        )
    except LLMError as exc:
        logger.error(
            "comprehension: LLM transport failure for user_id=%d "
            "enable_rag=%s: %s",
            current_user.id,
            payload.enable_rag,
            exc,
        )
        raise HTTPException(
            status_code=502,
            detail=f"comprehension generation failed: {exc}",
        )
    except ComprehensionGenerationError as exc:
        # The trace is already recorded by ``generate_comprehension``
        # before it raises; we don't call ``_trace_comprehension`` a
        # second time. ``exercise`` is unbound in this branch (the
        # exception fired before the assignment); log the trace
        # metadata the dead-letter carries instead of guessing the
        # target.
        logger.error(
            "comprehension: schema dead-letter for user_id=%d "
            "enable_rag=%s after %d attempt(s): %s",
            current_user.id,
            payload.enable_rag,
            exc.schema_retry_count,
            exc.last_validation_error,
        )
        raise HTTPException(
            status_code=502,
            detail={
                "error": "comprehension_generation_failed",
                "schema_retry_count": exc.schema_retry_count,
                "last_validation_error": exc.last_validation_error,
            },
        )
    except ValueError as exc:
        # Corpus inconsistency (e.g. axis-mapped word_type has zero
        # rows, or ``force_word_id`` doesn't resolve). 500 — operator
        # needs to look at the seed data.
        logger.error(
            "comprehension: corpus inconsistency for user_id=%d: %s",
            current_user.id,
            exc,
        )
        raise HTTPException(status_code=500, detail=str(exc))

    # Stamp the wire-only fields: ``exercise_id`` (server-minted per
    # generation, mirrors Phase 6.6's plan to re-use it on the
    # ``grade_logs`` row) and ``exercise_type`` (the
    # ``BaseExerciseFields`` default already sets this to
    # ``"comprehension"``). Lock ``prompt_template_version`` on the
    # way out so a future ``generate_comprehension`` that forgets to
    # set it can't desync the contract.
    exercise_id = int.from_bytes(os.urandom(8), "big", signed=True)
    compr_latency_ms = int((time.perf_counter() - _compr_started) * 1000)
    return schemas.ComprehensionExerciseOut.model_validate(
        {
            "exercise_id": exercise_id,
            "target_word_id": exercise.target_word_id,
            "prompt_template_version": PROMPT_TEMPLATE_VERSION,
            "enable_rag": payload.enable_rag,
            "trace_id": None,
            "latency_ms": compr_latency_ms,
            "passage": exercise.passage,
            "question": exercise.question,
            "choices": exercise.choices,
            "correct_choice": exercise.correct_choice,
            "rationale": exercise.rationale,
        }
    )


# ---------------------------------------------------------------------------
# Phase 5.4 — ``GET /exercises/due`` (card t_e8548d6d)
#
# The closed study loop's right half. Given an authenticated user,
# return the next cloze they're due to review (FSRS-driven) OR a fresh
# word they've never seen (first-encounter warm-up).
#
# Wire contract:
#   - Auth: 401 if no / invalid JWT (raised by ``get_current_user``).
#   - 204 No Content if no ``fsrs_cards`` rows are due AND no fresh
#     words exist in the corpus (the user has nothing to study; the
#     frontend shows an honest empty state).
#   - 200 + ``ClozeDueExerciseOut`` (the cloze payload plus a
#     ``due_from_fsrs: bool`` discriminator) otherwise.
#   - 502 if the cloze LLM call fails (transport / persistent schema
#     violation); 500 on corpus inconsistency (the due-queue picked
#     a word id that's no longer in the words table).
#
# Single-user dev assumption:
#   Phase 5's ``fsrs_cards`` table has no ``user_id`` column, so the
#   query is global — any due card in the system is returned to any
#   authenticated user. Phase 6 may add the column for per-user
#   scoping. The model assumes a single-user dev environment; the
#   route is documented as such.
#
# Two branches:
#   1. **FSRS-driven (due_from_fsrs=True).** Pick the row from
#      ``fsrs_cards`` whose ``due_date <= now()`` and that has the
#      earliest due_date. The picked word has a row in fsrs_cards
#      (we just queried it) — we don't need to create one. Pass its
#      word_id to ``generate_cloze(force_word_id=...)`` so the cloze
#      is built for THIS word, not a fresh selection.
#   2. **First-encounter (due_from_fsrs=False).** No card is due.
#      Pick the lowest-id ``Word`` row that has NO ``fsrs_cards`` row
#      yet — i.e. the user has never seen this word. Create a fresh
#      Learning row inline (state=1, due=now) so the next
#      ``POST /exercises/grade`` has a row to update. Pass its
#      word_id to ``generate_cloze(force_word_id=...)``.
#
# The route is intentionally thin — all word selection + cloze
# generation logic lives in ``app.cloze``. The route is responsible
# only for the due-queue SQL + the inline Learning-row creation.
# ---------------------------------------------------------------------------


@app.get("/exercises/due", response_model=schemas.ClozeDueExerciseOut)
def get_due_exercise(
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db),
) -> Any:
    """Return the next cloze the authenticated user is due to review.

    See the section header above for the full wire contract. The
    two-branch logic is documented inline below.
    """
    from app.cloze import (
        ClozeGenerationError,
        generate_cloze,
        PROMPT_TEMPLATE_VERSION,
    )
    from app.llm import LLMError

    # Phase 6.1 — start the wall-clock for ``latency_ms`` on the
    # response. The route wraps the FSRS pick + the LLM call.
    _due_started = time.perf_counter()
    now = datetime.utcnow()

    # Branch 1: FSRS-driven. Pick the earliest-due fsrs_cards row.
    # The query is intentionally global (no user_id filter) per the
    # single-user dev assumption documented in the section header.
    # Tie-break on ``id`` ASC so the deterministic ordering survives
    # any clock skew between request time and ``due_date``.
    due_row = (
        db.query(models.FsrsCard)
        .filter(models.FsrsCard.due_date <= now)
        .order_by(
            models.FsrsCard.due_date.asc(),
            models.FsrsCard.id.asc(),
        )
        .first()
    )
    if due_row is not None:
        picked_word_id: int = due_row.word_id
        due_from_fsrs = True
    else:
        # Branch 2: First-encounter. Pick a corpus word that has no
        # fsrs_cards row yet. The ``NOT IN`` subquery is the simplest
        # portable shape; both SQLite and Postgres support it. We use
        # ``select(...)`` explicitly (SQLAlchemy 2.x's preferred
        # form) rather than passing the subquery object directly to
        # ``.in_()`` — the latter emits a SAWarning and is deprecated.
        graded_word_ids_subq = (
            select(models.FsrsCard.word_id)
            .where(models.FsrsCard.word_id.isnot(None))
            .scalar_subquery()
        )
        fresh_word = (
            db.query(models.Word)
            .filter(~models.Word.id.in_(graded_word_ids_subq))
            .order_by(models.Word.id.asc())
            .first()
        )
        if fresh_word is None:
            # No due cards AND no fresh words in the corpus. The user
            # has literally nothing to study. 204 keeps the contract
            # clean — the frontend's "All caught up" empty state is
            # the right surface, not a fake exercise.
            return Response(status_code=204)

        picked_word_id = fresh_word.id
        # Inline-create a fresh Learning row so the next
        # ``POST /exercises/grade`` (Phase 5.3) has a row to update.
        # ``state=1`` is py-fsrs's ``State.Learning`` (1/2/3 in the
        # 4.1.2 enum); ``due_date=now`` makes the card immediately
        # due so a same-session grade works. All other numeric
        # columns start at 0 / None — py-fsrs will populate them on
        # the first ``review_card`` call.
        new_card = models.FsrsCard(
            word_id=picked_word_id,
            difficulty=None,
            stability=None,
            retrievability=None,
            due_date=now,
            last_review=None,
            reps=0,
            lapses=0,
            state=1,
            elapsed_days=0,
            scheduled_days=0,
        )
        db.add(new_card)
        try:
            db.commit()
        except IntegrityError:
            # Concurrent insert beat us to it — another request just
            # created an fsrs_cards row for this word. The unique
            # constraint (Phase 5.2's ``UNIQUE(word_id)``) fires here.
            # Roll back and re-query: the row exists now, so the
            # *next* /exercises/due call will see it via Branch 1.
            # For THIS call, we still want to give the user a cloze
            # for the picked word — keep due_from_fsrs=False because
            # semantically this is still a first-encounter from the
            # user's perspective. The fresh Learning row exists now
            # regardless of who won the race.
            db.rollback()
            logger.info(
                "exercises/due: concurrent fsrs_cards insert for "
                "word_id=%d; rolling back the inline create",
                picked_word_id,
            )
        due_from_fsrs = False

    # Generate the cloze for the picked word. ``force_word_id`` is
    # the 5.4-only knob on ``generate_cloze``; the deterministic
    # seed path stays untouched for the 4.5 ``POST /exercises/cloze``
    # caller.
    try:
        exercise = generate_cloze(
            db, current_user.id, force_word_id=picked_word_id
        )
    except LLMError as exc:
        logger.error(
            "exercises/due: LLM transport failure for user_id=%d: %s",
            current_user.id,
            exc,
        )
        raise HTTPException(
            status_code=502,
            detail=f"cloze generation failed: {exc}",
        )
    except ClozeGenerationError as exc:
        logger.error(
            "exercises/due: schema dead-letter for user_id=%d after %d "
            "attempt(s): %s",
            current_user.id,
            exc.schema_retry_count,
            exc.last_validation_error,
        )
        raise HTTPException(
            status_code=502,
            detail={
                "error": "cloze_generation_failed",
                "schema_retry_count": exc.schema_retry_count,
                "last_validation_error": exc.last_validation_error,
            },
        )
    except ValueError as exc:
        # Corpus inconsistency — the picked word_id is no longer in
        # the words table. The route picked it from fsrs_cards (a
        # join that should never go stale) but defensive: surface
        # 500 so the operator notices.
        logger.error(
            "exercises/due: corpus inconsistency for user_id=%d "
            "word_id=%d: %s",
            current_user.id,
            picked_word_id,
            exc,
        )
        raise HTTPException(status_code=500, detail=str(exc))

    # Lock the prompt_template_version (same as POST /exercises/cloze)
    # and attach the due_from_fsrs discriminator plus the Phase 6.1
    # shared metadata fields. FastAPI's
    # ``response_model=schemas.ClozeDueExerciseOut`` validates the
    # returned dict against the merged Pydantic schema.
    #
    # Note: the due-queue surface (5.4) is non-RAG by default — it
    # doesn't pass ``enable_rag=True`` to ``generate_cloze`` (the
    # due-queue picks the word for the user, not the user picking
    # "I want RAG on"). The new RAG flag stays default-False.
    latency_ms_due = int((time.perf_counter() - _due_started) * 1000)
    return {
        **exercise.model_dump(),
        "exercise_type": "cloze",
        "target_word_id": exercise.answer_word_id,
        "enable_rag": False,
        "trace_id": None,
        "latency_ms": latency_ms_due,
        "due_from_fsrs": due_from_fsrs,
    }


# ---------------------------------------------------------------------------
# Phase 5.3 / 6.6 — ``POST /exercises/grade`` (cards t_5160eecf, t_d11d0011)
#
# The closed study loop's left half. The cloze UI (Phase 4.5), the
# due-queue endpoint (Phase 5.4), and the Phase 6 matching /
# comprehension UIs all call this route to record a grade and
# persist the post-FSRS state.
#
# Phase 6.6 widens the route from a cloze-only handler to a 3-way
# fan-out. The cloze handler (``_grade_cloze``) keeps its
# byte-for-byte logic from Phase 5.3 — only the dispatch wrapper
# was added. ``_grade_matching`` and ``_grade_comprehension`` are
# sibling wrappers around the same ``apply_grade`` + ``grade_logs``
# write path. The fan-out is per-exercise-type at the trace-span
# level (cloze keeps ``exercise.grade``, matching uses
# ``match.grade``, comprehension uses ``comprehension.grade``) —
# the FSRS scheduling math is exercise-type-agnostic.
#
# Wire contract:
#   - Auth: 401 if no / invalid JWT (raised by ``get_current_user``).
#   - 200 + ``GradeResponse`` JSON on success. The response carries the
#     next due date, the post-review card state, the FSRS scalars
#     (stability, difficulty), and the Langfuse ``trace_id`` (or
#     ``None`` when keys are unset).
#   - 422 on Pydantic validation failure (out-of-range grade, bad
#     exercise_type, non-positive exercise_id) — FastAPI default
#     handler. The 3-way ``Literal["cloze", "matching",
#     "comprehension"]`` is the wire-level gate; ``"speaking"`` /
#     ``"CLOZE"`` / empty string all reject at the schema layer.
#   - 500 on DB integrity failure (e.g. concurrent insert on the
#     fsrs_cards unique constraint), with a structured error body.
#
# Branching:
#   - The request body carries only ``grade``; the ``word_id`` is
#     derived from ``exercise_id`` (for the cloze kind in Phase 5,
#     the cloze's ``answer_word_id`` is the exercise id — the wire
#     schema docstring in ``schemas.GradeRequest`` documents this).
#     For matching / comprehension, the same derivation holds:
#     the FSRS card row keys on ``word_id``, and the exercise
#     type only changes the trace span name and the
#     ``grade_logs.exercise_type`` label.
#   - The ``fsrs_cards`` row is looked up by ``word_id``. If no row
#     exists (first encounter), a fresh Learning row is created
#     inline (matching the ``/exercises/due`` first-encounter
#     shape).
#   - ``apply_grade`` (5.1) computes the post-review Card; the four
#     columns py-fsrs 4.1.2 doesn't carry (``reps``, ``lapses``,
#     ``elapsed_days``, ``scheduled_days``) are filled in from the
#     row's pre-review state + the post-review snapshot.
#
# Hard rules enforced here:
#   1. py-fsrs only (no inline scheduling). All scheduling via
#      ``apply_grade``.
#   2. Closed 3-way exercise-type literal (cloze / matching /
#      comprehension) — the route trusts the schema and does not
#      re-validate.
#   4. Every grade is traced via ``_trace_grade``; ``trace_id``
#      propagates to the ``grade_logs`` row. Span name is
#      per-exercise-type (``exercise.grade`` for cloze,
#      ``match.grade`` for matching, ``comprehension.grade`` for
#      comprehension).
#   5. Pydantic v2 validates input; the handler does no extra
#      validation beyond what the schema enforces.
# ---------------------------------------------------------------------------


@app.post("/exercises/grade", response_model=schemas.GradeResponse)
def grade_exercise(
    payload: schemas.GradeRequest,
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db),
) -> Any:
    """Apply a 1-4 FSRS grade to the user's card for the requested
    exercise type.

    Phase 6.6 fan-out: the route dispatches on
    ``payload.exercise_type`` via a ``match`` statement to the
    per-type handler. Each handler is a thin wrapper around
    ``_grade_one`` that pins the trace span name and the
    ``grade_logs.exercise_type`` label — the FSRS scheduling path
    is shared. The cloze handler (``_grade_cloze``) is the same
    code that 5.3 shipped, lifted verbatim.
    """
    match payload.exercise_type:
        case "cloze":
            return _grade_cloze(db, current_user, payload)
        case "matching":
            return _grade_matching(db, current_user, payload)
        case "comprehension":
            return _grade_comprehension(db, current_user, payload)
        case _:  # pragma: no cover — schema gate rejects this
            # Defensive fallback. The ``GradeRequest`` schema's
            # 3-way ``ExerciseType`` literal should have rejected
            # anything outside the union upstream; if we reach
            # here, the schema gate has been bypassed and we want
            # a clean 422 rather than a 500.
            raise HTTPException(
                status_code=422,
                detail=f"unsupported exercise_type: {payload.exercise_type}",
            )


def _grade_cloze(
    db: Session,
    current_user: models.User,
    payload: schemas.GradeRequest,
) -> schemas.GradeResponse:
    """Phase 5.3's cloze-grade logic, lifted byte-for-byte into a
    per-type handler.

    Trace span name: ``exercise.grade`` (unchanged from 5.3).
    ``grade_logs.exercise_type`` is ``"cloze"``.
    """
    return _grade_one(
        db=db,
        current_user=current_user,
        payload=payload,
        span_name="exercise.grade",
    )


def _grade_matching(
    db: Session,
    current_user: models.User,
    payload: schemas.GradeRequest,
) -> schemas.GradeResponse:
    """Phase 6.6 matching-grade handler.

    Sibling wrapper around ``_grade_one``. Trace span name:
    ``match.grade``. ``grade_logs.exercise_type`` is
    ``"matching"``. The FSRS scheduling path is shared with
    ``_grade_cloze`` and ``_grade_comprehension`` — only the
    span name and the audit-row label differ.
    """
    return _grade_one(
        db=db,
        current_user=current_user,
        payload=payload,
        span_name="match.grade",
    )


def _grade_comprehension(
    db: Session,
    current_user: models.User,
    payload: schemas.GradeRequest,
) -> schemas.GradeResponse:
    """Phase 6.6 comprehension-grade handler.

    Sibling wrapper around ``_grade_one``. Trace span name:
    ``comprehension.grade``. ``grade_logs.exercise_type`` is
    ``"comprehension"``. The FSRS scheduling path is shared
    with ``_grade_cloze`` and ``_grade_matching`` — only the
    span name and the audit-row label differ.
    """
    return _grade_one(
        db=db,
        current_user=current_user,
        payload=payload,
        span_name="comprehension.grade",
    )


def _grade_one(
    *,
    db: Session,
    current_user: models.User,
    payload: schemas.GradeRequest,
    span_name: str,
) -> schemas.GradeResponse:
    """Apply a 1-4 FSRS grade to the user's card for the requested
    exercise.

    This is the shared body that all three per-type handlers
    (``_grade_cloze`` / ``_grade_matching`` / ``_grade_comprehension``)
    route through. The Phase 5.3 cloze-only body is preserved
    byte-for-byte — the per-type handlers exist only to pin the
    ``span_name`` and (via ``payload.exercise_type``) the
    ``grade_logs.exercise_type`` label.

    See the section header above for the full wire contract.
    """
    started = time.perf_counter()
    word_id: int = payload.exercise_id  # cloze: answer_word_id == exercise_id

    # ------------------------------------------------------------------
    # Step 1: Look up (or create) the fsrs_cards row.
    # ------------------------------------------------------------------
    card_row = (
        db.query(models.FsrsCard)
        .filter(models.FsrsCard.word_id == word_id)
        .first()
    )

    # ``prev_due_at`` for the audit row. For a fresh row it's
    # ``datetime.utcnow()`` (the moment the inline row is created);
    # for an existing row it's the row's current ``due_date``. The
    # audit row uses this for interval-delta observability.
    if card_row is None:
        # First-encounter: create a fresh Learning row inline. The
        # shape mirrors the /exercises/due first-encounter branch.
        # ``prev_due_at`` is the moment of the grade — there's no
        # prior schedule.
        now_for_audit = datetime.utcnow()
        prev_due_at: datetime = now_for_audit
        # Initial counters: 0 for everything; py-fsrs will fill
        # ``stability`` / ``difficulty`` / ``last_review`` on the
        # first ``review_card`` call.
        card_row = models.FsrsCard(
            word_id=word_id,
            difficulty=None,
            stability=None,
            retrievability=None,
            due_date=now_for_audit,
            last_review=None,
            reps=0,
            lapses=0,
            state=1,  # py-fsrs State.Learning
            elapsed_days=0,
            scheduled_days=0,
        )
        db.add(card_row)
        try:
            db.flush()  # surface the IntegrityError here, not on commit
        except IntegrityError:
            # Concurrent insert beat us to it — re-query and reuse
            # the winner's row.
            db.rollback()
            card_row = (
                db.query(models.FsrsCard)
                .filter(models.FsrsCard.word_id == word_id)
                .first()
            )
            if card_row is None:
                # Extremely unlikely: the row vanished between the
                # failed insert and the re-query. Surface 500.
                logger.error(
                    "grade: fsrs_cards row vanished for word_id=%d after "
                    "concurrent-insert race",
                    word_id,
                )
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error": "fsrs_card_vanished",
                        "word_id": word_id,
                    },
                )
            # We lost the race — but we have a valid card_row now.
            # Recompute prev_due_at from the existing row.
            prev_due_at = card_row.due_date or datetime.utcnow()
    else:
        prev_due_at = card_row.due_date or datetime.utcnow()

    # ------------------------------------------------------------------
    # Step 2: Run the FSRS review.
    #
    # The ``fsrs_cards`` row stores ``last_review`` / ``due_date`` as
    # naive UTC (because ``datetime.utcnow()`` is the project's
    # convention and SQLAlchemy drops tzinfo on round-trip). py-fsrs
    # 4.1.2 compares ``review_datetime`` (tz-aware UTC) against
    # ``card.last_review`` and crashes on naive-vs-aware. We
    # normalize the card's timestamps to tz-aware UTC in place before
    # the library call so the comparison stays well-defined.
    # ------------------------------------------------------------------
    card = fsrs.row_to_card(card_row)
    if card.last_review is not None and card.last_review.tzinfo is None:
        card.last_review = card.last_review.replace(tzinfo=timezone.utc)
    if card.due is not None and card.due.tzinfo is None:
        card.due = card.due.replace(tzinfo=timezone.utc)
    try:
        new_card, _review_log = fsrs.apply_grade(card, payload.grade)
    except ValueError as exc:
        # ``apply_grade`` raises ValueError on out-of-range grades.
        # The schema's ``Literal[1,2,3,4]`` should have caught this
        # upstream; this branch is a defence-in-depth fallback.
        logger.error(
            "grade: apply_grade rejected payload for user_id=%d word_id=%d "
            "grade=%d: %s",
            current_user.id, word_id, payload.grade, exc,
        )
        raise HTTPException(status_code=422, detail=str(exc))

    # ------------------------------------------------------------------
    # Step 3: Compute the four columns py-fsrs 4.1.2 doesn't carry.
    #
    # ``reps`` / ``lapses`` are scalar counters: increment from the
    # pre-review row. ``scheduled_days`` / ``elapsed_days`` are
    # deltas derived from the post-review ``card.due`` and
    # ``card.last_review``.
    # ------------------------------------------------------------------
    prev_reps: int = int(card_row.reps or 0)
    prev_lapses: int = int(card_row.lapses or 0)
    new_reps: int = prev_reps + 1
    new_lapses: int = prev_lapses + (1 if payload.grade == 1 else 0)

    # ``scheduled_days`` — the gap from ``last_review`` to the next
    # due. Both fields are tz-aware UTC after a review.
    if new_card.last_review is not None and new_card.due is not None:
        new_scheduled_days: int = (
            new_card.due - new_card.last_review
        ).days
    else:
        new_scheduled_days = 0

    # ``elapsed_days`` — the gap from the prior due to the current
    # review. For a fresh card there's no prior schedule, so it's 0.
    # ``card_row.last_review`` is naive UTC (DB convention) and
    # ``new_card.last_review`` is tz-aware UTC (py-fsrs convention);
    # normalize both sides so the subtraction stays well-defined.
    if (
        card_row.last_review is not None
        and new_card.last_review is not None
    ):
        prev_last_review = card_row.last_review
        if prev_last_review.tzinfo is None:
            prev_last_review = prev_last_review.replace(tzinfo=timezone.utc)
        new_elapsed_days: int = (
            new_card.last_review - prev_last_review
        ).days
    else:
        new_elapsed_days = 0

    # ------------------------------------------------------------------
    # Step 4: UPDATE the fsrs_cards row in place.
    #
    # Gotcha #4: every field ``card_to_row_dict`` returns must be
    # included in the UPDATE so the Python object and the row agree.
    # We splat ``card_to_row_dict`` then overwrite the four columns
    # 5.1 leaves as ``None`` with our computed values.
    # ------------------------------------------------------------------
    row_dict = fsrs.card_to_row_dict(new_card, word_id)
    row_dict["reps"] = new_reps
    row_dict["lapses"] = new_lapses
    row_dict["elapsed_days"] = new_elapsed_days
    row_dict["scheduled_days"] = new_scheduled_days

    for col, val in row_dict.items():
        setattr(card_row, col, val)

    # ------------------------------------------------------------------
    # Step 5: Build the metadata payload + trace the grade.
    # ------------------------------------------------------------------
    latency_ms = int((time.perf_counter() - started) * 1000)

    # The ``trace_id`` placeholder is filled in by ``_trace_grade``
    # AFTER the span is opened. We assemble the metadata first
    # (without ``trace_id``) so the span sees the same shape the
    # audit row sees.
    metadata: dict[str, Any] = {
        "user_id": current_user.id,
        "exercise_id": payload.exercise_id,
        "exercise_type": payload.exercise_type,
        "word_id": word_id,
        "grade": payload.grade,
        "scheduled_next_due_at": new_card.due,
        "prev_due_at": prev_due_at,
        "state": int(new_card.state),
        "stability": new_card.stability,
        "difficulty": new_card.difficulty,
        "reps": new_reps,
        "lapses": new_lapses,
        "trace_id": None,  # filled by _trace_grade
        "latency_ms": latency_ms,
    }

    trace_id = _trace_grade(
        metadata=metadata,
        latency_ms=latency_ms,
        grade=payload.grade,
        exercise_id=payload.exercise_id,
        span_name=span_name,
    )

    # ------------------------------------------------------------------
    # Step 6: INSERT the grade_logs row.
    # ------------------------------------------------------------------
    log_row = models.GradeLog(
        user_id=current_user.id,
        exercise_id=payload.exercise_id,
        exercise_type=payload.exercise_type,
        word_id=word_id,
        grade=payload.grade,
        scheduled_next_due_at=new_card.due,
        prev_due_at=prev_due_at,
        state=int(new_card.state),
        stability=float(new_card.stability or 0.0),
        difficulty=float(new_card.difficulty or 0.0),
        reps=new_reps,
        lapses=new_lapses,
        trace_id=trace_id,
        latency_ms=latency_ms,
    )
    db.add(log_row)

    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        logger.error(
            "grade: DB integrity failure for user_id=%d word_id=%d grade=%d: %s",
            current_user.id, word_id, payload.grade, exc,
        )
        raise HTTPException(
            status_code=500,
            detail={
                "error": "grade_log_insert_failed",
                "user_id": current_user.id,
                "word_id": word_id,
                "grade": payload.grade,
            },
        )

    return schemas.GradeResponse(
        exercise_id=payload.exercise_id,
        exercise_type=payload.exercise_type,
        next_due_at=new_card.due,
        card_state=int(new_card.state),
        stability=float(new_card.stability or 0.0),
        difficulty=float(new_card.difficulty or 0.0),
        trace_id=trace_id,
    )


def _trace_grade(
    *,
    metadata: dict[str, Any],
    latency_ms: int,
    grade: int,
    exercise_id: int,
    span_name: str = "exercise.grade",
) -> str | None:
    """Emit one Langfuse span per grade request.

    Phase 6.6 widens the span name from a fixed
    ``"exercise.grade"`` to a per-exercise-type value
    (``exercise.grade`` for cloze, ``match.grade`` for matching,
    ``comprehension.grade`` for comprehension). The default
    keeps the Phase 5.3 cloze path byte-identical for callers
    that don't pass the new argument.

    Mirrors ``_trace_retrieval`` and ``_trace_cloze``'s shape: a
    v2 SDK ``client.span(...)`` + ``span.update(metadata=...)`` +
    ``span.end()`` + ``client.flush()`` sequence. The
    ``start_as_current_span`` context-manager API referenced in
    the card body is a v3-only method — we follow the v2 pattern
    that 4.3 (cloze) and Phase 1 (retrieval) both use.

    **SDK version note.** The v2 SDK exposes the span id via
    ``span.id`` on the handle returned by ``client.span(...)``.
    The card body sketch returned ``span.id``; we keep that
    choice. If a future SDK upgrade changes the attribute name,
    this function is the single edit point.

    **Graceful degradation.** When ``LANGFUSE_PUBLIC_KEY`` /
    ``LANGFUSE_SECRET_KEY`` are missing, ``get_langfuse()`` returns
    ``None``. We early-return ``None`` and the grade_logs row is
    inserted with ``trace_id=None`` — observability is best-effort,
    never blocking.

    **Failure mode.** Any exception raised by the Langfuse SDK is
    caught and logged at WARNING — the grade has already succeeded
    at this point, so a trace failure must never break the request.

    Returns:
        ``str``: the Langfuse span id (the ``grade_logs.trace_id``
            value when Langfuse keys are set).
        ``None``: when Langfuse keys are unset, or when the SDK
            raises an exception (we log the error and continue).
    """
    client = get_langfuse()
    if client is None:
        # Keys missing — already warned at startup. Don't spam per call.
        return None

    span = None
    try:
        span = client.span(name=span_name)
        span.update(
            # The ``input`` is the minimum discriminator pair (grade +
            # exercise_id); the full metadata contract lives in
            # ``metadata`` so a Langfuse UI filter on any field
            # surfaces the matching spans.
            input={"grade": grade, "exercise_id": exercise_id},
            metadata=metadata,
        )
        # Fill the ``trace_id`` key in the metadata dict IN PLACE so
        # the caller can read it back. We do this before ``end()``
        # so the span record carries its own id in its metadata.
        span_id = getattr(span, "id", None) or getattr(
            span, "span_id", None
        )
        metadata["trace_id"] = span_id
        span.end()
        # Force a flush so the trace is queryable in the UI before
        # the request returns — mirrors _trace_retrieval / _trace_cloze.
        client.flush()
        return span_id
    except Exception as exc:  # noqa: BLE001 — tracing must never break the activity
        logger.warning(
            "grade: Langfuse trace failed (non-fatal): %s", exc
        )
        if span is not None:
            try:
                span.end()
            except Exception:
                pass
        return None


# ---------------------------------------------------------------------------
# Phase 3.1 — Diagnostic probe router (card t_41d85c32)
#
# Four auth-gated routes mounted at the ``/diagnostic`` prefix:
#   - POST /diagnostic/start   — create / resume a session, return the
#                                stripped question bank.
#   - POST /diagnostic/answer  — record one (question, choice) answer.
#   - GET  /diagnostic/result  — recompute the deterministic score.
#   - POST /diagnostic/apply   — UPSERT the score into the caller's
#                                WeaknessProfile (existing helper).
#
# Handlers live in ``app.diagnostic.routes``. NO LLM call, NO Langfuse
# tracing — the probe is fully deterministic. Existing endpoints stay
# untouched and unauthenticated.
# ---------------------------------------------------------------------------

app.include_router(
    diagnostic_router, prefix="/diagnostic", tags=["diagnostic"]
)

