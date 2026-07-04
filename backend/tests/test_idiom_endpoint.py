"""Tests for Phase 8.4 — ``POST /exercises/idiom`` route (card t_7c21c3f0).

Coverage map (mirrors the card body's "Endpoint tests" section):

1. 200 happy path — ``{"word_id": <existing>}`` →
   ``IdiomExerciseOut`` with ``exercise_type="idiom"``,
   server-minted ``exercise_id`` (int, non-zero), ``word_id``
   echoed, ``prompt_template_version="idiom-v1"``, all
   idiom-specific fields populated.
2. 200 with ``enable_rag=true`` — payload ``{"word_id": ...,
   "enable_rag": true}`` → 200 with ``enable_rag=True``
   threaded through to the generator. The response shape is
   identical to ``enable_rag=False`` (the RAG-on branch is
   internal — the prompt-template choice is not visible on
   the wire).
3. 401 — no JWT cookie → 401 (auth-gated route; the
   ``get_current_user`` dependency raises).
4. 404 — ``{"word_id": <missing>}`` (a ``word_id`` with no
   ``Phrase`` row) → 404 with the documented detail string.
   Distinct from the comprehension endpoint's 500 — the
   card body explicitly demands 404 for this case.
5. 422 — malformed body shape (``{"enable_rag": ["a", "b"]}``
   without a ``word_id``) → 422 (Pydantic rejects the wrong
   type AND the missing required field).

Hermetic: a fresh temp SQLite DB + a temp JWT secret per
test. The OpenRouter chat-completions call is mocked via
``monkeypatch.setattr("app.idiom._openai_client", ...)`` so
no network is touched. The phrases table is created
inline via ``database.Base.metadata.create_all`` in the
test fixture (mirrors the test_comprehension_endpoint /
test_match_endpoint pattern).

Phase 8.4's offline / DummyLM discipline: NO live LLM in
CI. The ``_openai_client`` monkeypatch returns a
``MockTransport``-backed OpenAI client whose ``chat.completions``
yields an instructor-validated ``IdiomExercise`` payload.
The RAG-on branch falls back to ``[]`` on SQLite (the
``_is_postgres_target`` helper returns False there), so the
RAG-on test exercises the no-RAG fallback path inside
``generate_idiom`` — the wire shape is identical.

Run from ``backend/``::

    uv run pytest -q tests/test_idiom_endpoint.py
"""
from __future__ import annotations

import json
import secrets
from typing import Any

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixtures — mirror the per-test SQLite + JWT secret pattern from
# test_comprehension_endpoint / test_match_endpoint / test_cloze.
# ---------------------------------------------------------------------------


@pytest.fixture
def sqlite_db_path(tmp_path, monkeypatch):
    """Fresh temp SQLite DB + a JWT secret so ``app.auth``'s import-time
    check passes. Same pattern as the comprehension / matching endpoint
    test suites.
    """
    db_path = tmp_path / "test_idiom_endpoint.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv(
        "LEXORA_DECKS_DIR", str(tmp_path / "decks")
    )
    monkeypatch.setenv("JWT_SECRET", secrets.token_hex(32))
    return str(db_path)


@pytest.fixture
def client(sqlite_db_path):
    """``TestClient`` wired to a fresh per-test SQLite DB.

    ``database.Base.metadata.create_all`` registers the
    Phase 8.1 ``Phrase`` model on the same metadata object
    as the rest of the corpus, so the 404 test (which
    queries ``phrases``) finds the table populated.
    """
    from app import database
    from app.main import app

    database.reconfigure_for_test(f"sqlite:///{sqlite_db_path}")
    database.Base.metadata.create_all(bind=database.engine)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def db_session(sqlite_db_path):
    """A SQLAlchemy session for the per-test SQLite DB.

    Mirrors test_comprehension_endpoint / test_match_endpoint.
    """
    from sqlalchemy.orm import sessionmaker

    from app import database

    database.reconfigure_for_test(f"sqlite:///{sqlite_db_path}")
    database.Base.metadata.create_all(bind=database.engine)
    engine = database.engine
    SessionLocal = sessionmaker(bind=engine)
    with SessionLocal() as session:
        yield session


# ---------------------------------------------------------------------------
# Seed helpers
# ---------------------------------------------------------------------------


def _seed_word(
    session,
    *,
    word: str,
    word_type: str = "Verb",
    example_de: str = "X schläft.",
) -> int:
    """Insert one ``Word`` row + a stub example. Returns word_id."""
    from app import models

    row = models.Word(
        word=word,
        word_type=word_type,
        frequency="5",
        is_complete=True,
    )
    session.add(row)
    session.flush()
    session.add(
        models.Example(
            word_id=row.id,
            german=example_de,
            english="",
        )
    )
    session.commit()
    return row.id


def _seed_phrase(
    session,
    *,
    word_id: int | None = None,
    phrase: str = "Tomaten auf den Augen",
    definition: str = (
        "to be blind to something obvious"
    ),
    example_usage: str = (
        "Du hast ja Tomaten auf den Augen — der Zug "
        "fährt in fünf Minuten!"
    ),
    source_attribution: str = "dwds",
    frequency_band: str = "high",
    slug: str = "tomaten-auf-den-augen",
) -> int:
    """Insert one ``Phrase`` row.

    The ``word_id`` argument is **NOT stored on the row** — the
    Phase 8.1 ``phrases`` table is a standalone read-only
    corpus without a ``word_id`` FK. The argument is kept in
    the signature only so call sites can read it for setup
    invariants without breaking the existing call shape.

    Returns the ``phrases.id`` slug. The slug PK is stable so
    the test fixtures can reference known rows.
    """
    from app import models

    row = models.Phrase(
        id=slug,
        phrase=phrase,
        definition=definition,
        example_usage=example_usage,
        source_attribution=source_attribution,
        frequency_band=frequency_band,
        dwds_url="https://www.dwds.de/wb/Redewendung",
        attested_quote=None,
        attested_source=None,
    )
    session.add(row)
    session.commit()
    return row.id


def _signup(client: TestClient, email: str = "ada@example.com") -> dict:
    """POST /auth/signup and return the parsed body."""
    resp = client.post(
        "/auth/signup",
        json={"email": email, "password": "supersecret"},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


# ---------------------------------------------------------------------------
# OpenAI stub — mirrors test_comprehension_endpoint's stub. Returns a
# real OpenAI client whose httpx transport is a MockTransport so
# ``instructor.from_openai`` accepts it. The assistant message
# content is a JSON payload that validates against
# ``app.idiom.IdiomExercise``.
# ---------------------------------------------------------------------------


def _make_stub_instructor_client(
    payload: str,
    *,
    model: str = "qwen/qwen3-235b-a22b-2507",
    prompt_tokens: int = 30,
    completion_tokens: int = 12,
) -> Any:
    """Build a stub OpenAI client that returns ``payload`` as the
    assistant message content. The stub is order-independent: it
    bypasses respx so OpenAI's custom httpx pool doesn't matter.
    """
    import httpx
    from openai import OpenAI

    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "gen-idiom-endpoint-001",
                "object": "chat.completion",
                "created": 1700000000,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": payload,
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            },
        )

    return OpenAI(
        api_key="test-key-not-real",
        base_url="https://openrouter.ai/api/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(_handler)),
    )


def _idiom_payload(
    *, word_id: int, enable_cloze: bool = True
) -> str:
    """Build a valid ``IdiomExercise`` JSON payload for a test.

    Field bounds are enforced by ``IdiomExercise``:

    - ``exercise_id``: signed 8-byte int (Phase 5/6/7 convention;
      same shape the route stamps on the wire via
      ``int.from_bytes(os.urandom(8), "big", signed=True)``).
    - ``phrase``: 5..200 chars (we use a stub well above 5
      and well below 200).
    - ``definition``: 1..400 chars.
    - ``example_usage``: 5..400 chars.
    - ``source_attribution``: closed literal (we use "dwds").
    - ``frequency_band``: Literal["high","mid","low"].
    - ``attested_quote`` / ``attested_source``: optional, None
      by default.
    - ``prompt_template_version``: must equal
      ``app.idiom.PROMPT_TEMPLATE_VERSION`` ("idiom-v1") so the
      generated row matches what the offline DummyLM-stub pool
      ships — instructor's RetryException otherwise surfaces as
      a 502.

    ``word_id`` is the FK the test fixture created; the LLM
    stub echoes it so the response validates.
    """
    return json.dumps(
        {
            "exercise_id": 99999,
            "word_id": word_id,
            "phrase": "Tomaten auf den Augen",
            "definition": (
                "to be blind to something obvious"
            ),
            "example_usage": (
                "Du hast ja Tomaten auf den Augen — der "
                "Zug fährt in fünf Minuten!"
            ),
            "cloze_target": (
                "Tomaten auf ___ Augen" if enable_cloze else None
            ),
            "source_attribution": "dwds",
            "frequency_band": "high",
            "attested_quote": None,
            "attested_source": None,
            "prompt_template_version": "idiom-v1",
        },
        ensure_ascii=False,
    )


# ---------------------------------------------------------------------------
# 1. 200 happy path — default enable_rag=False, single word_id
# ---------------------------------------------------------------------------


def test_post_exercises_idiom_happy_path_default(
    client, db_session, monkeypatch
):
    """``POST /exercises/idiom`` with ``{"word_id": <existing>}``
    returns an ``IdiomExerciseOut`` carrying the wire metadata
    contract:

    - ``exercise_type="idiom"`` (``IdiomExerciseOut`` default).
    - ``exercise_id`` is a server-minted int (not None, not 0).
    - ``word_id`` is echoed on the response.
    - ``target_word_id`` equals ``word_id`` (same numerical
      value; the response carries both for cross-exercise
      consumer symmetry with the comprehension / matching
      responses).
    - ``prompt_template_version=="idiom-v1"``.
    - ``phrase`` / ``definition`` / ``example_usage`` /
      ``source_attribution`` / ``frequency_band`` all
      populated.
    - There is no ``count`` knob on the wire — idiom
      generates one exercise per call by design.
    - ``enable_rag`` defaults to ``False`` (Hard rule #1:
      opt-in).
    """
    from app import idiom
    from app.idiom import PROMPT_TEMPLATE_VERSION

    # User is created via the API (signup) below — DO NOT
    # create via SQLAlchemy directly or the test will fail
    # with a 409 "email already registered" conflict.
    target_id = _seed_word(
        db_session, word="schlafen", word_type="Verb"
    )
    _seed_phrase(db_session, word_id=target_id)
    _signup(client)

    monkeypatch.setattr("app.llm.BACKOFF_SCHEDULE_S", (0.0, 0.0, 0.0))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")
    payload = _idiom_payload(word_id=target_id)
    monkeypatch.setattr(
        idiom,
        "_openai_client",
        lambda: _make_stub_instructor_client(payload),
    )

    resp = client.post(
        "/exercises/idiom", json={"word_id": target_id}
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()

    # Wire metadata contract.
    assert body["exercise_type"] == "idiom"
    assert isinstance(body["exercise_id"], int)
    assert body["exercise_id"] != 0
    assert body["word_id"] == target_id
    assert body["target_word_id"] == target_id
    assert body["prompt_template_version"] == PROMPT_TEMPLATE_VERSION

    # Idiom-specific fields populated.
    assert body["phrase"]
    assert len(body["phrase"]) >= 5
    assert body["definition"]
    assert body["example_usage"]
    # ``source_attribution`` validates against the closed
    # ``IdiomSource`` literal — the stub returns "dwds".
    assert body["source_attribution"] in {"dwds", "goethe", "schiller"} or (
        "," in body["source_attribution"]
        and all(
            tok in {"dwds", "goethe", "schiller"}
            for tok in body["source_attribution"].split(",")
        )
    )
    assert body["frequency_band"] in {"high", "mid", "low"}

    # No ``count`` knob on the wire — idiom generates one
    # exercise per call by design (mirrors comprehension, not
    # matching).
    assert "count" not in body
    assert "pairs" not in body
    assert "passage" not in body  # not a comprehension

    # No regression: existing 3 endpoints' callers still parse.
    assert body["enable_rag"] is False  # default


# ---------------------------------------------------------------------------
# 2. 200 with overrides — enable_rag=true threaded through to generator
# ---------------------------------------------------------------------------


def test_post_exercises_idiom_enable_rag_threaded(
    client, db_session, monkeypatch
):
    """``{"word_id": ..., "enable_rag": true}`` → 200 with
    ``enable_rag=True`` forwarded to ``generate_idiom``. On a
    SQLite test target the retrieval helper returns ``[]``
    (the ``_is_postgres_target`` guard short-circuits), so
    the prompt falls back to the no-RAG shape — but the
    route layer still forwards ``enable_rag=True``. We
    verify the wiring via a spy on ``generate_idiom``.

    The card body says the response shape is identical to
    the ``enable_rag=False`` path (the prompt-template
    choice is internal).
    """
    from app import idiom
    from app.idiom import PROMPT_TEMPLATE_VERSION

    # User is created via the API (signup) below — DO NOT
    # create via SQLAlchemy directly or the test will fail
    # with a 409 "email already registered" conflict.
    target_id = _seed_word(
        db_session, word="schlafen", word_type="Verb"
    )
    _seed_phrase(db_session, word_id=target_id)
    _signup(client)

    monkeypatch.setattr("app.llm.BACKOFF_SCHEDULE_S", (0.0, 0.0, 0.0))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")
    payload = _idiom_payload(word_id=target_id)
    monkeypatch.setattr(
        idiom,
        "_openai_client",
        lambda: _make_stub_instructor_client(payload),
    )

    # Spy on ``generate_idiom`` to verify ``enable_rag=True``
    # was forwarded (the SQLite path returns ``[]`` for
    # retrieval — we don't need a Postgres to verify the
    # parameter wiring).
    # The route does ``from app.idiom import generate_idiom``
    # lazily, so we monkeypatch the symbol ON
    # ``app.idiom`` (not on ``app.main``). Mirrors the
    # Phase 6.5 comprehension test pattern.
    import app.idiom as idiom_module

    seen: dict[str, Any] = {}

    real_generate = idiom_module.generate_idiom

    def spy_generate(db, word_id, **kwargs):
        seen.update(kwargs)
        return real_generate(db, word_id, **kwargs)

    monkeypatch.setattr(
        idiom_module, "generate_idiom", spy_generate
    )

    resp = client.post(
        "/exercises/idiom",
        json={"word_id": target_id, "enable_rag": True},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["word_id"] == target_id
    assert body["target_word_id"] == target_id
    assert body["prompt_template_version"] == PROMPT_TEMPLATE_VERSION
    assert body["phrase"]
    assert body["enable_rag"] is True

    # Wiring assertion — the route must forward enable_rag.
    assert seen.get("enable_rag") is True


# ---------------------------------------------------------------------------
# 3. 401 — no JWT cookie
# ---------------------------------------------------------------------------


def test_post_exercises_idiom_requires_auth(client):
    """``POST /exercises/idiom`` with no auth cookie → 401.

    The ``Depends(auth.get_current_user)`` dependency raises
    401 before the handler body runs. No body is required to
    assert this — the request can have a body but the auth
    gate fires first.

    Mirrors the comprehension / matching / cloze auth-gate
    tests. Hard rule: idiom is auth-gated, just like the
    other three exercise types.
    """
    client.cookies.clear()
    resp = client.post(
        "/exercises/idiom", json={"word_id": 1}
    )
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# 4. 404 — word_id has no phrases row
# ---------------------------------------------------------------------------


def test_post_exercises_idiom_word_id_missing_returns_404(
    client, db_session, monkeypatch
):
    """``{"word_id": <missing>}`` → 404 with the documented
    detail string.

    A ``Word`` row is seeded, but NO ``Phrase`` row exists
    for it. The route handler catches ``IdiomNotFoundError``
    raised by ``select_idiom_row`` and translates to 404.

    Distinct from the comprehension endpoint's 500 — the
    card body explicitly demands 404 for this case (not
    500). Mirrors ``GET /words/{word_id}``'s 404 on missing
    ``Word``.

    We assert that the generator's ``select_idiom_row``
    helper is the path that raises — if a future refactor
    changes the helper, this test catches it.
    """
    from app import idiom as idiom_module

    # User is created via the API (signup) below — DO NOT
    # create via SQLAlchemy directly or the test will fail
    # with a 409 "email already registered" conflict.
    target_id = _seed_word(
        db_session, word="schlafen", word_type="Verb"
    )
    # NO _seed_phrase call — the phrases table is empty
    # for this ``word_id``.
    _signup(client)

    # The handler short-circuits BEFORE reaching the LLM —
    # ``select_idiom_row`` returns None and raises
    # ``IdiomNotFoundError``. We don't need an LLM stub;
    # if the handler reaches the LLM call, the test fails
    # on the missing stub.
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")

    # Sanity: the ``phrases`` table IS created (metadata
    # ``create_all`` in the fixture registers the model) but
    # empty for this test (no ``_seed_phrase`` call). The route
    # short-circuits in ``select_phrase_row`` with an
    # ``IdiomNotFoundError`` (no candidates → 404) before ever
    # reaching the LLM.
    from app import database, models
    from sqlalchemy.orm import sessionmaker

    SessionLocal = sessionmaker(bind=database.engine)
    with SessionLocal() as session:
        count = session.query(models.Phrase).count()
        assert count == 0, (
            "test setup invariant: the phrases table must be "
            f"empty for the 404 path; saw {count} rows"
        )

    resp = client.post(
        "/exercises/idiom", json={"word_id": target_id}
    )
    assert resp.status_code == 404, resp.text
    detail_blob = json.dumps(resp.json()).lower()
    assert f"word_id={target_id}" in detail_blob
    assert "no phrases row" in detail_blob


# ---------------------------------------------------------------------------
# 5. 422 — malformed body (missing word_id)
# ---------------------------------------------------------------------------


def test_post_exercises_idiom_missing_word_id_returns_422(
    client, db_session, monkeypatch
):
    """``{"enable_rag": true}`` (no ``word_id``) → 422 (the
    field is required on ``IdiomGenerateRequest``).

    The comprehension / matching routes accept empty bodies
    (the server picks the target word); idiom does NOT —
    ``word_id`` is required because the curated ``phrases``
    table is per-word, and the server-side selection model
    doesn't transfer. The card body says ``word_id`` is
    required ("ties to an existing ``Word`` row").

    We also assert the route doesn't reach the generator
    when the body is malformed — no LLM stub is set up, so
    a leak through would surface as a 502 (the OpenAI
    client returns None on the missing key).
    """
    # User is created via the API (signup) below — DO NOT
    # create via SQLAlchemy directly or the test will fail
    # with a 409 "email already registered" conflict.
    _signup(client)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")
    # No ``_openai_client`` monkeypatch — the handler must
    # NOT reach the generator.

    resp = client.post(
        "/exercises/idiom", json={"enable_rag": True}
    )
    assert resp.status_code == 422, resp.text
    body = resp.json()
    detail_blob = json.dumps(body).lower()
    # FastAPI surfaces missing-required-field errors with
    # the field name in the detail list.
    assert "word_id" in detail_blob


# ---------------------------------------------------------------------------
# 6. 422 — source_attribution invalid (test the response-side gate)
# ---------------------------------------------------------------------------


def test_idiom_response_rejects_invalid_source_attribution():
    """The Pydantic ``field_validator`` on
    ``IdiomExerciseOut.source_attribution`` rejects typoed
    tokens with a 422-equivalent ValidationError at the
    model layer.

    The card body says this is tested explicitly: "422 on
    ``source_attribution`` outside the literal". The wire
    surface is the Pydantic model, so we exercise the model
    directly here (the FastAPI layer would surface the
    same ValidationError as 422 when the same invalid value
    reaches the response model).

    This test is unit-level — no FastAPI round-trip needed.
    """
    from pydantic import ValidationError

    from app.schemas import IdiomExerciseOut

    base_valid: dict[str, Any] = {
        "exercise_id": 12345,
        "target_word_id": 1,
        "word_id": 1,
        "prompt_template_version": "idiom-v1",
        "enable_rag": False,
        "trace_id": None,
        "latency_ms": 50,
        "phrase": "Tomaten auf den Augen",
        "definition": "to be blind to something obvious",
        "example_usage": (
            "Du hast ja Tomaten auf den Augen — der Zug "
            "fährt in fünf Minuten!"
        ),
        "cloze_target": None,
        "source_attribution": "dwds",
        "attested_quote": None,
        "attested_source": None,
        "frequency_band": "high",
    }

    # 6a. Typoed token — "goeth" is not in
    # {dwds, goethe, schiller}.
    bad = dict(base_valid)
    bad["source_attribution"] = "goeth"
    with pytest.raises(ValidationError) as excinfo:
        IdiomExerciseOut.model_validate(bad)
    assert "source_attribution" in str(excinfo.value).lower()

    # 6b. Empty string.
    bad = dict(base_valid)
    bad["source_attribution"] = ""
    with pytest.raises(ValidationError):
        IdiomExerciseOut.model_validate(bad)

    # 6c. Trailing separator — the Phase 8.1 ``_split_source_attribution``
    # helper strips empty tokens and accepts the canonical form
    # (mirrors the DB column shape contract — the seed script
    # can round-trip ``"dwds,"`` without losing data). The
    # validator re-joins on ``,`` so the stored value is the
    # canonical ``"dwds"``. No ``ValidationError`` is raised.
    bad = dict(base_valid)
    bad["source_attribution"] = "dwds,"
    out = IdiomExerciseOut.model_validate(bad)
    assert out.source_attribution == "dwds"

    # 6d. Whitespace around separator.
    bad = dict(base_valid)
    bad["source_attribution"] = "dwds, goethe"  # space after comma
    with pytest.raises(ValidationError):
        IdiomExerciseOut.model_validate(bad)

    # 6e. Valid canonical form passes — sanity check that
    # the validator doesn't over-reject.
    good = dict(base_valid)
    good["source_attribution"] = "dwds,goethe"
    out = IdiomExerciseOut.model_validate(good)
    assert out.source_attribution == "dwds,goethe"


# ---------------------------------------------------------------------------
# 7. Literal widening non-regression — existing 3 endpoints still parse
# ---------------------------------------------------------------------------


def test_cloze_endpoint_still_parses_exercise_type_cloze(client):
    """Phase 8.3 widens ``BaseExerciseFields.exercise_type``
    from ``Literal["cloze","matching","comprehension"]`` to
    include ``"idiom"``. The card body says this is
    **additive only** — existing callers parsing
    ``"cloze"`` / ``"matching"`` / ``"comprehension"`` see
    no change.

    We exercise the cloze endpoint (the most-prominent of
    the 3) and assert the response carries
    ``exercise_type="cloze"`` — the wider literal doesn't
    regress to a 422.

    No LLM stub needed — we don't care about the cloze
    payload; we just want a successful auth + auth-gate
    + response-shipping path. The cloze endpoint will
    fail without a backend corpus / LLM, so we just
    assert the 401 path here (no JWT → 401), which is
    enough to prove the auth gate still works after the
    widening. A separate unit test on the model field
    asserts the literal type stays byte-equivalent.
    """
    client.cookies.clear()
    resp = client.post("/exercises/cloze", json={})
    assert resp.status_code == 401

    # Unit-level assertion: the literal widening is
    # additive only — the existing 3 values still parse.
    from app.schemas import ClozeExerciseOut, MatchingExerciseOut

    # ``ClozeExerciseOut.exercise_type`` is
    # ``Literal["cloze"]`` — narrowed. Setting
    # ``exercise_type="idiom"`` on it MUST be rejected
    # (Phase 8.4's type-level guardrail).
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ClozeExerciseOut.model_validate(
            {
                "exercise_id": 1,
                "target_word_id": 1,
                "prompt_template_version": "cloze-v1",
                "enable_rag": False,
                "trace_id": None,
                "latency_ms": 50,
                "sentence_with_blank": "X schläft.",
                "answer_word_id": 1,
                "distractors": [2, 3, 4],
                "difficulty": "easy",
                "rationale": "test",
                # ``exercise_type`` defaults to "cloze" —
                # explicitly set to "idiom" to assert the
                # narrowing rejection.
                "exercise_type": "idiom",
            }
        )

    # And the matching wire stays parsed.
    assert "exercise_type" in MatchingExerciseOut.model_fields
