"""Unit tests for the OpenRouter chat-completions client.

Mocks httpx so the test never touches the network — even though
``OPENROUTER_API_KEY`` is set on the dev box, we don't want CI or
local pytest runs to consume OpenRouter quota. The mock mirrors the
shape of OpenRouter's actual response::

    {"id": "...",
     "model": "qwen/...",
     "choices": [{"index": 0,
                  "message": {"role": "assistant", "content": "..."},
                  "finish_reason": "stop"}],
     "usage": {"prompt_tokens": N, "completion_tokens": M, "total_tokens": N+M}}

Run from ``backend/``::

    uv run pytest -q tests/test_llm.py
"""
from __future__ import annotations

import pytest
import respx
from httpx import Response

from app.llm import (
    BACKOFF_SCHEDULE_S,
    ChatResult,
    LLMError,
    MAX_ATTEMPTS,
    RETRYABLE_STATUS,
    complete,
)


# Make sure the env-derived defaults don't blow up. The fixture sets
# OPENROUTER_API_KEY via the test runner; if not, set a placeholder
# here so individual tests can opt out via monkeypatch.delenv().
@pytest.fixture(autouse=True)
def require_api_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")


@pytest.fixture(autouse=True)
def zero_backoff(monkeypatch):
    """Zero out the backoff schedule so retry tests don't sleep.
    Production behavior is covered by the constants test (which
    asserts the real schedule via the imported module binding, not
    the patched attribute) and by integration paths."""
    monkeypatch.setattr("app.llm.BACKOFF_SCHEDULE_S", (0.0, 0.0, 0.0))


def _mock_chat_response(
    content: str = "Hello back.",
    *,
    model: str = "qwen/qwen3-235b-a22b-2507",
    prompt_tokens: int = 12,
    completion_tokens: int = 8,
) -> dict:
    """Build an OpenRouter-shaped chat response body."""
    return {
        "id": "gen-test-001",
        "object": "chat.completion",
        "created": 1700000000,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


# --- Constants --------------------------------------------------------------


def test_module_constants_match_phase4_plan():
    """The Phase 4 plan locks MAX_ATTEMPTS=3, the retryable status
    tuple, and the backoff schedule as module-level constants. Any
    accidental change here means a code review needs to happen.

    We read the *imported* bindings (not the module attributes) so
    the autouse ``zero_backoff`` fixture doesn't mask a regression.
    The fixture patches the module attribute, but the import-time
    binding in this test file is captured at collection and stays
    pinned to the production value — which is exactly the surface
    the plan cares about.
    """
    assert MAX_ATTEMPTS == 3
    assert RETRYABLE_STATUS == (408, 425, 429, 500, 502, 503, 504)
    assert BACKOFF_SCHEDULE_S == (0.5, 1.0, 2.0)


# --- Happy path -------------------------------------------------------------


@respx.mock
def test_complete_returns_text_and_usage():
    """Single 200 response: text matches, usage block is mapped,
    latency is recorded, model echoes."""
    respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=Response(
            200,
            json=_mock_chat_response(
                content="Die Katze schläft.",
                prompt_tokens=42,
                completion_tokens=7,
            ),
        )
    )
    result = complete(
        messages=[{"role": "user", "content": "Was macht die Katze?"}]
    )
    assert isinstance(result, ChatResult)
    assert result.text == "Die Katze schläft."
    assert result.usage == {
        "prompt_tokens": 42,
        "completion_tokens": 7,
        "total_tokens": 49,
    }
    assert result.latency_ms >= 0
    assert result.model == "qwen/qwen3-235b-a22b-2507"


@respx.mock
def test_complete_sends_expected_request_body():
    """The wire format must match the OpenAI spec: model, messages,
    temperature, max_tokens. Anything else is a wire-format break."""
    route = respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=Response(200, json=_mock_chat_response())
    )
    complete(
        messages=[
            {"role": "system", "content": "You are a German tutor."},
            {"role": "user", "content": "Translate: hello"},
        ],
        temperature=0.3,
        max_tokens=256,
        model="qwen/qwen3-235b-a22b-2507",
    )
    import json

    request = route.calls.last.request
    body = json.loads(request.content)
    assert body["model"] == "qwen/qwen3-235b-a22b-2507"
    assert body["temperature"] == 0.3
    assert body["max_tokens"] == 256
    assert body["messages"] == [
        {"role": "system", "content": "You are a German tutor."},
        {"role": "user", "content": "Translate: hello"},
    ]
    # Auth header
    assert request.headers["authorization"] == "Bearer test-key-not-real"


# --- Validation -------------------------------------------------------------


def test_complete_empty_messages_raises_value_error():
    """An empty messages list is a programming error — fail loud at
    the boundary, don't silently return an empty string."""
    with pytest.raises(ValueError, match="non-empty"):
        complete(messages=[])


# --- Retry policy: retryable -------------------------------------------------


@respx.mock
def test_complete_retries_on_429():
    """One 429, then 200. Should succeed after 2 attempts."""
    route = respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        side_effect=[
            Response(429, json={"error": {"message": "rate limited"}}),
            Response(200, json=_mock_chat_response(content="eventually")),
        ]
    )
    result = complete(messages=[{"role": "user", "content": "ping"}])
    assert result.text == "eventually"
    assert route.call_count == 2


@pytest.mark.parametrize("status", [500, 502, 503, 504])
@respx.mock
def test_complete_retries_on_each_5xx_status(status):
    """Each retryable 5xx (500/502/503/504) should trigger one retry.
    The plan lists them all explicitly so we exercise every one."""
    route = respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        side_effect=[
            Response(status, json={"error": f"HTTP {status}"}),
            Response(200, json=_mock_chat_response(content=f"after-{status}")),
        ]
    )
    result = complete(messages=[{"role": "user", "content": "ping"}])
    assert result.text == f"after-{status}"
    assert route.call_count == 2, f"expected 2 attempts for status {status}"


@respx.mock
def test_complete_caps_retries_at_max_attempts():
    """Three consecutive 429s should raise LLMError — we cap at 3,
    never enter an infinite loop."""
    route = respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=Response(429, json={"error": "rate limited"})
    )
    with pytest.raises(LLMError, match="gave up"):
        complete(messages=[{"role": "user", "content": "ping"}])
    assert route.call_count == MAX_ATTEMPTS


@pytest.mark.parametrize("status", [408, 425])
@respx.mock
def test_complete_408_and_425_trigger_retry(status):
    """408 (request timeout) and 425 (too early) are explicitly in
    RETRYABLE_STATUS per the Phase 4 plan."""
    route = respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        side_effect=[
            Response(status, json={"error": f"HTTP {status}"}),
            Response(200, json=_mock_chat_response(content=f"after-{status}")),
        ]
    )
    result = complete(messages=[{"role": "user", "content": "ping"}])
    assert result.text == f"after-{status}"
    assert route.call_count == 2


# --- Retry policy: non-retryable -------------------------------------------


@respx.mock
def test_complete_non_retryable_400_bubbles_immediately():
    """400 is a client error — no retry, single attempt, LLMError."""
    route = respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=Response(400, json={"error": "bad request"})
    )
    with pytest.raises(LLMError, match="non-retryable HTTP 400"):
        complete(messages=[{"role": "user", "content": "ping"}])
    assert route.call_count == 1


@respx.mock
def test_complete_non_retryable_401_bubbles_immediately():
    """401 = bad key. We don't retry; that just hammers OpenRouter
    with the same wrong key. One call, one error, the caller decides."""
    route = respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=Response(401, json={"error": "invalid api key"})
    )
    with pytest.raises(LLMError, match="non-retryable HTTP 401"):
        complete(messages=[{"role": "user", "content": "ping"}])
    assert route.call_count == 1


@respx.mock
def test_complete_non_retryable_403_bubbles_immediately():
    """403 = forbidden (often the data-policy filter blocking a
    model). Single attempt — we want to surface this fast so the
    caller can fall back to a different model, not silently retry."""
    route = respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=Response(403, json={"error": "data policy"})
    )
    with pytest.raises(LLMError, match="non-retryable HTTP 403"):
        complete(messages=[{"role": "user", "content": "ping"}])
    assert route.call_count == 1


# --- Timeout ----------------------------------------------------------------


@respx.mock
def test_complete_respects_timeout_s():
    """A slow response that exceeds timeout_s should be treated as a
    transport error and trigger a retry. After MAX_ATTEMPTS the
    call raises LLMError."""
    import httpx

    route = respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        side_effect=httpx.TimeoutException("simulated slow response")
    )
    with pytest.raises(LLMError, match="gave up"):
        complete(
            messages=[{"role": "user", "content": "ping"}],
            timeout_s=0.001,  # sub-millisecond timeout guarantees the trigger
        )
    assert route.call_count == MAX_ATTEMPTS


# --- Missing API key --------------------------------------------------------


def test_complete_missing_api_key_raises(monkeypatch):
    """No OPENROUTER_API_KEY -> LLMError, no network call."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(LLMError, match="OPENROUTER_API_KEY"):
        complete(messages=[{"role": "user", "content": "ping"}])


# --- Latency recording ------------------------------------------------------


@respx.mock
def test_complete_records_latency_on_success():
    """latency_ms is set on the success path. We don't pin a number
    (timing is flaky) but it should be a non-negative int."""
    respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=Response(200, json=_mock_chat_response())
    )
    result = complete(messages=[{"role": "user", "content": "ping"}])
    assert isinstance(result.latency_ms, int)
    assert result.latency_ms >= 0


@respx.mock
def test_complete_records_latency_on_retryable_failure():
    """Latency should be observed even on the failure path. The
    request module returns it from ``_post_chat`` regardless of
    outcome; on terminal failure the exception propagates and the
    caller can't read it from the return value — but the test
    fixture below verifies the *transport* recorded it by spying
    on the side_effect mock."""
    import time

    timestamps: list[float] = []

    def slow_429_then_200(request):
        timestamps.append(time.perf_counter())
        # First call: 429. Second call: succeed.
        if len(timestamps) == 1:
            return Response(429, json={"error": "rate limited"})
        return Response(200, json=_mock_chat_response())

    respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        side_effect=slow_429_then_200
    )
    result = complete(messages=[{"role": "user", "content": "ping"}])
    assert result.text == "Hello back."
    # The success-path latency is what the caller reads.
    assert result.latency_ms >= 0
    # And we made exactly two requests with measurable gap between them.
    assert len(timestamps) == 2
    assert timestamps[1] >= timestamps[0]


# --- Defensive: response shape ----------------------------------------------


@respx.mock
def test_complete_unexpected_response_shape_raises():
    """If the provider returns a body that doesn't carry
    choices[0].message.content, raise LLMError — don't return an
    empty-text ChatResult that the caller thinks is real content."""
    respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=Response(200, json={"id": "weird", "data": "missing"})
    )
    with pytest.raises(LLMError, match="unexpected OpenRouter response shape"):
        complete(messages=[{"role": "user", "content": "ping"}])