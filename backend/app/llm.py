"""OpenRouter chat-completions client for Lexora.

Phase 4 — first real chat-model surface in the lexora backend.
Used by the cloze-exercise generator (``app.cloze.generate_cloze``,
Phase 4.2) to produce a single cloze item from a target word + the
user's weakness axes. Phase 6 will route this client through the
Langfuse wrapper at the call site; Phase 4 keeps the transport
dependency-free so unit tests stay network-free.

The transport is OpenAI-compatible (OpenRouter) — same wire format
as ``app/embeddings.py``. No LangChain. No JSON-mode here: that's
``instructor``'s job (Phase 4.2).

## Build-time model probe result (Phase 4.1, 2026-07-03)

The Phase 4 plan named ``qwen/qwen3-235b-a22b-2507`` as the default
chat model. A probe of the public OpenRouter ``/v1/models`` endpoint
(no auth required) at build time returned::

    Probed 5 candidates against /v1/models (no auth required).
    Total models in catalog: 340
      [OK]      qwen/qwen3-235b-a22b-2507
      [OK]      qwen/qwen3-235b-a22b
      [OK]      openai/gpt-4o-mini
      [BLOCKED] anthropic/claude-3.5-sonnet
      [OK]      openai/gpt-4o
    RESULT: default chat model is 'qwen/qwen3-235b-a22b-2507'.

So the Phase 4 default holds. Note the same ``claude-3.5-sonnet``
block that hit ``baai/bge-m3`` in Phase 1 — the account's
data-policy filter excludes that provider. The model id is wired
through ``OPENROUTER_CHAT_MODEL`` so swapping once the policy is
relaxed (or once the team prefers a different model) is a one-env-var
change, no code change.

The probe is public (no API key needed). A real ``chat/completions``
call would need the key and might still hit the privacy filter at
the inference layer; Phase 4.1 does not call chat, so this probe is
the strongest signal available at the foundation-build stage.

Phase 6.2 (card t_ddaf9cf9) — DSPy adapter extraction.

The ``_DSPyOpenAICompatLM`` adapter was originally private to
``app.cloze`` (Phase 4.2). Phase 6.2 extracts it to ``app.llm`` so
the matching-exercise generator (``app.match``) can share the same
DSPy transport without duplicating the adapter. The class is defined
here and re-exported from ``app.cloze`` for one release so any
out-of-tree caller that imported ``app.cloze._DSPyOpenAICompatLM``
keeps working; new code should import from ``app.llm`` directly.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger(__name__)


# --- Type-level guardrails (Hard rule #11: hard-coded module
# constants, not env-derived). The .env.example has *defaults*
# for the chat model + base URL only; the retry policy lives here
# because changing it should require a code review, not a
# container restart.
MAX_ATTEMPTS = 3
RETRYABLE_STATUS = (408, 425, 429, 500, 502, 503, 504)
BACKOFF_SCHEDULE_S = (0.5, 1.0, 2.0)


@dataclass(frozen=True)
class ChatResult:
    """Structured response from a successful chat completion.

    ``text`` is the assistant message content. ``usage`` carries the
    token-count block from OpenRouter (prompt_tokens,
    completion_tokens, total_tokens) so the caller can attribute
    cost. ``latency_ms`` is wall-clock from the start of the HTTP
    request to receipt of the response — recorded on every attempt,
    so the activity layer can read latency from the final result
    without timing the call again.

    On retryable failure the call raises ``LLMError`` and does not
    produce a ``ChatResult`` — callers that need latency on the
    error path should time the call themselves or use the
    ``respx``-based test fixture to verify the recorded latency.
    """

    text: str
    usage: dict = field(default_factory=dict)
    latency_ms: int = 0
    model: str = ""


class LLMError(RuntimeError):
    """Raised when the OpenRouter chat endpoint fails unrecoverably.

    Bubbles up immediately on non-retryable status (4xx other than
    the ``RETRYABLE_STATUS`` set) and after ``MAX_ATTEMPTS`` on a
    persistent retryable failure.
    """


def _get_api_key() -> str:
    """Resolve the OpenRouter API key from env.

    Same name convention as ``embeddings.py`` — ``OPENROUTER_API_KEY``
    is intentional, not ``*_SECRET_*`` / ``*_TOKEN_*`` (the harness
    redacts those substrings even when the value isn't a secret).
    """
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise LLMError(
            "OPENROUTER_API_KEY is not set. Add it to ~/.lexora/.env "
            "and restart the backend container."
        )
    return key


def _default_base_url() -> str:
    """Env-derived, with the OpenRouter default if unset."""
    return os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")


def _default_model() -> str:
    """Env-derived, with the Phase 4 plan default if unset.

    Default was confirmed available at build time via the public
    ``/v1/models`` probe (see module docstring).
    """
    return os.getenv("OPENROUTER_CHAT_MODEL", "qwen/qwen3-235b-a22b-2507")


def _post_chat(
    client: httpx.Client,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    timeout_s: float,
) -> tuple[int, dict, int]:
    """POST one ``chat/completions`` request with retry.

    Returns ``(status_code, response_json, latency_ms)`` on the final
    attempt. Raises ``LLMError`` on non-retryable status or after
    the retry budget is exhausted.

    The latency is recorded per attempt; the returned value is the
    latency of the *successful* attempt (or the final attempt on
    failure, so the caller can read the wall-clock regardless of
    outcome — useful for the Langfuse trace in Phase 4.3).
    """
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    last_exc: Exception | None = None
    last_status: int | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        start = time.perf_counter()
        try:
            resp = client.post(url, json=payload, headers=headers, timeout=timeout_s)
            elapsed_ms = int((time.perf_counter() - start) * 1000)
        except httpx.HTTPError as exc:
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            last_exc = exc
            if attempt >= MAX_ATTEMPTS:
                break
            backoff = BACKOFF_SCHEDULE_S[attempt - 1]
            logger.warning(
                "complete: attempt %d/%d transport error (%s); retrying in %.1fs",
                attempt,
                MAX_ATTEMPTS,
                exc,
                backoff,
            )
            time.sleep(backoff)
            continue

        if resp.status_code in RETRYABLE_STATUS:
            last_status = resp.status_code
            last_exc = LLMError(
                f"retryable HTTP {resp.status_code}: {resp.text[:200]}"
            )
            if attempt >= MAX_ATTEMPTS:
                break
            backoff = BACKOFF_SCHEDULE_S[attempt - 1]
            logger.warning(
                "complete: attempt %d/%d got HTTP %d; retrying in %.1fs",
                attempt,
                MAX_ATTEMPTS,
                resp.status_code,
                backoff,
            )
            time.sleep(backoff)
            continue

        # Non-retryable — bubble immediately on 4xx other than the
        # retryable set, and surface 5xx that we somehow missed.
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise LLMError(
                f"non-retryable HTTP {resp.status_code}: {resp.text[:200]}"
            ) from exc

        # Success — return JSON body + latency so the caller can
        # parse the assistant message + usage block.
        try:
            return resp.status_code, resp.json(), elapsed_ms
        except Exception as exc:
            raise LLMError(
                f"failed to decode OpenRouter response as JSON: {exc}: "
                f"{resp.text[:200]}"
            ) from exc

    # Out of attempts.
    if last_status is not None:
        raise LLMError(
            f"complete: gave up after {MAX_ATTEMPTS} attempts; "
            f"last HTTP {last_status}: {last_exc}"
        )
    raise LLMError(
        f"complete: gave up after {MAX_ATTEMPTS} attempts; "
        f"last error: {last_exc}"
    )


def complete(
    messages: list[dict],
    *,
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 512,
    timeout_s: float = 30.0,
    base_url: str | None = None,
) -> ChatResult:
    """One-shot chat completion against OpenRouter.

    Parameters
    ----------
    messages
        OpenAI-style message list, e.g.
        ``[{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]``.
        Must be non-empty; an empty list raises ``ValueError`` so
        callers don't silently get back an empty ``text``.
    model, base_url
        Override the env-derived defaults. Useful in tests and for
        one-off probes; production code should rely on the env vars.
    temperature, max_tokens
        Passed straight through to OpenRouter. No clamping — the
        provider rejects out-of-range values.
    timeout_s
        Per-request timeout in seconds. Applies to *each* attempt,
        not the total retry budget. With the default 30s and the
        default 3 attempts + (0.5+1+2)s backoff, the worst-case wall
        clock is ~93s.

    Returns
    -------
    ChatResult
        Carries ``text`` (assistant content), ``usage``
        (prompt_tokens / completion_tokens / total_tokens),
        ``latency_ms`` (last attempt's wall clock), and ``model``
        (the model id that was used).

    Raises
    ------
    ValueError
        If ``messages`` is empty.
    LLMError
        If the API key is missing, the request hits a non-retryable
        HTTP status, or the retry budget is exhausted on transient
        failures.
    """
    if not messages:
        raise ValueError("complete: messages must be a non-empty list")

    api_key = _get_api_key()
    use_model = model or _default_model()
    use_base = base_url or _default_base_url()

    # Reuse one HTTP client for the (single) request — keeps the
    # connection-pool warm and lets the test harness inject a
    # transport via httpx.MockTransport if needed in the future.
    with httpx.Client() as client:
        status, body, latency_ms = _post_chat(
            client,
            use_base,
            api_key,
            use_model,
            messages,
            temperature,
            max_tokens,
            timeout_s,
        )

    # OpenRouter's response shape:
    # {
    #   "id": "...",
    #   "model": "qwen/...",
    #   "choices": [{"index": 0, "message": {"role": "assistant",
    #                                        "content": "..."},
    #                "finish_reason": "stop"}],
    #   "usage": {"prompt_tokens": N, "completion_tokens": M,
    #             "total_tokens": N+M}
    # }
    try:
        choices = body["choices"]
        text = choices[0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError(
            f"unexpected OpenRouter response shape (missing choices[0].message.content): "
            f"{exc}: {str(body)[:300]}"
        ) from exc

    usage = body.get("usage", {}) or {}
    # OpenRouter may not always echo back the exact model id (alias
    # resolution). Prefer the response body's ``model`` if present,
    # fall back to the request-side ``use_model``.
    returned_model = body.get("model") or use_model

    return ChatResult(
        text=text,
        usage={
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "completion_tokens": int(usage.get("completion_tokens", 0)),
            "total_tokens": int(usage.get("total_tokens", 0)),
        },
        latency_ms=latency_ms,
        model=returned_model,
    )


# ---------------------------------------------------------------------------
# Phase 6.2 (card t_ddaf9cf9) — DSPy adapter.
#
# Extracted from ``app.cloze`` so ``app.match`` (Phase 6.2) can share the
# same DSPy transport without duplicating the adapter. The class is
# functionally identical to the Phase 4.2 implementation; only the module
# ownership changed. ``app.cloze`` re-exports the symbol under its old
# name for one release — see the module docstring of ``app.cloze``.
# ---------------------------------------------------------------------------


class _DSPyOpenAICompatLM:
    """Thin DSPy adapter that routes through ``app.llm.complete``.

    DSPy 3.x has a built-in ``dspy.LM`` for OpenAI-compatible
    endpoints, but it imports the ``openai`` SDK directly and bypasses
    our retry + latency-recording wrapper. Using a hand-written
    adapter lets us keep every chat call going through
    ``app.llm.complete`` (Hard rule #4 + #5: "every LLM call goes
    through app/llm.py").

    We don't subclass ``dspy.BaseLM`` because DSPy 3.x's ``BaseLM``
    enforces a constructor signature and ``__call__`` shape that
    depends on the active DSPy version; a duck-typed adapter that
    ``dspy.Predict`` accepts via ``settings.configure(lm=...)`` is
    more portable across DSPy releases.

    The adapter is only constructed when an OpenRouter key is
    present, so the offline path (``DummyLM``) doesn't pay the
    import cost.
    """

    # DSPy reads ``model`` off the LM instance when it builds
    # ``dspy.Predict`` calls.
    model: str

    def __init__(self) -> None:
        self.model = _default_model()

    def __call__(self, prompt: str | None = None, **kwargs: Any) -> list[str]:
        """DSPy v3.x entry point.

        DSPy calls the LM with either ``prompt=...`` (legacy) or
        ``messages=...`` (newer protocol). We accept both and
        normalise into a messages-shaped ``app.llm.complete`` call.
        Returns a list of strings — one per generation — which is the
        shape DSPy 3.x expects from a custom LM.
        """
        messages = kwargs.get("messages")
        if not messages:
            messages = [{"role": "user", "content": prompt or ""}]
        result = complete(messages=messages)
        # DSPy expects a list of strings (one per sample).
        return [result.text]

    # DSPy 3.x sometimes probes ``basic_request`` directly; provide
    # a passthrough so the optimiser can talk to the LM without
    # knowing about our internal shape.
    def basic_request(self, prompt: str | None = None, **kwargs: Any) -> list[dict]:
        text = self.__call__(prompt=prompt, **kwargs)[0]
        return [{"text": text}]