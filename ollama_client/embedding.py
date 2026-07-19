"""Embedding vectors (``/api/embeddings``, with ``/api/embed`` 404 fallback).

Embeddings are deliberately NOT cached (callers usually need fresh vectors and
manage their own index). See ``topics/cache-policy.md``.
"""

from __future__ import annotations

from ._config import DEFAULT_EMBED_MODEL, DEFAULT_URL, EMBED_TIMEOUT
from ._transport import OllamaRequestError, OllamaUnavailable, _post


def _vector_from_response(data: dict) -> list[float] | None:
    """Parse either legacy ``embedding`` or new-style ``embeddings[0]``."""
    vec = data.get("embedding")
    if isinstance(vec, list) and vec:
        return [float(x) for x in vec]
    embeddings = data.get("embeddings")
    if isinstance(embeddings, list) and embeddings:
        first = embeddings[0]
        if isinstance(first, list) and first:
            return [float(x) for x in first]
    return None


def _embed_once(
    text: str,
    model: str,
    timeout: float,
    base_url: str,
) -> list[float] | None:
    """Single embedding attempt.

    Prefers ``/api/embeddings`` with ``prompt``. On HTTP 404 only (endpoint
    removed/renamed on some Ollama builds), retries once with ``/api/embed``
    and ``input``.

    Returns the vector, or ``None`` if the text is blank, or Ollama returned no
    embedding field.

    Raises:
        OllamaRequestError: if the request failed (HTTP 4xx/5xx/timeout).
        OllamaUnavailable: if the daemon is unreachable.
    """
    if not text.strip():
        return None
    try:
        data = _post(
            "/api/embeddings",
            {"model": model, "prompt": text},
            base_url,
            timeout,
        )
    except OllamaRequestError as exc:
        if exc.status != 404:
            raise
        data = _post(
            "/api/embed",
            {"model": model, "input": text},
            base_url,
            timeout,
        )
    return _vector_from_response(data)


def embed(
    text: str,
    *,
    model: str = DEFAULT_EMBED_MODEL,
    timeout: float = EMBED_TIMEOUT,
    base_url: str = DEFAULT_URL,
) -> list[float] | None:
    """Embedding vector for a single text via ``/api/embeddings``.

    Returns ``None`` if the daemon is down or returned nothing.

    Resilience: a long token-dense chunk (commit hashes, dotted version strings
    like ``v1.5.21.18.19.54``) can exceed the embed model's token context even
    under a char budget — Ollama then answers HTTP 500 and the chunk would be
    skipped. Rather than drop it, retry once with the text halved: a
    half-content embedding still serves recall better than none.

    On HTTP 404 from ``/api/embeddings``, :func:`_embed_once` retries once with
    the newer ``/api/embed`` endpoint (``input`` field).
    """
    try:
        vec = _embed_once(text, model, timeout, base_url)
        if vec is not None:
            return vec
        if len(text) > 512:
            try:
                return _embed_once(text[: len(text) // 2], model, timeout, base_url)
            except OllamaUnavailable:
                pass
    except OllamaRequestError:
        # Request failed (e.g. context limit HTTP 500) -> try once with halved text.
        if len(text) > 512:
            try:
                return _embed_once(text[: len(text) // 2], model, timeout, base_url)
            except OllamaUnavailable:
                pass
    except OllamaUnavailable:
        # Daemon is unreachable -> abort immediately, do not make a redundant call.
        pass
    return None
