# vs-soft-allow  — frozen public API: chat / chat_fallback carry up to 9 params
# each (the four ecosystem consumers depend on this exact signature; collapsing
# to a param object would break the contract). Graduated lib, not greenfield.
"""Chat-style completion (``/api/chat``) with a messages list."""

from __future__ import annotations

from pathlib import Path

from ._cache import _cache_key_chat, _cache_path, _prune_cache, _strip_think_tags, _write_cache_text
from ._config import (
    CACHE_MAX_TEMP,
    DEFAULT_GEN_MODEL,
    DEFAULT_TIMEOUT,
    DEFAULT_URL,
    OLLAMA_CACHE_DIR,
)
from ._transport import OllamaRequestError, OllamaUnavailable, _post


def chat(
    messages: list[dict],
    model: str = DEFAULT_GEN_MODEL,
    *,
    temperature: float = 0.2,
    num_predict: int | None = None,
    think: bool = False,
    timeout: float = DEFAULT_TIMEOUT,
    base_url: str = DEFAULT_URL,
    cache: bool = True,
    num_ctx: int | None = None,
) -> str | None:
    """Chat-style completion (``/api/chat``) with a messages list.

    Returns the assistant message content (stripped), or ``None`` if
    empty/unavailable. Cached when ``temperature <= CACHE_MAX_TEMP``. Equivalent
    to :func:`ollama_client.generate` but for callers that need system/user role
    separation. Pass ``num_ctx`` to extend the context window for large message
    bodies (e.g. session compaction).
    """
    cacheable = cache and temperature <= CACHE_MAX_TEMP
    cached_path: Path | None = (
        _cache_path(_cache_key_chat(model, messages, temperature, num_predict, num_ctx))
        if cacheable
        else None
    )
    if cached_path is not None:
        OLLAMA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        if cached_path.exists():
            try:
                return cached_path.read_text(encoding="utf-8")
            except OSError:
                pass

    options: dict = {"temperature": temperature}
    if num_predict is not None:
        options["num_predict"] = num_predict
    if num_ctx is not None:
        options["num_ctx"] = num_ctx
    data = _post(
        "/api/chat",
        {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": think,
            "options": options,
        },
        base_url,
        timeout,
    )
    msg = data.get("message")
    content = msg.get("content", "") if isinstance(msg, dict) else ""
    content = _strip_think_tags(str(content).strip()) or None

    if content and cached_path is not None:
        try:
            _write_cache_text(cached_path, content)
            _prune_cache()
        except OSError:
            pass
    return content


def chat_fallback(
    messages: list[dict],
    models: list[str],
    *,
    temperature: float = 0.2,
    num_predict: int | None = None,
    think: bool = False,
    timeout: float = DEFAULT_TIMEOUT,
    base_url: str = DEFAULT_URL,
    cache: bool = True,
    num_ctx: int | None = None,
) -> tuple[str | None, str | None]:
    """Try each chat model in order; return ``(content, model_used)`` or
    ``(None, None)``.

    ``chat`` is resolved via the **public package attribute** (late import) so
    ``@patch("ollama_client.chat")`` also intercepts this internal call. See
    ``topics/mocking-semantics.md``.
    """
    if not models:
        return None, None
    from ollama_client import chat as _chat  # late binding (mock-patchable)

    for model in models:
        try:
            content = _chat(
                messages,
                model=model,
                temperature=temperature,
                num_predict=num_predict,
                think=think,
                timeout=timeout,
                base_url=base_url,
                cache=cache,
                num_ctx=num_ctx,
            )
        except OllamaRequestError:
            continue  # model-specific failure (load/OOM/404) -> try next model
        except OllamaUnavailable:
            return None, None  # daemon down -> no point trying more models
        if content:
            return content, model
    return None, None
