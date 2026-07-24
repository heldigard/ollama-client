# vs-soft-allow  — frozen public API: generate / generate_fallback carry up to 8
# params each because the four ecosystem consumers (codeq, smart-trim,
# prompt-improve, web-research) already depend on this exact signature.
# Collapsing to a param object would break the contract. Graduated lib, not
# greenfield. Mirrors the original flat module's allowance.
"""One-shot text completion (``/api/generate``)."""

from __future__ import annotations

from ._cache import _cache_key, _cache_replay, _cache_store, _strip_think_tags
from ._config import (
    CACHE_MAX_TEMP,
    DEFAULT_GEN_MODEL,
    DEFAULT_TIMEOUT,
    DEFAULT_URL,
)
from ._transport import OllamaRequestError, OllamaUnavailable, _post


def generate(
    prompt: str,
    model: str = DEFAULT_GEN_MODEL,
    *,
    temperature: float = 0.2,
    timeout: float = DEFAULT_TIMEOUT,
    base_url: str = DEFAULT_URL,
    cache: bool = True,
    num_ctx: int | None = None,
) -> str | None:
    """One-shot completion. Returns stripped text, or ``None`` if the model
    returned nothing.

    Raises :class:`~ollama_client.OllamaUnavailable` if the daemon is down
    (callers should catch and fall back). Cached when
    ``temperature <= CACHE_MAX_TEMP``. Pass ``num_ctx`` to extend the context
    window for large prompts (default model ctx is often 8192, which silently
    truncates ~7K-token inputs to empty).
    """
    cached_path, hit = _cache_replay(
        cache and temperature <= CACHE_MAX_TEMP,
        _cache_key(model, prompt, temperature, num_ctx),
    )
    if hit:
        return hit

    options: dict = {"temperature": temperature}
    if num_ctx is not None:
        options["num_ctx"] = num_ctx
    data = _post(
        "/api/generate",
        {"model": model, "prompt": prompt, "stream": False, "options": options},
        base_url,
        timeout,
    )
    out = _strip_think_tags(str(data.get("response", "")).strip()) or None
    _cache_store(cached_path, out)
    return out


def generate_fallback(
    prompt: str,
    models: list[str],
    *,
    temperature: float = 0.2,
    timeout: float = DEFAULT_TIMEOUT,
    base_url: str = DEFAULT_URL,
    cache: bool = True,
    num_ctx: int | None = None,
) -> tuple[str | None, str | None]:
    """Try each model in order; return ``(text, model_used)`` from the first that
    answers. Returns ``(None, None)`` if all fail or the daemon is down.

    ``generate`` is resolved via the **public package attribute** (late import)
    so ``@patch("ollama_client.generate")`` also intercepts this internal call —
    preserving the flat-module mocking semantics the ecosystem tests rely on.
    See ``topics/mocking-semantics.md``.
    """
    if not models:
        return None, None
    # Late import: resolve ollama_client.generate at call time so a test that
    # patches the public attribute is honored (a top-level import would bind a
    # static reference and defeat the patch).
    from ollama_client import generate as _generate

    for model in models:
        try:
            text = _generate(
                prompt,
                model=model,
                temperature=temperature,
                timeout=timeout,
                base_url=base_url,
                cache=cache,
                num_ctx=num_ctx,
            )
        except OllamaRequestError:
            continue  # model-specific failure (load/OOM/404) -> try next model
        except OllamaUnavailable:
            return None, None  # daemon down -> no point trying more models
        if text:
            return text, model
    return None, None
