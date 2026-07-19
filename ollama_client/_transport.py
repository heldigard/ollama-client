"""Low-level HTTP transport + exceptions.

One responsibility: turn an Ollama endpoint + JSON payload into a parsed dict,
mapping the three failure modes (daemon down / HTTP error / timeout) onto a
two-class exception taxonomy that drives the fallback-chain behavior.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request

from ._config import DEFAULT_URL
from ._lock import serialize_lock


class OllamaUnavailable(RuntimeError):
    """Raised when the local Ollama daemon is unreachable (connection refused,
    DNS, timeout-on-connect) — there is no point trying more models, the daemon
    itself is down or unreachable. Callers walking a fallback chain should abort.
    """


class OllamaRequestError(OllamaUnavailable):
    """Raised when the daemon WAS reachable but a single request failed
    (HTTP 4xx/5xx: model failed to load, OOM, model-not-found) or the request
    timed out (slow cold load). This is model-specific, not daemon-wide —
    callers walking a fallback chain should try the NEXT model, not abort.

    Subclass of :class:`OllamaUnavailable` for backward compatibility: existing
    ``except OllamaUnavailable`` handlers still catch it (preserving prior
    abort-on-any-failure behavior). New handlers can ``except OllamaRequestError``
    first to continue the chain on model-specific failures.
    """

    def __init__(self, status: int, body: str = "") -> None:
        super().__init__(f"HTTP {status}: {body[:200]}" if body else f"HTTP {status}")
        self.status = status
        self.body = body


def is_alive(base_url: str = DEFAULT_URL, timeout: float = 2.0) -> bool:
    """True iff the Ollama daemon answers ``/api/tags`` quickly."""
    try:
        req = urllib.request.Request(f"{_normalize_base_url(base_url)}/api/tags", method="GET")
        # nosemgrep: dynamic-urllib-use-detected
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except (ValueError, urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
        return False


def _normalize_base_url(base_url: str) -> str:
    """Accept either a base URL (``http://host:11434``) or a legacy full endpoint
    URL (``.../api/generate``, ``.../api/chat``, ``.../api/embeddings``,
    ``.../api/embed``, ``.../api/tags``, ``.../api/ps``, ``.../api/show``) and
    return the base. Lets old callers that pass a full URL keep working after
    the consolidation of the four inline clients.
    """
    url = base_url.rstrip("/")
    for suffix in (
        "/api/generate",
        "/api/chat",
        "/api/embeddings",
        "/api/embed",
        "/api/tags",
        "/api/ps",
        "/api/show",
    ):
        if url.endswith(suffix):
            url = url[: -len(suffix)]
            break
    scheme = urllib.parse.urlsplit(url).scheme
    if scheme not in {"http", "https"}:
        raise ValueError(f"unsupported Ollama base_url scheme: {scheme or '<missing>'}")
    return url


_TRANSIENT_RETRY_CODES = frozenset({408, 500, 503, 504})
# One retry: production callers (smart-trim/codeq hooks) are latency-sensitive,
# so we add at most one extra round-trip. A transient drop (GPU contention,
# cold-load timeout, 5xx blip) clears within the backoff window; a genuinely
# broken or missing model raises again on the second attempt and the caller
# falls over to the next model in its chain. See the ollama-bench 2026-07-13
# incident where a champion emptied under GPU contention and was nearly
# dethroned by a false hard-DQ.
_TRANSIENT_RETRIES = 1
_TRANSIENT_BACKOFF_SEC = 3.0


def _post(path: str, payload: dict, base_url: str, timeout: float) -> dict:
    """POST JSON to an Ollama endpoint; return parsed JSON.

    Retries transient failures (HTTP 408/5xx — cold-load timeout, GPU
    contention, server blip) once before raising. Permanent 4xx errors and
    daemon-down (``OllamaUnavailable``) propagate immediately — retrying them
    cannot help. When ``OLLAMA_SERIALIZE_LOCK`` is set, the whole call holds a
    cross-process exclusive lock so concurrent consumers (hooks) don't load two
    models into VRAM at once (opt-in; default off — see ``_lock``).

    Raises:
        OllamaRequestError: daemon reachable but the request failed on every
            attempt (HTTP 4xx/5xx or timeout) — model-specific, try next.
        OllamaUnavailable: daemon unreachable / network down — abort the chain.
    """
    with serialize_lock():
        return _post_with_retry(path, payload, base_url, timeout)


def _post_with_retry(path: str, payload: dict, base_url: str, timeout: float) -> dict:
    """Transient-retry loop, extracted so :func:`_post`'s ``with serialize_lock()``
    does not push this loop past the vertical-slice nesting budget."""
    for attempt in range(_TRANSIENT_RETRIES + 1):
        try:
            return _post_once(path, payload, base_url, timeout)
        except OllamaRequestError as exc:
            if exc.status not in _TRANSIENT_RETRY_CODES or attempt >= _TRANSIENT_RETRIES:
                raise
            time.sleep(_TRANSIENT_BACKOFF_SEC)
    raise RuntimeError("unreachable")  # pragma: no cover


def _post_once(path: str, payload: dict, base_url: str, timeout: float) -> dict:
    """POST JSON to an Ollama endpoint; return parsed JSON (single attempt).

    Raises:
        OllamaRequestError: daemon reachable but this request failed (HTTP 4xx/5xx)
            or the request timed out (slow cold load) — model-specific, try next.
        OllamaUnavailable: daemon unreachable / network down — abort the chain.
    """
    url = f"{_normalize_base_url(base_url)}{path}"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        # nosemgrep: dynamic-urllib-use-detected
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw_body = resp.read()
            try:
                body = raw_body.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise OllamaRequestError(502, "invalid UTF-8 response") from exc
            try:
                parsed = json.loads(body)
            except json.JSONDecodeError as exc:
                raise OllamaRequestError(502, f"invalid JSON response: {body[:200]}") from exc
            if not isinstance(parsed, dict):
                raise OllamaRequestError(502, f"unexpected JSON response: {body[:200]}")
            return parsed
    except urllib.error.HTTPError as exc:
        # Daemon reachable but THIS request failed (load failure, OOM,
        # model-not-found). Model-specific → caller tries the next model.
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except OSError:
            body = ""
        raise OllamaRequestError(exc.code, body) from exc
    except TimeoutError as exc:
        # A timed-out request is almost always a slow model LOAD (daemon IS up
        # and reachable), not a dead daemon — a dead daemon gives an instant
        # URLError (connection refused), never a timeout. Treat as model-specific
        # (HTTP 408 convention) so a fallback chain tries the NEXT (faster) model.
        raise OllamaRequestError(408, f"timeout: {exc}") from exc
    except (urllib.error.URLError, OSError) as exc:
        # Daemon unreachable / network down → no point trying more models.
        raise OllamaUnavailable(str(exc)) from exc


def _get(path: str, base_url: str, timeout: float) -> dict:
    """GET JSON from an Ollama endpoint; return parsed JSON.

    Used by read-only ops (``/api/tags`` model listing). No transient retry —
    these are interactive diagnostics, not hook-critical calls.

    Raises:
        OllamaRequestError: daemon reachable but the request failed (HTTP 4xx/5xx
            or timeout).
        OllamaUnavailable: daemon unreachable / network down.
    """
    url = f"{_normalize_base_url(base_url)}{path}"
    req = urllib.request.Request(url, method="GET")
    try:
        # nosemgrep: dynamic-urllib-use-detected
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw_body = resp.read()
        try:
            parsed = json.loads(raw_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise OllamaRequestError(502, f"invalid JSON response: {raw_body[:200]!r}") from exc
        if not isinstance(parsed, dict):
            raise OllamaRequestError(502, f"unexpected JSON response: {raw_body[:200]!r}")
        return parsed
    except urllib.error.HTTPError as exc:
        raise OllamaRequestError(exc.code, "") from exc
    except TimeoutError as exc:
        raise OllamaRequestError(408, f"timeout: {exc}") from exc
    except (urllib.error.URLError, OSError) as exc:
        raise OllamaUnavailable(str(exc)) from exc
