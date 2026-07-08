"""Low-level HTTP transport + exceptions.

One responsibility: turn an Ollama endpoint + JSON payload into a parsed dict,
mapping the three failure modes (daemon down / HTTP error / timeout) onto a
two-class exception taxonomy that drives the fallback-chain behavior.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from ._config import DEFAULT_URL


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
        req = urllib.request.Request(f"{base_url.rstrip('/')}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
        return False


def _normalize_base_url(base_url: str) -> str:
    """Accept either a base URL (``http://host:11434``) or a legacy full endpoint
    URL (``.../api/generate``, ``.../api/chat``, ``.../api/embeddings``) and
    return the base. Lets old callers that pass a full URL keep working after
    the consolidation of the four inline clients.
    """
    url = base_url.rstrip("/")
    for suffix in ("/api/generate", "/api/chat", "/api/embeddings"):
        if url.endswith(suffix):
            url = url[: -len(suffix)]
            break
    return url


def _post(path: str, payload: dict, base_url: str, timeout: float) -> dict:
    """POST JSON to an Ollama endpoint; return parsed JSON.

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
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
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
