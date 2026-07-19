"""Model inventory ops (``/api/tags``, ``/api/ps``).

Read-only helpers used by the CLI and library callers that need the installed
or currently-loaded model list without re-implementing GET + filtering.
"""

from __future__ import annotations

from ._config import DEFAULT_URL
from ._transport import _get


def _dict_entries(items: object) -> list[dict]:
    """Return only dict elements from a models/ps array; empty if missing/non-list."""
    if not isinstance(items, list):
        return []
    return [x for x in items if isinstance(x, dict)]


def list_models(base_url: str = DEFAULT_URL, *, timeout: float = 5.0) -> list[dict]:
    """Installed models via ``GET /api/tags`` (``models`` array)."""
    data = _get("/api/tags", base_url, timeout)
    return _dict_entries(data.get("models"))


def list_running(base_url: str = DEFAULT_URL, *, timeout: float = 5.0) -> list[dict]:
    """Currently loaded models via ``GET /api/ps`` (``models`` array)."""
    data = _get("/api/ps", base_url, timeout)
    return _dict_entries(data.get("models"))
