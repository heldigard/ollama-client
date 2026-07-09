"""Versioned public API (SemVer).

Consumers gate with :func:`require` to fail fast on version drift instead of a
cryptic mid-run error. Mirrors ``cheap_llm``.
"""

from __future__ import annotations

import re

__version__ = "1.1.0"


def _parse_version(v: str) -> tuple[int, ...]:
    """Best-effort dotted-integer parse (ignores pre-release suffixes)."""
    parts: list[int] = []
    for p in v.split("."):
        match = re.search(r"\d+", p)
        parts.append(int(match.group(0)) if match else 0)
    return tuple(parts)


def require(min_version: str | None = None) -> str:
    """Return ``__version__``, raising ``RuntimeError`` if older than ``min_version``.

    Consumers call ``ollama_client.require("1.0.0")`` at import time so a stale
    install fails immediately with a clear message instead of a mid-run
    ``AttributeError`` on a feature added in a newer version.
    """
    if min_version and _parse_version(__version__) < _parse_version(min_version):
        raise RuntimeError(
            f"ollama_client {__version__} is older than required {min_version}. "
            f"Upgrade the ollama-client project (~/ollama-client/)."
        )
    return __version__
