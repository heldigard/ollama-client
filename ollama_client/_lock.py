"""Opt-in cross-process serialization lock for Ollama calls.

When multiple consumers of ollama-client run concurrently (e.g. a smart-trim
PreCompact hook firing while codeq is mid-summary), two models can be resident
in VRAM at once. On a 16GB card that fragments memory and large / context-hungry
models collapse to empty responses — the same GPU-contention artifact the bench
hit (2026-07-13 Qwythos incident).

This module provides a best-effort file lock (``fcntl.flock``) that, when
enabled, serializes every ``_post`` across processes so only one model is in
VRAM at a time. It is **opt-in** (env ``OLLAMA_SERIALIZE_LOCK=1``): default
behavior is unchanged so existing callers pay no latency and need no config.
The lock is auto-released when the holding process exits (kernel drops the
flock on fd close), so a crash never leaves a stale lock.

Unix-only (``fcntl``). On platforms without ``fcntl`` the lock silently
degrades to a no-op rather than breaking callers.
"""

from __future__ import annotations

import os
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

try:  # Unix
    import fcntl

    _HAVE_FCNTL = True
except ImportError:  # Windows / unsupported
    fcntl = None  # type: ignore[assignment]
    _HAVE_FCNTL = False


def _lock_path() -> Path:
    """Lock file path, overridable via env for tests/isolation. Evaluated at
    call time (not import) so monkeypatching the env works in tests."""
    return Path(
        os.environ.get(
            "OLLAMA_LOCK_FILE",
            str(Path.home() / ".cache" / "ollama-client" / "ollama.lock"),
        )
    )


@contextmanager
def serialize_lock() -> Generator[None, None, None]:
    """Acquire the cross-process Ollama lock if ``OLLAMA_SERIALIZE_LOCK`` is
    set; otherwise yield immediately (no-op, default behavior).

    Usage::

        with serialize_lock():
            data = _post_once(...)

    Exclusive (``LOCK_EX``). Released on exit or process death. No timeout —
    callers block until the holder releases; Ollama calls are seconds, not
    minutes, so this is acceptable for hook-driven consumers.
    """
    if not os.environ.get("OLLAMA_SERIALIZE_LOCK") or not _HAVE_FCNTL:
        yield
        return
    assert fcntl is not None  # _HAVE_FCNTL is True in this branch
    path = _lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_CREAT | os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
