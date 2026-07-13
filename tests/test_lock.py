"""Opt-in cross-process serialize lock (default off; flock when env set)."""

from __future__ import annotations

import fcntl
from unittest.mock import patch

from ollama_client._lock import serialize_lock


def test_serialize_lock_noop_without_env(monkeypatch):
    """Default (no env set): the lock is a no-op — fcntl is never touched, so
    existing callers pay no latency and need no config."""
    monkeypatch.delenv("OLLAMA_SERIALIZE_LOCK", raising=False)
    with patch("ollama_client._lock.fcntl.flock") as mock_flock:
        with serialize_lock():
            pass
    mock_flock.assert_not_called()


def test_serialize_lock_acquired_and_released_with_env(monkeypatch, tmp_path):
    """With OLLAMA_SERIALIZE_LOCK set: an exclusive flock is acquired on enter
    and released on exit. Prevents two concurrent consumers from holding two
    models in VRAM (2026-07-13 contention fix)."""
    monkeypatch.setenv("OLLAMA_SERIALIZE_LOCK", "1")
    monkeypatch.setenv("OLLAMA_LOCK_FILE", str(tmp_path / "ollama.lock"))
    with patch("ollama_client._lock.fcntl.flock") as mock_flock:
        with serialize_lock():
            pass
    calls = mock_flock.call_args_list
    assert len(calls) == 2  # LOCK_EX enter, LOCK_UN exit
    assert calls[0].args[1] == fcntl.LOCK_EX
    assert calls[1].args[1] == fcntl.LOCK_UN


def test_serialize_lock_noop_when_env_empty(monkeypatch):
    """An empty value is treated as unset (no lock)."""
    monkeypatch.setenv("OLLAMA_SERIALIZE_LOCK", "")
    with patch("ollama_client._lock.fcntl.flock") as mock_flock:
        with serialize_lock():
            pass
    mock_flock.assert_not_called()
