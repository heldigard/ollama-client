"""Shared pytest fixtures for the ollama-client test suite.

Tests run against the installed (editable) ``ollama_client`` package. No test
touches the network or the real cache dir unless explicitly opted in.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect ``OLLAMA_CACHE_DIR`` to a temp dir in the modules that read it.

    All cache I/O (key path, replay, write, prune) now lives in ``_cache`` and
    reads ``_cache.OLLAMA_CACHE_DIR`` — a *static* copy bound at import from
    ``_config`` — so patching ``_cache`` is what makes tests hermetic. ``_config``
    is patched too for belt-and-suspenders. ``generation``/``chat`` no longer
    reference the cache root directly (they call ``_cache_replay``/``_cache_store``),
    so they are not patched.
    """
    cache_dir = tmp_path / "ollama-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    import importlib

    cache_mod = importlib.import_module("ollama_client._cache")
    config_mod = importlib.import_module("ollama_client._config")

    monkeypatch.setattr(config_mod, "OLLAMA_CACHE_DIR", cache_dir)
    monkeypatch.setattr(cache_mod, "OLLAMA_CACHE_DIR", cache_dir)
    return cache_dir


@pytest.fixture
def fake_urlopen():
    """Patch ``urllib.request.urlopen`` (the one real network call) with a mock.

    Yields a recorder so a test can (a) set the canned response body and (b)
    assert on the URL/payload that was sent. Usage::

        def test_x(fake_urlopen):
            fake_urlopen.set_response({"response": "hi"})
            ...
            assert fake_urlopen.calls[0].path == "/api/generate"
    """
    calls: list[_Request] = []

    class _Recorder:
        def __init__(self) -> None:
            self.response_body: dict | bytes = {}
            self.response_status: int = 200
            self.error: BaseException | None = None

        def set_response(self, body: dict, status: int = 200) -> None:
            self.response_body = body
            self.response_status = status

        def set_raw_response(self, body: bytes, status: int = 200) -> None:
            self.response_body = body
            self.response_status = status

        def raise_error(self, exc: BaseException) -> None:
            self.error = exc

        @property
        def calls(self) -> list[_Request]:
            return calls

    recorder = _Recorder()

    class _FakeResponse:
        def __init__(self, body: dict | bytes, status: int) -> None:
            self._body = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
            self.status = status

        def read(self) -> bytes:
            return self._body

        def __enter__(self) -> "_FakeResponse":
            return self

        def __exit__(self, *exc: object) -> None:
            pass

    class _Request:
        def __init__(self, req: object) -> None:
            self.url = req.full_url if hasattr(req, "full_url") else str(req)  # type: ignore[attr-defined]
            try:
                self.payload = json.loads(req.data.decode("utf-8"))  # type: ignore[attr-defined]
            except (AttributeError, ValueError):
                self.payload = {}
            self.path = (
                self.url.split("/api/", 1)[-1].join(("/api/", ""))
                if "/api/" in self.url
                else self.url
            )
            self.method = req.get_method() if hasattr(req, "get_method") else "POST"  # type: ignore[attr-defined]

    def _fake_urlopen(req: object, timeout: float | None = None) -> _FakeResponse:  # noqa: ARG001
        calls.append(_Request(req))
        if recorder.error is not None:
            raise recorder.error
        return _FakeResponse(recorder.response_body, recorder.response_status)

    with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
        yield recorder
