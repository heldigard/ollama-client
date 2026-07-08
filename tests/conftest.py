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
    """Redirect ``OLLAMA_CACHE_DIR`` to a temp dir in every module that imported
    it at load time (``_config``, ``_cache``, ``generation``, ``chat``).

    Each ``from ._config import OLLAMA_CACHE_DIR`` binds a *static* reference, so
    patching only ``_config.OLLAMA_CACHE_DIR`` is not enough — the consumers keep
    their original copy. Patching all four keeps cache tests hermetic.
    """
    cache_dir = tmp_path / "ollama-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    # ``chat``/``generation`` COLLIDE with same-named functions re-exported in
    # __init__ (``from .chat import chat``). Both ``from ollama_client import
    # chat`` AND ``import ollama_client.chat as x`` resolve the FUNCTION, not the
    # module, because __init__ rebound the package attribute. importlib always
    # returns the real submodule object from sys.modules. _cache/_config don't
    # collide, but we use importlib uniformly for clarity.
    import importlib

    cache_mod = importlib.import_module("ollama_client._cache")
    config_mod = importlib.import_module("ollama_client._config")
    chat_mod = importlib.import_module("ollama_client.chat")
    gen_mod = importlib.import_module("ollama_client.generation")

    monkeypatch.setattr(config_mod, "OLLAMA_CACHE_DIR", cache_dir)
    monkeypatch.setattr(cache_mod, "OLLAMA_CACHE_DIR", cache_dir)
    monkeypatch.setattr(gen_mod, "OLLAMA_CACHE_DIR", cache_dir)
    monkeypatch.setattr(chat_mod, "OLLAMA_CACHE_DIR", cache_dir)
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
            self.response_body: dict = {}
            self.response_status: int = 200
            self.error: BaseException | None = None

        def set_response(self, body: dict, status: int = 200) -> None:
            self.response_body = body
            self.response_status = status

        def raise_error(self, exc: BaseException) -> None:
            self.error = exc

        @property
        def calls(self) -> list[_Request]:
            return calls

    recorder = _Recorder()

    class _FakeResponse:
        def __init__(self, body: dict, status: int) -> None:
            self._body = json.dumps(body).encode("utf-8")
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
