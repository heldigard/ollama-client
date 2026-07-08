"""Transport layer: URL normalization, exception taxonomy, liveness."""

from __future__ import annotations

import io
import urllib.error

import pytest

import ollama_client as o
from ollama_client._transport import _normalize_base_url, _post

# --- _normalize_base_url (legacy full-URL callers keep working) ---


def test_normalize_plain_base():
    assert _normalize_base_url("http://h:11434") == "http://h:11434"


def test_normalize_trailing_slash():
    assert _normalize_base_url("http://h:11434/") == "http://h:11434"


def test_normalize_generate_suffix():
    assert _normalize_base_url("http://h:11434/api/generate") == "http://h:11434"


def test_normalize_chat_suffix():
    assert _normalize_base_url("http://h:11434/api/chat") == "http://h:11434"


def test_normalize_embeddings_suffix():
    assert _normalize_base_url("http://h:11434/api/embeddings") == "http://h:11434"


# --- _post exception taxonomy (drives fallback behavior) ---


def test_post_success_returns_parsed_json(fake_urlopen):
    fake_urlopen.set_response({"response": "hi", "done": True})
    data = _post("/api/generate", {"model": "m"}, "http://h:11434", 10)
    assert data == {"response": "hi", "done": True}
    # _post forwards the payload verbatim (generate/chat add stream upstream)
    assert fake_urlopen.calls[0].payload["model"] == "m"


def test_post_http_error_becomes_request_error(fake_urlopen):
    """HTTPError -> OllamaRequestError (model-specific -> try next)."""
    fp = io.BytesIO(b'{"error":"model not found"}')
    err = urllib.error.HTTPError("http://h/api/generate", 404, "Not Found", {}, fp)
    fake_urlopen.raise_error(err)
    with pytest.raises(o.OllamaRequestError) as ei:
        _post("/api/generate", {"model": "m"}, "http://h:11434", 10)
    assert ei.value.status == 404
    assert "model not found" in ei.value.body


def test_post_url_error_becomes_unavailable(fake_urlopen):
    """URLError (connection refused) -> OllamaUnavailable (daemon down -> abort)."""
    fake_urlopen.raise_error(urllib.error.URLError("connection refused"))
    with pytest.raises(o.OllamaUnavailable) as ei:
        _post("/api/generate", {"model": "m"}, "http://h:11434", 10)
    # Must NOT be the model-specific subclass (callers check this to decide abort)
    assert not isinstance(ei.value, o.OllamaRequestError)


def test_post_timeout_becomes_request_error_408(fake_urlopen):
    """TimeoutError -> OllamaRequestError(408) (slow cold load -> try next,
    NOT abort). This mapping is load-bearing — see systemPatterns.md."""
    fake_urlopen.raise_error(TimeoutError("read timed out"))
    with pytest.raises(o.OllamaRequestError) as ei:
        _post("/api/generate", {"model": "m"}, "http://h:11434", 10)
    assert ei.value.status == 408


def test_post_uses_normalized_base_url(fake_urlopen):
    fake_urlopen.set_response({"ok": 1})
    _post("/api/generate", {}, "http://h:11434/api/chat", 10)  # legacy full URL
    assert fake_urlopen.calls[0].url == "http://h:11434/api/generate"


# --- is_alive ---


def test_is_alive_true_on_200(fake_urlopen):
    fake_urlopen.set_response({"models": []}, status=200)
    assert o.is_alive("http://h:11434") is True


def test_is_alive_false_on_url_error(fake_urlopen):
    fake_urlopen.raise_error(urllib.error.URLError("refused"))
    assert o.is_alive("http://h:11434") is False


def test_is_alive_false_on_timeout(fake_urlopen):
    fake_urlopen.raise_error(TimeoutError("slow"))
    assert o.is_alive("http://h:11434") is False
