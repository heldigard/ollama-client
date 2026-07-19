"""Transport layer: URL normalization, exception taxonomy, liveness."""

from __future__ import annotations

import io
import urllib.error
from email.message import Message
from unittest.mock import MagicMock, patch

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


def test_normalize_other_endpoint_suffixes():
    for suffix in ("/api/embed", "/api/tags", "/api/ps", "/api/show"):
        assert _normalize_base_url(f"http://h:11434{suffix}") == "http://h:11434"


def test_normalize_rejects_non_http_scheme():
    with pytest.raises(ValueError):
        _normalize_base_url("file:///tmp/ollama")


def test_normalize_rejects_missing_scheme():
    with pytest.raises(ValueError):
        _normalize_base_url("localhost:11434")


# --- _post exception taxonomy (drives fallback behavior) ---


def test_post_success_returns_parsed_json(fake_urlopen):
    fake_urlopen.set_response({"response": "hi", "done": True})
    data = _post("/api/generate", {"model": "m"}, "http://h:11434", 10)
    assert data == {"response": "hi", "done": True}
    # _post forwards the payload verbatim (generate/chat add stream upstream)
    assert fake_urlopen.calls[0].payload["model"] == "m"
    assert fake_urlopen.calls[0].method == "POST"


def test_post_invalid_json_becomes_request_error(fake_urlopen):
    """Malformed 200 responses are model/request failures, not raw JSON errors."""
    fake_urlopen.set_raw_response(b"not-json")
    with pytest.raises(o.OllamaRequestError) as ei:
        _post("/api/generate", {"model": "m"}, "http://h:11434", 10)
    assert ei.value.status == 502
    assert "invalid JSON response" in ei.value.body


def test_post_invalid_utf8_becomes_request_error(fake_urlopen):
    fake_urlopen.set_raw_response(b"\xff")
    with pytest.raises(o.OllamaRequestError) as ei:
        _post("/api/generate", {"model": "m"}, "http://h:11434", 10)
    assert ei.value.status == 502
    assert ei.value.body == "invalid UTF-8 response"


def test_post_non_object_json_becomes_request_error(fake_urlopen):
    fake_urlopen.set_raw_response(b'["not", "an", "object"]')
    with pytest.raises(o.OllamaRequestError) as ei:
        _post("/api/generate", {"model": "m"}, "http://h:11434", 10)
    assert ei.value.status == 502
    assert "unexpected JSON response" in ei.value.body


def test_post_http_error_becomes_request_error(fake_urlopen):
    """HTTPError -> OllamaRequestError (model-specific -> try next)."""
    fp = io.BytesIO(b'{"error":"model not found"}')
    err = urllib.error.HTTPError("http://h/api/generate", 404, "Not Found", Message(), fp)
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
    NOT abort). This mapping is load-bearing — see systemPatterns.md.
    Backoff patched to 0 so the retry does not stall the test suite."""
    fake_urlopen.raise_error(TimeoutError("read timed out"))
    with patch("ollama_client._transport._TRANSIENT_BACKOFF_SEC", 0.0):
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
    assert fake_urlopen.calls[0].method == "GET"


def test_is_alive_uses_normalized_base_url(fake_urlopen):
    fake_urlopen.set_response({"models": []}, status=200)
    assert o.is_alive("http://h:11434/api/generate") is True
    assert fake_urlopen.calls[0].url == "http://h:11434/api/tags"


def test_is_alive_false_on_invalid_base_url():
    assert o.is_alive("file:///tmp/ollama") is False


def test_is_alive_false_on_url_error(fake_urlopen):
    fake_urlopen.raise_error(urllib.error.URLError("refused"))
    assert o.is_alive("http://h:11434") is False


def test_is_alive_false_on_timeout(fake_urlopen):
    fake_urlopen.raise_error(TimeoutError("slow"))
    assert o.is_alive("http://h:11434") is False


# --- transient-load retry (2026-07-13: stop false hard-DQ / false fallback) ---


def test_post_retries_transient_500_then_succeeds():
    """A transient 5xx (server blip / GPU contention) is retried once and the
    second attempt wins. Mirrors the ollama-bench 2026-07-13 Qwythos incident
    where a champion emptied under load and was nearly dethroned."""
    err = urllib.error.HTTPError("u", 500, "Server Error", Message(), io.BytesIO(b""))
    good = MagicMock()
    good.__enter__.return_value.read.return_value = b'{"response": "ok", "done": true}'
    good.__exit__.return_value = None
    with (
        patch("urllib.request.urlopen", side_effect=[err, good]),
        patch("ollama_client._transport._TRANSIENT_BACKOFF_SEC", 0.0),
    ):
        data = _post("/api/generate", {"model": "m"}, "http://h:11434", 10)
    assert data["response"] == "ok"


def test_post_retries_transient_408_then_raises():
    """A persistent transient timeout exhausts the retry and still raises the
    model-specific error so the caller's fallback chain can proceed."""
    with (
        patch("urllib.request.urlopen", side_effect=TimeoutError("timed out")) as mock_open,
        patch("ollama_client._transport._TRANSIENT_BACKOFF_SEC", 0.0),
    ):
        with pytest.raises(o.OllamaRequestError) as ei:
            _post("/api/generate", {"model": "m"}, "http://h:11434", 10)
    assert ei.value.status == 408
    assert mock_open.call_count == 2  # 1 initial + 1 retry


def test_post_no_retry_on_permanent_4xx():
    """Permanent 4xx (model not found / bad request) is NOT retried — one call."""
    err = urllib.error.HTTPError("u", 404, "Not Found", Message(), io.BytesIO(b"nope"))
    with patch("urllib.request.urlopen", side_effect=err) as mock_open:
        with pytest.raises(o.OllamaRequestError) as ei:
            _post("/api/generate", {"model": "m"}, "http://h:11434", 10)
    assert ei.value.status == 404
    assert mock_open.call_count == 1
