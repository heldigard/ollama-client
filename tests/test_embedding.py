"""Tests for the embedding module (_embed_once and embed)."""

from __future__ import annotations

import urllib.error
from email.message import Message

import pytest

import ollama_client as o
from ollama_client.embedding import _embed_once, embed


def test_embed_once_success(fake_urlopen):
    fake_urlopen.set_response({"embedding": [0.1, 0.2, 0.3]})
    vec = _embed_once("hello", model="m", timeout=10, base_url="http://h:11434")
    assert vec == [0.1, 0.2, 0.3]
    assert len(fake_urlopen.calls) == 1
    assert fake_urlopen.calls[0].path == "/api/embeddings"
    assert fake_urlopen.calls[0].payload == {"model": "m", "prompt": "hello"}


def test_embed_once_blank_text():
    assert _embed_once("", model="m", timeout=10, base_url="http://h:11434") is None
    assert _embed_once("   ", model="m", timeout=10, base_url="http://h:11434") is None


def test_embed_once_propagates_request_error(fake_urlopen):
    fake_urlopen.raise_error(urllib.error.HTTPError("url", 500, "Error", Message(), None))
    with pytest.raises(o.OllamaRequestError):
        _embed_once("hello", model="m", timeout=10, base_url="http://h:11434")


def test_embed_once_propagates_unavailable(fake_urlopen):
    fake_urlopen.raise_error(urllib.error.URLError("connection refused"))
    with pytest.raises(o.OllamaUnavailable) as exc_info:
        _embed_once("hello", model="m", timeout=10, base_url="http://h:11434")
    # Must NOT be request error subclass
    assert not isinstance(exc_info.value, o.OllamaRequestError)


def test_embed_success_first_try(fake_urlopen):
    fake_urlopen.set_response({"embedding": [0.5, 0.6]})
    vec = embed("hello", model="m", timeout=10, base_url="http://h:11434")
    assert vec == [0.5, 0.6]
    assert len(fake_urlopen.calls) == 1


def test_embed_request_error_short_text_no_retry(fake_urlopen):
    """Text <= 512: fails on first try, does not retry."""
    fake_urlopen.raise_error(urllib.error.HTTPError("url", 404, "Not Found", Message(), None))
    vec = embed("hello", model="m", timeout=10, base_url="http://h:11434")
    assert vec is None
    assert len(fake_urlopen.calls) == 1  # only 1 attempt


def test_embed_request_error_long_text_retries(fake_urlopen):
    """Text > 512: HTTP 500 triggers a retry with halved text."""
    # First call: raise HTTP 500. Second call: succeed.
    calls = []

    def fake_post(path, payload, base_url, timeout):
        calls.append(payload)
        if len(calls) == 1:
            raise o.OllamaRequestError(500, "context length exceeded")
        return {"embedding": [0.9, 0.8]}

    # Mock _post inside embedding module to control responses
    import ollama_client.embedding as emb_mod
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(emb_mod, "_post", fake_post)
        long_text = "a" * 600
        vec = embed(long_text, model="m", timeout=10, base_url="http://h:11434")
        assert vec == [0.9, 0.8]
        assert len(calls) == 2
        assert calls[0]["prompt"] == long_text
        assert calls[1]["prompt"] == "a" * 300


def test_embed_unavailable_long_text_no_retry(fake_urlopen):
    """Daemon unreachable (OllamaUnavailable) -> abort immediately, do NOT retry."""
    fake_urlopen.raise_error(urllib.error.URLError("connection refused"))
    long_text = "a" * 600
    vec = embed(long_text, model="m", timeout=10, base_url="http://h:11434")
    assert vec is None
    assert len(fake_urlopen.calls) == 1  # NO retry


def test_embed_none_response_long_text_retries(fake_urlopen):
    """If first call returns None without raising (e.g. missing embedding key),
    retry with halved text if len > 512."""
    calls = []

    def fake_post(path, payload, base_url, timeout):
        calls.append(payload)
        if len(calls) == 1:
            return {}  # missing key -> _embed_once returns None
        return {"embedding": [0.7, 0.7]}

    import ollama_client.embedding as emb_mod
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(emb_mod, "_post", fake_post)
        long_text = "b" * 600
        vec = embed(long_text, model="m", timeout=10, base_url="http://h:11434")
        assert vec == [0.7, 0.7]
        assert len(calls) == 2
        assert calls[0]["prompt"] == long_text
        assert calls[1]["prompt"] == "b" * 300
