"""Tests for list_models / list_running (GET /api/tags, /api/ps)."""

from __future__ import annotations

import urllib.error

import pytest

import ollama_client as o
import ollama_client.models as models_mod
from ollama_client.models import list_models, list_running


def test_list_models_filters_dicts(monkeypatch):
    def fake_get(path: str, base_url: str, timeout: float) -> dict:
        assert path == "/api/tags"
        assert timeout == 5.0
        return {
            "models": [
                {"name": "a"},
                "skip-me",
                {"name": "b"},
                None,
            ]
        }

    monkeypatch.setattr(models_mod, "_get", fake_get)
    assert list_models(base_url="http://h:11434") == [{"name": "a"}, {"name": "b"}]


def test_list_models_missing_or_non_list(monkeypatch):
    monkeypatch.setattr(models_mod, "_get", lambda *_a, **_k: {})
    assert list_models() == []
    monkeypatch.setattr(models_mod, "_get", lambda *_a, **_k: {"models": "nope"})
    assert list_models() == []


def test_list_running_filters_dicts(monkeypatch):
    def fake_get(path: str, base_url: str, timeout: float) -> dict:
        assert path == "/api/ps"
        return {"models": [{"name": "running"}, 42]}

    monkeypatch.setattr(models_mod, "_get", fake_get)
    assert list_running(base_url="http://h:11434", timeout=3.0) == [{"name": "running"}]


def test_list_models_propagates_unavailable(monkeypatch):
    def boom(*_a, **_k):
        raise o.OllamaUnavailable("down")

    monkeypatch.setattr(models_mod, "_get", boom)
    with pytest.raises(o.OllamaUnavailable):
        list_models()


def test_list_models_via_fake_urlopen(fake_urlopen):
    fake_urlopen.set_response({"models": [{"name": "x", "size": 1}]})
    out = list_models(base_url="http://h:11434")
    assert out == [{"name": "x", "size": 1}]
    assert fake_urlopen.calls[0].path == "/api/tags"


def test_list_running_via_fake_urlopen(fake_urlopen):
    fake_urlopen.set_response({"models": [{"name": "y"}]})
    out = list_running(base_url="http://h:11434")
    assert out == [{"name": "y"}]
    assert fake_urlopen.calls[0].path == "/api/ps"


def test_list_running_unavailable(fake_urlopen):
    fake_urlopen.raise_error(urllib.error.URLError("connection refused"))
    with pytest.raises(o.OllamaUnavailable):
        list_running(base_url="http://h:11434")
