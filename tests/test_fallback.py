"""Fallback-chain behavior + the late-binding mocking guard.

These are the regression guards for the package split: ``generate_fallback`` /
``chat_fallback`` MUST resolve their inner ``generate`` / ``chat`` via the public
package attribute (late import) so ``@patch("ollama_client.generate")`` keeps
intercepting. A naive top-level import would defeat the patch.

See ``topics/mocking-semantics.md``.
"""

from __future__ import annotations

import importlib

import pytest

import ollama_client as o

# --- generate_fallback ---


def test_generate_fallback_respects_patched_generate(monkeypatch):
    """Late-binding: the fallback must call the (patched) public generate."""
    seen: list[str] = []

    def fake(prompt: str, **kw: object) -> str:
        seen.append(kw["model"])  # type: ignore[arg-type]
        return f"out-{kw['model']}"

    monkeypatch.setattr(o, "generate", fake)
    text, model = o.generate_fallback("p", ["mA", "mB"])
    assert seen == ["mA"]  # first model answers -> stops, never tries mB
    assert text == "out-mA" and model == "mA"


def test_generate_fallback_empty_models():
    assert o.generate_fallback("p", []) == (None, None)


def test_generate_fallback_continues_on_request_error(monkeypatch):
    """OllamaRequestError is model-specific -> try the NEXT model."""
    calls: list[str] = []

    def fake(prompt: str, **kw: object) -> str:
        calls.append(kw["model"])  # type: ignore[arg-type]
        if kw["model"] == "mA":
            raise o.OllamaRequestError(500, "oom")
        return "ok"

    monkeypatch.setattr(o, "generate", fake)
    text, model = o.generate_fallback("p", ["mA", "mB"])
    assert calls == ["mA", "mB"]
    assert text == "ok" and model == "mB"


def test_generate_fallback_aborts_on_unavailable(monkeypatch):
    """OllamaUnavailable means the daemon is down -> no point trying more models."""

    def fake(prompt: str, **kw: object) -> str:
        raise o.OllamaUnavailable("down")

    monkeypatch.setattr(o, "generate", fake)
    assert o.generate_fallback("p", ["mA", "mB"]) == (None, None)


def test_generate_fallback_all_return_none(monkeypatch):
    monkeypatch.setattr(o, "generate", lambda *a, **_kw: None)
    assert o.generate_fallback("p", ["mA", "mB"]) == (None, None)


def test_generate_fallback_skips_empty(monkeypatch):
    """A model returning None (empty output) is skipped, not selected."""
    calls: list[str] = []

    def fake(prompt: str, **kw: object) -> str | None:
        calls.append(kw["model"])  # type: ignore[arg-type]
        return None if kw["model"] == "mA" else "real"

    monkeypatch.setattr(o, "generate", fake)
    text, model = o.generate_fallback("p", ["mA", "mB"])
    assert calls == ["mA", "mB"]
    assert text == "real" and model == "mB"


# --- chat_fallback (same contract) ---


def test_chat_fallback_respects_patched_chat(monkeypatch):
    seen: list[str] = []

    def fake(messages: list, **kw: object) -> str:
        seen.append(kw["model"])  # type: ignore[arg-type]
        return f"c-{kw['model']}"

    monkeypatch.setattr(o, "chat", fake)
    content, model = o.chat_fallback([{"role": "user", "content": "x"}], ["cA", "cB"])
    assert seen == ["cA"]
    assert content == "c-cA" and model == "cA"


def test_chat_fallback_continues_on_request_error(monkeypatch):
    def fake(messages: list, **kw: object) -> str:
        if kw["model"] == "cA":
            raise o.OllamaRequestError(404, "not found")
        return "good"

    monkeypatch.setattr(o, "chat", fake)
    content, model = o.chat_fallback([{"role": "user", "content": "x"}], ["cA", "cB"])
    assert content == "good" and model == "cB"


def test_chat_fallback_aborts_on_unavailable(monkeypatch):
    def fake(messages: list, **kw: object) -> str:
        raise o.OllamaUnavailable("down")

    monkeypatch.setattr(o, "chat", fake)
    assert o.chat_fallback([{"role": "user", "content": "x"}], ["cA", "cB"]) == (None, None)


def test_chat_retries_generate_for_template_parser_error(monkeypatch):
    """Ollama 0.31.x can reject tool-capable templates on /api/chat even
    though the same model works on /api/generate."""
    chat_module = importlib.import_module("ollama_client.chat")
    calls: list[tuple[str, dict]] = []

    def fake_post(path, payload, _base_url, _timeout):
        calls.append((path, payload))
        if path == "/api/chat":
            raise o.OllamaRequestError(400, "Unable to generate parser for this template")
        return {"response": "grounded result"}

    monkeypatch.setattr(chat_module, "_post", fake_post)
    messages = [
        {"role": "system", "content": "Do not invent."},
        {"role": "user", "content": "Review ~/ollama-bench/."},
    ]
    assert o.chat(messages, model="m", cache=False) == "grounded result"
    assert [path for path, _ in calls] == ["/api/chat", "/api/generate"]
    assert calls[1][1]["system"] == "Do not invent."
    assert calls[1][1]["prompt"] == "Review ~/ollama-bench/."
    assert calls[1][1]["think"] is False


def test_chat_does_not_mask_other_bad_requests(monkeypatch):
    chat_module = importlib.import_module("ollama_client.chat")

    def fake_post(*_args, **_kwargs):
        raise o.OllamaRequestError(400, "invalid option")

    monkeypatch.setattr(chat_module, "_post", fake_post)
    with pytest.raises(o.OllamaRequestError, match="invalid option"):
        o.chat([{"role": "user", "content": "x"}], model="m", cache=False)
