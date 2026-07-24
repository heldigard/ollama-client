"""Cache helpers: key derivation, pruning, think-tag stripping."""

from __future__ import annotations

import hashlib
import importlib
import json

from ollama_client._cache import _cache_key, _cache_key_chat, _strip_think_tags, _write_cache_text

# --- _cache_key (generate) ---


def test_cache_key_deterministic():
    a = _cache_key("m", "p", 0.2)
    b = _cache_key("m", "p", 0.2)
    assert a == b


def test_cache_key_changes_on_model():
    assert _cache_key("m1", "p", 0.2) != _cache_key("m2", "p", 0.2)


def test_cache_key_changes_on_prompt():
    assert _cache_key("m", "p1", 0.2) != _cache_key("m", "p2", 0.2)


def test_cache_key_changes_on_temperature():
    assert _cache_key("m", "p", 0.2) != _cache_key("m", "p", 0.3)


def test_cache_key_changes_on_num_ctx():
    """num_ctx MUST be in the key: an overflowed (empty) result must NOT replay
    at a larger ctx."""
    assert _cache_key("m", "p", 0.2, num_ctx=8192) != _cache_key("m", "p", 0.2, num_ctx=32768)


def test_cache_key_none_vs_default_num_ctx():
    assert _cache_key("m", "p", 0.2) != _cache_key("m", "p", 0.2, num_ctx=8192)


# --- _cache_key_chat (chat) ---


def test_cache_key_chat_changes_on_num_predict():
    """num_predict bounds output length -> MUST be in the key."""
    msgs = [{"role": "user", "content": "x"}]
    assert _cache_key_chat("m", msgs, 0.2, num_predict=100) != _cache_key_chat(
        "m", msgs, 0.2, num_predict=500
    )


def test_cache_key_chat_same_messages_same_key():
    msgs = [{"role": "user", "content": "x"}]
    assert _cache_key_chat("m", msgs, 0.2) == _cache_key_chat("m", msgs, 0.2)


def test_cache_key_chat_dict_key_order_irrelevant():
    """sort_keys=True: ``{"role":..,"content":..}`` and ``{"content":..,"role":..}``
    hash identically (dict key order does not matter)."""
    a = [{"role": "user", "content": "x"}]
    b = [{"content": "x", "role": "user"}]
    assert _cache_key_chat("m", a, 0.2) == _cache_key_chat("m", b, 0.2)


def test_cache_key_chat_message_list_order_matters():
    """The message LIST order matters (semantically: chat history order).
    sort_keys only sorts dict keys, not list elements — so reordering messages
    MUST yield a different key."""
    a = [{"role": "system", "content": "s"}, {"role": "user", "content": "x"}]
    b = [{"role": "user", "content": "x"}, {"role": "system", "content": "s"}]
    assert _cache_key_chat("m", a, 0.2) != _cache_key_chat("m", b, 0.2)


def test_cache_key_chat_changes_on_content():
    a = [{"role": "user", "content": "x"}]
    b = [{"role": "user", "content": "y"}]
    assert _cache_key_chat("m", a, 0.2) != _cache_key_chat("m", b, 0.2)


def test_cache_key_chat_changes_on_think_mode():
    messages = [{"role": "user", "content": "x"}]
    legacy_shape = _cache_key_chat("m", messages, 0.2, None, None)

    assert legacy_shape == _cache_key_chat("m", messages, 0.2, think=False)
    assert legacy_shape != _cache_key_chat("m", messages, 0.2, think=True)


# --- _strip_think_tags ---


def test_strip_empty():
    assert _strip_think_tags("") == ""


def test_strip_think_block():
    assert _strip_think_tags("<think>hidden</think>visible") == "visible"


def test_strip_reasoning_block():
    assert _strip_think_tags("<reasoning>r</reasoning>answer") == "answer"


def test_strip_pipe_think():
    assert _strip_think_tags("<|think|>h<|/think|>real") == "real"


def test_strip_gemma_channel():
    assert _strip_think_tags("<|channel>thought\n<channel|>real answer") == "real answer"


def test_preserves_untagged_prose():
    assert _strip_think_tags("just normal text") == "just normal text"


def test_strip_unclosed_think_to_eof():
    assert _strip_think_tags("<think>leaked all the way") == ""


def test_strip_visible_preamble():
    out = _strip_think_tags("Thinking process: let me reason.\nFinal answer: 42")
    assert "42" in out


# --- _prune_cache ---


def test_prune_keeps_bounded(isolated_cache):
    # Write 5 files with EXPLICIT ascending mtimes (fast writes share a mtime
    # second -> non-deterministic sort). os.utime forces file0 oldest, file4 newest.
    import os

    for i in range(5):
        p = isolated_cache / f"file{i}.txt"
        p.write_text(str(i))
        os.utime(p, (i, i))
    from ollama_client._cache import _prune_cache

    _prune_cache(max_entries=2)
    remaining = sorted(isolated_cache.glob("*.txt"))
    assert len(remaining) == 2
    # oldest (file0, file1, file2) deleted; newest (file3, file4) survive
    assert {p.name for p in remaining} == {"file3.txt", "file4.txt"}


def test_prune_noop_under_limit(isolated_cache):
    (isolated_cache / "a.txt").write_text("a")
    from ollama_client._cache import _prune_cache

    _prune_cache(max_entries=10)
    assert (isolated_cache / "a.txt").exists()


def test_write_cache_text_replaces_without_tmp_leftovers(isolated_cache):
    path = isolated_cache / "entry.txt"
    path.write_text("old", encoding="utf-8")

    _write_cache_text(path, "new")

    assert path.read_text(encoding="utf-8") == "new"
    assert list(isolated_cache.glob("*.tmp")) == []


def test_generate_empty_cache_entry_falls_through(isolated_cache, fake_urlopen):
    """An empty cached file (truncated write) must not short-circuit the live
    call forever: it is dropped and the call regenerates."""
    from ollama_client._cache import _cache_key, _cache_path
    from ollama_client.generation import generate

    key = _cache_key("m", "prompt", 0.0, None)
    empty = _cache_path(key)
    empty.parent.mkdir(parents=True, exist_ok=True)
    empty.write_text("", encoding="utf-8")

    fake_urlopen.set_response({"response": "live answer"})
    out = generate("prompt", model="m", temperature=0.0)
    assert out == "live answer"
    assert fake_urlopen.calls, "live call must happen despite the cached entry"
    assert empty.read_text(encoding="utf-8") == "live answer"


def test_chat_empty_cache_entry_falls_through(isolated_cache, fake_urlopen):
    from ollama_client._cache import _cache_key_chat, _cache_path
    from ollama_client.chat import chat

    messages = [{"role": "user", "content": "q"}]
    key = _cache_key_chat("m", messages, 0.0, None, None)
    empty = _cache_path(key)
    empty.parent.mkdir(parents=True, exist_ok=True)
    empty.write_text("", encoding="utf-8")

    fake_urlopen.set_response({"message": {"role": "assistant", "content": "live"}})
    assert chat(messages, model="m", temperature=0.0) == "live"
    assert fake_urlopen.calls


def test_chat_cache_separates_think_modes(isolated_cache, fake_urlopen):
    from ollama_client.chat import chat

    messages = [{"role": "user", "content": "q"}]
    fake_urlopen.set_response({"message": {"role": "assistant", "content": "direct"}})
    assert chat(messages, model="m", temperature=0.0, think=False) == "direct"

    fake_urlopen.set_response({"message": {"role": "assistant", "content": "reasoned"}})
    assert chat(messages, model="m", temperature=0.0, think=True) == "reasoned"
    assert chat(messages, model="m", temperature=0.0, think=False) == "direct"

    assert len(fake_urlopen.calls) == 2
    assert [call.payload["think"] for call in fake_urlopen.calls] == [False, True]


def test_chat_ignores_legacy_cache_key(isolated_cache, fake_urlopen):
    from ollama_client.chat import chat

    messages = [{"role": "user", "content": "q"}]
    serialized = json.dumps(messages, sort_keys=True, ensure_ascii=False)
    legacy_raw = f"m\x1f0.0\x1fNone\x1fNone\x1f{serialized}"
    legacy_key = hashlib.sha256(legacy_raw.encode("utf-8")).hexdigest()
    (isolated_cache / f"{legacy_key}.txt").write_text("ambiguous legacy answer", encoding="utf-8")

    fake_urlopen.set_response({"message": {"role": "assistant", "content": "live"}})
    assert chat(messages, model="m", temperature=0.0, think=False) == "live"
    assert len(fake_urlopen.calls) == 1


def _set_cache_root(monkeypatch, cache_root):
    # Cache I/O lives in _cache (reads _cache.OLLAMA_CACHE_DIR); _config patched
    # for belt-and-suspenders. generation/chat no longer reference the cache root.
    for module_name in (
        "ollama_client._cache",
        "ollama_client._config",
    ):
        monkeypatch.setattr(importlib.import_module(module_name), "OLLAMA_CACHE_DIR", cache_root)


def test_generate_cache_root_failure_falls_through(tmp_path, monkeypatch, fake_urlopen):
    from ollama_client.generation import generate

    cache_root = tmp_path / "cache-root"
    cache_root.write_text("not a directory", encoding="utf-8")
    _set_cache_root(monkeypatch, cache_root)
    fake_urlopen.set_response({"response": "live"})

    assert generate("prompt", model="m", temperature=0.0) == "live"
    assert len(fake_urlopen.calls) == 1


def test_chat_cache_root_failure_falls_through(tmp_path, monkeypatch, fake_urlopen):
    from ollama_client.chat import chat

    cache_root = tmp_path / "cache-root"
    cache_root.write_text("not a directory", encoding="utf-8")
    _set_cache_root(monkeypatch, cache_root)
    fake_urlopen.set_response({"message": {"role": "assistant", "content": "live"}})

    assert chat([{"role": "user", "content": "q"}], model="m", temperature=0.0) == "live"
    assert len(fake_urlopen.calls) == 1


def test_cache_false_does_not_create_cache_root(tmp_path, monkeypatch, fake_urlopen):
    from ollama_client.chat import chat
    from ollama_client.generation import generate

    cache_root = tmp_path / "missing-cache-root"
    _set_cache_root(monkeypatch, cache_root)

    fake_urlopen.set_response({"response": "generated"})
    assert generate("prompt", model="m", temperature=0.0, cache=False) == "generated"
    fake_urlopen.set_response({"message": {"role": "assistant", "content": "chatted"}})
    assert chat([{"role": "user", "content": "q"}], model="m", temperature=0.0, cache=False) == (
        "chatted"
    )
    assert not cache_root.exists()


# --- _cache_replay / _cache_store (shared generate/chat cache policy) ---


def test_cache_replay_not_cacheable_returns_none_pair(tmp_path, monkeypatch):
    """Disabled caching -> (None, None) and no cache root created."""
    from ollama_client._cache import _cache_replay

    cache_root = tmp_path / "should-not-exist"
    monkeypatch.setattr(
        importlib.import_module("ollama_client._cache"), "OLLAMA_CACHE_DIR", cache_root
    )
    assert _cache_replay(False, "k") == (None, None)
    assert not cache_root.exists()


def test_cache_replay_miss_returns_path_no_hit(isolated_cache):
    from ollama_client._cache import _cache_path, _cache_replay

    path, hit = _cache_replay(True, "abc")
    assert hit is None
    assert path == _cache_path("abc")


def test_cache_replay_hit_returns_stored_text(isolated_cache):
    from ollama_client._cache import _cache_path, _cache_replay

    _cache_path("abc").write_text("stored", encoding="utf-8")
    path, hit = _cache_replay(True, "abc")
    assert hit == "stored"
    assert path == _cache_path("abc")


def test_cache_replay_drops_empty_entry(isolated_cache):
    """A truncated (empty) entry is unlinked and falls through (no hit)."""
    from ollama_client._cache import _cache_path, _cache_replay

    entry = _cache_path("abc")
    entry.write_text("", encoding="utf-8")
    path, hit = _cache_replay(True, "abc")
    assert hit is None
    assert path == entry
    assert not entry.exists()


def test_cache_replay_root_failure_fails_open(tmp_path, monkeypatch):
    from ollama_client._cache import _cache_replay

    cache_root = tmp_path / "cache-root"
    cache_root.write_text("not a directory", encoding="utf-8")
    monkeypatch.setattr(
        importlib.import_module("ollama_client._cache"), "OLLAMA_CACHE_DIR", cache_root
    )
    assert _cache_replay(True, "k") == (None, None)


def test_cache_store_writes_and_noop_on_empty(isolated_cache):
    from ollama_client._cache import _cache_path, _cache_store

    _cache_store(_cache_path("abc"), "value")
    assert _cache_path("abc").read_text(encoding="utf-8") == "value"

    # Empty text / no path -> no-op, nothing written.
    _cache_store(_cache_path("empty"), None)
    _cache_store(None, "value")
    assert not _cache_path("empty").exists()


def test_prune_cache_sweeps_stale_tmp_files(isolated_cache):
    import os
    import time

    from ollama_client._cache import _prune_cache

    stale = isolated_cache / ".abc.txt.xyz.tmp"
    stale.write_text("partial", encoding="utf-8")
    old = time.time() - 7200
    os.utime(stale, (old, old))
    fresh = isolated_cache / ".def.txt.uvw.tmp"
    fresh.write_text("in-flight", encoding="utf-8")

    _prune_cache()
    assert not stale.exists(), "stale tmp leftovers must be swept"
    assert fresh.exists(), "fresh tmp files (in-flight writes) must survive"
