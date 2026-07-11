"""Cache helpers: key derivation, pruning, think-tag stripping."""

from __future__ import annotations

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
