# vs-soft-allow  — single-responsibility: the deterministic-prompt response cache
# (key derivation, pruning, think-tag stripping). Three helpers, one concern.
"""Deterministic-prompt response cache helpers.

Key derivation, bounded pruning, and the think-tag stripper. Generation and chat
call into here to decide cacheability and to clean model output. See
``topics/cache-policy.md``.
"""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
import time
from pathlib import Path

from ._config import CACHE_MAX_ENTRIES, OLLAMA_CACHE_DIR


def _cache_key(model: str, prompt: str, temperature: float, num_ctx: int | None = None) -> str:
    """sha256 key for a ``generate`` call.

    ``num_ctx`` is in the key: a call that overflowed (returned empty) must NOT
    be replayed for the same prompt at a larger ctx.
    """
    raw = f"{model}\x1f{temperature}\x1f{num_ctx}\x1f{prompt}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cache_key_chat(
    model: str,
    messages: list[dict],
    temperature: float,
    num_predict: int | None = None,
    num_ctx: int | None = None,
    think: bool = False,
) -> str:
    """sha256 key for a ``chat`` call.

    ``num_predict`` bounds output length, so it MUST be in the key — otherwise a
    short-bound call poisons the cache for a long-bound call with the same
    prompt. ``num_ctx`` too: an overflowed (empty) result must not replay at a
    larger ctx. ``think`` changes the model's reasoning mode and output, so it
    must not share entries across modes. ``messages`` is JSON-sorted for stable
    hashing.
    """
    raw = (
        f"{model}\x1f{temperature}\x1f{num_predict}\x1f{num_ctx}\x1f{think}"
        f"\x1f{json.dumps(messages, sort_keys=True, ensure_ascii=False)}"
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


#: Orphaned atomic-write temp files older than this are swept by ``_prune_cache``.
_TMP_MAX_AGE_SECONDS = 3600.0


def _prune_cache(max_entries: int = CACHE_MAX_ENTRIES) -> None:
    """Keep the cache bounded by deleting oldest entries beyond ``max_entries``.

    Also sweeps ``.tmp`` leftovers from atomic writes interrupted by a hard
    kill — they match no ``*.txt`` glob, so without the sweep they would
    accumulate forever. The age threshold keeps in-flight writes safe.
    """
    _sweep_stale_tmp_files()
    try:
        files = sorted(OLLAMA_CACHE_DIR.glob("*.txt"), key=lambda p: p.stat().st_mtime)
    except OSError:
        return
    excess = len(files) - max_entries
    if excess <= 0:
        return
    for path in files[:excess]:
        try:
            path.unlink()
        except OSError:
            pass


def _sweep_stale_tmp_files(max_age: float = _TMP_MAX_AGE_SECONDS) -> None:
    cutoff = time.time() - max_age
    try:
        tmp_files = list(OLLAMA_CACHE_DIR.glob("*.tmp"))
    except OSError:
        return
    for path in tmp_files:
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink(missing_ok=True)
        except OSError:
            pass


def _write_cache_text(path: Path, text: str) -> None:
    """Atomically replace a cache entry with ``text``."""
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            tmp.write(text)
            tmp_path = Path(tmp.name)
        tmp_path.replace(path)
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink()
            except OSError:
                pass


# Precompiled at import time: _strip_think_tags runs on every generate/chat/ocr
# response (hot path for the codeq/smart-trim hooks), and re.compile per call is
# pure overhead. Ordered to mirror the original inline substitutions.
_RE_TAG_BLOCKS = [
    re.compile(r"<think\b[^>]*>.*?(</think\s*>|$)", re.S | re.I),
    re.compile(r"<reasoning\b[^>]*>.*?(</reasoning\s*>|$)", re.S | re.I),
    re.compile(r"<reflection\b[^>]*>.*?(</reflection\s*>|$)", re.S | re.I),
    re.compile(r"^.*?</(?:think|reasoning|reflection)\s*>", re.S | re.I),
]
_RE_OUTPUT_BLOCK = re.compile(r"<output\b[^>]*>(.*?)</output\s*>", re.S | re.I)
_RE_PIPE_THINK = re.compile(r"<\|think\|>.*?<\|/think\|>", re.S | re.I)
_RE_CHANNEL_MIXED = re.compile(r"<\|channel>.*?<channel\|>", re.S | re.I)
_RE_CHANNEL_PIPED = re.compile(r"<\|channel\|>.*?(<\|channel\|>|$)", re.S | re.I)
_RE_VISIBLE_THINKING = re.compile(
    r"^\s*(thinking process|let me think)[: ].*?(final answer|answer|output)\s*:\s*",
    re.S | re.I,
)
_RE_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")


def _strip_think_tags(text: str) -> str:
    """Remove inline reasoning blocks some models emit in ``content`` even with
    ``think=False`` (Qwen-family ``<think>...</think>``, ``<reasoning>...</reasoning>``,
    ``<|think|>...<|/think|>``). Conservative: only TAG-delimited blocks are
    removed — never un-tagged prose. Returns the cleaned text.

    Also handles Gemma-4 abliterated/uncensored variants that leak a thinking
    channel (``<|channel>thought\\n<channel|>real answer``) and visible
    "Thinking Process:" preambles.
    """
    if not text:
        return ""
    for pattern in _RE_TAG_BLOCKS:
        text = pattern.sub("", text)
    text = _RE_OUTPUT_BLOCK.sub(r"\1", text)
    text = _RE_PIPE_THINK.sub("", text)
    # Gemma-4 abliterated/uncensored variants leak a thinking channel even with
    # think=False: ``<|channel>thought\n<channel|>real answer``. Strip the whole
    # channel block (opening <|channel> through closing <channel|>) so only the
    # real answer remains. Conservative: requires BOTH delimiters.
    text = _RE_CHANNEL_MIXED.sub("", text)
    text = _RE_CHANNEL_PIPED.sub("", text)
    visible = _RE_VISIBLE_THINKING.search(text)
    if visible:
        text = text[visible.end() :]
    else:
        low = text.lstrip().lower()
        if low.startswith(("thinking process:", "let me think:")):
            parts = _RE_PARAGRAPH_SPLIT.split(text.strip(), maxsplit=1)
            text = parts[1] if len(parts) == 2 else ""
    return text.strip()


def _cache_path(key: str) -> Path:
    """Resolve the on-disk cache file for a key (does not imply existence)."""
    return OLLAMA_CACHE_DIR / f"{key}.txt"
