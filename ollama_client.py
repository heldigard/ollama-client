#!/usr/bin/env python3
# vs-soft-allow  — single-responsibility local-Ollama HTTP client (one cohesion:
# talk to the daemon). Wide param lists on generate/chat are the frozen public
# API the four ecosystem consumers (codeq, smart-trim, prompt-improve,
# web-research) already depend on — collapsing to a param object would break
# the contract. This is a graduated lib, not greenfield.
"""Shared local-Ollama client for all Claude Code hooks/scripts.

Single source of truth for talking to the local Ollama daemon
(http://localhost:11434). Consolidates the four previously inline
`urllib.request` implementations (extract-tool-output, ollama-summarize,
smart-trim, prompt-improve) so they share one HTTP path, one response
cache, one fallback chain, and one embed call.

Three operations:
  - generate():  one-shot text completion (optionally cached)
  - generate_fallback():  try a list of models in order, return (text, model)
  - embed():     vector for a single text (/api/embeddings)

Cache policy: deterministic prompts (temperature <= CACHE_MAX_TEMP) are
keyed on sha256(model|temperature|prompt) and replayed without a network
round-trip. Higher temperatures bypass the cache. Generation cache lives
in OLLAMA_CACHE_DIR; embeddings are NOT cached here (callers usually need
fresh vectors and manage their own index).

CLI:
  python3 ollama_client.py is-alive
  python3 ollama_client.py generate --prompt "say hi" [--model X] [--no-cache]
  python3 ollama_client.py embed --text "auth token"
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_URL = "http://localhost:11434"
# 3.4GB, ~115 tok/s, universal fallback (best small model 2026-07-08 PM).
DEFAULT_GEN_MODEL = "cryptidbleh/gemma4-claude-opus-4.6:latest"
DEFAULT_STRUCTURED_MODEL = "SetneufPT/Qwopus3.5-4B-Coder-MTP_Q4_64k_8GB-GPU:latest"
DEFAULT_PDF_OCR_MODEL = "hf.co/sahilchachra/Unlimited-OCR-GGUF:Q4_K_M"
PDF_OCR_PROMPT = "ocr [img]"
# 768-dim, 621MB, ~2K functional ctx — eval winner (MRR 0.724, 12-query bilingual,
# 2026-06-28). Tied with nomic-embed-text on context (both ~2K functional — empirically
# verified, NOT the 8K the modelfile claims; nomic silently truncates @~2K);
# embeddinggemma wins on quality + consistency (worst-case rank 43 vs nomic 222).
# nomic kept installed as fallback. embed() halves+retries for token-dense chunks.
DEFAULT_EMBED_MODEL = "embeddinggemma"
DEFAULT_TIMEOUT = 120
EMBED_TIMEOUT = 60
CACHE_MAX_TEMP = 0.3  # only cache near-deterministic outputs
CACHE_MAX_ENTRIES = 2000  # prune oldest beyond this
OLLAMA_CACHE_DIR = Path.home() / ".claude" / "state" / "ollama-cache"

# ---------------------------------------------------------------------------
# Versioned public API (SemVer). Consumers gate with require("<min>") to fail
# fast on version drift instead of a cryptic mid-run error. Mirrors cheap_llm.
# ---------------------------------------------------------------------------
__version__ = "1.0.0"

__all__ = [
    "is_alive",
    "generate",
    "generate_fallback",
    "chat",
    "chat_fallback",
    "embed",
    "ocr_image",
    "OllamaUnavailable",
    "OllamaRequestError",
    "require",
    "__version__",
]


def _parse_version(v: str) -> tuple[int, ...]:
    """Best-effort dotted-integer parse (ignores pre-release suffixes)."""
    parts = []
    for p in v.split("."):
        digits = "".join(ch for ch in p if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def require(min_version: str | None = None) -> str:
    """Return ``__version__``, raising ``RuntimeError`` if older than ``min_version``."""
    if min_version and _parse_version(__version__) < _parse_version(min_version):
        raise RuntimeError(
            f"ollama_client {__version__} is older than required {min_version}. "
            f"Upgrade the ollama-client project (~/ollama-client/)."
        )
    return __version__


# Friendly error so callers can catch the transport class without importing urllib.
class OllamaUnavailable(RuntimeError):
    """Raised when the local Ollama daemon is unreachable (connection refused,
    DNS, timeout) — there is no point trying more models, the daemon itself is
    down or unreachable. Callers walking a fallback chain should abort."""


class OllamaRequestError(OllamaUnavailable):
    """Raised when the daemon WAS reachable but a single request failed
    (HTTP 4xx/5xx: model failed to load, OOM, model-not-found, etc.). This is
    model-specific, not daemon-wide — callers walking a fallback chain should
    try the NEXT model, not abort.

    Subclass of OllamaUnavailable for backward compatibility: existing
    ``except OllamaUnavailable`` handlers still catch it (preserving prior
    abort-on-any-failure behavior). New handlers can ``except OllamaRequestError``
    first to continue the chain on model-specific failures."""

    def __init__(self, status: int, body: str = "") -> None:
        super().__init__(f"HTTP {status}: {body[:200]}" if body else f"HTTP {status}")
        self.status = status
        self.body = body


def is_alive(base_url: str = DEFAULT_URL, timeout: float = 2.0) -> bool:
    """True iff the Ollama daemon answers /api/tags quickly."""
    try:
        req = urllib.request.Request(f"{base_url.rstrip('/')}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
        return False


def _normalize_base_url(base_url: str) -> str:
    """Accept either a base URL (http://host:11434) or a legacy full endpoint
    URL (.../api/generate, .../api/chat, .../api/embeddings) and return the base.
    Lets old callers that pass a full URL keep working after consolidation."""
    url = base_url.rstrip("/")
    for suffix in ("/api/generate", "/api/chat", "/api/embeddings"):
        if url.endswith(suffix):
            url = url[: -len(suffix)]
            break
    return url


def _post(path: str, payload: dict, base_url: str, timeout: float) -> dict:
    """POST JSON to an Ollama endpoint; return parsed JSON. Raise on transport error."""
    url = f"{_normalize_base_url(base_url)}{path}"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        # Daemon was reachable but THIS request failed (load failure, OOM,
        # model-not-found). Model-specific → caller should try the next model.
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except OSError:
            body = ""
        raise OllamaRequestError(exc.code, body) from exc
    except TimeoutError as exc:
        # A timed-out request is almost always a slow model LOAD (daemon IS up
        # and reachable), not a dead daemon — a dead daemon gives an instant
        # URLError (connection refused), never a timeout. Treat as model-specific
        # (HTTP 408 convention) so a fallback chain tries the NEXT (faster/smaller)
        # model instead of aborting on a slow cold load.
        raise OllamaRequestError(408, f"timeout: {exc}") from exc
    except (urllib.error.URLError, OSError) as exc:
        # Daemon unreachable / network down → no point trying more models.
        raise OllamaUnavailable(str(exc)) from exc


def _cache_key(model: str, prompt: str, temperature: float, num_ctx: int | None = None) -> str:
    # num_ctx is in the key: a call that overflowed (empty) must NOT be replayed
    # for the same prompt at a larger ctx.
    raw = f"{model}\x1f{temperature}\x1f{num_ctx}\x1f{prompt}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _prune_cache(max_entries: int = CACHE_MAX_ENTRIES) -> None:
    """Keep the cache bounded by deleting oldest entries beyond max_entries."""
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


def _strip_think_tags(text: str) -> str:
    """Remove inline reasoning blocks some models emit in `content` even with
    think=False (Qwen-family ``<think>...</think>``, ``<reasoning>...</reasoning>``,
    ``<|think|>...<|/think|>``). Conservative: only TAG-delimited blocks are
    removed — never un-tagged prose (carstenuhlig's 'Thinking Process:' preamble
    is solved by model choice, not by this). Returns the cleaned text."""
    if not text:
        return ""
    text = re.sub(r"<think\b[^>]*>.*?(</think\s*>|$)", "", text, flags=re.S | re.I)
    text = re.sub(r"<reasoning\b[^>]*>.*?(</reasoning\s*>|$)", "", text, flags=re.S | re.I)
    text = re.sub(r"<reflection\b[^>]*>.*?(</reflection\s*>|$)", "", text, flags=re.S | re.I)
    text = re.sub(r"^.*?</(?:think|reasoning|reflection)\s*>", "", text, flags=re.S | re.I)
    text = re.sub(r"<output\b[^>]*>(.*?)</output\s*>", r"\1", text, flags=re.S | re.I)
    text = re.sub(r"<\|think\|>.*?<\|/think\|>", "", text, flags=re.S | re.I)
    # Gemma-4 abliterated/uncensored variants leak a thinking channel even with
    # think=False: ``<|channel>thought\n<channel|>real answer``. Strip the whole
    # channel block (opening <|channel> through closing <channel|>) so only the
    # real answer remains. Conservative: requires BOTH delimiters.
    text = re.sub(r"<\|channel>.*?<channel\|>", "", text, flags=re.S | re.I)
    text = re.sub(r"<\|channel\|>.*?(<\|channel\|>|$)", "", text, flags=re.S | re.I)
    visible = re.search(
        r"^\s*(thinking process|let me think)[: ].*?(final answer|answer|output)\s*:\s*",
        text,
        flags=re.S | re.I,
    )
    if visible:
        text = text[visible.end() :]
    else:
        low = text.lstrip().lower()
        if low.startswith(("thinking process:", "let me think:")):
            parts = re.split(r"\n\s*\n", text.strip(), maxsplit=1)
            text = parts[1] if len(parts) == 2 else ""
    return text.strip()


def generate(
    prompt: str,
    model: str = DEFAULT_GEN_MODEL,
    *,
    temperature: float = 0.2,
    timeout: float = DEFAULT_TIMEOUT,
    base_url: str = DEFAULT_URL,
    cache: bool = True,
    num_ctx: int | None = None,
) -> str | None:
    """One-shot completion. Returns stripped text, or None if the model
    returned nothing. Raises OllamaUnavailable if the daemon is down (callers
    should catch and fall back). Cached when temperature <= CACHE_MAX_TEMP.
    Pass num_ctx to extend the context window for large prompts (default model
    ctx is often 8192, which silently truncates ~7K-token inputs to empty)."""
    cacheable = cache and temperature <= CACHE_MAX_TEMP
    cached_path: Path | None = None
    if cacheable:
        OLLAMA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cached_path = OLLAMA_CACHE_DIR / f"{_cache_key(model, prompt, temperature, num_ctx)}.txt"
        if cached_path.exists():
            try:
                return cached_path.read_text(encoding="utf-8")
            except OSError:
                pass  # corrupt entry; fall through to live call

    options: dict = {"temperature": temperature}
    if num_ctx is not None:
        options["num_ctx"] = num_ctx
    data = _post(
        "/api/generate",
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": options,
        },
        base_url,
        timeout,
    )
    out = _strip_think_tags(str(data.get("response", "")).strip()) or None

    if out and cacheable and cached_path is not None:
        try:
            cached_path.write_text(out, encoding="utf-8")
            _prune_cache()
        except OSError:
            pass
    return out


def generate_fallback(
    prompt: str,
    models: list[str],
    *,
    temperature: float = 0.2,
    timeout: float = DEFAULT_TIMEOUT,
    base_url: str = DEFAULT_URL,
    cache: bool = True,
    num_ctx: int | None = None,
) -> tuple[str | None, str | None]:
    """Try each model in order; return (text, model_used) from the first that
    answers. Returns (None, None) if all fail or the daemon is down."""
    if not models:
        return None, None
    for model in models:
        try:
            text = generate(
                prompt,
                model=model,
                temperature=temperature,
                timeout=timeout,
                base_url=base_url,
                cache=cache,
                num_ctx=num_ctx,
            )
        except OllamaRequestError:
            continue  # model-specific failure (load/OOM/404) -> try next model
        except OllamaUnavailable:
            return None, None  # daemon down -> no point trying more models
        if text:
            return text, model
    return None, None


def _cache_key_chat(
    model: str,
    messages: list[dict],
    temperature: float,
    num_predict: int | None = None,
    num_ctx: int | None = None,
) -> str:
    # num_predict bounds output length, so it MUST be in the key — otherwise a
    # short-bound call poisons the cache for a long-bound call with the same prompt.
    # num_ctx too: an overflowed (empty) result must not replay at a larger ctx.
    raw = (
        f"{model}\x1f{temperature}\x1f{num_predict}\x1f{num_ctx}"
        f"\x1f{json.dumps(messages, sort_keys=True, ensure_ascii=False)}"
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def chat(
    messages: list[dict],
    model: str = DEFAULT_GEN_MODEL,
    *,
    temperature: float = 0.2,
    num_predict: int | None = None,
    think: bool = False,
    timeout: float = DEFAULT_TIMEOUT,
    base_url: str = DEFAULT_URL,
    cache: bool = True,
    num_ctx: int | None = None,
) -> str | None:
    """Chat-style completion (/api/chat) with a messages list. Returns the
    assistant message content (stripped), or None if empty/unavailable.
    Cached when temperature <= CACHE_MAX_TEMP. Equivalent to generate() but
    for callers that need system/user role separation. Pass num_ctx to extend
    the context window for large message bodies (e.g. session compaction)."""
    cacheable = cache and temperature <= CACHE_MAX_TEMP
    cached_path: Path | None = None
    if cacheable:
        OLLAMA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cached_path = (
            OLLAMA_CACHE_DIR
            / f"{_cache_key_chat(model, messages, temperature, num_predict, num_ctx)}.txt"
        )
        if cached_path.exists():
            try:
                return cached_path.read_text(encoding="utf-8")
            except OSError:
                pass

    options: dict = {"temperature": temperature}
    if num_predict is not None:
        options["num_predict"] = num_predict
    if num_ctx is not None:
        options["num_ctx"] = num_ctx
    data = _post(
        "/api/chat",
        {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": think,
            "options": options,
        },
        base_url,
        timeout,
    )
    msg = data.get("message")
    content = msg.get("content", "") if isinstance(msg, dict) else ""
    content = _strip_think_tags(str(content).strip()) or None

    if content and cacheable and cached_path is not None:
        try:
            cached_path.write_text(content, encoding="utf-8")
            _prune_cache()
        except OSError:
            pass
    return content


def chat_fallback(
    messages: list[dict],
    models: list[str],
    *,
    temperature: float = 0.2,
    num_predict: int | None = None,
    think: bool = False,
    timeout: float = DEFAULT_TIMEOUT,
    base_url: str = DEFAULT_URL,
    cache: bool = True,
    num_ctx: int | None = None,
) -> tuple[str | None, str | None]:
    """Try each chat model in order; return (content, model_used) or (None, None)."""
    if not models:
        return None, None
    for model in models:
        try:
            content = chat(
                messages,
                model=model,
                temperature=temperature,
                num_predict=num_predict,
                think=think,
                timeout=timeout,
                base_url=base_url,
                cache=cache,
                num_ctx=num_ctx,
            )
        except OllamaRequestError:
            continue  # model-specific failure (load/OOM/404) -> try next model
        except OllamaUnavailable:
            return None, None  # daemon down -> no point trying more models
        if content:
            return content, model
    return None, None


def ocr_image(
    image_bytes: bytes,
    *,
    model: str = DEFAULT_PDF_OCR_MODEL,
    prompt: str = PDF_OCR_PROMPT,
    temperature: float = 0.0,
    num_predict: int = 1024,
    timeout: float = DEFAULT_TIMEOUT,
    base_url: str = DEFAULT_URL,
) -> str | None:
    """OCR a rendered PDF page/image using a vision-capable Ollama model.

    Unlimited-OCR is prompt-sensitive under Ollama: generic prompts like
    "Extract the text" can return an empty response, while the model-specific
    ``ocr [img]`` prompt returns OCR text with optional coordinates.
    """
    if not image_bytes:
        return None
    messages: list[dict] = [
        {
            "role": "user",
            "content": prompt,
            "images": [base64.b64encode(image_bytes).decode("ascii")],
        }
    ]
    data = _post(
        "/api/chat",
        {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {"temperature": temperature, "num_predict": num_predict},
        },
        base_url,
        timeout,
    )
    msg = data.get("message")
    content = msg.get("content", "") if isinstance(msg, dict) else ""
    return _strip_think_tags(str(content).strip()) or None


def _embed_once(
    text: str,
    model: str,
    timeout: float,
    base_url: str,
) -> list[float] | None:
    """Single embedding attempt. Returns the vector, or None if the daemon is
    down, the text is blank, or Ollama returned no `embedding` field."""
    if not text.strip():
        return None
    try:
        data = _post(
            "/api/embeddings",
            {"model": model, "prompt": text},
            base_url,
            timeout,
        )
    except OllamaUnavailable:
        return None
    vec = data.get("embedding")
    if isinstance(vec, list) and vec:
        return [float(x) for x in vec]
    return None


def embed(
    text: str,
    *,
    model: str = DEFAULT_EMBED_MODEL,
    timeout: float = EMBED_TIMEOUT,
    base_url: str = DEFAULT_URL,
) -> list[float] | None:
    """Embedding vector for a single text via /api/embeddings.
    Returns None if the daemon is down or returned nothing.

    Resilience: a long token-dense chunk (commit hashes, dotted version strings
    like v1.5.21.18.19.54) can exceed mxbai-embed-large's 512-token context
    even under MAX_CHUNK_CHARS (1200) — Ollama then answers HTTP 500 and the
    chunk would be skipped. Rather than drop it, retry once with the text
    halved: a half-content embedding still serves recall better than none.
    """
    vec = _embed_once(text, model, timeout, base_url)
    if vec is not None:
        return vec
    if len(text) > 512:
        vec = _embed_once(text[: len(text) // 2], model, timeout, base_url)
        if vec is not None:
            return vec
    return None


def _cli() -> int:
    ap = argparse.ArgumentParser(description="Shared local-Ollama client")
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("is-alive", help="Exit 0 if Ollama daemon answers")
    g = sub.add_parser("generate", help="One-shot completion")
    g.add_argument("--prompt", required=True)
    g.add_argument("--model", default=DEFAULT_GEN_MODEL)
    g.add_argument("--temperature", type=float, default=0.2)
    g.add_argument("--no-cache", action="store_true")
    g.add_argument("--base-url", default=DEFAULT_URL)
    o = sub.add_parser("ocr-image", help="OCR one rendered page/image")
    o.add_argument("--image", required=True, help="PNG/JPEG path")
    o.add_argument("--model", default=DEFAULT_PDF_OCR_MODEL)
    o.add_argument("--prompt", default=PDF_OCR_PROMPT)
    o.add_argument("--base-url", default=DEFAULT_URL)
    e = sub.add_parser("embed", help="Embedding vector for one text")
    e.add_argument("--text", required=True)
    e.add_argument("--model", default=DEFAULT_EMBED_MODEL)
    e.add_argument("--base-url", default=DEFAULT_URL)
    args = ap.parse_args()

    if args.command == "is-alive":
        return 0 if is_alive(base_url=getattr(args, "base_url", DEFAULT_URL)) else 1

    if args.command == "generate":
        try:
            out = generate(
                args.prompt,
                model=args.model,
                temperature=args.temperature,
                base_url=args.base_url,
                cache=not args.no_cache,
            )
        except OllamaUnavailable as exc:
            print(f"ollama unavailable: {exc}", file=sys.stderr)
            return 2
        if out:
            print(out)
        return 0

    if args.command == "embed":
        vec = embed(args.text, model=args.model, base_url=args.base_url)
        if vec is None:
            print("ollama unavailable or empty embedding", file=sys.stderr)
            return 2
        print(json.dumps(vec))
        return 0

    if args.command == "ocr-image":
        try:
            out = ocr_image(
                Path(args.image).read_bytes(),
                model=args.model,
                prompt=args.prompt,
                base_url=args.base_url,
            )
        except OllamaUnavailable as exc:
            print(f"ollama unavailable: {exc}", file=sys.stderr)
            return 2
        if out:
            print(out)
            return 0
        print("ollama returned empty OCR output", file=sys.stderr)
        return 3

    return 0


def main() -> int:
    """Public CLI entry point (alias for ``_cli``); used by the console script."""
    return _cli()


if __name__ == "__main__":
    sys.exit(_cli())
