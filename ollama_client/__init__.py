"""ollama_client — shared local-Ollama HTTP client for the cross-CLI harness.

Single source of truth for talking to the local Ollama daemon
(http://localhost:11434). Consolidates the four previously inline
``urllib.request`` implementations (codeq summary/context/relations, smart-trim,
prompt-improve, web-research) so they share one HTTP path, one response cache,
one fallback chain, and one embed call.

Public operations:
  - :func:`generate` / :func:`generate_fallback`  — one-shot text completion
  - :func:`chat` / :func:`chat_fallback`          — messages-style completion
  - :func:`embed`                                  — vector for a single text
  - :func:`ocr_image`                              — vision OCR of an image
  - :func:`is_alive`                               — daemon liveness

This is a vertical-slice package. ``__init__`` re-exports the full public
contract so ``import ollama_client`` behaves identically to the prior flat
module — the four ecosystem consumers migrate with zero breakage. Internals
(``_post``, ``_cache_key``, ...) are re-exported too so existing
``@patch("ollama_client.<name>")`` call sites keep resolving.

Cache policy: deterministic prompts (``temperature <= CACHE_MAX_TEMP``) are
keyed on ``sha256(model|temperature|prompt[|messages])`` and replayed without a
network round-trip. Higher temperatures bypass the cache. Embeddings are NOT
cached (callers manage their own index).
"""

from __future__ import annotations

# --- Cache internals (re-exported for mock-patching parity with the flat module) ---
from ._cache import (
    _cache_key,
    _cache_key_chat,
    _cache_path,
    _prune_cache,
    _strip_think_tags,
)

# --- Config (public constants) ---
from ._config import (
    CACHE_MAX_ENTRIES,
    CACHE_MAX_TEMP,
    DEFAULT_EMBED_MODEL,
    DEFAULT_GEN_MODEL,
    DEFAULT_PDF_OCR_MODEL,
    DEFAULT_STRUCTURED_MODEL,
    DEFAULT_SUMMARY_MODEL,
    DEFAULT_TIMEOUT,
    DEFAULT_URL,
    EMBED_TIMEOUT,
    OLLAMA_CACHE_DIR,
    PDF_OCR_PROMPT,
)

# --- Transport + exceptions ---
from ._transport import (
    OllamaRequestError,
    OllamaUnavailable,
    _normalize_base_url,
    _post,
    is_alive,
)

# --- Versioning (SemVer gate) ---
from ._version import __version__, _parse_version, require

# --- Domain operations ---
from .chat import chat, chat_fallback

# --- CLI ---
from .cli import _cli, main
from .embedding import _embed_once, embed
from .generation import generate, generate_fallback
from .vision import ocr_image

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
    "main",
    "DEFAULT_URL",
    "DEFAULT_GEN_MODEL",
    "DEFAULT_SUMMARY_MODEL",
    "DEFAULT_STRUCTURED_MODEL",
    "DEFAULT_PDF_OCR_MODEL",
    "PDF_OCR_PROMPT",
    "DEFAULT_EMBED_MODEL",
    "DEFAULT_TIMEOUT",
    "EMBED_TIMEOUT",
    "CACHE_MAX_TEMP",
    "CACHE_MAX_ENTRIES",
    "OLLAMA_CACHE_DIR",
]
