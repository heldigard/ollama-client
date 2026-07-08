# vs-soft-allow  — single-responsibility: the flat configuration table for the
# Ollama client (defaults + cache knobs). A config table is one reason to change.
"""Default models, endpoints, and cache knobs for the Ollama client.

Single source of truth for every tunable the package uses. Importing a name
from here (not from the package root) keeps the dependency graph honest and
makes ``git blame`` on a default land on the one file that owns it.
"""

from __future__ import annotations

from pathlib import Path

#: Daemon base URL. Override per-call with ``base_url=``.
DEFAULT_URL = "http://localhost:11434"

#: Default completion model (3.4GB, ~115 tok/s, universal fallback 2026-07-08).
DEFAULT_GEN_MODEL = "cryptidbleh/gemma4-claude-opus-4.6:latest"

#: Structured/code summary model (codeq summary/context/relations).
DEFAULT_STRUCTURED_MODEL = "SetneufPT/Qwopus3.5-4B-Coder-MTP_Q4_64k_8GB-GPU:latest"

#: Vision OCR model (Unlimited-OCR GGUF).
DEFAULT_PDF_OCR_MODEL = "hf.co/sahilchachra/Unlimited-OCR-GGUF:Q4_K_M"

#: Model-specific OCR trigger prompt. Unlimited-OCR is prompt-sensitive under
#: Ollama: generic prompts ("Extract the text") return empty; ``ocr [img]`` works.
PDF_OCR_PROMPT = "ocr [img]"

#: Embedding model (768-d, MRR 0.724 winner 2026-06-28). Tied with
#: nomic-embed-text on context (both ~2K functional, NOT the 8K the modelfile
#: claims); embeddinggemma wins on quality + consistency.
DEFAULT_EMBED_MODEL = "embeddinggemma"

#: Generate/chat request timeout (seconds). Cold model loads can take ~60s.
DEFAULT_TIMEOUT = 120

#: Embedding request timeout (seconds). Embeds are fast; fail sooner.
EMBED_TIMEOUT = 60

#: Cache only near-deterministic outputs (temperature at/below this).
CACHE_MAX_TEMP = 0.3

#: Prune oldest cache entries beyond this count.
CACHE_MAX_ENTRIES = 2000

#: Response cache root. Deterministic prompts (temp <= CACHE_MAX_TEMP) are
#: replayed from here without a network round-trip. Embeddings are NOT cached.
OLLAMA_CACHE_DIR = Path.home() / ".claude" / "state" / "ollama-cache"
