# REFERENCE — ollama-client

## Default models (2026-07-08)
| Constant | Value | Use |
|----------|-------|-----|
| `DEFAULT_GEN_MODEL` | `cryptidbleh/gemma4-claude-opus-4.6:latest` | default completion (3.4GB, ~115 tok/s) |
| `DEFAULT_SUMMARY_MODEL` | `batiai/gemma4-e4b:q4` | one-line code summaries |
| `DEFAULT_STRUCTURED_MODEL` | `SetneufPT/Qwopus3.5-4B-Coder-MTP_Q4_64k_8GB-GPU:latest` | structured output/tool calls |
| `DEFAULT_PDF_OCR_MODEL` | `hf.co/sahilchachra/Unlimited-OCR-GGUF:Q4_K_M` | vision OCR |
| `DEFAULT_EMBED_MODEL` | `embeddinggemma` | 768-d vectors (MRR 0.724 winner) |
| `PDF_OCR_PROMPT` | `ocr [img]` | model-specific OCR trigger |

## Config constants
| Constant | Value | Meaning |
|----------|-------|---------|
| `DEFAULT_URL` | `http://localhost:11434` | daemon base |
| `DEFAULT_TIMEOUT` | `120` | generate/chat seconds |
| `EMBED_TIMEOUT` | `60` | embed seconds |
| `CACHE_MAX_TEMP` | `0.3` | cache only near-deterministic |
| `CACHE_MAX_ENTRIES` | `2000` | prune oldest beyond |
| `OLLAMA_CACHE_DIR` | `~/.claude/state/ollama-cache/` | cache root |

## Version
- `__version__ = "1.2.0"` (SemVer). `require(min)` raises `RuntimeError` if older.

## CLI (`ollama-client`)
```
ollama-client is-alive [--base-url URL]
ollama-client generate --prompt "..." [--model X] [--temperature 0.2] [--no-cache] [--base-url URL]
ollama-client chat --prompt "..." [--system "..."] [--model X] [--temperature 0.2] [--num-predict N] [--no-cache] [--base-url URL]
ollama-client embed --text "..." [--model X] [--base-url URL]
ollama-client ocr-image --image path.png [--model X] [--prompt "..."] [--base-url URL]
```
Exit codes: 0 ok · 1 is-alive false · 2 unavailable · 3 empty OCR.

## Consumer wiring
- Shim: `~/.claude/scripts/ollama_client.py` → loads `~/ollama-client/ollama_client/` package.
- Override project location: `OLLAMA_CLIENT_HOME=/path/to/ollama-client`.
- codeq: `~/codeq/src/codeq/shared/llm.py` (direct import).
- prompt-improve: `~/prompt-improve/src/prompt_improve/shared/config.py` + `features/improve.py` (compat).
- web-research: `~/web-research/src/web_research/shared/compat.py`.

## Re-exported internals (NOT public contract, but kept for mock-patching parity)
`_post`, `_normalize_base_url`, `_cache_key`, `_cache_key_chat`, `_prune_cache`,
`_strip_think_tags`, `_embed_once`, `_parse_version`, `_cli`.
