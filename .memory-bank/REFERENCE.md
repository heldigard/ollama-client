# REFERENCE — ollama-client

## Default models (aligned ollama-bench RANKING.md, 2026-07-18)
Env overrides (`OLLAMA_*`) win; fallbacks below match bench PRIMARY roles.
Source: `~/ollama-bench/RANKING.md` · host map `~/.config/dev/ollama-roles.json`.
**Installed multi-model library is intentional winners — not for bulk prune.**

| Constant | Default | Bench role |
|----------|---------|------------|
| `DEFAULT_GEN_MODEL` | `cryptidbleh/gemma4-claude-opus-4.6:latest` | improve PRIMARY |
| `DEFAULT_SUMMARY_MODEL` | `hf.co/TeichAI/Qwen3.5-9B-Fable-5-v1-GGUF:Q4_K_M` | codeq_sum PRIMARY (was e4b; Qwythos is FALLBACK) |
| `DEFAULT_STRUCTURED_MODEL` | `SetneufPT/Qwopus3.5-4B-Coder-MTP_Q4_64k_8GB-GPU:latest` | tool_call / structured PRIMARY |
| `DEFAULT_PDF_OCR_MODEL` | `hf.co/sahilchachra/Unlimited-OCR-GGUF:Q4_K_M` | pdf_ocr PRIMARY |
| `DEFAULT_EMBED_MODEL` | `embeddinggemma` | embedding PRIMARY |
| `DEFAULT_URL` | env `OLLAMA_URL` / `OLLAMA_HOST` or `http://localhost:11434` | daemon |
| `PDF_OCR_PROMPT` | `ocr [img]` | Unlimited-OCR trigger |

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
