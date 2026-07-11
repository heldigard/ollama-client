# Project: ollama-client

Shared local-Ollama HTTP client for the cross-CLI harness. Graduated from
`~/.claude/scripts/ollama_client.py` (2026-07-08). Public:
https://github.com/heldigard/ollama-client

## Commands
- Install (dev): `pip install -e .[test]` (or `uv sync --extra test`)
- Test: `pytest`
- Lint: `ruff check .`
- Format check (read-only): `ruff format --check .`
- Smoke: `python3 -m ollama_client is-alive [--base-url http://localhost:11434]` → exit 0 if daemon answers
- Console: `ollama-client is-alive | generate --prompt "..." | chat --prompt "..." [--system "..."] | embed --text "..." | ocr-image --image x.png`

## Stack
- Python ≥ 3.11
- **stdlib only** (urllib.request, hashlib, json, argparse, base64, re) — zero runtime deps
- **Vertical-slice package** `ollama_client/` (one responsibility: talk to Ollama),
  split by sub-concern: `_config`, `_version`, `_transport`, `_cache`,
  `generation`, `chat`, `embedding`, `vision`, `cli`. `__init__.py` re-exports
  the full public contract so `import ollama_client` is identical to the prior
  flat module.

## Entry Points
- Console script: `ollama-client` → `ollama_client:main` (also `python -m ollama_client`)
- Library: `import ollama_client` (the four ecosystem consumers — codeq,
  smart-trim, prompt-improve, web-research — import via the
  `~/.claude/scripts/ollama_client.py` **shim**, which re-exports from here)
- `require(min_version)` SemVer gate for consumers
- `base_url` accepts only `http://` / `https://` Ollama endpoints. Legacy full
  endpoint URLs (`.../api/generate`, `.../api/chat`, `.../api/embeddings`) are
  normalized back to the daemon base.

## Architecture (why the package split)
- Each submodule is one sub-concern: transport (HTTP + exceptions), cache
  (key + prune + think-strip), version (SemVer gate), then one file per domain
  op (generation, chat, embedding, vision) + the CLI dispatcher. `__init__.py`
  is the public surface (re-exports + `__all__`).
- **Late-binding** in `generate_fallback` / `chat_fallback` (they late-import
  `generate` / `chat` from the public package) preserves the flat-module mocking
  semantics: `@patch("ollama_client.generate")` still intercepts the internal
  call. See `.memory-bank/topics/mocking-semantics.md`.
- Decisions, cache policy, and the shim pattern live in `.memory-bank/`.

## Things That Look Wrong But Aren't
- **`generate` / `chat` have 6-9 params** — frozen public API the four consumers
  already depend on; collapsing to a param object would break the contract
  (`vs-soft-allow` marker at the top of `generation.py` / `chat.py` / `vision.py`).
- **`__init__.py` "unused imports" (F401)** — silenced via ruff `per-file-ignores`:
  internals (`_post`, `_cache_key`, `_embed_once`, …) are re-exported on purpose
  for `@patch` parity with the prior flat module.
- **`main()` is a 1-line alias for `_cli()`** — exists so the console-script
  entry point has a public name.
- **`chat` / `generation` name-collision** — both a submodule and a function.
  `from ollama_client import chat` gives the FUNCTION (the `__init__` re-export
  wins). Tests reach the module via `importlib.import_module("ollama_client.chat")`.

## Key Decisions
- **SemVer 1.1.0 + `require()`** — consumers fail fast on version drift instead
  of cryptic mid-run errors. Mirrors `cheap_llm`.
- **Narrow chat fallback** — only Ollama's HTTP 400 automatic template-parser
  failure retries through `/api/generate`; unrelated bad requests still raise.
- **Vertical-slice package, contract preserved** — split for navigability;
  `__init__` re-export keeps `import ollama_client` byte-identical (verified by
  `tests/test_contract.py`).
- **Shim at `~/.claude/scripts/ollama_client.py`** — loads the package via
  `spec_from_file_location(submodule_search_locations=…)` so relative imports
  resolve and the wired import path is preserved for the four consumers.

## Cache policy
Deterministic prompts (temperature ≤ `CACHE_MAX_TEMP`) keyed on
`sha256(model|temperature|prompt[|messages])`, stored under `OLLAMA_CACHE_DIR`
(`~/.claude/state/ollama-cache/`), pruned beyond `CACHE_MAX_ENTRIES`.
Embeddings are NOT cached (callers need fresh vectors).
