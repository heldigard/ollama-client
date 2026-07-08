# Project: ollama-client

Shared local-Ollama HTTP client for the cross-CLI harness. Graduated from
`~/.claude/scripts/ollama_client.py` (2026-07-08). Public:
https://github.com/heldigard/ollama-client

## Commands
- Install (dev): `pip install -e .[test]` (or `uv sync`)
- Test: `pytest`
- Lint: `ruff check .`
- Smoke: `python3 ollama_client.py is-alive` → exit 0 if daemon answers
- Console: `ollama-client is-alive | generate --prompt "..." | embed --text "..."`

## Stack
- Python ≥ 3.11
- **stdlib only** (urllib.request, hashlib, json, argparse) — zero runtime deps
- Flat module `ollama_client.py` (one cohesive responsibility: talk to Ollama)

## Entry Points
- Console script: `ollama-client` → `ollama_client:main`
- Library: `import ollama_client` (the four ecosystem consumers: codeq,
  smart-trim, prompt-improve, web-research import it via the
  `~/.claude/scripts/ollama_client.py` **shim**, which re-exports from here)
- `require(min_version)` SemVer gate for consumers

## Things That Look Wrong But Aren't
- **`generate`/`chat` have 6-10 params** — this is the frozen public API the four
  consumers already depend on; collapsing to a param object would break the
  contract. `vs-soft-allow` documents it at the top of the module.
- **Nesting depth 4** in a few spots — same graduated-lib rationale.
- **`main()` is a 1-line alias for `_cli()`** — exists so the console-script
  entry point (`ollama_client:main`) has a public name.

## Key Decisions
- **SemVer 1.0.0 + `require()`** — consumers fail fast on version drift instead
  of cryptic mid-run errors. Mirrors `cheap_llm`.
- **Flat module, not vertical-slice** — mirrors `cheap-llm`; the file is one
  cohesive HTTP client (541 LOC), splitting by use-case would scatter one concern.
- **Shim at `~/.claude/scripts/ollama_client.py`** — preserves the wired import
  path so consumers migrate incrementally without breaking.

## Cache policy
Deterministic prompts (temperature ≤ `CACHE_MAX_TEMP`) keyed on
`sha256(model|temperature|prompt[|messages])`, stored under
`OLLAMA_CACHE_DIR` (`~/.claude/state/ollama-cache/`), pruned beyond
`CACHE_MAX_ENTRIES`. Embeddings are NOT cached (callers need fresh vectors).
