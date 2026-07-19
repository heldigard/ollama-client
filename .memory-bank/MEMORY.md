# MEMORY — ollama-client

> Project memory index. One line per fact. Detail lives in `topics/` + sibling
> core files. Loaded at session start.

## Identity
- **What**: shared local-Ollama HTTP client for the cross-CLI harness (Claude/Codex/Gemini/OpenCode).
- **Public**: https://github.com/heldigard/ollama-client · license MIT · SemVer 1.2.1.
- **Graduated** 2026-07-08 from `~/.claude/scripts/ollama_client.py` (was inline across 4 hooks/scripts).
- **Stack**: Python ≥3.11, **stdlib only** (urllib/hashlib/json/argparse/base64/re), zero runtime deps.

## Architecture (2026-07-08 refactor)
- **Vertical-slice package** `ollama_client/` (was flat 605-LOC module). Submodules by responsibility:
  `_config`, `_version`, `_transport`, `_cache`, `generation`, `chat`, `embedding`, `vision`, `models`, `cli`.
- `__init__.py` re-exports the **full public contract** → `import ollama_client` identical to flat module.
- **Late-binding** in `generate_fallback`/`chat_fallback` → `@patch("ollama_client.generate")` still
  intercepts internal calls (preserves ecosystem test semantics). See [[mocking-semantics]].
- Shim `~/.claude/scripts/ollama_client.py` loads the package via `spec_from_file_location` +
  `submodule_search_locations` (package-aware, no sys.path mutation). See [[shim-pattern]].

## Four ecosystem consumers (contract frozen)
- `codeq` (~/.claude/codeq) — imports `ollama_client` directly: `is_alive`, `generate`, `DEFAULT_URL`.
- `prompt-improve` (~/.claude/prompt-improve) — via `compat` layer: `chat`, `OllamaRequestError`,
  `OllamaUnavailable`, `DEFAULT_URL`.
- `web-research` (~/.claude/web-research) — via `compat` + tests `@patch("ollama_client.{embed,is_alive,generate}")`.
- `smart-trim` (~/.claude/hooks/smart-trim.py) — hook consumer.

## Commands
- Install dev: `pip install -e .[test]` (or `uv sync`).
- Test: `pytest` · Lint: `ruff check .` · Smoke: `python3 -c "import ollama_client; ollama_client.is_alive()"`.
- Console: `ollama-client is-alive | models | ps | generate --prompt "..." | chat --prompt "..." | embed --text "..." | ocr-image --image x.png`.

## Entry points
- Console: `ollama-client` → `ollama_client:main`.
- Library: `import ollama_client` (the shim path wires the four consumers).
- `require(min_version)` SemVer gate; version SSOT = `_version.py` (hatch dynamic).
- Public history: `CHANGELOG.md`.

## Related topics
- [[shim-pattern]] — how the graduated package stays wired to the harness.
- [[mocking-semantics]] — late-binding preserves `@patch` across the refactor.
- [[cache-policy]] — deterministic-prompt sha256 cache, embeddings NOT cached.
