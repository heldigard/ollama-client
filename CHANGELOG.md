# Changelog

All notable changes to `ollama-client` are documented here. The package stays
**stdlib-only** (zero runtime dependencies) and preserves the frozen public
function signatures that the harness consumers depend on.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed
- Internal: collapsed the duplicated generate/chat cache replay+write block into
  shared `_cache_replay` / `_cache_store` helpers (one fail-open + empty-entry-drop
  policy instead of two drifting copies). Public signatures unchanged.
- Internal: `embed` retry-once-with-halved-text logic consolidated into
  `_retry_halved`; `embed` is now total (never raises) and no longer re-halves
  twice on a repeated failure.
- CI: run the quality gate (pytest / ruff / pyright) across Python 3.11-3.14 to
  match the declared classifiers; wheel build + smoke stay on 3.11.

## [1.2.1] — 2026-07-19

### Added
- Library API: `list_models` / `list_running` (not CLI-only).
- CLI: `ps` subcommand (loaded models via `GET /api/ps`).
- Env knobs: `OLLAMA_EMBED_TIMEOUT`, `OLLAMA_CACHE_MAX_ENTRIES`, `OLLAMA_CACHE_DIR`.
- Packaging: Python 3.14 classifier; hatch dynamic version from `_version.py`.
- CI: concurrency cancel-in-progress; Dependabot for GitHub Actions.
- Public `CHANGELOG.md` and `[project.urls]` metadata.

### Changed
- Embed fallback default tag aligned to host map (`embeddinggemma:latest`).
- `_get` surfaces HTTP response body on errors (parity with `_post`).

### Fixed
- Embed: on HTTP 404 only, retry `/api/embed` + `input`; parse `embedding` or
  `embeddings[0]`.

## [1.2.0] — 2026-07-11

### Added
- CLI `chat` subcommand (system/user roles, `--num-predict`, `--think`).
- Opt-in cross-process serialize lock (`OLLAMA_SERIALIZE_LOCK` / `OLLAMA_LOCK_FILE`).
- Transient HTTP retry (one attempt) for 408/5xx before raising.
- Chat cache key includes `think` (legacy keys intentionally invalidated).
- GitHub Actions hermetic quality gate (pytest, ruff, pyright, wheel smoke).

### Fixed
- Empty cache entries dropped and regenerated.
- Orphan `.tmp` cache files swept after 1h in `_prune_cache`.
- Cache-root creation fails open (live request still runs).
- Invalid UTF-8 responses map to `OllamaRequestError(502)`.
- `ocr-image` handles `OSError` with exit 2.

## [1.1.0] — 2026-07-09

### Added
- Role-specific `DEFAULT_SUMMARY_MODEL` distinct from generation default.
- Narrow chat→generate fallback only for Ollama template-parser HTTP 400.

### Fixed
- Unrelated HTTP 400 errors remain visible (no silent generate fallback).

## [1.0.0] — 2026-07-08

### Added
- Graduated from `~/.claude/scripts/ollama_client.py` to versioned package.
- Vertical-slice layout: `_config`, `_version`, `_transport`, `_cache`,
  `generation`, `chat`, `embedding`, `vision`, `cli` (+ later `models`, `_lock`).
- Public ops: `generate` / `generate_fallback`, `chat` / `chat_fallback`,
  `embed`, `ocr_image`, `is_alive`, `require()`.
- Deterministic-prompt response cache; embeddings never cached.
- Console script `ollama-client` and `python -m ollama_client`.
- Shim at `~/.claude/scripts/ollama_client.py` keeps four consumers working.
