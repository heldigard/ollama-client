# CONTEXT — ollama-client

## Why this project exists
Four cross-CLI harness components (`codeq` summary/context/relations, `smart-trim`
PreCompact, `prompt-improve`, `web-research`) each had an inline `urllib.request`
call to the local Ollama daemon. Four copies → four bugs, four cache policies,
four fallback chains. This project is the **single source of truth** for talking
to `http://localhost:11434`: one HTTP path, one response cache, one fallback
chain, one embed call.

## Graduation (2026-07-08)
Was `~/.claude/scripts/ollama_client.py` (a script in the harness). Promoted to a
standalone SemVer'd project at `~/ollama-client/` so it can be:
- `pip install`ed / `uv sync`ed independently,
- version-gated by consumers (`require("1.0.0")`),
- released publicly (github.com/heldigard/ollama-client),
- tested in isolation (`pytest`).

The wired import path `import ollama_client` from `~/.claude/scripts/` is preserved
by a **compat shim** that re-exports from the graduated project — consumers migrate
incrementally without breaking. See [[shim-pattern]].

## Architecture refactor (2026-07-08, same session)
Flat 605-LOC module → **vertical-slice package** `ollama_client/`. Decision: the
module was marked `vs-soft-allow` (single-responsibility HTTP client); the user
explicitly requested vertical slices and the shim already declared
"(vertical-slice project)". Resolved by: package split by responsibility
(transport / cache / version / generation / chat / embedding / vision / models / cli)
with `__init__.py` re-exporting the **identical public contract**. Zero breaking
change to the four consumers. Late-binding preserves mocking semantics. See
[[mocking-semantics]].

## Operations (public API)
- `generate(prompt, model, *, temperature, timeout, base_url, cache, num_ctx) -> str|None`
- `generate_fallback(prompt, models, *, ...) -> tuple[str|None, str|None]`
- `chat(messages, model, *, temperature, num_predict, think, timeout, base_url, cache, num_ctx) -> str|None`
- `chat_fallback(messages, models, *, ...) -> tuple[str|None, str|None]`
- `embed(text, *, model, timeout, base_url) -> list[float]|None` — `/api/embeddings`, 404→`/api/embed`
- `list_models(base_url, *, timeout) -> list[dict]` — `GET /api/tags`
- `list_running(base_url, *, timeout) -> list[dict]` — `GET /api/ps`
- `ocr_image(image_bytes, *, model, prompt, temperature, num_predict, timeout, base_url) -> str|None`
- `is_alive(base_url, timeout) -> bool`
- `require(min_version) -> str` · `__version__`
- Exceptions: `OllamaUnavailable` (daemon down), `OllamaRequestError(status, body)` (model-specific, subclass).

## Endpoints used
- `GET  /api/tags` — liveness.
- `POST /api/generate` — one-shot completion.
- `POST /api/chat` — messages completion + vision OCR.
- `POST /api/embeddings` — vector.

## Error taxonomy (load-bearing)
- `URLError`/`OSError` → `OllamaUnavailable` (daemon unreachable → abort fallback chain).
- `HTTPError` → `OllamaRequestError` (model-specific → try NEXT model).
- `TimeoutError` → `OllamaRequestError(408)` (slow cold load → try next, NOT abort).

## Host environment
- Ollama daemon on WSL2 `localhost:11434`, alive (verified is-alive exit 0).
- Cache dir `~/.claude/state/ollama-cache/`.
