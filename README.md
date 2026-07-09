# ollama-client

Shared local-Ollama HTTP client for the cross-CLI harness. Single source of
truth for talking to a local Ollama daemon (`http://localhost:11434`).

**Graduated** from `~/.claude/scripts/ollama_client.py` (originally a flat
script consumed by four projects: `codeq`, `smart-trim`, `prompt-improve`,
`web-research`). This project gives it a SemVer contract (`require()`), a
`pyproject.toml`, a real install path, and a **vertical-slice package** layout
(submodules: `_config`, `_version`, `_transport`, `_cache`, `generation`,
`chat`, `embedding`, `vision`, `cli`) — mirroring how `cheap-llm` graduated
before it. The original path remains at `~/.claude/scripts/ollama_client.py`
as a thin **shim** that re-exports from the package here, so every existing
consumer keeps working untouched (`import ollama_client` is byte-identical).

Public repo: https://github.com/heldigard/ollama-client

## Operations

| Function | What |
|----------|------|
| `is_alive(base_url)` | daemon reachability probe (`http/https` base URLs only) |
| `generate(prompt, model, temperature, ...)` | one-shot completion (cached when temp ≤ `CACHE_MAX_TEMP`) |
| `generate_fallback(prompt, models, ...)` | try a model list in order → `(text, model)` |
| `chat(messages, model, ...)` / `chat_fallback(...)` | chat completion; retries `/api/generate` only for Ollama's template-parser HTTP 400 |
| `embed(text, model, base_url)` | vector via `/api/embeddings` |
| `ocr_image(bytes, model, prompt, base_url)` | vision OCR of a PNG/JPEG |

Cache: deterministic prompts keyed on `sha256(model|temperature|prompt)`, stored
under `OLLAMA_CACHE_DIR` (`~/.claude/state/ollama-cache/`), pruned beyond
`CACHE_MAX_ENTRIES`.

## Versioned contract

```python
import ollama_client
ollama_client.require("1.1")          # raises RuntimeError on drift
ollama_client.__version__             # "1.1.0"
```

Consumers gate with `require("<min>")` to fail fast instead of hitting a cryptic
mid-run error on version drift.

Role defaults are intentionally separate: `DEFAULT_SUMMARY_MODEL` is the
one-line summary champion, while `DEFAULT_STRUCTURED_MODEL` is the structured
tool-call champion. A single "best model" constant would hide that distinction.

## Install

```bash
pip install -e .          # or: uv sync
```

Console script: `ollama-client is-alive [--base-url http://localhost:11434] | generate --prompt "..." | embed --text "..." | ocr-image --image x.png`

Also: `python -m ollama_client <command>` and `ollama-client --version`.

## License

MIT © heldigard
