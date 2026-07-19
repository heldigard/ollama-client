# ollama-client

Shared local-Ollama HTTP client for the cross-CLI harness. Single source of
truth for talking to a local Ollama daemon (`http://localhost:11434`).

**Graduated** from `~/.claude/scripts/ollama_client.py` (originally a flat
script consumed by four projects: `codeq`, `smart-trim`, `prompt-improve`,
`web-research`). This project gives it a SemVer contract (`require()`), a
`pyproject.toml`, a real install path, and a **vertical-slice package** layout
(submodules: `_config`, `_version`, `_transport`, `_cache`, `generation`,
`chat`, `embedding`, `vision`, `models`, `cli`) — mirroring how `cheap-llm` graduated
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
| `embed(text, model, base_url)` | vector via `/api/embeddings` (404 → `/api/embed` + `input`) |
| `list_models(base_url)` | installed models (`GET /api/tags`) |
| `list_running(base_url)` | loaded models (`GET /api/ps`) |
| `ocr_image(bytes, model, prompt, base_url)` | vision OCR of a PNG/JPEG |

Cache: deterministic prompts are stored under `OLLAMA_CACHE_DIR`
(`~/.claude/state/ollama-cache/` by default; override with env `OLLAMA_CACHE_DIR`
for XDG) and pruned beyond `CACHE_MAX_ENTRIES` (env `OLLAMA_CACHE_MAX_ENTRIES`).
Generate keys include `(model, temperature, num_ctx, prompt)`; chat keys include
`(model, temperature, num_predict, num_ctx, think, messages)`. Embed timeout:
`OLLAMA_EMBED_TIMEOUT` (default 60s).

## Versioned contract

```python
import ollama_client
ollama_client.require("1.2")          # raises RuntimeError on drift
ollama_client.__version__             # "1.2.1"
```

Consumers gate with `require("<min>")` to fail fast instead of hitting a cryptic
mid-run error on version drift.

Role defaults are intentionally separate — a single "best model" constant would
hide the distinction. Current champions (wiring validated 2026-07-18; source
`~/ollama-bench/results_wiring_validation_20260709.md`):

| constant | model | role |
|---|---|---|
| `DEFAULT_GEN_MODEL` | `cryptidbleh/gemma4-claude-opus-4.6` | general generation (`chat`→`generate` fallback) |
| `DEFAULT_SUMMARY_MODEL` | `hf.co/TeichAI/Qwen3.5-9B-Fable-5-v1-GGUF:Q4_K_M` | one-line code summaries |
| `DEFAULT_STRUCTURED_MODEL` | `SetneufPT/Qwopus3.5-4B-Coder-MTP` | structured tool-call |
| `DEFAULT_PDF_OCR_MODEL` | `hf.co/sahilchachra/Unlimited-OCR-GGUF:Q4_K_M` | PDF OCR |
| `DEFAULT_EMBED_MODEL` | `embeddinggemma` | embeddings |

## Install

```bash
pip install -e .          # or: uv sync
```

Console script: `ollama-client is-alive [--base-url http://localhost:11434] | models [--json] | ps [--json] | generate --prompt "..." [--num-ctx N] [--timeout S] | chat --prompt "..." [--system "..."] [--think] [--num-ctx N] | embed --text "..." | ocr-image --image x.png`

Also: `python -m ollama_client <command>` and `ollama-client --version`.

## License

MIT © heldigard
