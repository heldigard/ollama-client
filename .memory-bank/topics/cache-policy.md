# cache-policy — deterministic-prompt response cache

> status: live (reference, never archive)

## Rule
Generation (`generate`) and chat (`chat`) responses are cached **iff**
`temperature <= CACHE_MAX_TEMP` (0.3). Keyed on `sha256(...)`, stored as
`<hash>.txt` under `OLLAMA_CACHE_DIR` (`~/.claude/state/ollama-cache/`).

## Why temperature gate
Higher-temperature outputs are non-deterministic — replaying a cached one would
silently degrade quality. Only near-deterministic calls cache.

## Key composition (load-bearing details)
- `generate`: `sha256(model | temperature | num_ctx | prompt)`.
- `chat`: `sha256(model | temperature | num_predict | num_ctx | think | messages_json_sorted)`.

`num_ctx` is in the key because a call that **overflowed** context (returned
empty) must NOT be replayed for the same prompt at a larger ctx.
`num_predict` (chat) bounds output length — a short-bound call must not poison a
long-bound call. `think` changes the model's reasoning mode and output, so modes
must not share entries. `messages` is JSON-serialized with `sort_keys=True` for
stable hashing.

## Pruning
`_prune_cache()` runs after each cache write, deleting oldest-mtime entries
beyond `CACHE_MAX_ENTRIES` (2000). Best-effort (OSError-tolerant).

## Embeddings: NEVER cached
`embed()` always hits the daemon. Callers (semantic-search indexes) usually need
fresh vectors and manage their own index. Caching here would add complexity for
no win — embeddings are deterministic per (model, text) but cheap enough and
callers want freshness.

## Corrupt-entry tolerance
If a cache file read fails (OSError) or the cached content is unreadable, the
call falls through to a live request (no crash). Cache writes that fail are
silently ignored. Cache-root creation is also fail-open: an unusable path disables
caching for that call but never blocks the live Ollama request.

Related: [[shim-pattern]], [[mocking-semantics]].
