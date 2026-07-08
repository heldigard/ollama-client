# topics index — ollama-client

| Topic | What |
|-------|------|
| [[shim-pattern]] | How the compat shim wires the graduated package to `import ollama_client` from `~/.claude/scripts/`. Package-aware spec + submodule_search_locations. |
| [[mocking-semantics]] | Why `generate_fallback`/`chat_fallback` use late-binding — preserves `@patch("ollama_client.generate")` interception across the package split. |
| [[cache-policy]] | Deterministic-prompt sha256 cache rules; why embeddings are never cached; num_ctx/num_predict in the key. |
