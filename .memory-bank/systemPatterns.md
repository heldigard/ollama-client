# systemPatterns — ollama-client decisions

## Decision: vertical-slice PACKAGE, not flat module (2026-07-08)
**Choice**: split the 605-LOC `ollama_client.py` into a package by responsibility.
**Why**: user explicitly requested vertical slices; the shim already declared
"(vertical-slice project)"; sub-responsibilities (transport / cache / version /
generation / chat / embedding / vision / cli) are genuine boundaries that improve
navigability without scattering.
**Constraint honored**: `__init__.py` re-exports the **identical public contract**.
Zero break to the four consumers. `import ollama_client; ollama_client.generate(...)`
and `from ollama_client import chat` work unchanged.
**Alternative rejected**: keep flat + `vs-soft-allow` (was the documented prior
decision). Overridden by explicit user request + shim declaration.
**Trade-off accepted**: mocking semantics for internal calls (generate_fallback →
generate) would change with a naive split. Mitigated by **late-binding**. See
[[mocking-semantics]].

## Decision: SemVer 1.0.0 + `require()` gate
**Choice**: consumers call `ollama_client.require("1.0.0")` to fail fast on drift.
**Why**: a consumer using a feature added in 1.1.0 against a 1.0.0 install gets a
clear RuntimeError, not a cryptic mid-run AttributeError. Mirrors `cheap_llm`.
**How**: `__version__` + `require(min) -> str` in `_version.py`.

## Decision: late-binding in fallback functions
**Choice**: `generate_fallback` does `from ollama_client import generate` *inside*
the function body (call-time), not at module top.
**Why**: preserves the flat-module semantics where `@patch("ollama_client.generate")`
also intercepts the call from `generate_fallback`. A top-level
`from .generation import generate` would bind a static reference and defeat the
patch. See [[mocking-semantics]].
**Cost**: one import lookup per fallback iteration (cached by Python import system).

## Decision: shim loads PACKAGE via submodule_search_locations
**Choice**: shim uses `spec_from_file_location("ollama_client", init_path,
submodule_search_locations=[pkg_dir])`.
**Why**: relative imports (`from ._transport import`) inside the package require
the loaded module to be registered as a PACKAGE, not a plain module. The old shim
loaded a flat `.py`; the package version needs `submodule_search_locations` so
Python resolves `ollama_client._transport`. No sys.path mutation, no recursion.
See [[shim-pattern]].

## Pattern: error taxonomy drives fallback behavior
`OllamaUnavailable` (daemon down) → abort chain. `OllamaRequestError` (model
404/OOM/timeout) → try next model. The `TimeoutError → OllamaRequestError(408)`
mapping is load-bearing: a timeout is a slow cold LOAD (daemon up), not a dead
daemon (which gives instant URLError). Getting this backwards aborts the chain
on a recoverable slow model.

## Pattern: deterministic-prompt cache, embeddings NEVER cached
Generation/chat cached when `temperature <= CACHE_MAX_TEMP` (keyed on
all output-affecting inputs). Embeddings fresh every call (callers manage their
own index). `_cache_key` includes `num_ctx`; `_cache_key_chat` includes
`num_predict` + `num_ctx` + `think` — an overflowed (empty) result must NOT poison
a larger-ctx call, and reasoning modes must never share answers. Cache I/O is
best-effort: an unusable cache root must not block a live request. See
[[cache-policy]].
