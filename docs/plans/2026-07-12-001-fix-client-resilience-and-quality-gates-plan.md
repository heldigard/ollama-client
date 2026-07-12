---
title: "fix: Harden cache and transport quality gates"
type: fix
status: completed
date: 2026-07-12
---

# fix: Harden cache and transport quality gates

## Enhancement Summary

**Deepened on:** 2026-07-12

**Sections enhanced:** cache compatibility, failure boundaries, regression tests,
CI packaging, and documentation/memory coverage

**Review lenses used:** Python correctness, architecture compatibility, code
simplicity, and specification flow

### Key Improvements

1. The private helper keeps its five existing positional arguments and appends
   `think: bool = False`; `chat()` passes it by keyword.
2. Cache setup catches only `OSError` and ends before `_post()`, so transport
   exceptions cannot be swallowed accidentally.
3. Invalid response encoding uses a constant bounded diagnostic rather than
   decoding corrupt bytes with replacement or retaining raw binary data.
4. CI remains one hermetic Python 3.11 job; external shim validation remains a
   local integration gate because the shim is not owned by this repository.

### Scope Guardrails

- Do not add a cache class/helper abstraction, key migration, schema prefix, new
  exception type, CI matrix, daemon setup, publishing step, or general typing
  cleanup.
- Do not rewrite historical memory entries that correctly describe earlier
  releases; update only stale current-state claims and canonical policy records.

## Overview

Review and strengthen the shared Ollama client without changing its frozen public
function signatures, stdlib-only runtime, late-binding fallback behavior, or
two-class exception taxonomy. The work corrects three confirmed resilience bugs,
brings documentation in sync with release 1.2.0, and adds a repository-level CI
gate that does not require a live Ollama daemon.

## Research Findings

- The current baseline is healthy: 98 tests pass; Ruff format/lint and Pyright are
  clean.
- `chat()` sends `think` to Ollama, but `_cache_key_chat()` omits it. Identical
  calls with different reasoning modes can share a cached answer.
- `generate()` and `chat()` create the cache directory outside their best-effort
  I/O handling. A read-only or invalid cache path can prevent the live HTTP call.
- `_post()` decodes successful response bytes as strict UTF-8 before its JSON error
  normalization. Invalid bytes escape as `UnicodeDecodeError` rather than
  `OllamaRequestError(502)`.
- `README.md`, `CLAUDE.md`, and the package docstring describe stale version/cache
  details. The canonical package version is 1.2.0.
- No GitHub Actions workflow currently enforces the project's existing test,
  lint, type, format, or build checks.
- No `docs/solutions/` knowledge base exists. Applicable constraints come from
  `.memory-bank/`: preserve late binding, cache best-effort semantics, transport
  taxonomy, shim compatibility, and the public contract.
- External research is unnecessary: the changes are governed by local contracts
  and do not depend on a changing third-party API.

## Proposed Solution

1. Extend `_cache_key_chat()` with a final optional `think: bool = False`
   parameter and include it unconditionally in the hashed material. Pass the
   current `chat()` value into the helper.
2. Make cache-directory creation fail open in both `generate()` and `chat()`:
   catch only `OSError`, disable caching for that invocation, and continue to the
   live request.
3. Convert invalid UTF-8 in a successful response into
   `OllamaRequestError(502, ...)` with a safe, bounded diagnostic body.
4. Update active documentation and project memory to version 1.2.0 and the exact
   cache formulas.
5. Add a minimal GitHub Actions workflow for Python 3.11 that runs pytest, Ruff,
   Pyright, and `uv build`, with read-only repository permissions and no live
   daemon smoke tests.
6. Add focused regression tests for each bug and keep all existing contract tests.

## Technical Approach

### Cache key compatibility

The existing positional order of `_cache_key_chat(model, messages, temperature,
num_predict, num_ctx)` remains intact; `think` is appended with a default. Both
`False` and `True` are serialized into new keys. This intentionally invalidates
all legacy chat cache entries because an old entry cannot reveal which `think`
mode produced it.

Exact formulas after the change:

```text
generate: sha256(model | temperature | num_ctx | prompt)
chat:     sha256(model | temperature | num_predict | num_ctx | think | messages_json_sorted)
```

### Cache failure propagation

Cache reads and writes remain optional optimizations. If the root cannot be
created, only the effective cache path becomes `None`; network errors and
programming errors retain their current behavior. No broad exception handling is
introduced.

Implementation boundary: wrap only `OLLAMA_CACHE_DIR.mkdir(...)` in
`try/except OSError`; set `cached_path = None` in the handler, then keep the
existing read/delete/write branches guarded by `cached_path is not None`.
`_post()` must remain outside this block. Keeping the small duplication in
`generation.py` and `chat.py` is simpler than introducing a cache manager for two
call sites.

### Transport failure propagation

An HTTP 200 response with invalid UTF-8 is a reachable-daemon/request failure,
not proof that the daemon is unavailable. `_post()` therefore maps it to the
model-specific `OllamaRequestError(502)` so fallback chains can try another model.
The hierarchy `OllamaRequestError < OllamaUnavailable` remains unchanged.

Read the response into `raw_body`, decode strictly in a dedicated
`try/except UnicodeDecodeError`, and raise
`OllamaRequestError(502, "invalid UTF-8 response")` chained from the decode
error. Do not use `errors="replace"`: corrupt wire data must not be transformed
into apparently valid JSON, and the exception body must not retain unbounded raw
bytes.

### CI boundary

The workflow is a reversible repository file only. It uses the declared minimum
Python 3.11, installs the test extra, executes hermetic checks, and builds wheel
and sdist. It does not deploy, publish, require credentials, or contact Ollama.

Use one job with `actions/checkout`, `astral-sh/setup-uv`, and
`uv sync --extra test --locked`. After quality checks, run `uv build`, create one
throwaway environment, install the wheel, and smoke both installed entry points.
This validates the stated packaging criterion without adding a matrix or publish
workflow. The host-specific shim at `~/.claude/scripts/ollama_client.py` stays out
of GitHub Actions and is validated locally at the final gate.

## System-Wide Impact

- **Interaction graph:** `chat()` -> `_cache_key_chat()` -> `_cache_path()`;
  cache miss -> `_post()` -> Ollama; request error -> `chat_fallback()` tries the
  next model. `generate()` follows the corresponding generate path.
- **Error propagation:** cache `OSError` is swallowed only at the cache boundary;
  invalid response encoding becomes `OllamaRequestError(502)`; daemon connection
  failures remain `OllamaUnavailable` and abort a fallback chain.
- **State lifecycle:** old chat cache files become unreachable and are eventually
  removed by normal bounded pruning. Atomic writes and stale-temp cleanup remain
  intact.
- **API parity:** public `generate`, `chat`, fallbacks, embed, OCR, exceptions,
  constants, signatures, and `__all__` do not change.
- **External compatibility:** no packaging/import changes are planned. Final
  validation must still exercise the compatibility shim and both CLI entry points.

## Acceptance Criteria

### Functional

- [x] Chat cache keys differ for `think=False` and `think=True`.
- [x] Same-mode repeated chat calls still hit cache; cross-mode calls do not.
- [x] Legacy chat keys are not reused after the change.
- [x] `_cache_key_chat()` remains callable with its original five positional
      arguments and accepts `think` only as an appended optional argument.
- [x] `generate(cache=True)` and `chat(cache=True)` still perform a live request
      when the cache root cannot be created.
- [x] `cache=False` never attempts to create the cache root.
- [x] Invalid UTF-8 in an HTTP 200 response raises `OllamaRequestError` with status
      502 and never leaks `UnicodeDecodeError`.
- [x] Existing fallback ordering, late binding, signatures, and exception hierarchy
      remain unchanged.

### Documentation and automation

- [x] README, CLAUDE guidance, package docstring, and cache-policy memory agree on
      version 1.2.0 and the exact cache key inputs.
- [x] CI runs pytest, Ruff lint, Ruff format check, Pyright, and `uv build` on
      Python 3.11 without a live daemon.
- [x] The workflow uses minimal `contents: read` permissions and no secrets.

### Quality gates

- [x] Full pytest suite passes.
- [x] `ruff check .` and `ruff format --check .` pass.
- [x] Pyright reports zero errors.
- [x] Narrow quality sensors report no actionable findings.
- [x] Wheel and sdist build successfully; the wheel installs into an isolated
      environment and both module/console version commands succeed.
- [x] Compatibility shim imports the package and reports version 1.2.0.

## Risks and Mitigations

- **Cache churn:** adding `think` invalidates prior chat entries. This is intended
  because legacy entries are semantically ambiguous; bounded pruning removes them.
- **Contract regression:** frozen-signature and late-binding tests remain mandatory.
- **Overbroad error swallowing:** catch only `OSError` at cache I/O boundaries and
  `UnicodeDecodeError` at response decoding.
- **CI network assumptions:** tests mock `urlopen`; do not add live-daemon smoke
  tests to CI.
- **Dirty worktree:** preserve pre-existing `.gitignore`, `uv.lock`, and
  `.memory-bank` coordination changes; do not revert or rewrite them.

## Validation Scenarios

1. Two chat calls with identical messages/options but opposite `think` values
   each reach the fake daemon and return their own result.
2. A third chat call using the first mode reads its corresponding cache entry;
   the observed request sequence is `False -> True`, with no third HTTP call.
3. A seeded legacy-format cache path is ignored and replaced by a live response.
4. A regular file placed at the configured cache-root path causes `mkdir()` to
   fail, but generation and chat return the fake live response.
5. With `cache=False` and a nonexistent cache root, neither public call creates
   the directory.
6. A fake 200 response containing `b"\xff"` becomes a bounded 502 request error.
7. All existing API contract, fallback, CLI, version, transport, and cache tests
   stay green.
8. Built wheel installation exposes `import ollama_client`,
   `python -m ollama_client --version`, and `ollama-client --version`.

## Files Expected to Change

- `ollama_client/_cache.py`
- `ollama_client/chat.py`
- `ollama_client/generation.py`
- `ollama_client/_transport.py`
- `ollama_client/__init__.py`
- `pyproject.toml`
- `tests/test_cache.py`
- `tests/test_transport.py`
- `README.md`
- `CLAUDE.md`
- `.memory-bank/MEMORY.md`
- `.memory-bank/REFERENCE.md`
- `.memory-bank/systemPatterns.md`
- `.memory-bank/topics/cache-policy.md`
- `.memory-bank/topics/_index.md`
- `.memory-bank/progress.md`
- `.github/workflows/ci.yml`

## Sources and References

- `ollama_client/chat.py:82-89` — chat cache key call and unguarded directory setup.
- `ollama_client/generation.py:42-48` — generate cache setup.
- `ollama_client/_transport.py:86-95` — response decode and JSON normalization.
- `.memory-bank/topics/cache-policy.md` — canonical cache policy.
- `.memory-bank/topics/mocking-semantics.md` — late-binding compatibility.
- `.memory-bank/systemPatterns.md` — exception/fallback and stdlib-only contracts.
- `tests/test_contract.py` and `tests/test_fallback.py` — frozen API guards.
