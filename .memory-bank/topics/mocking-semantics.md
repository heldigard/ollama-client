# mocking-semantics — why fallback functions use late-binding

> status: live (reference, never archive)

## The hazard
In the original **flat** module, `generate_fallback` called `generate` by bare
name — a module-global lookup. So a test doing `@patch("ollama_client.generate")`
would ALSO intercept the call made from inside `generate_fallback` (same
namespace).

A **naive** package split breaks this:
```python
# generation.py
def generate(...): ...
def generate_fallback(...):
    text = generate(...)   # binds to generation.generate (static), NOT ollama_client.generate
```
`@patch("ollama_client.generate")` replaces the attribute on the `ollama_client`
package (__init__), but `generate_fallback` holds a direct reference to
`generation.generate` — the patch is invisible to it. The web-research test
`@patch("ollama_client.generate")` would silently stop intercepting.

## Fix: late-binding (resolve the public attr at call time)
```python
# generation.py
def generate_fallback(...):
    from ollama_client import generate   # late: hits the (possibly patched) public attr
    ...
    text = generate(...)
```
The import runs at call time, resolving `ollama_client.generate` via
`sys.modules` — which IS the patched attribute. Semantics restored.

## Where it applies
- `generation.generate_fallback` → late-imports `generate`.
- `chat.chat_fallback` → late-imports `chat`.

## Where it does NOT apply
- `embedding.embed` → calls `_embed_once`. Both live in the same `embedding.py`;
  `_embed_once` is private (leading underscore), not part of `__all__`, nobody
  patches it externally. Local call is correct and cheaper.

## Verification
`tests/test_fallback.py::test_generate_fallback_respects_patched_generate` and
`test_chat_fallback_respects_patched_chat` assert the patch is seen. This is the
regression guard for the package split.

Related: [[shim-pattern]].
