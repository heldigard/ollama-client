# shim-pattern — graduated package stays wired to the harness

> status: live (reference, never archive)

## Problem
Four harness consumers do `import ollama_client` from `~/.claude/scripts/` (on
sys.path for hooks/scripts). The canonical source moved to `~/ollama-client/`
as a **package**. The shim must redirect without sys.path hacks or recursion.

## Solution (package-aware)
`~/.claude/scripts/ollama_client.py` builds a module spec for the package's
`__init__.py` with `submodule_search_locations=[pkg_dir]`:

```python
spec = importlib.util.spec_from_file_location(
    "ollama_client",
    str(pkg_dir / "__init__.py"),
    submodule_search_locations=[str(pkg_dir)],
)
_real = importlib.util.module_from_spec(spec)
sys.modules["ollama_client"] = _real   # replace THIS stub before exec
spec.loader.exec_module(_real)
from ollama_client import *  # noqa
```

## Why submodule_search_locations
Relative imports inside the package (`from ._transport import _post`) resolve
`ollama_client._transport` by searching `submodule_search_locations`. Without it
(spec of a plain module), every relative import raises
`ImportError: attempted relative import with no known parent package`. This is the
single difference from the flat-module shim.

## Why replace sys.modules BEFORE exec
The shim file is itself loaded as `ollama_client` (from scripts/). If we exec the
real package first, it would re-enter `ollama_client` (recursion). By setting
`sys.modules["ollama_client"] = _real` then exec, the real package sees itself
already registered and relative imports resolve to `_real`'s submodules.

## Override
`OLLAMA_CLIENT_HOME=/path/to/ollama-client` redirects to a different checkout
(useful for testing a branch without uninstalling).

## Consumers verified (2026-07-08)
- codeq `shared/llm.py`: direct `import ollama_client`.
- prompt-improve `compat.py`: `compat.ollama_client.chat`, `.OllamaRequestError`.
- web-research `compat.py` + tests `@patch("ollama_client.generate")`.
- smart-trim hook.

Related: [[mocking-semantics]], [[cache-policy]].
