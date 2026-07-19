"""Contract test: the public API is byte-identical to the graduated flat module.

If this test fails, a consumer (codeq / smart-trim / prompt-improve /
web-research) will break. It is the regression gate for ANY refactor.
"""

from __future__ import annotations

import inspect

import ollama_client as o
from ollama_client._config import DEFAULT_STRUCTURED_MODEL, DEFAULT_SUMMARY_MODEL

# --- Names that MUST exist on the public surface ---

PUBLIC_FUNCTIONS = [
    "is_alive",
    "generate",
    "generate_fallback",
    "chat",
    "chat_fallback",
    "embed",
    "ocr_image",
    "require",
    "main",
]
PUBLIC_CONSTANTS = [
    "DEFAULT_URL",
    "DEFAULT_GEN_MODEL",
    "DEFAULT_SUMMARY_MODEL",
    "DEFAULT_STRUCTURED_MODEL",
    "DEFAULT_PDF_OCR_MODEL",
    "PDF_OCR_PROMPT",
    "DEFAULT_EMBED_MODEL",
    "DEFAULT_TIMEOUT",
    "EMBED_TIMEOUT",
    "CACHE_MAX_TEMP",
    "CACHE_MAX_ENTRIES",
    "OLLAMA_CACHE_DIR",
]
PUBLIC_EXCEPTIONS = ["OllamaUnavailable", "OllamaRequestError"]


def test_public_functions_exist_and_callable():
    for name in PUBLIC_FUNCTIONS:
        obj = getattr(o, name, None)
        assert obj is not None, f"missing public function: {name}"
        assert callable(obj), f"{name} is not callable"


def test_public_constants_exist():
    for name in PUBLIC_CONSTANTS:
        assert hasattr(o, name), f"missing public constant: {name}"
    assert o.DEFAULT_STRUCTURED_MODEL == DEFAULT_STRUCTURED_MODEL
    assert o.DEFAULT_SUMMARY_MODEL == DEFAULT_SUMMARY_MODEL
    assert o.DEFAULT_SUMMARY_MODEL == "hf.co/TeichAI/Qwen3.5-9B-Fable-5-v1-GGUF:Q4_K_M"
    assert o.DEFAULT_STRUCTURED_MODEL != o.DEFAULT_SUMMARY_MODEL


def test_exceptions_exist_and_hierarchy():
    for name in PUBLIC_EXCEPTIONS:
        assert hasattr(o, name), f"missing public exception: {name}"
    assert issubclass(o.OllamaRequestError, o.OllamaUnavailable)
    assert issubclass(o.OllamaUnavailable, RuntimeError)
    # ctor carries status + body (prompt-improve reads these)
    exc = o.OllamaRequestError(503, "boom")
    assert exc.status == 503
    assert exc.body == "boom"


def test_version_attribute():
    assert isinstance(o.__version__, str)
    assert o.__version__ == "1.2.0"


def test_all_exports_resolve():
    """Every name in __all__ is actually present on the package."""
    for name in o.__all__:
        assert hasattr(o, name), f"__all__ lists {name} but it is not importable"


def test_all_includes_public_contract():
    """Star-import compatibility: the prior flat module exported public constants."""
    for name in [*PUBLIC_FUNCTIONS, *PUBLIC_CONSTANTS, *PUBLIC_EXCEPTIONS, "__version__"]:
        assert name in o.__all__, f"public contract name missing from __all__: {name}"


# --- Frozen signatures (the wide param lists are the contract) ---


def test_generate_signature_frozen():
    sig = inspect.signature(o.generate)
    params = list(sig.parameters)
    assert params == ["prompt", "model", "temperature", "timeout", "base_url", "cache", "num_ctx"]


def test_chat_signature_frozen():
    sig = inspect.signature(o.chat)
    params = list(sig.parameters)
    assert params == [
        "messages",
        "model",
        "temperature",
        "num_predict",
        "think",
        "timeout",
        "base_url",
        "cache",
        "num_ctx",
    ]


def test_embed_signature_frozen():
    sig = inspect.signature(o.embed)
    params = list(sig.parameters)
    assert params == ["text", "model", "timeout", "base_url"]


def test_internals_re_exported_for_mock_parity():
    """Internals are re-exported so existing @patch("ollama_client._post") etc.
    keep resolving (parity with the flat module)."""
    for name in [
        "_post",
        "_normalize_base_url",
        "_cache_key",
        "_cache_key_chat",
        "_prune_cache",
        "_strip_think_tags",
        "_embed_once",
        "_parse_version",
        "_cache_path",
    ]:
        assert hasattr(o, name), f"internal {name} not re-exported (mock parity broken)"


def test_cli_entry_point_callable():
    assert callable(o.main)
    assert callable(o._cli)
