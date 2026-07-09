# vs-soft-allow  — single-responsibility CLI dispatcher: one argparse tree that
# routes subcommands to the domain functions. A dispatcher is one reason to
# change; the per-command if-branches are the route table, not mixed concerns.
"""Command-line interface (``ollama-client`` console script, ``python -m ollama_client``)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ._config import (
    DEFAULT_EMBED_MODEL,
    DEFAULT_GEN_MODEL,
    DEFAULT_PDF_OCR_MODEL,
    DEFAULT_URL,
    PDF_OCR_PROMPT,
)
from ._transport import OllamaUnavailable, is_alive
from ._version import __version__
from .embedding import embed
from .generation import generate
from .vision import ocr_image


def _cli(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ollama-client", description="Shared local-Ollama client")
    ap.add_argument("--version", action="version", version=f"ollama-client {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)
    alive = sub.add_parser("is-alive", help="Exit 0 if Ollama daemon answers")
    alive.add_argument("--base-url", default=DEFAULT_URL)
    g = sub.add_parser("generate", help="One-shot completion")
    g.add_argument("--prompt", required=True)
    g.add_argument("--model", default=DEFAULT_GEN_MODEL)
    g.add_argument("--temperature", type=float, default=0.2)
    g.add_argument("--no-cache", action="store_true")
    g.add_argument("--base-url", default=DEFAULT_URL)
    o = sub.add_parser("ocr-image", help="OCR one rendered page/image")
    o.add_argument("--image", required=True, help="PNG/JPEG path")
    o.add_argument("--model", default=DEFAULT_PDF_OCR_MODEL)
    o.add_argument("--prompt", default=PDF_OCR_PROMPT)
    o.add_argument("--base-url", default=DEFAULT_URL)
    e = sub.add_parser("embed", help="Embedding vector for one text")
    e.add_argument("--text", required=True)
    e.add_argument("--model", default=DEFAULT_EMBED_MODEL)
    e.add_argument("--base-url", default=DEFAULT_URL)
    args = ap.parse_args(argv)

    if args.command == "is-alive":
        return 0 if is_alive(base_url=getattr(args, "base_url", DEFAULT_URL)) else 1

    if args.command == "generate":
        try:
            out = generate(
                args.prompt,
                model=args.model,
                temperature=args.temperature,
                base_url=args.base_url,
                cache=not args.no_cache,
            )
        except OllamaUnavailable as exc:
            print(f"ollama unavailable: {exc}", file=sys.stderr)
            return 2
        if out:
            print(out)
        return 0

    if args.command == "embed":
        vec = embed(args.text, model=args.model, base_url=args.base_url)
        if vec is None:
            print("ollama unavailable or empty embedding", file=sys.stderr)
            return 2
        print(json.dumps(vec))
        return 0

    if args.command == "ocr-image":
        try:
            out = ocr_image(
                Path(args.image).read_bytes(),
                model=args.model,
                prompt=args.prompt,
                base_url=args.base_url,
            )
        except OllamaUnavailable as exc:
            print(f"ollama unavailable: {exc}", file=sys.stderr)
            return 2
        if out:
            print(out)
            return 0
        print("ollama returned empty OCR output", file=sys.stderr)
        return 3

    return 0


def main() -> int:
    """Public CLI entry point (alias for :func:`_cli`); used by the console script."""
    return _cli()
