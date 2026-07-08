"""``python -m ollama_client`` entry point."""

from __future__ import annotations

import sys

from .cli import _cli

if __name__ == "__main__":
    sys.exit(_cli())
