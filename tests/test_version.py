"""Version + SemVer ``require()`` gate."""

from __future__ import annotations

import pytest

import ollama_client as o
from ollama_client._version import _parse_version


@pytest.mark.parametrize(
    "v,expected",
    [
        ("1.0.0", (1, 0, 0)),
        ("1.2.3", (1, 2, 3)),
        ("1.0", (1, 0)),
        ("2", (2,)),
        ("1.0.0-rc1", (1, 0, 0)),  # prerelease suffix ignored
        ("1.2rc3", (1, 2)),
        ("", (0,)),  # split("") -> [""] -> digits "" -> 0
        ("v1.2.3", (1, 2, 3)),  # leading 'v' dropped
    ],
)
def test_parse_version(v, expected):
    assert _parse_version(v) == expected


def test_require_returns_version():
    assert o.require() == o.__version__


def test_require_accepts_equal():
    assert o.require("1.1.0") == o.__version__


def test_require_accepts_older_min():
    assert o.require("0.9.0") == o.__version__


def test_require_accepts_shorter_min():
    # prompt-improve and smart-trim require the parser-fallback contract.
    assert o.require("1.1") == o.__version__


def test_require_rejects_newer_min():
    with pytest.raises(RuntimeError, match="older than required"):
        o.require("2.0.0")


def test_require_none_passes():
    assert o.require(None) == o.__version__
