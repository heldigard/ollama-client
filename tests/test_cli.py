"""CLI dispatch: subcommands + exit codes (no daemon required)."""

from __future__ import annotations

import json

import pytest

import ollama_client as o
import ollama_client.cli as cli_mod

# --- is-alive ---


def test_cli_is_alive_true(monkeypatch):
    monkeypatch.setattr(cli_mod, "is_alive", lambda *a, **_kw: True)
    assert cli_mod._cli(["is-alive"]) == 0


def test_cli_is_alive_false(monkeypatch):
    monkeypatch.setattr(cli_mod, "is_alive", lambda *a, **_kw: False)
    assert cli_mod._cli(["is-alive"]) == 1


def test_cli_is_alive_base_url(monkeypatch):
    captured: dict = {}

    def fake(**kw: object) -> bool:
        captured.update(kw)
        return True

    monkeypatch.setattr(cli_mod, "is_alive", fake)
    assert cli_mod._cli(["is-alive", "--base-url", "http://custom:11434"]) == 0
    assert captured["base_url"] == "http://custom:11434"


# --- generate ---


def test_cli_generate_ok(monkeypatch, capsys):
    monkeypatch.setattr(cli_mod, "generate", lambda *a, **_kw: "hello world")
    rc = cli_mod._cli(["generate", "--prompt", "hi"])
    assert rc == 0
    assert capsys.readouterr().out.strip() == "hello world"


def test_cli_generate_empty_exit_zero(monkeypatch, capsys):
    """Empty output is not an error for generate (exit 0, nothing printed)."""
    monkeypatch.setattr(cli_mod, "generate", lambda *a, **_kw: None)
    assert cli_mod._cli(["generate", "--prompt", "hi"]) == 0


def test_cli_generate_unavailable_exit_2(monkeypatch):
    def boom(*a, **_kw):
        raise o.OllamaUnavailable("down")

    monkeypatch.setattr(cli_mod, "generate", boom)
    assert cli_mod._cli(["generate", "--prompt", "hi"]) == 2


def test_cli_generate_no_cache_flag(monkeypatch):
    """--no-cache must pass cache=False through."""
    captured: dict = {}

    def fake(prompt: str, **kw: object) -> str:
        captured.update(kw)
        captured["prompt"] = prompt
        return "ok"

    monkeypatch.setattr(cli_mod, "generate", fake)
    cli_mod._cli(["generate", "--prompt", "hi", "--no-cache"])
    assert captured["cache"] is False


# --- embed ---


def test_cli_embed_ok(monkeypatch, capsys):
    monkeypatch.setattr(cli_mod, "embed", lambda *a, **_kw: [0.1, 0.2, 0.3])
    rc = cli_mod._cli(["embed", "--text", "word"])
    assert rc == 0
    assert json.loads(capsys.readouterr().out) == [0.1, 0.2, 0.3]


def test_cli_embed_none_exit_2(monkeypatch):
    monkeypatch.setattr(cli_mod, "embed", lambda *a, **_kw: None)
    assert cli_mod._cli(["embed", "--text", "word"]) == 2


# --- ocr-image ---


def test_cli_ocr_ok(monkeypatch, capsys, tmp_path):
    img = tmp_path / "page.png"
    img.write_bytes(b"\x89PNG fake")
    monkeypatch.setattr(cli_mod, "ocr_image", lambda *a, **_kw: "EXTRACTED TEXT")
    rc = cli_mod._cli(["ocr-image", "--image", str(img)])
    assert rc == 0
    assert capsys.readouterr().out.strip() == "EXTRACTED TEXT"


def test_cli_ocr_empty_exit_3(monkeypatch, tmp_path):
    img = tmp_path / "page.png"
    img.write_bytes(b"\x89PNG fake")
    monkeypatch.setattr(cli_mod, "ocr_image", lambda *a, **_kw: None)
    assert cli_mod._cli(["ocr-image", "--image", str(img)]) == 3


def test_cli_ocr_unavailable_exit_2(monkeypatch, tmp_path):
    img = tmp_path / "page.png"
    img.write_bytes(b"x")

    def boom(*a, **_kw):
        raise o.OllamaUnavailable("down")

    monkeypatch.setattr(cli_mod, "ocr_image", boom)
    assert cli_mod._cli(["ocr-image", "--image", str(img)]) == 2


# --- main() is the console-script entry ---


def test_main_delegates_to_cli(monkeypatch):
    """main() is the public alias for _cli() and reads sys.argv."""
    monkeypatch.setattr("sys.argv", ["ollama-client", "is-alive"])
    monkeypatch.setattr(cli_mod, "is_alive", lambda *a, **_kw: True)
    assert o.main() == 0


def test_cli_version_flag(capsys):
    """--version prints the SemVer string and exits 0 (argparse action=version)."""
    with pytest.raises(SystemExit) as ei:
        cli_mod._cli(["--version"])
    assert ei.value.code == 0
    out = capsys.readouterr().out
    assert "ollama-client" in out
    assert o.__version__ in out
