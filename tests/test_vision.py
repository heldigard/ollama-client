"""Tests for the vision OCR module (ocr_image)."""

from __future__ import annotations

import base64
import urllib.error

import pytest

import ollama_client as o
from ollama_client.vision import ocr_image


def test_ocr_image_success(fake_urlopen):
    fake_urlopen.set_response(
        {"message": {"role": "assistant", "content": "Extracted text from image"}}
    )
    img_data = b"fake-png-bytes"
    out = ocr_image(
        img_data,
        model="vision-model",
        prompt="Read this",
        temperature=0.1,
        num_predict=500,
    )
    assert out == "Extracted text from image"
    assert len(fake_urlopen.calls) == 1
    assert fake_urlopen.calls[0].path == "/api/chat"

    payload = fake_urlopen.calls[0].payload
    assert payload["model"] == "vision-model"
    assert payload["stream"] is False
    assert payload["think"] is False
    assert payload["options"] == {"temperature": 0.1, "num_predict": 500}

    messages = payload["messages"]
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Read this"

    expected_b64 = base64.b64encode(img_data).decode("ascii")
    assert messages[0]["images"] == [expected_b64]


def test_ocr_image_empty_bytes():
    assert ocr_image(b"") is None


def test_ocr_image_propagates_exceptions(fake_urlopen):
    fake_urlopen.raise_error(urllib.error.URLError("connection refused"))
    with pytest.raises(o.OllamaUnavailable):
        ocr_image(b"png-data")


def test_ocr_image_non_dict_message_returns_none(fake_urlopen):
    # message is not a dict
    fake_urlopen.set_response({"message": "not a dict"})
    assert ocr_image(b"png-data") is None


def test_ocr_image_missing_message_returns_none(fake_urlopen):
    # message key missing
    fake_urlopen.set_response({})
    assert ocr_image(b"png-data") is None


def test_ocr_image_strips_think_tags(fake_urlopen):
    fake_urlopen.set_response(
        {"message": {"role": "assistant", "content": "<think>reasoning</think>Final Result"}}
    )
    assert ocr_image(b"png-data") == "Final Result"
