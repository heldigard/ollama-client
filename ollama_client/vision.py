# vs-soft-allow  — frozen public API: ocr_image carries 7 params; the PDF/vision
# pipeline depends on this exact signature. Graduated lib, not greenfield.
"""Vision: OCR a rendered PDF page/image via a vision-capable Ollama model."""

from __future__ import annotations

import base64

from ._cache import _strip_think_tags
from ._config import (
    DEFAULT_PDF_OCR_MODEL,
    DEFAULT_TIMEOUT,
    DEFAULT_URL,
    PDF_OCR_PROMPT,
)
from ._transport import _post


def ocr_image(
    image_bytes: bytes,
    *,
    model: str = DEFAULT_PDF_OCR_MODEL,
    prompt: str = PDF_OCR_PROMPT,
    temperature: float = 0.0,
    num_predict: int = 1024,
    timeout: float = DEFAULT_TIMEOUT,
    base_url: str = DEFAULT_URL,
) -> str | None:
    """OCR a rendered PDF page/image using a vision-capable Ollama model.

    Unlimited-OCR is prompt-sensitive under Ollama: generic prompts like
    "Extract the text" can return an empty response, while the model-specific
    ``ocr [img]`` prompt returns OCR text with optional coordinates.
    """
    if not image_bytes:
        return None
    messages: list[dict] = [
        {
            "role": "user",
            "content": prompt,
            "images": [base64.b64encode(image_bytes).decode("ascii")],
        }
    ]
    data = _post(
        "/api/chat",
        {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {"temperature": temperature, "num_predict": num_predict},
        },
        base_url,
        timeout,
    )
    msg = data.get("message")
    content = msg.get("content", "") if isinstance(msg, dict) else ""
    return _strip_think_tags(str(content).strip()) or None
