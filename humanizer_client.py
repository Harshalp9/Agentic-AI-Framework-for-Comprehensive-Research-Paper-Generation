"""Client for an optional GoHumanize Hugging Face Gradio Space."""

from __future__ import annotations

import os
import re
from functools import lru_cache


MAX_REQUEST_CHARS = 5000


@lru_cache(maxsize=1)
def _client():
    """Create the remote client only when humanization is enabled."""
    from gradio_client import Client

    space_id = os.getenv("GOHUMANIZE_SPACE_ID", "").strip()
    if not space_id:
        return None
    token = os.getenv("HF_TOKEN", "").strip() or None
    return Client(space_id, hf_token=token)


def _protect(text: str) -> tuple[str, dict[str, str]]:
    """Mask references that should survive paraphrasing byte-for-byte."""
    protected: dict[str, str] = {}
    patterns = [
        r"```[\s\S]*?```",                 # fenced code
        r"`[^`\n]+`",                      # inline code
        r"\\begin\{(?:equation|align|gather)\*?\}[\s\S]*?\\end\{(?:equation|align|gather)\*?\}",
        r"\\\[[\s\S]*?\\\]",           # display math
        r"\\\([\s\S]*?\\\)",           # inline math
        r"\$\$[\s\S]*?\$\$",             # display math
        r"\$(?!\$)[^\n$]+\$",              # inline math
        r"\[(?:SRC\d+(?:\s*,\s*SRC\d+)*)\]",  # source citations
        r"https?://[^\s)]+",                 # URLs
    ]
    for pattern in patterns:
        text = re.sub(
            pattern,
            lambda match: _store(match.group(0), protected),
            text,
        )
    return text, protected


def _store(value: str, protected: dict[str, str]) -> str:
    marker = f"ZXQPROTECTED{len(protected)}QXZ"
    protected[marker] = value
    return marker


def _humanize_chunk(text: str) -> str:
    client = _client()
    if client is None:
        return text
    masked, protected = _protect(text)
    result = client.predict(masked, api_name="/predict")
    if not isinstance(result, str) or not result.strip():
        raise ValueError("GoHumanize returned empty text.")
    for marker, original in protected.items():
        if marker not in result:
            raise ValueError("GoHumanize changed a protected citation, equation, or URL.")
        result = result.replace(marker, original)
    return result.strip()


def _chunks(text: str):
    """Split at paragraph boundaries, then sentence boundaries for long paragraphs."""
    for paragraph in re.split(r"(\n\s*\n)", text):
        if not paragraph or paragraph.isspace():
            yield paragraph
            continue
        if len(paragraph) <= MAX_REQUEST_CHARS:
            yield _humanize_chunk(paragraph) if paragraph.strip() else paragraph
            continue
        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        batch = ""
        sent_batch = False
        for sentence in sentences:
            if len(sentence) > MAX_REQUEST_CHARS:
                raise ValueError("A single sentence exceeds the 5,000-character service limit.")
            candidate = f"{batch} {sentence}".strip()
            if batch and len(candidate) > MAX_REQUEST_CHARS:
                if sent_batch:
                    yield " "
                yield _humanize_chunk(batch)
                batch = sentence
                sent_batch = True
            else:
                batch = candidate
        if batch:
            if sent_batch:
                yield " "
            yield _humanize_chunk(batch)


def humanize_text(text: str) -> str:
    """Humanize prose while retaining paragraph breaks and protected spans."""
    if not text or not text.strip():
        return text
    if _client() is None:
        return text
    return "".join(_chunks(text))
