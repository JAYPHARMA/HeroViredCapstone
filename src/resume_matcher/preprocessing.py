"""Text normalization helpers for resumes and job descriptions."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable


_WHITESPACE_RE = re.compile(r"\s+")
_NON_TEXT_RE = re.compile(r"[^a-zA-Z0-9+#./\s-]")

_PERSONAL_LINE_RE = re.compile(
    r"^\s*(?:candidate name|full name|name|gender|sex|age|date of birth|dob|religion|caste|"
    r"marital status|nationality|home address|address)\s*[:=].*$",
    re.I | re.M,
)


def redact_personal_details(text: object) -> str:
    """Remove explicit demographic/contact fields, without inferring identity.

    This is a limited mitigation. Free-form mentions and proxy bias can remain.
    Candidate display names are stored separately and never modeled.
    """
    value = "" if text is None else str(text)
    value = _PERSONAL_LINE_RE.sub(" ", value)
    value = re.sub(r"[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}", " ", value)
    value = re.sub(r"(?<!\w)(?:\+?\d[\d ()-]{8,}\d)(?!\w)", " ", value)
    # Remove the boilerplate clause, not the entire JD when text is one long line.
    value = re.sub(
        r"\b(?:equal opportunity employer|all rights reserved|apply now|click here to apply)\b[^.!?\n]*(?:[.!?]|$)",
        " ",
        value,
        flags=re.I,
    )
    return value


def clean_text(text: object) -> str:
    """Normalize noisy resume/JD text while preserving common tech tokens."""
    if text is None:
        return ""
    return _clean_string(str(text))


@lru_cache(maxsize=4096)
def _clean_string(text: str) -> str:
    value = redact_personal_details(text).replace("\x00", " ")
    value = _NON_TEXT_RE.sub(" ", value)
    value = _WHITESPACE_RE.sub(" ", value).strip().lower()
    return value


def token_set(text: object) -> set[str]:
    cleaned = clean_text(text)
    return {token for token in re.findall(r"[a-zA-Z][a-zA-Z0-9+#./-]{1,}", cleaned)}


def join_texts(values: Iterable[object]) -> list[str]:
    return [clean_text(value) for value in values]
