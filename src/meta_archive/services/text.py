from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import ftfy


@dataclass(frozen=True, slots=True)
class NormalizedText:
    raw: str
    display: str
    search: str


def normalize_text(raw: str) -> NormalizedText:
    display = unicodedata.normalize("NFC", ftfy.fix_text(raw))
    search = unicodedata.normalize("NFKC", display).casefold()
    return NormalizedText(raw=raw, display=display, search=search)


def normalize_classifier_token(value: str) -> str:
    """Canonicalize source classifier labels identically in Python and SQLite."""

    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def safe_filesystem_label(value: str, *, fallback: str = "untitled", limit: int = 80) -> str:
    normalized = normalize_text(value).display
    ascii_hint = unicodedata.normalize("NFKD", normalized).encode("ascii", "ignore").decode("ascii")
    candidate = re.sub(r"[^A-Za-z0-9._-]+", "-", ascii_hint).strip("-._").casefold()
    return (candidate or fallback)[:limit].rstrip("-._") or fallback
