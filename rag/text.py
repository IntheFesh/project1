"""Tokenization shared by the dev embedder and BM25.

CJK-aware: ASCII word tokens are kept whole; CJK characters are tokenized per-character
(a simple, dependency-free approximation suitable for the offline dev pipeline).
"""

from __future__ import annotations

import re

_ASCII = re.compile(r"[a-z0-9]+")
_CJK = re.compile(r"[一-鿿]")


def tokenize(text: str) -> list[str]:
    """Return lowercase ASCII word tokens plus individual CJK characters."""
    lowered = text.lower()
    tokens = _ASCII.findall(lowered)
    tokens.extend(_CJK.findall(lowered))
    return tokens
