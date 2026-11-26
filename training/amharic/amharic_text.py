"""Amharic (Ge'ez script) text utilities for the Tibeb data pipeline.

Covers Unicode checks, light normalization, sentence splitting and
de-duplication keys. Nothing here changes spelling unless asked: homophone
folding is used only to build de-duplication keys by default.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

# Unicode blocks for Ethiopic script
ETHIOPIC_RANGES = (
    (0x1200, 0x137F),  # Ethiopic
    (0x1380, 0x139F),  # Ethiopic Supplement
    (0x2D80, 0x2DDF),  # Ethiopic Extended
    (0xAB00, 0xAB2F),  # Ethiopic Extended-A
)

# Ethiopic punctuation (word space, full stop, comma, semicolon, colon,
# preface colon, question mark, paragraph separator)
ETHIOPIC_PUNCT = "፡።፣፤፥፦፧፨"

# Homophone series that modern Amharic spelling treats as interchangeable.
# Each maps the first 7 orders of a series onto the first 7 orders of a
# canonical series (offsets 0..6).
_HOMOPHONE_SERIES = {
    0x1210: 0x1200,  # ሐ -> ሀ
    0x1280: 0x1200,  # ኀ -> ሀ
    0x1220: 0x1230,  # ሠ -> ሰ
    0x12D0: 0x12A0,  # ዐ -> አ
    0x1340: 0x1338,  # ፀ -> ጸ
}
_FOLD_TABLE = {
    src + k: dst + k for src, dst in _HOMOPHONE_SERIES.items() for k in range(7)
}

_SENT_END = re.compile(r"(?<=[።፧?!])\s+|\n+")
_WS = re.compile(r"[ \t ​]+")


def is_ethiopic_char(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in ETHIOPIC_RANGES)


def ethiopic_ratio(text: str) -> float:
    """Share of letters in `text` that are Ethiopic (0.0 when no letters)."""
    letters = [c for c in text if c.isalpha() or is_ethiopic_char(c)]
    if not letters:
        return 0.0
    return sum(is_ethiopic_char(c) for c in letters) / len(letters)


def looks_like_unicode_amharic(text: str, min_ratio: float = 0.5) -> bool:
    """True if `text` is mostly real Unicode Ethiopic.

    Text extracted from PDFs typed with legacy (non-Unicode) Amharic fonts
    comes out as Latin letters or symbols, so it fails this check.
    """
    return ethiopic_ratio(text) >= min_ratio


def fold_homophones(text: str) -> str:
    """Map homophone letter series (ሐ/ኀ->ሀ, ሠ->ሰ, ዐ->አ, ፀ->ጸ)."""
    return text.translate(_FOLD_TABLE)


def normalize(text: str, fold: bool = False) -> str:
    """NFC-normalize, standardize punctuation and whitespace.

    - '::' or '፡፡' typed for the full stop becomes '።'
    - runs of spaces collapse to one; line breaks are kept
    - `fold=True` also applies homophone folding (off by default)
    """
    text = unicodedata.normalize("NFC", text)
    text = text.replace("፡፡", "።").replace("::", "።")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(_WS.sub(" ", line).strip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if fold:
        text = fold_homophones(text)
    return text


def split_sentences(text: str) -> list[str]:
    """Split on Ethiopic/Latin sentence ends and line breaks."""
    return [s.strip() for s in _SENT_END.split(text) if s and s.strip()]


def dedup_key(text: str) -> str:
    """Stable key for near-exact duplicate removal."""
    t = fold_homophones(unicodedata.normalize("NFC", text)).lower()
    t = re.sub(rf"[\s{re.escape(ETHIOPIC_PUNCT)}\W_]+", "", t)
    return hashlib.sha1(t.encode("utf-8")).hexdigest()


def word_count(text: str) -> int:
    """Words separated by whitespace or the Ethiopic word space '፡'."""
    return len([w for w in re.split(r"[\s፡]+", text) if w])
