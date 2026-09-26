"""
Parallax Script & Language Detection Utility
============================================
Deterministic Unicode block identifier for Indic and non-Latin business records.
Handles pure Indic strings, pure Latin strings, and mixed-script tokens.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import NamedTuple

# Unicode block bounds for Brahmic / South Asian scripts
SCRIPT_UNICODE_RANGES: dict[str, tuple[int, int]] = {
    "DEVANAGARI": (0x0900, 0x097F),
    "BENGALI": (0x0980, 0x09FF),
    "GURMUKHI": (0x0A00, 0x0A7F),
    "GUJARATI": (0x0A80, 0x0AFF),
    "ORIYA": (0x0B00, 0x0B7F),
    "TAMIL": (0x0B80, 0x0BFF),
    "TELUGU": (0x0C00, 0x0C7F),
    "KANNADA": (0x0C80, 0x0CFF),
    "MALAYALAM": (0x0D00, 0x0D7F),
}

# Mapping from script name to default AI4Bharat / ISO language code
SCRIPT_TO_LANG_CODE: dict[str, str] = {
    "DEVANAGARI": "hi",
    "BENGALI": "bn",
    "GURMUKHI": "pa",
    "GUJARATI": "gu",
    "ORIYA": "or",
    "TAMIL": "ta",
    "TELUGU": "te",
    "KANNADA": "kn",
    "MALAYALAM": "ml",
}

_WORD_SPLIT_REGEX = re.compile(r"\s+|(?<=[^\w\s])|(?=[^\w\s])")


class TokenScriptInfo(NamedTuple):
    token: str
    script: str
    lang_code: str | None


def get_char_script(ch: str) -> str:
    """Return script name for a single Unicode character."""
    cp = ord(ch)
    for script_name, (start, end) in SCRIPT_UNICODE_RANGES.items():
        if start <= cp <= end:
            return script_name
    if cp < 0x0370:
        return "LATIN"
    return "OTHER"


def is_brahmic_char(ch: str) -> bool:
    """Return True if character belongs to any Brahmic script block."""
    cp = ord(ch)
    return 0x0900 <= cp <= 0x0D7F


def has_indic_script(text: str | None) -> bool:
    """Return True if text contains at least one Brahmic script character."""
    if not text:
        return False
    return any(is_brahmic_char(c) for c in str(text))


def detect_dominant_script(text: str | None) -> str:
    """
    Detect the dominant script in the given string based on character frequency.
    Returns the script name (e.g. 'DEVANAGARI', 'KANNADA', 'LATIN', 'UNKNOWN').
    """
    if not text:
        return "UNKNOWN"
    norm = unicodedata.normalize("NFKC", str(text))
    scripts: list[str] = []
    for c in norm:
        if c.isalnum():
            scripts.append(get_char_script(c))

    if not scripts:
        return "UNKNOWN"

    counts = Counter(scripts)
    # Prefer Indic script over Latin if present and substantial
    indic_counts = {k: v for k, v in counts.items() if k in SCRIPT_UNICODE_RANGES}
    if indic_counts:
        return max(indic_counts, key=indic_counts.get)  # type: ignore[arg-type]

    return counts.most_common(1)[0][0]


def get_language_for_text(text: str | None) -> str | None:
    """
    Determine the primary ISO language code for transliteration.
    Returns None if text is Latin/Unknown or has no Indic script.
    """
    dominant = detect_dominant_script(text)
    return SCRIPT_TO_LANG_CODE.get(dominant)


def segment_mixed_script(text: str | None) -> list[TokenScriptInfo]:
    """
    Segment a string into tokens and classify each token's script and language.
    Useful for Indian business names mixing English suffixes with Indic names
    (e.g., 'साउथ पायोनियर Engineering Pvt Ltd').
    """
    if not text:
        return []

    tokens = [t.strip() for t in str(text).split() if t.strip()]
    results: list[TokenScriptInfo] = []

    for tok in tokens:
        scripts = [get_char_script(c) for c in tok if c.isalnum()]
        if not scripts:
            results.append(TokenScriptInfo(token=tok, script="PUNCT", lang_code=None))
            continue
        c_indic = [s for s in scripts if s in SCRIPT_UNICODE_RANGES]
        if c_indic:
            dom = Counter(c_indic).most_common(1)[0][0]
            results.append(TokenScriptInfo(token=tok, script=dom, lang_code=SCRIPT_TO_LANG_CODE.get(dom)))
        else:
            results.append(TokenScriptInfo(token=tok, script="LATIN", lang_code=None))

    return results
