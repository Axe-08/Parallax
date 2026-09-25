"""
Parallax Brahmic Script Transliteration Module
==============================================
Deterministic, zero-dependency phonetic transliteration for Brahmic scripts
(Devanagari, Bengali, Gujarati, Gurmukhi, Kannada, Telugu, Tamil, Malayalam)
into normalized Latin phonemes.

Enables cross-script entity resolution without external APIs or ML overhead.
"""

from __future__ import annotations

import unicodedata

# Unicode offset mapping for Brahmic scripts (relative to script block start: cp % 0x80)
# Covers independent vowels, consonants, dependent matras, and special signs.
_BRAHMIC_OFFSET_MAP: dict[int, str] = {
    # Modifiers
    0x01: "n",   # Chandrabindu
    0x02: "n",   # Anusvara
    0x03: "h",   # Visarga
    0x04: "",    # Short vowel / sign
    # Independent vowels
    0x05: "a",
    0x06: "aa",
    0x07: "i",
    0x08: "ii",
    0x09: "u",
    0x0A: "uu",
    0x0B: "ri",
    0x0C: "li",
    0x0E: "e",
    0x0F: "e",
    0x10: "ai",
    0x12: "o",
    0x13: "o",
    0x14: "au",
    # Consonants (Gutturals)
    0x15: "k",
    0x16: "kh",
    0x17: "g",
    0x18: "gh",
    0x19: "ng",
    # Palatals
    0x1A: "ch",
    0x1B: "chh",
    0x1C: "j",
    0x1D: "jh",
    0x1E: "ny",
    # Retroflexes
    0x1F: "t",
    0x20: "th",
    0x21: "d",
    0x22: "dh",
    0x23: "n",
    # Dentals
    0x24: "t",
    0x25: "th",
    0x26: "d",
    0x27: "dh",
    0x28: "n",
    # Labials
    0x2A: "p",
    0x2B: "ph",
    0x2C: "b",
    0x2D: "bh",
    0x2E: "m",
    # Semivowels / Liquids / Sibilants
    0x2F: "y",
    0x30: "r",
    0x31: "r",
    0x32: "l",
    0x33: "l",
    0x35: "v",
    0x36: "sh",
    0x37: "sh",
    0x38: "s",
    0x39: "h",
    # Matras (Dependent Vowels)
    0x3E: "aa",
    0x3F: "i",
    0x40: "ii",
    0x41: "u",
    0x42: "uu",
    0x43: "ri",
    0x44: "ri",
    0x46: "e",
    0x47: "e",
    0x48: "ai",
    0x4A: "o",
    0x4B: "o",
    0x4C: "au",
    0x4D: "",    # Virama / Halant (suppresses default vowel)
    # Nukta / Extensions
    0x3C: "",    # Nukta
    0x58: "q",   # Urdu / Persianized consonants
    0x59: "kh",
    0x5A: "g",
    0x5B: "z",
    0x5C: "d",
    0x5D: "rh",
    0x5E: "f",
    0x5F: "y",
}


def is_brahmic_char(ch: str) -> bool:
    """Return True if character belongs to the South Asian Brahmic script range."""
    return 0x0900 <= ord(ch) <= 0x0D7F


def has_brahmic_script(text: str | None) -> bool:
    """Return True if text contains at least one Brahmic script character."""
    if not text:
        return False
    return any(0x0900 <= ord(c) <= 0x0D7F for c in str(text))


def transliterate_brahmic_to_latin(text: str | None) -> str:
    """
    Deterministically transliterate Brahmic script text into Latin phonetic representation.

    If text contains only ASCII or Latin characters, it is returned without modification.
    """
    if not text:
        return ""

    raw = unicodedata.normalize("NFKC", str(text).strip())
    # O(1) fast-path check
    if not any(0x0900 <= ord(c) <= 0x0D7F for c in raw):
        return raw

    out: list[str] = []
    for ch in raw:
        cp = ord(ch)
        if 0x0900 <= cp <= 0x0D7F:
            offset = cp % 0x80
            out.append(_BRAHMIC_OFFSET_MAP.get(offset, ""))
        else:
            out.append(ch)

    return "".join(out)
