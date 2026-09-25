"""
Parallax Preprocessing & Text Normalizer
========================================
Non-destructive data widening and multilingual text canonicalization:
- Unicode NFKC normalization (handles French accents and Indic scripts)
- Top-level web domain stripping (.com, .in, .fr, .org, .co)
- Structural building/unit number extraction
- Token-sorted views for word-order invariance
"""

from __future__ import annotations

import re
import unicodedata

import pandas as pd

from parallax.preprocessing.transliteration import transliterate_brahmic_to_latin

_DOMAIN_PATTERN = re.compile(
    r"\.(com|in|org|co|net|io|fr|gov|edu|biz|info)\b",
    re.IGNORECASE,
)
_PUNCTUATION_PATTERN = re.compile(
    r"[\r\n\t,;:\"\'\[\]\(\)\{\}\*\#\-\_\/\\]",
)
_NUMBER_PATTERN = re.compile(
    r"\b[a-z]{0,2}[0-9]{1,6}[a-z]{0,2}\b",
    re.IGNORECASE,
)


def normalize_unicode(text: str | None) -> str:
    """Standardize unicode representation using NFKC normalization."""
    if not text or pd.isna(text):
        return ""
    return unicodedata.normalize("NFKC", str(text).strip())


def clean_soft_name(text: str | None) -> str:
    """
    Produce a relaxed soft representation of business name:
    - Lowercase + NFKC unicode
    - Strips domain extensions (e.g. burgersolution.com -> burgersolution)
    - Replaces brackets and punctuation with spaces
    - Collapses repeated whitespace
    """
    if not text or pd.isna(text):
        return ""

    raw = unicodedata.normalize("NFKC", str(text).strip().lower())
    raw = _DOMAIN_PATTERN.sub("", raw)
    raw = _PUNCTUATION_PATTERN.sub(" ", raw)
    return " ".join(raw.split())


def get_token_sorted_name(soft_name: str) -> str:
    """Return words sorted alphabetically to provide word-order transposition invariance."""
    tokens = soft_name.split()
    tokens.sort()
    return " ".join(tokens)


def extract_numbers(address: str | None) -> set[str]:
    """Extract numeric and alphanumeric building/flat numbers from an address string."""
    if not address or pd.isna(address):
        return set()
    raw = str(address).lower()
    matches = _NUMBER_PATTERN.findall(raw)
    return {m.strip() for m in matches if m.strip()}


def clean_address(address: str | None) -> str:
    """Produce a normalized lowercase address string with collapsed whitespace."""
    if not address or pd.isna(address):
        return ""
    raw = unicodedata.normalize("NFKC", str(address).strip().lower())
    raw = _PUNCTUATION_PATTERN.sub(" ", raw)
    return " ".join(raw.split())


def clean_transliterated_text(text: str | None) -> str:
    """Produce normalized Latin phonetic string from potentially multilingual text."""
    if not text or pd.isna(text):
        return ""
    translit = transliterate_brahmic_to_latin(str(text))
    return clean_soft_name(translit)


def widen_records_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived, parallel representation columns to a business records DataFrame
    without modifying the original raw columns.
    """
    enriched = df.copy()
    enriched["soft_name"] = enriched["business_name"].apply(clean_soft_name)
    enriched["token_sorted_name"] = enriched["soft_name"].apply(get_token_sorted_name)
    enriched["translit_name"] = enriched["business_name"].apply(clean_transliterated_text)
    enriched["clean_address"] = enriched["business_address"].apply(clean_address)
    enriched["translit_address"] = enriched["business_address"].apply(clean_transliterated_text)
    enriched["numbers"] = enriched["business_address"].apply(extract_numbers)
    enriched["is_addr_null"] = enriched["business_address"].isna().astype(int)
    return enriched
