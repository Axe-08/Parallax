"""
Record serialization for neural encoders.

Two layers:
- Legacy helpers (normalize_text, serialize_full, ...) kept for the zero-shot reference
  channel in run_5k.py.
- RecordSerializer: field-bounded, missingness-explicit, token-aware serialization with
  multiple views (full / name / address / name_country / address_country). Each field is
  truncated independently on token boundaries of the encoder's tokenizer, so a long
  address can never push the name out of the context window.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def normalize_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def serialize_full(row):
    """
    Serializes a row into a canonical dense representation.
    Output: [COUNTRY] india [NAME] some business [ADDRESS] 123 street
    """
    country = normalize_text(row.get('country', ''))
    name = normalize_text(row.get('business_name', ''))
    address = normalize_text(row.get('business_address', ''))

    parts = []
    if country:
        parts.append(f"[COUNTRY] {country}")
    if name:
        parts.append(f"[NAME] {name}")
    if address:
        parts.append(f"[ADDRESS] {address}")

    # Default to something if completely empty, to avoid empty embeddings
    if not parts:
        return "[EMPTY]"
    return " ".join(parts)

def serialize_name_only(row):
    """
    Serializes a row using only country and name.
    """
    country = normalize_text(row.get('country', ''))
    name = normalize_text(row.get('business_name', ''))

    parts = []
    if country:
        parts.append(f"[COUNTRY] {country}")
    if name:
        parts.append(f"[NAME] {name}")

    if not parts:
        return "[EMPTY]"
    return " ".join(parts)

def serialize_address_only(row):
    """
    Serializes a row using only country and address.
    """
    country = normalize_text(row.get('country', ''))
    address = normalize_text(row.get('business_address', ''))

    parts = []
    if country:
        parts.append(f"[COUNTRY] {country}")
    if address:
        parts.append(f"[ADDRESS] {address}")

    if not parts:
        return "[EMPTY]"
    return " ".join(parts)

def truncate_text(text, max_chars=1024):
    """
    Truncates text to a maximum number of characters to protect embedding context length.
    """
    if len(text) > max_chars:
        return text[:max_chars]
    return text

def extract_numerics(text):
    """
    Extracts purely numeric tokens or alphanumeric blocks from an address.
    """
    if not isinstance(text, str):
        return []
    # Find all contiguous blocks of digits
    numerics = re.findall(r'\b\d+\b', text)
    return numerics


# ---------------------------------------------------------------------------------------
# Field-bounded, token-aware serialization (Phase 2)
# ---------------------------------------------------------------------------------------

VIEWS: dict[str, tuple[str, ...]] = {
    "full": ("name", "address", "country"),
    "name": ("name",),
    "address": ("address",),
    "name_country": ("name", "country"),
    "address_country": ("address", "country"),
}
_FIELD_COLUMNS = {"name": "business_name", "address": "business_address", "country": "country"}
_WS = re.compile(r"\s+")


def clean_field(value: object, lowercase: bool = True) -> str:
    """NFKC + whitespace collapse (+ lowercase). Keeps every script intact; never transliterates."""
    if value is None or (isinstance(value, float) and value != value):
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = _WS.sub(" ", text).strip()
    return text.lower() if lowercase else text


@dataclass(frozen=True)
class SerializerConfig:
    max_name_tokens: int = 40
    max_address_tokens: int = 80
    max_country_tokens: int = 8
    lowercase: bool = True
    missing_token: str = "<missing>"


class RecordSerializer:
    """
    Serializes records as e.g. ``name: acme llc | address: 12 main st | country: us``.

    Missing fields are explicit (``address: <missing>``) in the full view so that the encoder
    can distinguish "no address" from "short address". Single-field views of a missing field
    yield the bare missing token. Field truncation is token-aware when a tokenizer is given.
    """

    def __init__(self, tokenizer: Any = None, config: SerializerConfig | None = None) -> None:
        self.tokenizer = tokenizer
        self.config = config or SerializerConfig()
        self.truncation_counts: dict[str, int] = {}

    def _limit(self, field: str) -> int:
        return {
            "name": self.config.max_name_tokens,
            "address": self.config.max_address_tokens,
            "country": self.config.max_country_tokens,
        }[field]

    def _truncate(self, field: str, values: list[str]) -> list[str]:
        if self.tokenizer is None:
            return values
        limit = self._limit(field)
        enc = self.tokenizer(
            values, add_special_tokens=False, return_offsets_mapping=True, truncation=False
        )
        out: list[str] = []
        n_cut = 0
        for text, ids, offsets in zip(values, enc["input_ids"], enc["offset_mapping"]):
            if len(ids) > limit:
                n_cut += 1
                text = text[: offsets[limit - 1][1]].rstrip()
            out.append(text)
        self.truncation_counts[field] = self.truncation_counts.get(field, 0) + n_cut
        return out

    def clean_fields(self, df: pd.DataFrame, fields: Sequence[str]) -> dict[str, list[str]]:
        cleaned: dict[str, list[str]] = {}
        for f in fields:
            col = _FIELD_COLUMNS[f]
            if col not in df.columns:
                raise ValueError(f"Records are missing column {col!r} required by field {f!r}.")
            cleaned[f] = self._truncate(f, [clean_field(v, self.config.lowercase) for v in df[col].tolist()])
        return cleaned

    def serialize(self, df: pd.DataFrame, view: str = "full") -> list[str]:
        if view not in VIEWS:
            raise ValueError(f"Unknown view {view!r}. Available: {sorted(VIEWS)}")
        fields = VIEWS[view]
        cleaned = self.clean_fields(df, fields)
        miss = self.config.missing_token
        if len(fields) == 1:
            return [v if v else miss for v in cleaned[fields[0]]]
        return [
            " | ".join(f"{f}: {cleaned[f][i] or miss}" for f in fields)
            for i in range(len(df))
        ]
