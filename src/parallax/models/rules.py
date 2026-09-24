"""
Parallax Deterministic Rule Engine & Unit Normalizer
====================================================
Deterministic regex entity extraction and unit normalization.
Enforces strict separation of control and guardrail formatting.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import pandas as pd

from parallax.models.base import BaseModel

# Standard unit canonicalization mappings
UNIT_SYNONYMS: dict[str, str] = {
    # Mass
    "g": "gram",
    "gm": "gram",
    "gms": "gram",
    "gram": "gram",
    "grams": "gram",
    "kg": "kilogram",
    "kgs": "kilogram",
    "kilogram": "kilogram",
    "kilograms": "kilogram",
    "mg": "milligram",
    "milligram": "milligram",
    "oz": "ounce",
    "ounce": "ounce",
    "ounces": "ounce",
    "lb": "pound",
    "lbs": "pound",
    "pound": "pound",
    "pounds": "pound",
    # Volume
    "ml": "millilitre",
    "mls": "millilitre",
    "millilitre": "millilitre",
    "milliliter": "millilitre",
    "l": "litre",
    "liter": "litre",
    "litre": "litre",
    "litres": "litre",
    "fl oz": "fluid ounce",
    "fluid ounce": "fluid ounce",
    # Dimensions / Length
    "cm": "centimetre",
    "centimetre": "centimetre",
    "centimeter": "centimetre",
    "mm": "millimetre",
    "millimetre": "millimetre",
    "millimeter": "millimetre",
    "m": "metre",
    "meter": "metre",
    "metre": "metre",
    "in": "inch",
    "inch": "inch",
    "inches": "inch",
    "ft": "foot",
    "foot": "foot",
    "feet": "foot",
    # Electrical
    "v": "volt",
    "volt": "volt",
    "volts": "volt",
    "w": "watt",
    "watt": "watt",
    "watts": "watt",
    "kw": "kilowatt",
    "kilowatt": "kilowatt",
}


def normalize_unit(unit_str: str) -> str:
    """Normalize colloquial unit strings to standardized canonical forms."""
    clean = unit_str.strip().lower()
    return UNIT_SYNONYMS.get(clean, clean)


class RegexRuleEngine(BaseModel):
    """
    Deterministic rule-based extractor using regular expressions.
    Acts as a high-precision baseline and guardrail fallback.
    """

    def __init__(
        self,
        name: str = "regex_rule_engine",
        text_cols: list[str] | None = None,
        default_fallback: str = "",
    ) -> None:
        super().__init__(name=name)
        self.text_cols = text_cols or ["catalog_content", "title", "description"]
        self.default_fallback = default_fallback
        # Match e.g. "500 gram", "1.5 kg", "250ml", "10.2 cm"
        self._pattern = re.compile(
            r"(\b\d+(?:\.\d+)?)\s*([a-zA-Z]+(?:\s+[a-zA-Z]+)?)\b",
            re.IGNORECASE,
        )

    def fit(self, df: pd.DataFrame, target_col: str) -> RegexRuleEngine:
        """Compute most frequent baseline target as fallback."""
        if target_col in df.columns:
            mode_series = df[target_col].mode()
            if not mode_series.empty:
                self.default_fallback = str(mode_series.iloc[0])
        self.is_fitted = True
        return self

    def extract_from_text(self, text: str) -> str | None:
        """Extract value and normalized unit from a single text block."""
        if not text or not isinstance(text, str):
            return None

        # Find occurrences of number followed by potential unit words
        pattern = r"(\b\d+(?:\.\d+)?)\s*([a-zA-Z]+)(?:\s+([a-zA-Z]+))?"
        for match in re.finditer(pattern, text):
            val_str = match.group(1)
            word1 = match.group(2).lower()
            word2 = match.group(3).lower() if match.group(3) else None

            # Check two-word unit first (e.g. "fl oz", "fluid ounce")
            if word2:
                two_word = f"{word1} {word2}"
                if two_word in UNIT_SYNONYMS:
                    return f"{val_str} {normalize_unit(two_word)}"

            # Check single-word unit (e.g. "g", "kg", "cm", "ml")
            if word1 in UNIT_SYNONYMS:
                return f"{val_str} {normalize_unit(word1)}"

        return None

    def predict_single(self, row: pd.Series) -> str:
        """Predict for a single row across candidate text columns."""
        for col in self.text_cols:
            if col in row and pd.notna(row[col]):
                result = self.extract_from_text(str(row[col]))
                if result:
                    return result
        return self.default_fallback

    def predict(self, df: pd.DataFrame) -> Sequence[Any]:
        """Predict for all rows in dataframe."""
        return [self.predict_single(row) for _, row in df.iterrows()]
