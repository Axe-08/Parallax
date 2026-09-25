"""Tests for Parallax Preprocessing & Normalizer."""

import pandas as pd

from parallax.preprocessing.normalizer import (
    clean_soft_name,
    extract_numbers,
    get_token_sorted_name,
    normalize_unicode,
    widen_records_df,
)


def test_unicode_normalization_french():
    # French accented words
    raw = "Société Électrique de Paris"
    normalized = normalize_unicode(raw)
    assert "Société" in normalized
    soft = clean_soft_name(raw)
    assert soft == "société électrique de paris"


def test_unicode_preservation_indic():
    # Devanagari / Hindi
    hindi_text = "मॉडर्न कंस्ट्रक्शन प्राइवेट लिमिटेड"
    assert clean_soft_name(hindi_text) == hindi_text

    # Gujarati
    gujarati_text = "એપેક્સ ઇન્વેસ્ટમેન્ટ્સ"
    assert clean_soft_name(gujarati_text) == gujarati_text


def test_domain_stripping():
    assert clean_soft_name("burgersolution.com") == "burgersolution"
    assert clean_soft_name("Lumyn Oriental.co.in") == "lumyn oriental"


def test_token_sorted_name():
    assert get_token_sorted_name("robotics acme") == "acme robotics"
    assert get_token_sorted_name("delhi shahdara dandekar") == "dandekar delhi shahdara"


def test_number_extraction():
    numbers = extract_numbers("Flat No. A702, Door 25, 8th Cross, Koramangala")
    assert "a702" in numbers
    assert "25" in numbers
    assert "8th" in numbers or "8" in numbers


def test_widen_records_df():
    df = pd.DataFrame(
        [
            {
                "entity_id": "S1-1",
                "business_name": "Acme Robotics.com",
                "business_address": "500 Market St, Suite 4",
                "country": "US",
            }
        ]
    )
    widened = widen_records_df(df)
    assert "soft_name" in widened.columns
    assert widened.iloc[0]["soft_name"] == "acme robotics"
    assert "500" in widened.iloc[0]["numbers"]
