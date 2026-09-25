"""
Unit tests for Experiment A: Feature Extractor v2 & Normalizer additions.
Verifies the 28-feature matrix, primary building number discrimination,
postal code alignment, and canonical address folding.
"""

from __future__ import annotations

import pandas as pd
import pytest

from parallax.features.extractor import FEATURE_COLUMNS, PairwiseFeatureExtractor
from parallax.preprocessing.normalizer import (
    canonicalize_address,
    extract_numbers,
    extract_postal_code,
    extract_primary_number,
    widen_records_df,
)


def test_feature_columns_count():
    """Verify that FEATURE_COLUMNS contains exactly 28 features."""
    assert len(FEATURE_COLUMNS) == 28
    # Baseline 13 features must remain in exact initial order
    baseline_13 = [
        "raw_name_ratio",
        "soft_name_ratio",
        "token_sort_ratio",
        "token_set_ratio",
        "partial_ratio",
        "addr_token_set_ratio",
        "addr_ratio",
        "num_match_score",
        "is_s1_addr_null",
        "is_cand_addr_null",
        "both_addr_present",
        "len_diff_name",
        "len_ratio_name",
    ]
    assert FEATURE_COLUMNS[:13] == baseline_13


def test_primary_number_street_conflict_with_shared_unit():
    """Test 28 Levant St, Unit 2 vs 39 Levant St, # 2 (shared unit cannot hide street conflict)."""
    addr1 = "28 Levant Street, Unit 2, Boston, MA"
    addr2 = "39 Levant Street, # 2, Boston, MA"

    p1 = extract_primary_number(addr1)
    p2 = extract_primary_number(addr2)

    assert p1 == "28"
    assert p2 == "39"
    assert p1 != p2

    # End-to-end feature extraction test
    s1_df = pd.DataFrame(
        [
            {
                "entity_id": "S1-1",
                "business_name": "Parks Bureau",
                "business_address": addr1,
                "country": "US",
            }
        ]
    )
    cand_df = pd.DataFrame(
        [
            {
                "entity_id": "S2-1",
                "business_name": "Parks Bureau Co",
                "business_address": addr2,
                "country": "US",
            }
        ]
    )
    s1_w = widen_records_df(s1_df)
    cand_w = widen_records_df(cand_df)

    extractor = PairwiseFeatureExtractor()
    feat_df = extractor.extract_features_df({"S1-1": {"S2-1"}}, s1_w, cand_w)

    row = feat_df.iloc[0]
    assert row["primary_num_conflict"] == 1.0
    assert row["primary_num_match"] == 0.0
    assert row["primary_num_missing"] == 0.0
    # Jaccard over {28, 2} and {39, 2} is 1/3 ~ 0.33, NOT 1.0
    assert pytest.approx(row["num_jaccard"], 0.01) == 0.333
    assert row["num_conflict_count"] == 2.0


def test_adjacent_street_numbers():
    """Test 4711 vs 4712 Howell School Road (adjacent clinic false merge case)."""
    addr1 = "4711 Howell School Road, Jonesville, NC"
    addr2 = "4712 HOWELL SCHOOL ROAD, JONESVILLE, NC"

    p1 = extract_primary_number(addr1)
    p2 = extract_primary_number(addr2)

    assert p1 == "4711"
    assert p2 == "4712"
    assert p1 != p2

    s1_df = widen_records_df(
        pd.DataFrame(
            [
                {
                    "entity_id": "S1-1",
                    "business_name": "Surgical Care",
                    "business_address": addr1,
                    "country": "US",
                }
            ]
        )
    )
    cand_df = widen_records_df(
        pd.DataFrame(
            [
                {
                    "entity_id": "S2-1",
                    "business_name": "Surgical Care Ltd",
                    "business_address": addr2,
                    "country": "US",
                }
            ]
        )
    )
    extractor = PairwiseFeatureExtractor()
    feat_df = extractor.extract_features_df({"S1-1": {"S2-1"}}, s1_df, cand_df)

    row = feat_df.iloc[0]
    assert row["primary_num_conflict"] == 1.0
    assert row["primary_num_match"] == 0.0


def test_indian_flat_and_wing_conflict():
    """Test 706 vs 707 C Wing (Indian flat number false merge case)."""
    addr1 = "706 C Wing, Rna Royal Park, Mg Rd, Mumbai"
    addr2 = "H.NO 707 C WING, RNA ROYAL PARK, MG RD, MUMBAI"

    p1 = extract_primary_number(addr1)
    p2 = extract_primary_number(addr2)

    assert p1 == "706"
    assert p2 == "707"

    s1_df = widen_records_df(
        pd.DataFrame(
            [
                {
                    "entity_id": "S1-1",
                    "business_name": "Ravitech",
                    "business_address": addr1,
                    "country": "India",
                }
            ]
        )
    )
    cand_df = widen_records_df(
        pd.DataFrame(
            [
                {
                    "entity_id": "S2-1",
                    "business_name": "Ravitech Ltd",
                    "business_address": addr2,
                    "country": "India",
                }
            ]
        )
    )
    extractor = PairwiseFeatureExtractor()
    feat_df = extractor.extract_features_df({"S1-1": {"S2-1"}}, s1_df, cand_df)

    row = feat_df.iloc[0]
    assert row["primary_num_conflict"] == 1.0
    assert row["primary_num_match"] == 0.0


def test_ordinal_normalization_9th_vs_ninth():
    """Test 9th vs Ninth folding in address canonicalization."""
    addr1 = "711 9th Street, Etowah, TN"
    addr2 = "Ninth St, Etwah, Tennessee"

    c1 = canonicalize_address(addr1)
    c2 = canonicalize_address(addr2)

    assert "ninth" in c1
    assert "ninth" in c2
    assert "street" in c1
    assert "street" in c2


def test_state_abbreviation_normalization_tn_vs_tennessee():
    """Test US State abbreviation folding TN -> tennessee."""
    c1 = canonicalize_address("Etowah, TN")
    c2 = canonicalize_address("Etwah, Tennessee")

    assert "tennessee" in c1
    assert "tennessee" in c2


def test_missing_postal_codes():
    """Test missing postal code handling."""
    addr1 = "18 Ash Street, Danvers, MA"
    addr2 = "18 Ash Street, MA"

    post1 = extract_postal_code(addr1)
    post2 = extract_postal_code(addr2)

    assert post1 is None
    assert post2 is None

    s1_df = widen_records_df(
        pd.DataFrame(
            [
                {
                    "entity_id": "S1-1",
                    "business_name": "Acme",
                    "business_address": addr1,
                    "country": "US",
                }
            ]
        )
    )
    cand_df = widen_records_df(
        pd.DataFrame(
            [
                {
                    "entity_id": "S2-1",
                    "business_name": "Acme",
                    "business_address": addr2,
                    "country": "US",
                }
            ]
        )
    )
    extractor = PairwiseFeatureExtractor()
    feat_df = extractor.extract_features_df({"S1-1": {"S2-1"}}, s1_df, cand_df)

    row = feat_df.iloc[0]
    assert row["postal_missing"] == 1.0
    assert row["postal_match"] == 0.0
    assert row["postal_conflict"] == 0.0


def test_postal_code_match_and_conflict():
    """Test explicit postal code match and conflict."""
    addr_us1 = "180 Strunk Road, Corbin, KY 40701"
    addr_us2 = "180 Strunk Road, Corbin, KY 40701"
    addr_us3 = "180 Strunk Road, Corbin, KY 90210"

    assert extract_postal_code(addr_us1) == "40701"
    assert extract_postal_code(addr_us2) == "40701"
    assert extract_postal_code(addr_us3) == "90210"

    # India PIN code
    addr_in1 = "Khairatabad, Hyderabad 500034, Telangana"
    addr_in3 = "Khairatabad, Hyderabad 400001, Maharashtra"

    assert extract_postal_code(addr_in1) == "500034"
    assert extract_postal_code(addr_in3) == "400001"


def test_multiple_address_numbers():
    """Test parsing addresses with multiple numbers (building, floor, plot)."""
    addr = "210, 2nd Floor, Sim Lim Square, Plot-301, Mumbai 400001"
    p = extract_primary_number(addr)
    nums = extract_numbers(addr)
    pin = extract_postal_code(addr)

    # Primary number is 210
    assert p == "210"
    # All numbers captured in number_set
    assert "210" in nums
    assert "2nd" in nums
    assert "301" in nums
    assert pin == "400001"


def test_alphanumeric_flat_formats():
    """Test Indian alphanumeric flat formats such as 4A, Flat H-1, 303-A."""
    assert extract_primary_number("Bangalore, 4A, Gold Nest, Wind Tunnel Road") == "4a"
    assert extract_primary_number("Flat H-1, Rohan Garima, Pune") == "h1"
    assert extract_primary_number("303-A, Nbc Complex, Sector 11, Navi Mumbai") == "303a"
