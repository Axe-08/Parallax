"""
Unit tests for Experiment A: Feature Extractor v2 & Normalizer additions.
Verifies the 28-feature matrix, primary building number discrimination,
postal code alignment, and canonical address folding.
"""

from __future__ import annotations

import pandas as pd
import pytest

from parallax.features.extractor import (
    FEATURE_COLUMNS,
    PairwiseFeatureExtractor,
    extract_core_and_suffix,
)
from parallax.preprocessing.normalizer import (
    canonicalize_address,
    extract_numbers,
    extract_postal_code,
    extract_primary_number,
    widen_records_df,
)


def test_feature_columns_count():
    """Verify that FEATURE_COLUMNS contains exactly 34 features with baseline 28 intact."""
    assert len(FEATURE_COLUMNS) == 34
    baseline_28 = [
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
        "primary_num_match",
        "primary_num_conflict",
        "primary_num_missing",
        "num_jaccard",
        "num_conflict_count",
        "postal_match",
        "postal_conflict",
        "postal_missing",
        "jaro_winkler_soft",
        "jaro_winkler_raw",
        "token_jaccard_name",
        "token_overlap_name",
        "first_token_match",
        "canon_addr_ratio",
        "token_jaccard_addr",
    ]
    assert FEATURE_COLUMNS[:28] == baseline_28
    batch_1_new = [
        "translit_boost_name",
        "is_cross_script",
        "translit_name_ratio",
        "name_core_ratio",
        "suffix_match",
        "s1_candidate_count",
    ]
    assert FEATURE_COLUMNS[28:] == batch_1_new


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


def test_translit_boost_and_cross_script():
    """Verify translit_boost_name, is_cross_script, and translit_name_ratio on cross-script pair."""
    s1_df = pd.DataFrame(
        [
            {
                "entity_id": "S1-HI",
                "business_name": "Dynamic Engineering",
                "business_address": "123 Main St, New Delhi",
                "country": "India",
            }
        ]
    )
    cand_df = pd.DataFrame(
        [
            {
                "entity_id": "S2-HI",
                "business_name": "डायनामिक इंजीनियरिंग",
                "business_address": "123 Main St, New Delhi",
                "country": "India",
            }
        ]
    )
    s1_w = widen_records_df(s1_df)
    cand_w = widen_records_df(cand_df)

    extractor = PairwiseFeatureExtractor()
    feat_df = extractor.extract_features_df({"S1-HI": {"S2-HI"}}, s1_w, cand_w)
    row = feat_df.iloc[0]

    assert row["is_cross_script"] == 1.0
    assert row["translit_boost_name"] > 0.4
    assert row["translit_name_ratio"] > 0.65

    cand_latin_df = pd.DataFrame(
        [
            {
                "entity_id": "S2-LAT",
                "business_name": "Dynamic Engineering Inc",
                "business_address": "123 Main St, New Delhi",
                "country": "India",
            }
        ]
    )
    cand_latin_w = widen_records_df(cand_latin_df)
    feat_latin = extractor.extract_features_df({"S1-HI": {"S2-LAT"}}, s1_w, cand_latin_w)
    row_latin = feat_latin.iloc[0]
    assert row_latin["is_cross_script"] == 0.0
    assert row_latin["translit_boost_name"] == 0.0


def test_name_core_and_suffix_match():
    """Verify name_core_ratio and suffix_match discriminate corporate legal entities."""
    s1_df = pd.DataFrame(
        [
            {
                "entity_id": "S1-CORP",
                "business_name": "Google, Inc.",
                "business_address": "1600 Amphitheatre Pkwy",
                "country": "US",
            }
        ]
    )
    cand_df = pd.DataFrame(
        [
            {
                "entity_id": "S2-LLC",
                "business_name": "Google LLC",
                "business_address": "1600 Amphitheatre Pkwy",
                "country": "US",
            }
        ]
    )
    s1_w = widen_records_df(s1_df)
    cand_w = widen_records_df(cand_df)

    extractor = PairwiseFeatureExtractor()
    feat_df = extractor.extract_features_df({"S1-CORP": {"S2-LLC"}}, s1_w, cand_w)
    row = feat_df.iloc[0]

    assert row["name_core_ratio"] == 1.0
    assert row["suffix_match"] == 0.0

    cand_inc_df = pd.DataFrame(
        [
            {
                "entity_id": "S2-INC",
                "business_name": "Google Incorporated",
                "business_address": "1600 Amphitheatre Pkwy",
                "country": "US",
            }
        ]
    )
    cand_inc_w = widen_records_df(cand_inc_df)
    feat_inc = extractor.extract_features_df({"S1-CORP": {"S2-INC"}}, s1_w, cand_inc_w)
    assert feat_inc.iloc[0]["suffix_match"] == 1.0
    assert feat_inc.iloc[0]["name_core_ratio"] == 1.0

    s1_dr = pd.DataFrame(
        [
            {
                "entity_id": "S1-DR",
                "business_name": "Dr. Agarwal Clinic",
                "business_address": "MG Road",
                "country": "India",
            }
        ]
    )
    cand_plain = pd.DataFrame(
        [
            {
                "entity_id": "S2-PLAIN",
                "business_name": "Agarwal Clinic",
                "business_address": "MG Road",
                "country": "India",
            }
        ]
    )
    feat_dr = extractor.extract_features_df(
        {"S1-DR": {"S2-PLAIN"}}, widen_records_df(s1_dr), widen_records_df(cand_plain)
    )
    assert feat_dr.iloc[0]["name_core_ratio"] == 1.0
    assert feat_dr.iloc[0]["suffix_match"] == 0.5


def test_suffix_stripping_safety_on_brand_tokens():
    """Verify corporate suffix extraction does NOT remove core brand tokens."""
    core, suf = extract_core_and_suffix("zinc")
    assert core == "zinc"
    assert suf is None

    core, suf = extract_core_and_suffix("incite solutions")
    assert core == "incite solutions"
    assert suf is None

    core, suf = extract_core_and_suffix("llc")
    assert core == "llc"
    assert suf == "llc"

    core, suf = extract_core_and_suffix("tata motors private limited")
    assert core == "tata motors"
    assert suf == "pvt_ltd"

    core, suf = extract_core_and_suffix("m s sharma trading co")
    assert core == "sharma trading"
    assert suf == "co"


def test_s1_candidate_count():
    """Verify s1_candidate_count accurately reflects blocker pool cardinality."""
    s1_df = pd.DataFrame(
        [
            {
                "entity_id": "S1-A",
                "business_name": "Acme Alpha",
                "business_address": "1st Ave",
                "country": "US",
            },
            {
                "entity_id": "S1-B",
                "business_name": "Beta Bravo",
                "business_address": "2nd Ave",
                "country": "US",
            },
        ]
    )
    cand_df = pd.DataFrame(
        [
            {
                "entity_id": "C-1",
                "business_name": "Acme Alpha 1",
                "business_address": "1st Ave",
                "country": "US",
            },
            {
                "entity_id": "C-2",
                "business_name": "Acme Alpha 2",
                "business_address": "1st Ave",
                "country": "US",
            },
            {
                "entity_id": "C-3",
                "business_name": "Acme Alpha 3",
                "business_address": "1st Ave",
                "country": "US",
            },
            {
                "entity_id": "C-4",
                "business_name": "Beta Bravo 1",
                "business_address": "2nd Ave",
                "country": "US",
            },
        ]
    )
    s1_w = widen_records_df(s1_df)
    cand_w = widen_records_df(cand_df)

    pairs = {
        "S1-A": {"C-1", "C-2", "C-3"},  # 3 candidates
        "S1-B": {"C-4"},  # 1 candidate
    }
    extractor = PairwiseFeatureExtractor()
    feat_df = extractor.extract_features_df(pairs, s1_w, cand_w)

    a_rows = feat_df[feat_df["s1_id"] == "S1-A"]
    b_rows = feat_df[feat_df["s1_id"] == "S1-B"]

    assert len(a_rows) == 3
    assert (a_rows["s1_candidate_count"] == 3.0).all()

    assert len(b_rows) == 1
    assert (b_rows["s1_candidate_count"] == 1.0).all()

