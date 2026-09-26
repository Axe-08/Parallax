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
from parallax.features.meta_features import (
    META_FEATURE_COLUMNS,
    compute_entity_meta_features,
)
from parallax.preprocessing.normalizer import (
    canonicalize_address,
    extract_city_token,
    extract_numbers,
    extract_postal_code,
    extract_primary_number,
    widen_records_df,
)


def test_feature_columns_count():
    """Verify that FEATURE_COLUMNS contains exactly 49 features."""
    assert len(FEATURE_COLUMNS) == 49
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
    assert "translit_boost_name" in FEATURE_COLUMNS
    assert "name_core_ratio" in FEATURE_COLUMNS
    assert "phonetic_name_match" in FEATURE_COLUMNS
    assert "blocking_sim_score" in FEATURE_COLUMNS


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


def test_primary_num_distance():
    """Test primary_num_distance for matching, conflicting, and missing numbers."""
    s1_df = widen_records_df(
        pd.DataFrame(
            [
                {
                    "entity_id": "S1-1",
                    "business_name": "Clinic Alpha",
                    "business_address": "5 Central Market, Delhi",
                    "country": "India",
                },
                {
                    "entity_id": "S1-2",
                    "business_name": "Custom Special",
                    "business_address": "2693 Fairfield Pike, Wartrace, TN",
                    "country": "US",
                },
            ]
        )
    )
    cand_df = widen_records_df(
        pd.DataFrame(
            [
                {
                    "entity_id": "S2-1",
                    "business_name": "Clinic Beta",
                    "business_address": "9 Central Market, Delhi",
                    "country": "India",
                },
                {
                    "entity_id": "S2-2",
                    "business_name": "Custom Special Co",
                    "business_address": "2698 Fairfield Pike, Wartrace, TN",
                    "country": "US",
                },
            ]
        )
    )
    extractor = PairwiseFeatureExtractor()
    feat_df = extractor.extract_features_df(
        {"S1-1": {"S2-1"}, "S1-2": {"S2-2"}}, s1_df, cand_df, show_progress=False
    )
    row_5_9 = feat_df[feat_df["s1_id"] == "S1-1"].iloc[0]
    row_close = feat_df[feat_df["s1_id"] == "S1-2"].iloc[0]

    # 5 vs 9 has significant distance (>= 0.4)
    assert row_5_9["primary_num_distance"] >= 0.4
    assert row_5_9["primary_num_conflict"] == 1.0

    # 2693 vs 2698 has low distance (<= 0.25)
    assert row_close["primary_num_distance"] <= 0.25
    assert row_close["primary_num_conflict"] == 1.0


def test_city_token_extraction_and_matching():
    """Test city extraction, match, and conflict features."""
    addr_us_s1 = "18 Ash Street, Danvers, MA"
    addr_us_cand_match = "18 Ash Street, Danvers, MA"
    addr_us_cand_conflict = "18 Ash Street, Boston, MA"

    assert extract_city_token(addr_us_s1) == "danvers"
    assert extract_city_token(addr_us_cand_conflict) == "boston"

    s1_df = widen_records_df(
        pd.DataFrame(
            [
                {
                    "entity_id": "S1-1",
                    "business_name": "Bakery",
                    "business_address": addr_us_s1,
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
                    "business_name": "Bakery",
                    "business_address": addr_us_cand_match,
                    "country": "US",
                },
                {
                    "entity_id": "S2-2",
                    "business_name": "Bakery",
                    "business_address": addr_us_cand_conflict,
                    "country": "US",
                },
            ]
        )
    )
    extractor = PairwiseFeatureExtractor()
    feat_df = extractor.extract_features_df(
        {"S1-1": {"S2-1", "S2-2"}}, s1_df, cand_df, show_progress=False
    )
    match_row = feat_df[feat_df["cand_id"] == "S2-1"].iloc[0]
    conflict_row = feat_df[feat_df["cand_id"] == "S2-2"].iloc[0]

    assert match_row["city_match"] == 1.0
    assert match_row["city_conflict"] == 0.0

    assert conflict_row["city_match"] == 0.0
    assert conflict_row["city_conflict"] == 1.0


def test_missing_address_interaction_features():
    """Test null-address interaction feature values when address is present vs missing."""
    s1_df = widen_records_df(
        pd.DataFrame(
            [
                {
                    "entity_id": "S1-1",
                    "business_name": "Acme Global Solutions",
                    "business_address": "100 Market St, San Francisco, CA",
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
                    "business_name": "Acme Global Solutions LLC",
                    "business_address": None,
                    "country": "US",
                },
                {
                    "entity_id": "S2-2",
                    "business_name": "Acme Global Solutions LLC",
                    "business_address": "100 Market St, San Francisco, CA",
                    "country": "US",
                },
            ]
        )
    )
    extractor = PairwiseFeatureExtractor()
    feat_df = extractor.extract_features_df(
        {"S1-1": {"S2-1", "S2-2"}}, s1_df, cand_df, show_progress=False
    )

    null_addr_row = feat_df[feat_df["cand_id"] == "S2-1"].iloc[0]
    present_addr_row = feat_df[feat_df["cand_id"] == "S2-2"].iloc[0]

    # For null candidate address:
    assert null_addr_row["is_cand_addr_null"] == 1.0
    assert null_addr_row["name_ratio_null_cand_addr"] == null_addr_row["soft_name_ratio"]
    assert null_addr_row["name_ratio_null_either_addr"] == null_addr_row["soft_name_ratio"]
    assert null_addr_row["high_conf_name_no_addr"] == 1.0

    # For present address:
    assert present_addr_row["is_cand_addr_null"] == 0.0
    assert present_addr_row["name_ratio_null_cand_addr"] == 0.0
    assert present_addr_row["name_ratio_null_either_addr"] == 0.0
    assert present_addr_row["high_conf_name_no_addr"] == 0.0


def test_indian_state_canonicalization():
    """Test that canonicalize_address standardizes Indian state abbreviations and Indic scripts."""
    c1 = canonicalize_address("Kolkata, WB")
    c2 = canonicalize_address("Door No 73 Suryoday Vihar, Thane, महाराष्ट्र")
    c3 = canonicalize_address("Bangalore, KA")

    assert "west bengal" in c1
    assert "maharashtra" in c2
    assert "karnataka" in c3


def test_legal_suffix_and_core_extraction():
    """Test prefix and legal suffix stripping and matching."""
    core1, s1 = extract_core_and_suffix("dr dandekar vidyalaya industries")
    assert core1 == "dandekar vidyalaya"
    assert s1 == "industries"

    core2, s2 = extract_core_and_suffix("dandekar vidyalaya")
    assert core2 == "dandekar vidyalaya"
    assert s2 is None

    core3, s3 = extract_core_and_suffix("m/s custom special pvt ltd")
    assert core3 == "custom special"
    assert s3 == "ltd"

    # End-to-end extraction test
    s1_df = widen_records_df(
        pd.DataFrame(
            [
                {
                    "entity_id": "S1-1",
                    "business_name": "Custom Special LLC",
                    "business_address": None,
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
                    "business_name": "The Custom Special Inc",
                    "business_address": None,
                    "country": "US",
                },
                {
                    "entity_id": "S2-2",
                    "business_name": "Custom Special LLC",
                    "business_address": None,
                    "country": "US",
                },
            ]
        )
    )
    extractor = PairwiseFeatureExtractor()
    feat_df = extractor.extract_features_df(
        {"S1-1": {"S2-1", "S2-2"}}, s1_df, cand_df, show_progress=False
    )

    row_inc = feat_df[feat_df["cand_id"] == "S2-1"].iloc[0]
    row_llc = feat_df[feat_df["cand_id"] == "S2-2"].iloc[0]

    assert row_inc["name_core_ratio"] == 1.0
    assert row_inc["suffix_match"] == 0.0  # LLC vs Inc conflict
    assert row_llc["suffix_match"] == 1.0  # LLC vs LLC match


def test_transliteration_and_cross_script_features():
    """Test cross script detection and transliteration boost."""
    s1_df = widen_records_df(
        pd.DataFrame(
            [
                {
                    "entity_id": "S1-1",
                    "business_name": "Creative Logistics",
                    "business_address": None,
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
                    "business_name": "क्रिएटिव लॉजिस्टिक्स",
                    "business_address": None,
                    "country": "India",
                }
            ]
        )
    )
    extractor = PairwiseFeatureExtractor()
    feat_df = extractor.extract_features_df({"S1-1": {"S2-1"}}, s1_df, cand_df, show_progress=False)

    row = feat_df.iloc[0]
    assert row["is_cross_script"] == 1.0
    assert row["translit_name_ratio"] > 0.70
    assert row["translit_boost_name"] > 0.0


def test_phonetic_and_edit_features():
    """Test phonetic similarity and edit geometry features."""
    s1_df = widen_records_df(
        pd.DataFrame(
            [
                {
                    "entity_id": "S1-1",
                    "business_name": "Phonetic Dental Clinic",
                    "business_address": None,
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
                    "business_name": "Fonetic Dental Care",
                    "business_address": None,
                    "country": "US",
                }
            ]
        )
    )
    extractor = PairwiseFeatureExtractor()
    feat_df = extractor.extract_features_df({"S1-1": {"S2-1"}}, s1_df, cand_df, show_progress=False)

    row = feat_df.iloc[0]
    assert row["phonetic_name_match"] == 1.0
    assert row["common_prefix_len"] == 0.0  # P vs F
    assert row["name_edit_distance"] > 0


def test_entity_meta_features():
    """Test computation of 5 entity-level meta-features."""
    df = pd.DataFrame(
        {
            "s1_id": ["S1-1", "S1-1", "S1-1", "S1-2"],
            "cand_id": ["C1", "C2", "C3", "C4"],
            "prob": [0.90, 0.40, 0.20, 0.75],
        }
    )
    res = compute_entity_meta_features(df, prob_col="prob", s1_id_col="s1_id")

    for col in META_FEATURE_COLUMNS:
        assert col in res.columns

    s1_rows = res[res["s1_id"] == "S1-1"]
    assert pytest.approx(s1_rows["s1_max_score"].iloc[0], 0.01) == 0.90
    assert pytest.approx(s1_rows["score_gap_to_best"].iloc[0], 0.01) == 0.0  # Top cand has 0 gap
    assert pytest.approx(s1_rows["score_gap_to_best"].iloc[1], 0.01) == 0.50
    assert s1_rows["score_rank_pct"].iloc[0] < s1_rows["score_rank_pct"].iloc[1]
