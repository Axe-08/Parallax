"""
Unit tests for multi-process parallelization across Steps 2, 3, and 4.
Verifies exact data equivalence between serial (n_jobs=1) and multi-process (n_jobs=2)
implementations.
"""

from __future__ import annotations

import pandas as pd
import pytest

from parallax.blocking.tfidf_blocker import DualChannelTFIDFBlocker
from parallax.features.extractor import PairwiseFeatureExtractor
from parallax.preprocessing.normalizer import widen_records_df


@pytest.fixture
def sample_business_df() -> pd.DataFrame:
    """Generate 2,500 sample records to trigger multi-process chunking (threshold >= 2000)."""
    rows = []
    base_names = [
        "Acme Industrial Solutions LLC",
        "Apex Global Technologies Inc",
        "Blue Horizon Enterprises Ltd",
        "Metro Logistics & Distribution",
        "Continental Elm Center",
        "ओम शांति एंटरप्राइजेज",
        "Reliance Digital Services Pvt Ltd",
        "Zenith Medical Clinic",
    ]
    base_addrs = [
        "123 Main Street, Suite 400, New York, NY 10001",
        "456 Elm Boulevard, Floor 2, Austin, TX 78701",
        "789 Industrial Pkwy, Bldg C, Chicago, IL 60601",
        "100 MG Road, Indiranagar, Bengaluru, Karnataka 560038",
        "200 Connaught Place, Block B, New Delhi, Delhi 110001",
        "Plot 12, Sector 18, Gurugram, Haryana 122015",
        "500 Market St, San Francisco, CA 94105",
        None,
    ]
    for i in range(2500):
        name = base_names[i % len(base_names)] + f" #{i}"
        addr = base_addrs[i % len(base_addrs)]
        if addr:
            addr = addr + f" Room {i}"
        country = "US" if i % 2 == 0 else "India"
        rows.append({
            "entity_id": f"ENT-{i:05d}",
            "business_name": name,
            "business_address": addr,
            "country": country,
        })
    return pd.DataFrame(rows)


def test_widen_records_parallel_equivalence(sample_business_df: pd.DataFrame) -> None:
    """Verify widen_records_df(n_jobs=2) yields identical output to n_jobs=1."""
    serial_wide = widen_records_df(sample_business_df, n_jobs=1)
    parallel_wide = widen_records_df(sample_business_df, n_jobs=2)

    pd.testing.assert_frame_equal(serial_wide, parallel_wide)
    assert len(parallel_wide) == 2500
    assert "soft_name" in parallel_wide.columns
    assert "clean_address" in parallel_wide.columns
    assert "primary_number" in parallel_wide.columns
    assert "city_token" in parallel_wide.columns


def test_tfidf_blocker_parallel_equivalence(sample_business_df: pd.DataFrame) -> None:
    """Verify DualChannelTFIDFBlocker(n_jobs=2) produces candidate matches identical to n_jobs=1."""
    wide_df = widen_records_df(sample_business_df, n_jobs=1)
    s1_df = wide_df.iloc[:2000].reset_index(drop=True)
    target_df = wide_df.iloc[1500:].reset_index(drop=True)

    blocker_serial = DualChannelTFIDFBlocker(
        name_top_k=5,
        addr_top_k=5,
        name_min_sim=0.10,
        show_progress=False,
        max_candidates_per_query=10,
        n_jobs=1,
    )
    cands_serial = blocker_serial.generate_candidates(s1_df, target_df, n_jobs=1)

    blocker_parallel = DualChannelTFIDFBlocker(
        name_top_k=5,
        addr_top_k=5,
        name_min_sim=0.10,
        show_progress=False,
        max_candidates_per_query=10,
        n_jobs=2,
    )
    cands_parallel = blocker_parallel.generate_candidates(s1_df, target_df, n_jobs=2)

    assert set(cands_serial.keys()) == set(cands_parallel.keys())
    for s1_id in cands_serial:
        assert set(cands_serial[s1_id].keys()) == set(cands_parallel[s1_id].keys())
        for cand_id in cands_serial[s1_id]:
            expected_score = cands_serial[s1_id][cand_id]
            assert pytest.approx(expected_score, rel=1e-4) == cands_parallel[s1_id][cand_id]


def test_feature_extractor_parallel_equivalence(sample_business_df: pd.DataFrame) -> None:
    """Verify PairwiseFeatureExtractor(n_jobs=2) produces identical features to n_jobs=1."""
    wide_df = widen_records_df(sample_business_df, n_jobs=1)
    s1_df = wide_df.iloc[:1200].reset_index(drop=True)
    target_df = wide_df.iloc[1000:].reset_index(drop=True)

    # Generate synthetic candidates mapping
    candidate_pairs: dict[str, dict[str, float]] = {}
    for i in range(1200):
        s1_id = f"ENT-{i:05d}"
        cand_id = f"ENT-{(1000 + (i % 500)):05d}"
        candidate_pairs[s1_id] = {cand_id: 0.85}

    extractor = PairwiseFeatureExtractor()
    feat_serial = extractor.extract_features_df(
        candidate_pairs, s1_df, target_df, show_progress=False, n_jobs=1
    )
    feat_parallel = extractor.extract_features_df(
        candidate_pairs, s1_df, target_df, show_progress=False, n_jobs=2
    )

    pd.testing.assert_frame_equal(feat_serial, feat_parallel)
    assert len(feat_parallel) == 1200
    assert "raw_name_ratio" in feat_parallel.columns
    assert "blocking_sim_score" in feat_parallel.columns
