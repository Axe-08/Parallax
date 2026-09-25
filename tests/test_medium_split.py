"""Unit tests for the medium split generator logic."""

import json

# Import the core function from notebooks/colab_build_medium_split.py
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "notebooks"))
from colab_build_medium_split import build_medium_split


@pytest.fixture
def mock_raw_data(tmp_path: Path) -> Path:
    """Create a synthetic mini raw dataset mimicking official S1, S2, S3, and GT."""
    raw_dir = tmp_path / "raw_train"
    raw_dir.mkdir()

    # 100 S1 entities (60 US, 40 India)
    s1_rows = []
    for i in range(100):
        country = "US" if i < 60 else "India"
        s1_rows.append(
            {
                "entity_id": f"S1-{i:05d}",
                "business_name": f"Business Name {i}",
                "business_address": f"{i} Main St, State",
                "country": country,
            }
        )
    pd.DataFrame(s1_rows).to_csv(raw_dir / "train_source1.tsv", sep="\t", index=False)

    # Ground truth: 80 matching S1s (with S2 and S3), 20 singletons (no matches)
    gt_rows = []
    for i in range(100):
        s1_id = f"S1-{i:05d}"
        if i < 80:
            mids = f"S2-{i:05d},S3-{i:05d}"
            cnt = 2
        else:
            mids = ""
            cnt = 0
        gt_rows.append(
            {
                "source1_entity_id": s1_id,
                "matched_entity_ids": mids,
                "match_count": cnt,
            }
        )
    pd.DataFrame(gt_rows).to_csv(raw_dir / "train_ground_truth.tsv", sep="\t", index=False)

    # S2 entities: 80 true positives + 120 distractors = 200 total
    s2_rows = []
    for i in range(200):
        country = "US" if i % 2 == 0 else "India"
        s2_rows.append(
            {
                "entity_id": f"S2-{i:05d}",
                "business_name": f"Candidate S2 {i}",
                "business_address": f"{i} Candidate Ave",
                "country": country,
            }
        )
    pd.DataFrame(s2_rows).to_csv(raw_dir / "train_source2.tsv", sep="\t", index=False)

    # S3 entities: 80 true positives + 120 distractors = 200 total
    s3_rows = []
    for i in range(200):
        country = "US" if i % 2 == 0 else "India"
        s3_rows.append(
            {
                "entity_id": f"S3-{i:05d}",
                "business_name": f"Candidate S3 {i}",
                "business_address": f"{i} Candidate Blvd",
                "country": country,
            }
        )
    pd.DataFrame(s3_rows).to_csv(raw_dir / "train_source3.tsv", sep="\t", index=False)

    return raw_dir


def test_build_medium_split_sampling_and_integrity(mock_raw_data: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "medium_split"

    meta = build_medium_split(
        raw_dir=mock_raw_data,
        output_dir=output_dir,
        s1_target_size=40,  # Sample 40 out of 100 S1 entities
        s2_target_size=80,  # Target 80 S2 entities
        s3_target_size=80,  # Target 80 S3 entities
        n_splits=5,
        seed=42,
    )
    assert meta["entity_counts"]["source1"] == 40

    # 1. Output files exist
    assert (output_dir / "train_source1.tsv").exists()
    assert (output_dir / "train_source2.tsv").exists()
    assert (output_dir / "train_source3.tsv").exists()
    assert (output_dir / "train_ground_truth.tsv").exists()
    assert (output_dir / "cv_folds_source1.tsv").exists()
    assert (output_dir / "split_metadata.json").exists()

    # 2. Check S1 sample size and country stratification
    s1_out = pd.read_csv(output_dir / "train_source1.tsv", sep="\t")
    assert len(s1_out) == 40
    # Expected ~60% US and ~40% India
    us_count = (s1_out["country"] == "US").sum()
    india_count = (s1_out["country"] == "India").sum()
    assert 20 <= us_count <= 28
    assert 12 <= india_count <= 20

    # 3. Check Ground Truth matches S1
    gt_out = pd.read_csv(output_dir / "train_ground_truth.tsv", sep="\t")
    assert len(gt_out) == 40
    assert set(gt_out["source1_entity_id"]) == set(s1_out["entity_id"])

    # 4. Check 100% preservation of required positive matches in S2 and S3
    s2_out = pd.read_csv(output_dir / "train_source2.tsv", sep="\t")
    s3_out = pd.read_csv(output_dir / "train_source3.tsv", sep="\t")
    assert len(s2_out) == 80
    assert len(s3_out) == 80

    for mids in gt_out["matched_entity_ids"].dropna():
        if not mids or mids == "nan":
            continue
        for mid in mids.split(","):
            mid = mid.strip()
            if mid.startswith("S2-"):
                assert mid in set(s2_out["entity_id"]), f"Dangling S2 match: {mid}"
            elif mid.startswith("S3-"):
                assert mid in set(s3_out["entity_id"]), f"Dangling S3 match: {mid}"

    # 5. Check 5-Fold CV
    cv_out = pd.read_csv(output_dir / "cv_folds_source1.tsv", sep="\t")
    assert len(cv_out) == 40
    assert set(cv_out["fold"].unique()) == {0, 1, 2, 3, 4}
    for fold in range(5):
        assert (cv_out["fold"] == fold).sum() == 8

    # 6. Check metadata
    with open(output_dir / "split_metadata.json") as f:
        saved_meta = json.load(f)
    assert saved_meta["entity_counts"]["source1"] == 40
    assert saved_meta["entity_counts"]["source2"] == 80
    assert saved_meta["entity_counts"]["source3"] == 80
