"""Smoke test for the production training and streaming test inference pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from parallax.submission.test_inference import run_streaming_test_inference
from parallax.submission.validator import validate_official_er_submission
from parallax.training.full_trainer import train_production_pipeline


def test_full_pipeline_smoke(tmp_path: Path):
    train_dir = tmp_path / "raw_train"
    test_dir = tmp_path / "raw_test"
    output_dir = tmp_path / "output"
    submissions_dir = tmp_path / "submissions"

    train_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    submissions_dir.mkdir(parents=True)

    # 1. Create small training data
    s1_train = pd.DataFrame(
        {
            "entity_id": [f"S1-{i}" for i in range(10)],
            "business_name": [
                "Acme Corp",
                "Beta Logistics",
                "Omega Retail",
                "Delta Tech",
                "Zeta Coffee",
                "Alpha Dental",
                "Gamma Health",
                "Sigma Auto",
                "Theta Solar",
                "Kappa Books",
            ],
            "business_address": [
                "100 Market St",
                "200 Main Ave",
                "300 Broadway",
                "400 Pine Rd",
                "500 Oak St",
                "600 Elm Dr",
                "700 Cedar Ln",
                "800 Maple Way",
                "900 Birch Blvd",
                "1000 Walnut Ct",
            ],
            "country": ["US"] * 5 + ["IN"] * 5,
        }
    )
    s1_train.to_csv(train_dir / "train_source1.tsv", sep="\t", index=False)

    s2_train = pd.DataFrame(
        {
            "entity_id": ["S2-10", "S2-11", "S2-12", "S2-15", "S2-16"],
            "business_name": [
                "Acme Corporation",
                "Beta Logistics LLC",
                "Omega Supermarket",
                "Alpha Dental Care",
                "Gamma Clinic",
            ],
            "business_address": [
                "100 Market Street",
                "200 Main Avenue",
                "300 Broadway St",
                "600 Elm Drive",
                "700 Cedar Lane",
            ],
            "country": ["US"] * 3 + ["IN"] * 2,
        }
    )
    s2_train.to_csv(train_dir / "train_source2.tsv", sep="\t", index=False)

    s3_train = pd.DataFrame(
        {
            "entity_id": ["S3-20", "S3-23", "S3-24"],
            "business_name": ["Acme Inc", "Delta Technologies", "Zeta Cafe"],
            "business_address": ["100 Market", "400 Pine Road", "500 Oak"],
            "country": ["US"] * 3,
        }
    )
    s3_train.to_csv(train_dir / "train_source3.tsv", sep="\t", index=False)

    # Ground truth: S1-0 matches S2-10, S3-20; S1-1 matches S2-11; S1-3 matches S3-23;
    # S1-4 has no match (singleton)
    gt_lines = [
        "source1_entity_id\tmatched_entity_ids\tmatch_count\n",
        "S1-0\tS2-10,S3-20\t2\n",
        "S1-1\tS2-11\t1\n",
        "S1-2\tS2-12\t1\n",
        "S1-3\tS3-23\t1\n",
        "S1-4\t\t0\n",
        "S1-5\tS2-15\t1\n",
        "S1-6\tS2-16\t1\n",
        "S1-7\t\t0\n",
        "S1-8\t\t0\n",
        "S1-9\t\t0\n",
    ]
    (train_dir / "train_ground_truth.tsv").write_text("".join(gt_lines), encoding="utf-8")

    # 2. Train production pipeline on dummy data
    metadata = train_production_pipeline(
        train_dir=train_dir,
        output_dir=output_dir,
        holdout_size=3,
        seed=42,
    )
    assert metadata.optimal_tau > 0.0
    assert (output_dir / "models" / "production_pass1.txt").is_file()
    assert (output_dir / "models" / "production_pass2.txt").is_file()
    assert (output_dir / "models" / "production_metadata.json").is_file()

    # 3. Create dummy test data
    s1_test = pd.DataFrame(
        {
            "entity_id": ["S1-100", "S1-101", "S1-102"],
            "business_name": ["Acme Corp", "Random Unknown Store", "Alpha Dental"],
            "business_address": ["100 Market St", "999 Nowhere Rd", "600 Elm Dr"],
            "country": ["US", "US", "IN"],
        }
    )
    s1_test.to_csv(test_dir / "test_source1.tsv", sep="\t", index=False)

    s2_test = pd.DataFrame(
        {
            "entity_id": ["S2-200", "S2-205"],
            "business_name": ["Acme Global", "Alpha Dental Services"],
            "business_address": ["100 Market St", "600 Elm Drive"],
            "country": ["US", "IN"],
        }
    )
    s2_test.to_csv(test_dir / "test_source2.tsv", sep="\t", index=False)

    s3_test = pd.DataFrame(
        {
            "entity_id": ["S3-300"],
            "business_name": ["Acme Services"],
            "business_address": ["100 Market"],
            "country": ["US"],
        }
    )
    s3_test.to_csv(test_dir / "test_source3.tsv", sep="\t", index=False)

    # 4. Run test inference
    submission_tsv = submissions_dir / "matching_results.tsv"
    run_streaming_test_inference(
        test_dir=test_dir,
        models_dir=output_dir / "models",
        output_tsv_path=submission_tsv,
        chunk_size=2,
    )
    assert submission_tsv.is_file()

    # 5. Validate submission against official invariants
    is_valid, errors, warnings = validate_official_er_submission(
        matching_tsv_path=submission_tsv,
        test_dir=test_dir,
        check_ids=True,
    )
    assert is_valid is True
    assert len(errors) == 0

    # 6. Check row ordering and singleton formatting
    lines = submission_tsv.read_text(encoding="utf-8").strip().split("\n")
    assert lines[0] == "source1_entity_id\tmatched_entity_ids"
    assert len(lines) == 4  # Header + 3 entities
    assert lines[1].startswith("S1-100\t")
    assert lines[2].startswith("S1-101\t")
    assert lines[3].startswith("S1-102\t")
