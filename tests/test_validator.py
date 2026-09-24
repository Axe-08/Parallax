"""Unit tests for submission integrity validator."""

from pathlib import Path

import pandas as pd

from parallax.submission.formatter import format_and_save_submission
from parallax.submission.validator import SubmissionValidator


def test_submission_validator_valid(tmp_path: Path):
    sub_file = tmp_path / "valid_submission.csv"
    sample_sub_file = tmp_path / "sample_submission.csv"

    data = pd.DataFrame({"index": [1, 2, 3], "prediction": ["10 g", "20 kg", "30 cm"]})
    data.to_csv(sub_file, index=False)
    data.to_csv(sample_sub_file, index=False)

    validator = SubmissionValidator(id_col="index", target_col="prediction")
    report = validator.validate(sub_file, sample_submission_path=sample_sub_file)

    assert report.is_valid is True
    assert report.total_rows == 3
    assert report.null_count == 0
    assert report.id_mismatches == 0
    assert len(report.errors) == 0


def test_submission_validator_detects_nans_and_mismatches(tmp_path: Path):
    sub_file = tmp_path / "bad_submission.csv"
    sample_sub_file = tmp_path / "sample_submission.csv"

    # sub_file has a NaN and row count mismatch
    sub_data = pd.DataFrame({"index": [1, 2], "prediction": ["10 g", None]})
    sub_data.to_csv(sub_file, index=False)

    sample_data = pd.DataFrame({"index": [1, 2, 3], "prediction": ["10 g", "20 kg", "30 cm"]})
    sample_data.to_csv(sample_sub_file, index=False)

    validator = SubmissionValidator(id_col="index", target_col="prediction")
    report = validator.validate(sub_file, sample_submission_path=sample_sub_file)

    assert report.is_valid is False
    assert report.null_count == 1
    assert report.expected_rows == 3
    assert report.total_rows == 2
    assert any("Row count mismatch" in err for err in report.errors)
    assert any("NaN/Null" in err for err in report.errors)


def test_format_and_save_submission(tmp_path: Path):
    out_file = tmp_path / "formatted_sub.csv"
    ids = [101, 102, 103]
    preds = ["500 gram", "10 metre", None]

    path, report = format_and_save_submission(
        ids=ids,
        predictions=preds,
        output_path=out_file,
        default_fallback="fallback_value",
    )

    assert path.exists()
    assert report.is_valid is True
    saved_df = pd.read_csv(path)
    assert len(saved_df) == 3
    assert saved_df["prediction"].iloc[2] == "fallback_value"
