"""
Parallax Submission Integrity Validator
=======================================
Strict verification gate for challenge submission files before uploading.
Catches schema mismatches, nulls, row count errors, and ID drift.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field


class ValidationReport(BaseModel):
    """Structured report of submission validity."""

    is_valid: bool = True
    submission_path: str
    total_rows: int = 0
    expected_rows: int | None = None
    columns: list[str] = Field(default_factory=list)
    null_count: int = 0
    empty_string_count: int = 0
    id_mismatches: int = 0
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class SubmissionValidator:
    """Performs deterministic validation on submission CSV files."""

    def __init__(
        self,
        id_col: str = "index",
        target_col: str = "prediction",
    ) -> None:
        self.id_col = id_col
        self.target_col = target_col

    def validate(
        self,
        submission_path: Path,
        sample_submission_path: Path | None = None,
        test_data_path: Path | None = None,
    ) -> ValidationReport:
        """Validate submission file against invariants and reference datasets."""
        report = ValidationReport(submission_path=str(submission_path))

        # 1. File existence
        if not submission_path.exists():
            report.is_valid = False
            report.errors.append(f"Submission file does not exist: {submission_path}")
            return report

        try:
            sub_df = pd.read_csv(submission_path)
        except Exception as exc:
            report.is_valid = False
            report.errors.append(f"Failed to parse CSV: {exc}")
            return report

        report.total_rows = len(sub_df)
        report.columns = list(sub_df.columns)

        # 2. Check essential columns
        if self.id_col not in sub_df.columns:
            report.is_valid = False
            report.errors.append(f"Missing required ID column: '{self.id_col}'")

        if self.target_col not in sub_df.columns:
            report.is_valid = False
            report.errors.append(f"Missing required target column: '{self.target_col}'")

        # 3. Check for NaNs and Nulls
        null_count = int(sub_df.isna().sum().sum())
        report.null_count = null_count
        if null_count > 0:
            report.is_valid = False
            report.errors.append(f"Submission contains {null_count} NaN/Null values!")

        # 4. Check for empty strings in target column
        if self.target_col in sub_df.columns:
            empty_count = int((sub_df[self.target_col].astype(str).str.strip() == "").sum())
            report.empty_string_count = empty_count
            if empty_count > 0:
                report.is_valid = False
                report.errors.append(
                    f"Submission contains {empty_count} blank/empty string predictions!"
                )

        # 5. Check against sample submission or test reference
        reference_df = None
        if sample_submission_path and sample_submission_path.exists():
            reference_df = pd.read_csv(sample_submission_path)
        elif test_data_path and test_data_path.exists():
            reference_df = pd.read_csv(test_data_path)

        if reference_df is not None:
            report.expected_rows = len(reference_df)

            # Row count check
            if len(sub_df) != len(reference_df):
                report.is_valid = False
                report.errors.append(
                    f"Row count mismatch! Expected {len(reference_df)}, got {len(sub_df)}"
                )

            # Column header check (if checking against sample_submission)
            if sample_submission_path and list(sub_df.columns) != list(reference_df.columns):
                report.is_valid = False
                expected_cols = list(reference_df.columns)
                actual_cols = list(sub_df.columns)
                report.errors.append(
                    f"Column names mismatch! Expected {expected_cols}, got {actual_cols}"
                )

            # ID alignment check
            if self.id_col in reference_df.columns and self.id_col in sub_df.columns:
                ref_ids = set(reference_df[self.id_col].astype(str))
                sub_ids = set(sub_df[self.id_col].astype(str))
                diff = ref_ids.symmetric_difference(sub_ids)
                report.id_mismatches = len(diff)
                if diff:
                    report.is_valid = False
                    report.errors.append(
                        f"ID set mismatch! {len(diff)} IDs differ from reference dataset."
                    )

        return report


def validate_official_er_submission(
    matching_tsv_path: Path | str,
    test_dir: Path | str,
    candidate_tsv_path: Path | str | None = None,
    check_ids: bool = True,
) -> tuple[bool, list[str], list[str]]:
    """
    Validate matching_results.tsv against official competition invariants.
    Returns (is_valid, errors, warnings).
    """
    from utils.validate_submission import validate

    cand_str = str(candidate_tsv_path) if candidate_tsv_path else None
    errors, warnings = validate(
        matching_path=str(matching_tsv_path),
        candidate_path=cand_str,
        test_dir=str(test_dir),
        check_ids=check_ids,
    )
    return len(errors) == 0, errors, warnings


def main() -> None:
    """CLI to validate a submission file."""
    parser = argparse.ArgumentParser(description="Parallax Submission Integrity Validator")
    parser.add_argument("submission", type=Path, help="Path to submission CSV")
    parser.add_argument("--sample-sub", "-s", type=Path, default=None, help="Sample submission CSV")
    parser.add_argument("--test-data", "-t", type=Path, default=None, help="Test CSV")
    parser.add_argument("--id-col", default="index", help="Index ID column")
    parser.add_argument("--target-col", default="prediction", help="Target column")
    args = parser.parse_args()

    validator = SubmissionValidator(id_col=args.id_col, target_col=args.target_col)
    report = validator.validate(
        submission_path=args.submission,
        sample_submission_path=args.sample_sub,
        test_data_path=args.test_data,
    )

    print("\n" + "=" * 60)
    print(f"  SUBMISSION INTEGRITY REPORT: {args.submission.name}")
    print("=" * 60)
    print(f"  Total Rows:           {report.total_rows}")
    print(f"  Expected Rows:        {report.expected_rows or 'N/A'}")
    print(f"  Columns:              {report.columns}")
    print(f"  Null / NaN Count:     {report.null_count}")
    print(f"  Empty String Count:   {report.empty_string_count}")
    print(f"  ID Mismatches:        {report.id_mismatches}")
    print("-" * 60)

    if report.is_valid:
        print("  🎉 VERDICT: PASS - File is safe and compliant for leaderboard submission!")
        print("=" * 60)
        raise SystemExit(0)
    else:
        print("  ❌ VERDICT: BLOCKED - Fix the following errors before submitting:")
        for err in report.errors:
            print(f"     • {err}")
        print("=" * 60)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
