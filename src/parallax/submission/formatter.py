"""
Parallax Submission Formatter
=============================
Standardized submission generation pipeline with automated integrity verification.
"""

from __future__ import annotations

import datetime
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from parallax.config import get_config
from parallax.submission.validator import SubmissionValidator, ValidationReport


def format_and_save_submission(
    ids: Sequence[Any],
    predictions: Sequence[Any],
    output_path: Path | None = None,
    id_col: str = "index",
    target_col: str = "prediction",
    sample_sub_path: Path | None = None,
    default_fallback: str = "",
) -> tuple[Path, ValidationReport]:
    """
    Format predictions into compliant submission DataFrame, write to CSV,
    and validate integrity.
    """
    cfg = get_config()
    if output_path is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = cfg.paths.submissions_dir / f"submission_{timestamp}.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Clean and sanitize predictions
    cleaned_preds = []
    for pred in predictions:
        val = str(pred).strip() if pred is not None else ""
        if not val or val.lower() == "nan":
            val = default_fallback
        cleaned_preds.append(val)

    sub_df = pd.DataFrame(
        {
            id_col: ids,
            target_col: cleaned_preds,
        }
    )

    sub_df.to_csv(output_path, index=False)

    validator = SubmissionValidator(id_col=id_col, target_col=target_col)
    report = validator.validate(
        submission_path=output_path,
        sample_submission_path=sample_sub_path,
    )

    return output_path, report
