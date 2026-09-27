"""
Identity, population and ground-truth contracts for the hybrid ER pipeline.

Every check here fails loudly (ValueError). Nothing is silently repaired or dropped:
an invalid candidate identity indicates a pipeline bug, and hiding it would corrupt
both candidate-recall accounting and the matcher metric.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

ALLOWED_SOURCES = ("S2", "S3")
CANONICAL_KEY = ("s1_id", "source", "cand_id")


def validate_candidate_schema(df: pd.DataFrame, unique_on: Sequence[str] = CANONICAL_KEY) -> None:
    """
    Asserts that the candidate DataFrame contains the mandatory canonical identity columns,
    validates types, checks for nulls/empty strings, enforces allowed sources, and ensures
    uniqueness of `unique_on` (the canonical tuple by default).

    Identity columns are cast to str in place.
    """
    if df is None or df.empty:
        return

    required_cols = set(CANONICAL_KEY) | set(unique_on)
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Candidate DataFrame is missing required columns: {missing}")

    for col in CANONICAL_KEY:
        if df[col].isnull().any():
            raise ValueError(f"Candidate DataFrame contains null values in canonical column: {col}")

    # Normalize to string before validation
    for col in CANONICAL_KEY:
        df[col] = df[col].astype(str)

    for col in CANONICAL_KEY:
        if (df[col].str.strip() == "").any():
            raise ValueError(f"Candidate DataFrame contains empty strings in canonical column: {col}")

    invalid_sources = df[~df["source"].isin(ALLOWED_SOURCES)]
    if not invalid_sources.empty:
        raise ValueError(
            f"Invalid source found in candidates. Allowed: 'S2', 'S3'. "
            f"Found: {invalid_sources['source'].unique()}"
        )

    dups = df.duplicated(subset=list(unique_on))
    if dups.any():
        raise ValueError(
            f"Candidate DataFrame contains duplicate canonical tuples {tuple(unique_on)}. "
            f"Found {int(dups.sum())} duplicates."
        )


def validate_source_contract(df: pd.DataFrame, s2_ids: set[str], s3_ids: set[str]) -> None:
    """
    Asserts that all 'S2' sources map to valid S2 IDs, and 'S3' to valid S3 IDs.
    """
    if df is None or df.empty:
        return

    cand = df["cand_id"].astype(str)
    s2_mask = df["source"] == "S2"
    s3_mask = df["source"] == "S3"

    invalid_s2 = s2_mask & ~cand.isin(s2_ids)
    if invalid_s2.any():
        raise ValueError(
            f"Found {int(invalid_s2.sum())} candidates labeled 'S2' whose cand_id is not in S2 ID population."
        )

    invalid_s3 = s3_mask & ~cand.isin(s3_ids)
    if invalid_s3.any():
        raise ValueError(
            f"Found {int(invalid_s3.sum())} candidates labeled 'S3' whose cand_id is not in S3 ID population."
        )

    invalid_other = ~(s2_mask | s3_mask)
    if invalid_other.any():
        raise ValueError(f"Found {int(invalid_other.sum())} candidates with source other than S2 or S3.")


def assign_source_by_membership(df: pd.DataFrame, s2_ids: set[str], s3_ids: set[str]) -> pd.DataFrame:
    """
    Returns a copy of `df` with an authoritative `source` column derived from population
    membership (never from ID prefixes). Raises if any cand_id belongs to neither or both
    populations.
    """
    out = df.copy()
    cand = out["cand_id"].astype(str)
    in_s2 = cand.isin(s2_ids).to_numpy()
    in_s3 = cand.isin(s3_ids).to_numpy()

    ambiguous = in_s2 & in_s3
    if ambiguous.any():
        raise ValueError(f"{int(ambiguous.sum())} cand_ids belong to both S2 and S3 populations.")
    unknown = ~(in_s2 | in_s3)
    if unknown.any():
        examples = cand[unknown].head(5).tolist()
        raise ValueError(
            f"{int(unknown.sum())} cand_ids belong to neither S2 nor S3 population. Examples: {examples}"
        )

    out["cand_id"] = cand
    out["source"] = np.where(in_s2, "S2", "S3")
    return out


def validate_evaluation_population(df: pd.DataFrame, eval_s1_ids: Sequence[str]) -> None:
    """
    Asserts that the input DataFrame covers exactly the required evaluation population.
    """
    if df is None or df.empty:
        raise ValueError("DataFrame is empty but expected evaluation population.")

    id_col = "s1_id" if "s1_id" in df.columns else "entity_id"
    present_ids = set(df[id_col].astype(str).unique())
    required_ids = {str(x) for x in eval_s1_ids}

    missing = required_ids - present_ids
    extra = present_ids - required_ids

    if missing:
        raise ValueError(
            f"Evaluation population validation failed: Missing {len(missing)} S1 entities. "
            f"Example: {list(missing)[:5]}"
        )
    if extra:
        raise ValueError(
            f"Evaluation population validation failed: Found {len(extra)} extra S1 entities not in evaluation set."
        )


def validate_prediction_population(predictions: Mapping[str, set[str]], eval_s1_ids: Sequence[str]) -> None:
    """
    Asserts that prediction S1 IDs == evaluation S1 IDs precisely.
    Zero-match S1s must have an empty set.
    """
    pred_keys = {str(k) for k in predictions.keys()}
    required_ids = {str(x) for x in eval_s1_ids}

    missing = required_ids - pred_keys
    extra = pred_keys - required_ids

    if missing:
        raise ValueError(
            f"Prediction population validation failed: Missing {len(missing)} S1 entities. "
            f"Zero-match S1s must have an empty set."
        )
    if extra:
        raise ValueError(
            f"Prediction population validation failed: Found {len(extra)} extra S1 entities not in evaluation set."
        )


def validate_prediction_targets(
    predictions: Mapping[str, Iterable[str]], s2_ids: set[str], s3_ids: set[str]
) -> None:
    """
    Asserts every predicted target is a real S2/S3 entity (never an S1 ID, never unknown).
    """
    for s1, targets in predictions.items():
        for t in targets:
            t = str(t)
            if t not in s2_ids and t not in s3_ids:
                raise ValueError(f"Prediction for S1 {s1} contains target {t} that is not in S2 or S3 populations.")


def validate_predictions_subset_of_candidates(
    predictions: Mapping[str, Iterable[str]], candidates: Mapping[str, Iterable[str]]
) -> None:
    """
    Asserts final predictions ⊆ final candidates for every S1. A prediction outside the
    candidate set means the matcher saw records that candidate_pairs.tsv does not declare.
    """
    violations = 0
    example = None
    for s1, targets in predictions.items():
        cand_set = set(candidates.get(str(s1), ()))
        outside = {str(t) for t in targets} - cand_set
        if outside:
            violations += len(outside)
            if example is None:
                example = (s1, sorted(outside)[:3])
    if violations:
        raise ValueError(
            f"{violations} predicted pairs are not in the candidate set (predictions must be ⊆ candidates). "
            f"Example: {example}"
        )


def validate_gt_global_contract(
    gt: Mapping[str, set[str]],
    s2_ids: set[str],
    s3_ids: set[str],
    s1_ids: set[str] | None = None,
) -> None:
    """
    Validates the *global* ground truth of a split (not the evaluation subset):

    - every target is an S2 or S3 entity,
    - no target is an S1 ID,
    - if `s1_ids` is given, GT keys equal the split's S1 population exactly
      (the GT file lists every S1, zero-match entities included).

    Never compares the global GT to an evaluation subset; use scope_ground_truth_to_eval for that.
    """
    if gt is None:
        raise ValueError("Ground truth is None.")

    if s1_ids is not None:
        gt_keys = {str(k) for k in gt.keys()}
        unknown = gt_keys - s1_ids
        if unknown:
            raise ValueError(f"GT contract violation: {len(unknown)} GT keys are not in the S1 population.")
        uncovered = s1_ids - gt_keys
        if uncovered:
            raise ValueError(f"GT contract violation: {len(uncovered)} S1 entities have no GT row.")

    for s1, targets in gt.items():
        for t in targets:
            t = str(t)
            if s1_ids is not None and t in s1_ids:
                raise ValueError(f"GT contract violation: target ID {t} for S1 {s1} is an S1 ID.")
            if t not in s2_ids and t not in s3_ids:
                raise ValueError(f"GT contract violation: target ID {t} for S1 {s1} is not in S2 or S3 populations.")


def scope_ground_truth_to_eval(gt: Mapping[str, set[str]], eval_s1_ids: Sequence[str]) -> dict[str, set[str]]:
    """
    Extracts the subset of GT matching eval_s1_ids exactly.
    Raises ValueError if any eval S1 is missing from the global GT (a missing S1 must never
    be silently treated as a zero-match entity).
    """
    scoped_gt: dict[str, set[str]] = {}
    for s1 in eval_s1_ids:
        s1_str = str(s1)
        if s1_str not in gt:
            raise ValueError(f"Evaluation S1 ID {s1_str} missing from global ground truth.")
        scoped_gt[s1_str] = {str(x) for x in gt[s1_str]}

    if set(scoped_gt.keys()) != {str(x) for x in eval_s1_ids}:
        raise ValueError("Scoped GT keys do not exactly match eval_s1_ids.")

    return scoped_gt
