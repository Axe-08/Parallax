"""
Parallax Metric Evaluator Module
================================
Official competition metrics engine for Amazon ML Challenge 2026:
- Exact Macro F0.5 per Source 1 entity with strict singleton scoring (1.0 vs 0.0)
- Blocking diagnostic metrics: Pair Completeness (Recall), Reduction Ratio, Pairs Quality.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from typing import Any

import numpy as np
from pydantic import BaseModel, Field


class EvaluationReport(BaseModel):
    """Structured report of Macro F0.5 resolution performance."""

    macro_f05: float = Field(..., description="Primary competition leaderboard metric: Macro F0.5")
    singleton_score: float = Field(
        ..., description="Average score on singletons (should be close to 1.0)"
    )
    non_singleton_f05: float = Field(..., description="Average F0.5 on non-singletons")
    total_entities: int = Field(..., description="Total Source 1 entities evaluated")
    total_singletons: int = Field(..., description="Total ground truth singletons")
    total_true_pairs: int = Field(..., description="Total ground truth match pairs")
    total_predicted_pairs: int = Field(..., description="Total predicted match pairs")
    total_correct_pairs: int = Field(..., description="Total true positive pairs")


class BlockingReport(BaseModel):
    """Diagnostic report evaluating the candidate generation / blocking stage."""

    pair_completeness: float = Field(
        ..., description="Blocking recall ceiling (captured / total true matches)"
    )
    reduction_ratio: float = Field(..., description="Percentage of comparison space pruned")
    pairs_quality: float = Field(
        ..., description="Blocking precision (true matches / total candidates)"
    )
    total_candidates: int = Field(..., description="Total candidate pairs generated")
    avg_candidates_per_s1: float = Field(
        ..., description="Average candidate pool size per Source 1 entity"
    )
    max_candidates_per_s1: int = Field(
        ..., description="Maximum candidates assigned to any single S1"
    )


def calculate_entity_f05(true_matches: set[str], pred_matches: set[str]) -> float:
    """
    Compute F0.5 for a single Source 1 entity.
    - True singleton: 1.0 if predicted empty, 0.0 if any match predicted.
    - Non-singleton: 0.0 if predicted empty, else (1.25 * P * R) / (0.25 * P + R).
    """
    if len(true_matches) == 0:
        return 1.0 if len(pred_matches) == 0 else 0.0

    if len(pred_matches) == 0:
        return 0.0

    tp = len(true_matches.intersection(pred_matches))
    if tp == 0:
        return 0.0

    precision = tp / len(pred_matches)
    recall = tp / len(true_matches)
    denominator = (0.25 * precision) + recall
    if denominator <= 0.0:
        return 0.0

    return (1.25 * precision * recall) / denominator


def evaluate_resolution_predictions(
    ground_truth: Mapping[str, set[str]],
    predictions: Mapping[str, set[str]],
) -> EvaluationReport:
    """
    Evaluate predicted entity matches against ground truth labels across all S1 entities.
    """
    scores: list[float] = []
    singleton_scores: list[float] = []
    non_singleton_scores: list[float] = []

    total_true_pairs = 0
    total_pred_pairs = 0
    total_correct_pairs = 0

    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())

        total_true_pairs += len(true_set)
        total_pred_pairs += len(pred_set)
        total_correct_pairs += len(true_set.intersection(pred_set))

        score = calculate_entity_f05(true_set, pred_set)
        scores.append(score)

        if len(true_set) == 0:
            singleton_scores.append(score)
        else:
            non_singleton_scores.append(score)

    return EvaluationReport(
        macro_f05=float(np.mean(scores)) if scores else 0.0,
        singleton_score=float(np.mean(singleton_scores)) if singleton_scores else 1.0,
        non_singleton_f05=float(np.mean(non_singleton_scores)) if non_singleton_scores else 0.0,
        total_entities=len(scores),
        total_singletons=len(singleton_scores),
        total_true_pairs=total_true_pairs,
        total_predicted_pairs=total_pred_pairs,
        total_correct_pairs=total_correct_pairs,
    )


def evaluate_blocking_candidates(
    ground_truth: Mapping[str, set[str]],
    candidates: Mapping[str, Collection[str]],
    total_target_records: int,
) -> BlockingReport:
    """
    Evaluate blocking stage candidate generation:
    - Pair Completeness (PC): captured true matches / total true matches
    - Reduction Ratio (RR): 1 - (total_candidates / total_possible_comparisons)
    - Pairs Quality (PQ): captured true matches / total_candidates
    """
    total_s1 = len(ground_truth)
    total_true_pairs = sum(len(v) for v in ground_truth.values())
    total_candidates = sum(len(v) for v in candidates.values())

    captured_true_pairs = 0
    cand_counts: list[int] = []

    for s1_id, true_set in ground_truth.items():
        cands = set(candidates.get(s1_id, ()))
        cand_counts.append(len(cands))
        captured_true_pairs += len(true_set.intersection(cands))

    pc = (captured_true_pairs / total_true_pairs) if total_true_pairs > 0 else 1.0
    total_possible = total_s1 * total_target_records
    rr = (1.0 - (total_candidates / total_possible)) if total_possible > 0 else 1.0
    pq = (captured_true_pairs / total_candidates) if total_candidates > 0 else 0.0

    return BlockingReport(
        pair_completeness=float(pc),
        reduction_ratio=float(rr),
        pairs_quality=float(pq),
        total_candidates=total_candidates,
        avg_candidates_per_s1=float(np.mean(cand_counts)) if cand_counts else 0.0,
        max_candidates_per_s1=int(max(cand_counts)) if cand_counts else 0,
    )


# --- Legacy Metric Helpers for Backwards Compatibility ---


def calculate_f1_score(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    average: str = "macro",
) -> float:
    """Calculate F1 score across classes (legacy)."""
    from sklearn.metrics import f1_score

    y_true_str = [str(x).strip() for x in y_true]
    y_pred_str = [str(x).strip() for x in y_pred]
    return float(f1_score(y_true_str, y_pred_str, average=average, zero_division=0.0))


def calculate_smape(
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
    epsilon: float = 1e-8,
) -> float:
    """Calculate Symmetric Mean Absolute Percentage Error (legacy)."""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    denominator = (np.abs(yt) + np.abs(yp)) / 2.0 + epsilon
    diff = np.abs(yp - yt)
    return float(np.mean(diff / denominator) * 100.0)


def calculate_exact_match(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
) -> float:
    """Calculate exact match ratio (legacy)."""
    from sklearn.metrics import accuracy_score

    y_true_norm = [str(x).strip().lower() for x in y_true]
    y_pred_norm = [str(x).strip().lower() for x in y_pred]
    return float(accuracy_score(y_true_norm, y_pred_norm))


def evaluate_predictions(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
) -> dict[str, float]:
    """Evaluate multi-metric dictionary (legacy)."""
    res = {
        "f1_macro": calculate_f1_score(y_true, y_pred, average="macro"),
        "exact_match": calculate_exact_match(y_true, y_pred),
    }
    try:
        y_true_float = [float(x) for x in y_true]
        y_pred_float = [float(x) for x in y_pred]
        res["smape"] = calculate_smape(y_true_float, y_pred_float)
    except (ValueError, TypeError):
        pass
    return res
