"""
Parallax Metric Evaluator Module
================================
Unified evaluation metrics suite: F1, SMAPE, Exact Match,
and entity-unit consistency scoring.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def calculate_f1_score(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    average: str = "macro",
) -> float:
    """Calculate F1 score across classes."""
    y_true_str = [str(x).strip() for x in y_true]
    y_pred_str = [str(x).strip() for x in y_pred]
    return float(f1_score(y_true_str, y_pred_str, average=average, zero_division=0.0))


def calculate_smape(
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
    epsilon: float = 1e-8,
) -> float:
    """
    Calculate Symmetric Mean Absolute Percentage Error (SMAPE).
    Formula: 100 * mean( 2 * |y_true - y_pred| / (|y_true| + |y_pred| + eps) )
    Range: [0, 200%]
    """
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    denominator = (np.abs(yt) + np.abs(yp)) / 2.0 + epsilon
    diff = np.abs(yp - yt)
    return float(np.mean(diff / denominator) * 100.0)


def calculate_exact_match(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
) -> float:
    """Calculate exact match ratio."""
    y_true_norm = [str(x).strip().lower() for x in y_true]
    y_pred_norm = [str(x).strip().lower() for x in y_pred]
    return float(accuracy_score(y_true_norm, y_pred_norm))


def evaluate_predictions(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
) -> dict[str, float]:
    """Compute comprehensive diagnostic metrics between ground truth and predictions."""
    metrics: dict[str, float] = {
        "exact_match": calculate_exact_match(y_true, y_pred),
        "f1_macro": calculate_f1_score(y_true, y_pred, average="macro"),
        "f1_micro": calculate_f1_score(y_true, y_pred, average="micro"),
        "f1_weighted": calculate_f1_score(y_true, y_pred, average="weighted"),
    }

    # If targets are numeric or can be coerced to float, compute SMAPE
    try:
        y_true_float = [float(x) for x in y_true]
        y_pred_float = [float(x) for x in y_pred]
        metrics["smape"] = calculate_smape(y_true_float, y_pred_float)
    except (ValueError, TypeError):
        pass

    return metrics


def main() -> None:
    """CLI to evaluate a prediction CSV against a ground truth CSV."""
    parser = argparse.ArgumentParser(description="Parallax Local Metric Evaluator")
    parser.add_argument("--ground-truth", "-g", required=True, help="Path to ground truth CSV")
    parser.add_argument("--predictions", "-p", required=True, help="Path to predictions CSV")
    parser.add_argument("--target-col", "-t", default="prediction", help="Target column name")
    parser.add_argument("--id-col", default="index", help="Index ID column")
    args = parser.parse_args()

    gt_df = pd.read_csv(args.ground_truth)
    pred_df = pd.read_csv(args.predictions)

    # Align on ID
    merged = pd.merge(gt_df, pred_df, on=args.id_col, suffixes=("_true", "_pred"))
    results = evaluate_predictions(
        y_true=merged[f"{args.target_col}_true"],
        y_pred=merged[f"{args.target_col}_pred"],
    )

    print("\n📊 Evaluation Results:")
    for metric_name, val in results.items():
        print(f"   {metric_name:<16}: {val:.4f}")


if __name__ == "__main__":
    main()
