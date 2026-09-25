"""
Parallax LightGBM Pairwise Matcher & Threshold Calibrator
=========================================================
Asymmetric gradient-boosted decision tree classifier tuned for Macro F0.5
with automated decision threshold calibration.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from parallax.features.extractor import FEATURE_COLUMNS
from parallax.metrics.evaluator import evaluate_resolution_predictions


class LightGBMMatcher:
    """Pairwise match classifier using LightGBM."""

    def __init__(
        self,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        max_depth: int = 6,
        n_estimators: int = 150,
        decision_threshold: float = 0.75,
        seed: int = 42,
    ) -> None:
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.decision_threshold = decision_threshold
        self.seed = seed
        self.model: lgb.Booster | None = None

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame | None = None,
    ) -> None:
        """Train LightGBM binary classifier on candidate pair features."""
        x_train = train_df[FEATURE_COLUMNS]
        y_train = train_df["target"].astype(int)

        train_data = lgb.Dataset(x_train, label=y_train)

        params: dict[str, Any] = {
            "objective": "binary",
            "metric": "binary_logloss",
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "verbose": -1,
            "seed": self.seed,
            "scale_pos_weight": 1.0,  # Standard or calibrated
        }

        valid_sets = [train_data]
        if val_df is not None:
            x_val = val_df[FEATURE_COLUMNS]
            y_val = val_df["target"].astype(int)
            val_data = lgb.Dataset(x_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)

        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=self.n_estimators,
            valid_sets=valid_sets,
        )

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict match probability for candidate pairs."""
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")
        x = df[FEATURE_COLUMNS]
        preds = self.model.predict(x)
        return np.asarray(preds, dtype=np.float64)

    def optimize_threshold(
        self,
        val_df: pd.DataFrame,
        val_gt: Mapping[str, set[str]],
        search_range: Sequence[float] = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90),
    ) -> tuple[float, float]:
        """
        Locate optimal decision threshold tau maximizing Macro F0.5 on validation data.
        Returns (best_threshold, best_macro_f05).
        """
        probs = self.predict_proba(val_df)
        df_eval = val_df[["s1_id", "cand_id"]].copy()
        df_eval["prob"] = probs

        all_s1_ids = list(val_gt.keys())
        best_tau = self.decision_threshold
        best_score = -1.0

        for tau in search_range:
            preds: dict[str, set[str]] = {s1: set() for s1 in all_s1_ids}
            filtered = df_eval[df_eval["prob"] >= tau]
            for _, row in filtered.iterrows():
                preds[str(row["s1_id"])].add(str(row["cand_id"]))

            report = evaluate_resolution_predictions(val_gt, preds)
            if report.macro_f05 > best_score:
                best_score = report.macro_f05
                best_tau = float(tau)

        self.decision_threshold = best_tau
        return best_tau, best_score

    def save_model(self, path: Path | str) -> None:
        """Save model booster to disk."""
        if self.model is None:
            raise RuntimeError("No model to save.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))

    def load_model(self, path: Path | str) -> None:
        """Load model booster from disk."""
        self.model = lgb.Booster(model_file=str(path))
