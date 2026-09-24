"""
Parallax Base Model Interface
=============================
Abstract model interface for pluggable baseline shootouts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import pandas as pd

from parallax.metrics.evaluator import evaluate_predictions


class BaseModel(ABC):
    """Abstract baseline model interface."""

    def __init__(self, name: str, config: dict[str, Any] | None = None) -> None:
        self.name = name
        self.config = config or {}
        self.is_fitted = False

    @abstractmethod
    def fit(self, df: pd.DataFrame, target_col: str) -> BaseModel:
        """Fit model to training data."""
        pass

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> Sequence[Any]:
        """Generate predictions for given dataframe."""
        pass

    def evaluate(self, df: pd.DataFrame, target_col: str) -> dict[str, float]:
        """Generate predictions and evaluate against ground truth."""
        preds = self.predict(df)
        return evaluate_predictions(y_true=df[target_col].tolist(), y_pred=preds)
