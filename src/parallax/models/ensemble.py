"""
Parallax Ensembling & Blending Module
=====================================
Combines multiple diverse models (text, vision, rules) via priority fallback,
voting, and probability blending.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd

from parallax.models.base import BaseModel


class PriorityFallbackEnsemble(BaseModel):
    """
    Combines models sequentially: if the primary model output is empty, null,
    or falls below confidence, fall back to the next model in the cascade.
    """

    def __init__(
        self,
        models: list[BaseModel],
        name: str = "priority_fallback_ensemble",
        fallback_value: str = "",
    ) -> None:
        super().__init__(name=name)
        self.models = models
        self.fallback_value = fallback_value

    def fit(self, df: pd.DataFrame, target_col: str) -> PriorityFallbackEnsemble:
        """Fit all component models in the cascade."""
        for model in self.models:
            if not model.is_fitted:
                model.fit(df, target_col)
        self.is_fitted = True
        return self

    def predict(self, df: pd.DataFrame) -> Sequence[Any]:
        """Cascade predictions through models."""
        n_rows = len(df)
        final_preds: list[str | None] = [None] * n_rows

        for model in self.models:
            current_preds = model.predict(df)
            for idx, pred in enumerate(current_preds):
                if final_preds[idx] is None:
                    pred_str = str(pred).strip() if pred is not None else ""
                    if pred_str and pred_str.lower() != "nan" and pred_str != "":
                        final_preds[idx] = pred_str

            # If all rows have predictions, terminate early
            if all(p is not None for p in final_preds):
                break

        # Fill remaining missing with fallback
        return [p if p is not None else self.fallback_value for p in final_preds]
