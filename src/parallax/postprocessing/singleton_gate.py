"""
Parallax Post-Processing & Singleton Gating Engine
==================================================
High-precision filtering and singleton protection:
- Ensures all Source 1 entities have a row
- Gating rule: if max candidate probability < tau, predicts empty list (1.0 singleton score)
- Formats final output adhering to official matching_results.tsv schema.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from parallax.data.contracts import write_matching_results_tsv


class SingletonGatedPredictor:
    """Filters candidate pairs with calibrated confidence gating."""

    def __init__(self, decision_threshold: float = 0.75) -> None:
        self.decision_threshold = decision_threshold

    def filter_predictions(
        self,
        scored_pairs_df: pd.DataFrame,
        all_s1_ids: Sequence[str],
        threshold: float | None = None,
    ) -> dict[str, set[str]]:
        """
        Produce final match sets for all Source 1 entities with calibrated threshold gating.
        Ensures all Source 1 entities are present in the output mapping.
        """
        tau = threshold if threshold is not None else self.decision_threshold
        predictions: dict[str, set[str]] = {s1_id: set() for s1_id in all_s1_ids}

        if len(scored_pairs_df) == 0:
            return predictions

        # Filter by calibrated probability threshold
        base_matches = scored_pairs_df[scored_pairs_df["prob"] >= tau]

        for s1, cand in zip(
            base_matches["s1_id"].to_numpy(), base_matches["cand_id"].to_numpy(), strict=False
        ):
            s1_str = str(s1)
            if s1_str in predictions:
                predictions[s1_str].add(str(cand))

        return predictions

    def export_matching_results(
        self,
        output_path: Path | str,
        predictions: Mapping[str, set[str]],
    ) -> None:
        """Serialize predictions to official matching_results.tsv."""
        write_matching_results_tsv(output_path, dict(predictions))
