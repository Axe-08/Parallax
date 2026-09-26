"""
Parallax Entity-Level Meta-Feature Engineering (Track A)
========================================================
Extracts entity-level distribution and comparative ranking features from
Pass 1 model match probabilities:
- s1_max_score: Maximum candidate match probability for the S1 query
- s1_mean_score: Average candidate match probability for the S1 query
- s1_std_score: Spread / standard deviation of candidate probabilities
- score_rank_pct: Percentile rank of this candidate within S1 candidate pool
- score_gap_to_best: Margin between top candidate and current candidate
"""

from __future__ import annotations

import numpy as np
import pandas as pd

META_FEATURE_COLUMNS: list[str] = [
    "s1_max_score",
    "s1_mean_score",
    "s1_std_score",
    "score_rank_pct",
    "score_gap_to_best",
]


def compute_entity_meta_features(
    df: pd.DataFrame,
    prob_col: str = "prob",
    s1_id_col: str = "s1_id",
) -> pd.DataFrame:
    """
    Compute entity-level distribution and ranking meta-features for candidate pairs.
    Appends META_FEATURE_COLUMNS in-place and returns df.
    """
    if df.empty:
        for col in META_FEATURE_COLUMNS:
            df[col] = pd.Series(dtype=np.float32)
        return df

    g = df.groupby(s1_id_col)[prob_col]

    max_score = g.transform("max").astype(np.float32)
    mean_score = g.transform("mean").astype(np.float32)
    std_score = g.transform("std").fillna(0.0).astype(np.float32)
    rank_pct = g.rank(ascending=False, pct=True).astype(np.float32)
    gap_to_best = (max_score - df[prob_col].astype(np.float32)).astype(np.float32)

    df["s1_max_score"] = max_score
    df["s1_mean_score"] = mean_score
    df["s1_std_score"] = std_score
    df["score_rank_pct"] = rank_pct
    df["score_gap_to_best"] = gap_to_best

    return df
