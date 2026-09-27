"""
Deterministic, S1-grouped matcher harness (outer CV + inner-OOF threshold selection).

Protocol (identical to the trusted E0 baseline in experiments/neural_text/scripts/run_experiments.py):

    outer fold k (fixed fold file)
        -> inner grouped K-fold on outer-train S1s -> pooled inner OOF probabilities
        -> select tau on the inner OOF Macro F0.5 (ties -> higher tau)
        -> refit on full outer train -> predict outer validation with the frozen tau

Differences from the legacy script, all for correctness:
- Inner folds are built from *sorted* S1 IDs with a seeded Generator. The legacy script
  shuffled `list(set(ids))`, whose order depends on PYTHONHASHSEED, so its thresholds and
  final metric changed from process to process.
- Labels are derived from the evaluation-scoped GT, not trusted from a precomputed column.
- Returns row-level outer-OOF probabilities for downstream stacking.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from parallax.metrics.evaluator import evaluate_resolution_predictions
from parallax.postprocessing.singleton_gate import SingletonGatedPredictor

BASELINE_LGBM_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.08,
    "num_leaves": 31,
    "max_depth": 6,
    "verbose": -1,
    "n_jobs": 8,
    "deterministic": True,
    "force_col_wise": True,
}
BASELINE_NUM_BOOST_ROUND = 100
BASELINE_THRESHOLD_GRID: tuple[float, ...] = (0.60, 0.65, 0.70, 0.74, 0.78, 0.82, 0.86, 0.90)
BASELINE_INNER_K = 3


def deterministic_group_partition(s1_ids: Sequence[str], n_splits: int, seed: int) -> list[set[str]]:
    """Partitions S1 IDs into n_splits disjoint groups, independent of hash seed / input order."""
    ids = np.array(sorted({str(s) for s in s1_ids}), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    return [set(chunk.tolist()) for chunk in np.array_split(ids, n_splits)]


def label_pairs(pairs_df: pd.DataFrame, gt: Mapping[str, set[str]]) -> np.ndarray:
    """is_match for each (s1_id, cand_id) row, derived from GT only."""
    s1 = pairs_df["s1_id"].astype(str).to_numpy()
    cand = pairs_df["cand_id"].astype(str).to_numpy()
    return np.fromiter((c in gt.get(s, ()) for s, c in zip(s1, cand)), dtype=np.int8, count=len(pairs_df))


def predictions_at(scored: pd.DataFrame, s1_ids: Sequence[str], tau: float) -> dict[str, set[str]]:
    return SingletonGatedPredictor().filter_predictions(scored, list(s1_ids), threshold=tau)


def select_threshold(
    scored: pd.DataFrame,
    gt: Mapping[str, set[str]],
    s1_ids: Sequence[str],
    grid: Sequence[float] = BASELINE_THRESHOLD_GRID,
) -> tuple[float, float]:
    """Returns (tau, macro_f05) maximizing Macro F0.5 on `scored`; ties go to the higher tau."""
    gt_scope = {s: gt[s] for s in s1_ids}
    best_tau, best = float(grid[0]), -1.0
    for tau in sorted(grid):
        f = evaluate_resolution_predictions(gt_scope, predictions_at(scored, s1_ids, tau)).macro_f05
        if f >= best:
            best_tau, best = float(tau), f
    return best_tau, best


@dataclass
class FoldOutcome:
    fold: int
    tau: float
    inner_oof_f05: float
    n_train_s1: int
    n_val_s1: int
    n_train_rows: int
    n_val_rows: int


@dataclass
class MatcherRun:
    seed: int
    feature_cols: list[str]
    predictions: dict[str, set[str]]
    oof_prob: np.ndarray
    folds: list[FoldOutcome] = field(default_factory=list)

    @property
    def taus(self) -> list[float]:
        return [f.tau for f in self.folds]


def _train(x: pd.DataFrame, y: np.ndarray, params: dict[str, Any], rounds: int, seed: int) -> lgb.Booster:
    p = dict(params)
    p["seed"] = seed
    return lgb.train(p, lgb.Dataset(x, label=y), num_boost_round=rounds)


def run_grouped_matcher(
    pairs_df: pd.DataFrame,
    feature_cols: Sequence[str],
    gt_eval: Mapping[str, set[str]],
    s1_to_fold: Mapping[str, int],
    *,
    seed: int = 42,
    inner_k: int = BASELINE_INNER_K,
    grid: Sequence[float] = BASELINE_THRESHOLD_GRID,
    params: dict[str, Any] | None = None,
    num_boost_round: int = BASELINE_NUM_BOOST_ROUND,
) -> MatcherRun:
    """
    Runs the outer/inner grouped protocol over the evaluation population `gt_eval`.

    `s1_to_fold` must assign every evaluation S1 to an outer fold. S1s with no candidate
    rows still belong to their fold (they are scored as empty predictions).
    """
    params = BASELINE_LGBM_PARAMS if params is None else params
    feature_cols = list(feature_cols)
    eval_ids = sorted(gt_eval.keys())
    unassigned = [s for s in eval_ids if s not in s1_to_fold]
    if unassigned:
        raise ValueError(f"{len(unassigned)} evaluation S1s have no outer fold assignment.")
    row_s1 = pairs_df["s1_id"].astype(str).to_numpy()
    outside = set(row_s1) - set(eval_ids)
    if outside:
        raise ValueError(f"pairs_df contains {len(outside)} S1s outside the evaluation population.")

    y = label_pairs(pairs_df, gt_eval)
    x = pairs_df[feature_cols]
    row_fold = np.array([s1_to_fold[s] for s in row_s1])
    folds = sorted({s1_to_fold[s] for s in eval_ids})

    oof_prob = np.full(len(pairs_df), np.nan)
    predictions: dict[str, set[str]] = {}
    outcomes: list[FoldOutcome] = []

    for k in folds:
        train_s1 = [s for s in eval_ids if s1_to_fold[s] != k]
        val_s1 = [s for s in eval_ids if s1_to_fold[s] == k]
        if set(train_s1) & set(val_s1):
            raise AssertionError("Outer train/validation S1 overlap.")
        tr = row_fold != k
        va = row_fold == k

        # Inner OOF on outer-train S1s only (no validation GT is touched).
        inner_prob = np.full(int(tr.sum()), np.nan)
        tr_idx = np.flatnonzero(tr)
        tr_s1 = row_s1[tr_idx]
        for i, inner_val in enumerate(deterministic_group_partition(train_s1, inner_k, seed=seed * 1000 + k)):
            iv = np.isin(tr_s1, list(inner_val))
            booster = _train(x.iloc[tr_idx[~iv]], y[tr_idx[~iv]], params, num_boost_round, seed + k)
            inner_prob[iv] = booster.predict(x.iloc[tr_idx[iv]])
        if np.isnan(inner_prob).any():
            raise AssertionError("Inner OOF probabilities incomplete.")

        inner_scored = pd.DataFrame({"s1_id": tr_s1, "cand_id": pairs_df["cand_id"].to_numpy()[tr_idx], "prob": inner_prob})
        tau, inner_f = select_threshold(inner_scored, gt_eval, train_s1, grid)

        booster = _train(x.iloc[tr_idx], y[tr_idx], params, num_boost_round, seed + k)
        va_idx = np.flatnonzero(va)
        oof_prob[va_idx] = booster.predict(x.iloc[va_idx])
        val_scored = pd.DataFrame(
            {"s1_id": row_s1[va_idx], "cand_id": pairs_df["cand_id"].to_numpy()[va_idx], "prob": oof_prob[va_idx]}
        )
        predictions.update(predictions_at(val_scored, val_s1, tau))
        outcomes.append(
            FoldOutcome(
                fold=int(k),
                tau=tau,
                inner_oof_f05=inner_f,
                n_train_s1=len(train_s1),
                n_val_s1=len(val_s1),
                n_train_rows=int(tr.sum()),
                n_val_rows=int(va.sum()),
            )
        )

    if np.isnan(oof_prob).any():
        raise AssertionError("Outer OOF probabilities incomplete.")
    return MatcherRun(seed=seed, feature_cols=feature_cols, predictions=predictions, oof_prob=oof_prob, folds=outcomes)
