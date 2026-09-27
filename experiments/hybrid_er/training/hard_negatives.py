"""
Multi-channel hard-negative mining with provenance.

Input is a long-format retrieval pool (s1_id, source, cand_id, blocker, rank, score) that
combines several channels (e.g. zero-shot BGE, lexical name, lexical address). For each S1:

    negatives = pool candidates NOT in that S1's GT
    hardness  = rank inside the channel that found it

Channels are interleaved by rank (rank-1 of every channel, then rank-2, ...) so no single
score decides what is "hard"; a candidate found by several channels keeps its best rank and
records every channel in `found_by`. `hard_negative_source` is the channel that surfaced it
first in the interleave (ties broken by the configured channel priority).

Safety: a GT pair is never emitted as a negative (asserted, not assumed), and S1s that are
absent from the supplied GT raise instead of being treated as zero-match.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

POOL_COLS = ("s1_id", "source", "cand_id", "blocker", "rank", "score")


@dataclass
class MiningSummary:
    n_s1: int
    n_s1_with_negatives: int
    pool_rows: int
    pool_pairs: int
    pool_positive_pairs: int
    negatives_emitted: int
    negatives_by_source: dict[str, int]
    negatives_found_by_multiple: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def label_pool(pool: pd.DataFrame, gt: Mapping[str, set[str]]) -> np.ndarray:
    s1 = pool["s1_id"].astype(str).to_numpy()
    missing = set(s1) - set(gt)
    if missing:
        raise ValueError(f"{len(missing)} pool S1s are missing from the supplied GT (e.g. {sorted(missing)[:3]}).")
    cand = pool["cand_id"].astype(str).to_numpy()
    return np.fromiter((c in gt[s] for s, c in zip(s1, cand)), dtype=bool, count=len(pool))


def mine_hard_negatives(
    pool: pd.DataFrame,
    gt: Mapping[str, set[str]],
    n_per_s1: int = 30,
    channel_priority: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, MiningSummary]:
    """
    Returns (negatives, summary). `negatives` has one row per (s1_id, cand_id) with columns
    s1_id, source, cand_id, hard_negative_source, found_by, best_rank, max_score, neg_order
    (0 = hardest). At most `n_per_s1` negatives per S1.
    """
    missing_cols = set(POOL_COLS) - set(pool.columns)
    if missing_cols:
        raise ValueError(f"Pool is missing columns {missing_cols}")
    df = pool[list(POOL_COLS)].copy()
    df["s1_id"] = df["s1_id"].astype(str)
    df["cand_id"] = df["cand_id"].astype(str)
    is_pos = label_pool(df, gt)

    blockers = sorted(df["blocker"].unique())
    priority = list(channel_priority) if channel_priority else blockers
    prio = {b: i for i, b in enumerate(priority)}
    unknown = set(blockers) - set(prio)
    if unknown:
        raise ValueError(f"Blockers {unknown} missing from channel_priority {priority}")

    if df.duplicated(["s1_id", "cand_id", "blocker"]).any():
        raise ValueError("Pool has duplicate (s1_id, cand_id, blocker) rows.")
    neg = df[~is_pos].copy()
    neg["_prio"] = neg["blocker"].map(prio)
    neg = neg.sort_values(["s1_id", "rank", "_prio"], kind="stable")

    neg["_bit"] = np.left_shift(1, neg["_prio"].to_numpy(dtype=np.int64))
    agg = neg.groupby(["s1_id", "cand_id"], sort=False).agg(
        source=("source", "first"),
        hard_negative_source=("blocker", "first"),
        _mask=("_bit", "sum"),  # (s1, cand, blocker) is unique, so sum == bitwise OR
        best_rank=("rank", "min"),
        max_score=("score", "max"),
    )
    # groupby(sort=False) keeps first-appearance order = interleaved hardness order.
    agg = agg.reset_index()
    names = {m: "|".join(sorted(b for b, i in prio.items() if m >> i & 1)) for m in agg["_mask"].unique()}
    agg["found_by"] = agg["_mask"].map(names)
    if agg.duplicated(["s1_id", "cand_id"]).any():
        raise AssertionError("Duplicate (s1_id, cand_id) after aggregation.")
    agg["neg_order"] = agg.groupby("s1_id", sort=False).cumcount()
    out = agg[agg["neg_order"] < n_per_s1].reset_index(drop=True)

    leaked = label_pool(out, gt)
    if leaked.any():
        raise AssertionError(f"{int(leaked.sum())} GT pairs were emitted as hard negatives.")

    pos_pairs = df.loc[is_pos, ["s1_id", "cand_id"]].drop_duplicates()
    summary = MiningSummary(
        n_s1=int(df["s1_id"].nunique()),
        n_s1_with_negatives=int(out["s1_id"].nunique()),
        pool_rows=len(df),
        pool_pairs=int(df[["s1_id", "cand_id"]].drop_duplicates().shape[0]),
        pool_positive_pairs=len(pos_pairs),
        negatives_emitted=len(out),
        negatives_by_source={str(k): int(v) for k, v in out["hard_negative_source"].value_counts().items()},
        negatives_found_by_multiple=int(out["found_by"].str.contains("|", regex=False).sum()),
    )
    return out[["s1_id", "source", "cand_id", "hard_negative_source", "found_by", "best_rank", "max_score", "neg_order"]], summary
