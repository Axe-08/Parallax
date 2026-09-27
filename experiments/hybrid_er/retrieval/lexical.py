"""
Country-partitioned char-3gram TF-IDF retrieval that keeps ranks and scores.

Mirrors the E0 blocker's name/address channels (same normalizers, n-grams, K, min_sim and
address df-pruning) but returns long-format hits per channel instead of an unscored set,
so it can serve as a provenance-tracked hard-negative source. It is NOT used to replace
the frozen E0 candidate pool.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from parallax.preprocessing.normalizer import clean_address, clean_soft_name

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LexicalChannel:
    name: str
    field: str  # "name" | "address"
    top_k: int
    min_sim: float


E0_LEXICAL_CHANNELS = (
    LexicalChannel("lex_name", "name", top_k=25, min_sim=0.15),
    LexicalChannel("lex_addr", "address", top_k=20, min_sim=0.20),
)


def _texts(df: pd.DataFrame, field: str) -> list[str]:
    if field == "name":
        return [clean_soft_name(x) or "" for x in df["business_name"].tolist()]
    return [clean_address(x if x else None) or "" for x in df["business_address"].tolist()]


def _topk_sparse_rows(sims, k: int, min_sim: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Row-wise top-k of a CSR matrix. Returns (row, col, score) arrays."""
    rows, cols, vals = [], [], []
    indptr, indices, data = sims.indptr, sims.indices, sims.data
    for r in range(sims.shape[0]):
        a, b = indptr[r], indptr[r + 1]
        if a == b:
            continue
        v = data[a:b]
        c = indices[a:b]
        keep = v >= min_sim
        if not keep.any():
            continue
        v, c = v[keep], c[keep]
        if len(v) > k:
            top = np.argpartition(v, -k)[-k:]
            v, c = v[top], c[top]
        rows.append(np.full(len(v), r))
        cols.append(c)
        vals.append(v)
    if not rows:
        return np.empty(0, int), np.empty(0, int), np.empty(0, np.float32)
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals).astype(np.float32)


def lexical_retrieve(
    s1_df: pd.DataFrame,
    targets: pd.DataFrame,
    channels: tuple[LexicalChannel, ...] = E0_LEXICAL_CHANNELS,
    batch_size: int = 500,
) -> pd.DataFrame:
    """
    `targets` must have entity_id, business_name, business_address, country and source.
    Returns hits (s1_id, source, cand_id, blocker, rank, score); like E0, top-k is taken over
    the combined S2+S3 target set of the S1's country, and rank is 1-based per (s1, blocker).
    """
    frames = []
    for country in sorted(set(s1_df["country"].astype(str))):
        q = s1_df[s1_df["country"].astype(str) == country].reset_index(drop=True)
        t = targets[targets["country"].astype(str) == country].reset_index(drop=True)
        if q.empty or t.empty:
            logger.warning("Lexical: no targets for country %r (%d S1s)", country, len(q))
            continue
        t_ids = t["entity_id"].to_numpy(dtype=object)
        t_src = t["source"].to_numpy(dtype=object)
        q_ids = q["entity_id"].to_numpy(dtype=object)
        for ch in channels:
            if ch.field == "address" and len(t) > 500:
                vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 3), min_df=2, max_df=0.40, sublinear_tf=True)
            else:
                vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 3), min_df=1, sublinear_tf=True)
            t_mat = vec.fit_transform(_texts(t, ch.field)).T.tocsr()
            q_txt = _texts(q, ch.field)
            for start in range(0, len(q), batch_size):
                sims = vec.transform(q_txt[start : start + batch_size]).dot(t_mat).tocsr()
                r, c, v = _topk_sparse_rows(sims, ch.top_k, ch.min_sim)
                frames.append(
                    pd.DataFrame(
                        {"s1_id": q_ids[start + r], "source": t_src[c], "cand_id": t_ids[c], "blocker": ch.name, "score": v}
                    )
                )
            logger.info("Lexical %s/%s done (%d S1 x %d targets)", country, ch.name, len(q), len(t))
    if not frames:
        return pd.DataFrame(columns=["s1_id", "source", "cand_id", "blocker", "rank", "score"])
    hits = pd.concat(frames, ignore_index=True)
    hits = hits.sort_values(["s1_id", "blocker", "score"], ascending=[True, True, False], kind="stable")
    hits["rank"] = hits.groupby(["s1_id", "blocker"], sort=False).cumcount() + 1
    return hits.reset_index(drop=True)[["s1_id", "source", "cand_id", "blocker", "rank", "score"]]
