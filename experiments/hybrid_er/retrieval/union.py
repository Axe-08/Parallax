from __future__ import annotations

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from experiments.hybrid_er.core.validation import (
    CANONICAL_KEY,
    validate_candidate_schema,
    validate_source_contract,
)

CHANNEL_REQUIRED_COLS = ("s1_id", "source", "cand_id", "blocker", "rank", "score")
NOT_FOUND_RANK = 9999
NOT_FOUND_SCORE = 0.0


def validate_channel_table(df: pd.DataFrame, s2_ids: set[str], s3_ids: set[str]) -> None:
    """
    Validates one retrieval channel's output before it enters the union.

    Each (s1_id, source, cand_id, blocker) must be unique: a blocker that emits the same
    candidate twice (e.g. from two embedding views) must use distinct blocker names
    (e.g. 'bge_full', 'bge_name') so view-level evidence is preserved, not collapsed.
    """
    missing = set(CHANNEL_REQUIRED_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Candidate channel table is missing required columns {missing}. Columns: {list(df.columns)}")
    if df["blocker"].isnull().any() or (df["blocker"].astype(str).str.strip() == "").any():
        raise ValueError("Candidate channel table contains null/empty blocker names.")
    validate_candidate_schema(df, unique_on=(*CANONICAL_KEY, "blocker"))
    validate_source_contract(df, s2_ids, s3_ids)


def merge_candidate_tables(
    candidate_dfs: list[pd.DataFrame],
    e0_features_df: pd.DataFrame | None = None,
    *,
    s2_ids: set[str],
    s3_ids: set[str],
) -> pd.DataFrame:
    """
    Merges multiple candidate dataframes into a single table keyed by the canonical
    identity (s1_id, source, cand_id). This is the common union boundary: every input is
    validated for schema, allowed source and S2/S3 population membership here, regardless
    of which channel produced it.

    Each input dataframe MUST have:
    - s1_id (str)
    - source (str: 'S2' or 'S3')
    - cand_id (str)
    - blocker (str) - e.g., 'e0', 'bge_full', 'channel_f_name'
    - rank (int) - rank within that blocker
    - score (float) - similarity score within that blocker

    Returns a unified candidate table with columns:
    s1_id, source, cand_id,
    found_by_<blocker>, rank_<blocker>, score_<blocker> for every blocker.
    """
    if s2_ids is None or s3_ids is None:
        raise ValueError("merge_candidate_tables requires s2_ids and s3_ids to enforce the source contract.")

    std_dfs = []
    for df in candidate_dfs:
        if df is None or len(df) == 0:
            continue
        d = df.copy()
        validate_channel_table(d, s2_ids, s3_ids)
        d["blocker"] = d["blocker"].astype(str)
        std_dfs.append(d[list(CHANNEL_REQUIRED_COLS)])

    if not std_dfs:
        return pd.DataFrame(columns=list(CANONICAL_KEY))

    master_df = pd.concat(std_dfs, ignore_index=True)
    # The same blocker name arriving from two different input tables is also a collision.
    validate_candidate_schema(master_df, unique_on=(*CANONICAL_KEY, "blocker"))
    master_df["found_by"] = 1

    pivoted = master_df.pivot(
        index=list(CANONICAL_KEY),
        columns="blocker",
        values=["found_by", "rank", "score"],
    )
    pivoted.columns = [f"{col[0]}_{col[1]}" for col in pivoted.columns]
    pivoted = pivoted.reset_index()

    # Provenance comes from presence, not from rank/score: a channel may legitimately not
    # provide a score (left NaN), and must still count as having found the candidate.
    # Sentinels are applied only to candidates the blocker did NOT find.
    for b in sorted(master_df["blocker"].unique()):
        found = pivoted[f"found_by_{b}"].notna()
        pivoted[f"found_by_{b}"] = found.astype(int)
        pivoted.loc[~found, f"rank_{b}"] = NOT_FOUND_RANK
        pivoted.loc[~found, f"score_{b}"] = NOT_FOUND_SCORE

    if e0_features_df is not None:
        e0 = e0_features_df.copy()
        validate_candidate_schema(e0)
        n_before = len(pivoted)
        pivoted = pivoted.merge(e0, on=list(CANONICAL_KEY), how="left")
        if len(pivoted) != n_before:
            raise ValueError("Joining E0 features changed the candidate row count.")

    validate_candidate_schema(pivoted)
    return pivoted


def candidates_to_dict(df: pd.DataFrame, eval_s1_ids: list[str] | None = None) -> dict[str, set[str]]:
    """
    Collapses a validated candidate table to {s1_id: {cand_id, ...}}. When eval_s1_ids is
    given, every eval S1 gets an entry (empty set when no candidates were retrieved).
    """
    out: dict[str, set[str]] = {str(s): set() for s in eval_s1_ids} if eval_s1_ids is not None else {}
    if df is None or df.empty:
        return out
    for s1, cand in zip(df["s1_id"].astype(str).to_numpy(), df["cand_id"].astype(str).to_numpy()):
        if eval_s1_ids is not None and s1 not in out:
            raise ValueError(f"Candidate table contains S1 {s1} outside the evaluation population.")
        out.setdefault(s1, set()).add(cand)
    return out


def save_candidates(df: pd.DataFrame, filepath: str) -> None:
    """
    Saves candidates to parquet.
    """
    table = pa.Table.from_pandas(df)
    pq.write_table(table, filepath)


def load_candidates(filepath: str) -> pd.DataFrame:
    """
    Loads candidates from parquet.
    """
    table = pq.read_table(filepath)
    return table.to_pandas()
