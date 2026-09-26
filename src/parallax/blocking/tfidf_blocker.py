"""
Parallax Dual-Channel Sparse TF-IDF Blocker
===========================================
High-recall, memory-efficient candidate generator:
- Dynamic country-partitioned indexing (open-set: US, India, France)
- Dual-channel Character 3-Gram TF-IDF (Name Channel + Address Channel)
- Batched sparse matrix multiplication to prevent out-of-memory errors
- Pruning and union aggregation producing candidate_pairs.tsv
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from parallax.data.contracts import write_candidate_pairs_tsv

if TYPE_CHECKING:
    from parallax.utils.checkpoint_manager import CheckpointManager


class DualChannelTFIDFBlocker:
    """
    Generates candidate match pairs using sparse Character 3-Gram TF-IDF
    over both business name and business address fields.
    """

    def __init__(
        self,
        name_top_k: int = 35,
        addr_top_k: int = 25,
        translit_top_k: int = 5,
        name_min_sim: float = 0.15,
        addr_min_sim: float = 0.20,
        translit_min_sim: float = 0.30,
        batch_size: int = 2000,
        show_progress: bool = True,
    ) -> None:
        self.name_top_k = name_top_k
        self.addr_top_k = addr_top_k
        self.translit_top_k = translit_top_k
        self.name_min_sim = name_min_sim
        self.addr_min_sim = addr_min_sim
        self.translit_min_sim = translit_min_sim
        self.batch_size = batch_size
        self.show_progress = show_progress

    def generate_candidates(
        self,
        s1_df: pd.DataFrame,
        target_df: pd.DataFrame,
        checkpoint_mgr: CheckpointManager | None = None,
    ) -> dict[str, set[str]]:
        """
        Generate candidate pairs partitioned by country.
        Returns a dictionary mapping source1_entity_id -> set of candidate entity_ids.
        """
        candidate_pairs: dict[str, set[str]] = {s1_id: set() for s1_id in s1_df["entity_id"]}

        # Process each country partition independently
        countries = s1_df["country"].unique()

        def _build_texts(
            df: pd.DataFrame, primary_col: str, fallback_col: str, translit_col: str
        ) -> list[str]:
            if primary_col in df:
                p_s = df[primary_col].fillna("").astype(str).str.strip()
                if fallback_col in df:
                    empty = p_s == ""
                    if empty.any():
                        p_s[empty] = df.loc[empty, fallback_col].fillna("").astype(str).str.strip()
            elif fallback_col in df:
                p_s = df[fallback_col].fillna("").astype(str).str.strip()
            else:
                p_s = pd.Series([""] * len(df), index=df.index)

            if translit_col in df:
                t_s = df[translit_col].fillna("").astype(str).str.strip()
                diff_mask = (t_s != "") & (t_s != p_s)
                res = p_s.copy()
                if diff_mask.any():
                    res[diff_mask] = p_s[diff_mask] + " " + t_s[diff_mask]
                return [str(x) for x in res.tolist()]
            return [str(x) for x in p_s.tolist()]

        for country in countries:
            cp_key = f"candidates_{country}"
            if checkpoint_mgr is not None and checkpoint_mgr.has_checkpoint(cp_key):
                loaded_df = checkpoint_mgr.load_dataframe(cp_key)
                if loaded_df is not None:
                    s1_arr = loaded_df["s1_id"].to_numpy()
                    cand_arr = loaded_df["cand_id"].to_numpy()
                    for s1, cand in zip(s1_arr, cand_arr, strict=False):
                        if s1 in candidate_pairs:
                            candidate_pairs[s1].add(cand)
                    msg = f"  ⚡ [Checkpoint] Loaded {len(loaded_df):,} candidates for [{country}]."
                    print(msg)
                    continue

            s1_c = s1_df[s1_df["country"] == country].reset_index(drop=True)
            tgt_c = target_df[target_df["country"] == country].reset_index(drop=True)

            if len(s1_c) == 0 or len(tgt_c) == 0:
                continue

            # Build dual-representation blocking texts (native + transliterated Latin)
            s1_names = _build_texts(s1_c, "soft_name", "business_name", "translit_name")
            tgt_names = _build_texts(tgt_c, "soft_name", "business_name", "translit_name")
            s1_addrs = _build_texts(s1_c, "clean_address", "business_address", "translit_address")
            tgt_addrs = _build_texts(tgt_c, "clean_address", "business_address", "translit_address")

            # --- Channel A: Name Character 3-Gram TF-IDF ---
            vec_name = TfidfVectorizer(
                analyzer="char",
                ngram_range=(3, 3),
                min_df=1,
                sublinear_tf=True,
            )
            tgt_name_mat = vec_name.fit_transform(tgt_names).T  # shape (vocab, n_tgt)
            tgt_ids = tgt_c["entity_id"].tolist()

            batch_ranges_name = list(range(0, len(s1_c), self.batch_size))
            pbar_name = tqdm(
                batch_ranges_name,
                desc=f"  ⚡ Blocking [{country}|Name TF-IDF]",
                unit="batch",
                leave=False,
                disable=not self.show_progress,
            )
            for start_idx in pbar_name:
                end_idx = min(start_idx + self.batch_size, len(s1_c))
                s1_batch_names = s1_names[start_idx:end_idx]
                s1_batch_ids = s1_c["entity_id"].iloc[start_idx:end_idx].tolist()

                batch_mat = vec_name.transform(s1_batch_names)
                # Pure sparse dot product: (batch_size, n_tgt)
                batch_sims = batch_mat.dot(tgt_name_mat)

                for i, s1_id in enumerate(s1_batch_ids):
                    r_start = batch_sims.indptr[i]
                    r_end = batch_sims.indptr[i + 1]
                    if r_start == r_end:
                        continue
                    scores = batch_sims.data[r_start:r_end]
                    col_idx = batch_sims.indices[r_start:r_end]

                    valid_mask = scores >= self.name_min_sim
                    if not np.any(valid_mask):
                        continue
                    scores = scores[valid_mask]
                    col_idx = col_idx[valid_mask]

                    if len(scores) > self.name_top_k:
                        top_sub = np.argpartition(scores, -self.name_top_k)[-self.name_top_k :]
                        top_cols = col_idx[top_sub]
                    else:
                        top_cols = col_idx

                    for c_idx in top_cols:
                        candidate_pairs[s1_id].add(tgt_ids[c_idx])

            # --- Channel B: Address Character 3-Gram TF-IDF ---
            if any(a.strip() for a in tgt_addrs):
                min_df_addr = 2 if len(tgt_c) > 500 else 1
                max_df_addr = 0.40 if len(tgt_c) > 500 else 1.0
                vec_addr = TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(3, 3),
                    min_df=min_df_addr,
                    max_df=max_df_addr,
                    sublinear_tf=True,
                )
                tgt_addr_mat = vec_addr.fit_transform(tgt_addrs).T

                batch_ranges_addr = list(range(0, len(s1_c), self.batch_size))
                pbar_addr = tqdm(
                    batch_ranges_addr,
                    desc=f"  ⚡ Blocking [{country}|Addr TF-IDF]",
                    unit="batch",
                    leave=False,
                    disable=not self.show_progress,
                )
                for start_idx in pbar_addr:
                    end_idx = min(start_idx + self.batch_size, len(s1_c))
                    s1_batch_addrs = s1_addrs[start_idx:end_idx]
                    s1_batch_ids = s1_c["entity_id"].iloc[start_idx:end_idx].tolist()

                    batch_mat = vec_addr.transform(s1_batch_addrs)
                    batch_sims = batch_mat.dot(tgt_addr_mat)

                    for i, s1_id in enumerate(s1_batch_ids):
                        if not s1_batch_addrs[i].strip():
                            continue
                        r_start = batch_sims.indptr[i]
                        r_end = batch_sims.indptr[i + 1]
                        if r_start == r_end:
                            continue
                        scores = batch_sims.data[r_start:r_end]
                        col_idx = batch_sims.indices[r_start:r_end]

                        valid_mask = scores >= self.addr_min_sim
                        if not np.any(valid_mask):
                            continue
                        scores = scores[valid_mask]
                        col_idx = col_idx[valid_mask]

                        if len(scores) > self.addr_top_k:
                            top_sub = np.argpartition(scores, -self.addr_top_k)[-self.addr_top_k :]
                            top_cols = col_idx[top_sub]
                        else:
                            top_cols = col_idx

                        for c_idx in top_cols:
                            candidate_pairs[s1_id].add(tgt_ids[c_idx])

            # --- Channel C: Building Number Match with Leading Character Prefix ---
            if "numbers" in s1_c and "numbers" in tgt_c:
                tgt_ids = tgt_c["entity_id"].astype(str).tolist()
                tgt_names_ser = (
                    tgt_c["soft_name"].fillna("").astype(str)
                    if "soft_name" in tgt_c
                    else tgt_c["business_name"].fillna("").astype(str)
                )
                tgt_prefixes = dict(zip(tgt_ids, tgt_names_ser.str[:2].tolist(), strict=False))

                num_to_tgt: dict[str, list[str]] = defaultdict(list)
                for eid, num_set in zip(tgt_ids, tgt_c["numbers"].tolist(), strict=False):
                    for num in num_set:
                        num_str = str(num)
                        if len(num_str) >= 2:
                            num_to_tgt[num_str].append(eid)

                s1_ids = s1_c["entity_id"].astype(str).tolist()
                s1_names_ser = (
                    s1_c["soft_name"].fillna("").astype(str)
                    if "soft_name" in s1_c
                    else s1_c["business_name"].fillna("").astype(str)
                )
                s1_prefixes = s1_names_ser.str[:2].tolist()

                pbar_num = tqdm(
                    zip(s1_ids, s1_prefixes, s1_c["numbers"].tolist(), strict=False),
                    total=len(s1_ids),
                    desc=f"  ⚡ Blocking [{country}|BldgNum]",
                    unit="entity",
                    leave=False,
                    disable=not self.show_progress,
                )
                for s1_id, s1_pfx, num_set in pbar_num:
                    if not s1_pfx:
                        continue
                    for num in num_set:
                        num_str = str(num)
                        if len(num_str) < 2:
                            continue
                        cands = num_to_tgt.get(num_str, [])
                        if len(cands) <= 50:
                            for cand_id in cands:
                                if tgt_prefixes.get(cand_id, "") == s1_pfx:
                                    candidate_pairs[s1_id].add(cand_id)

            # --- Channel D: Transliterated Name Character 3-Gram TF-IDF ---
            if "translit_name" in s1_c and "translit_name" in tgt_c:
                s1_translit = s1_c["translit_name"].fillna("").astype(str).tolist()
                tgt_translit = tgt_c["translit_name"].fillna("").astype(str).tolist()

                if any(t.strip() for t in tgt_translit):
                    vec_translit = TfidfVectorizer(
                        analyzer="char",
                        ngram_range=(3, 3),
                        min_df=1,
                        sublinear_tf=True,
                    )
                    tgt_translit_mat = vec_translit.fit_transform(tgt_translit).T
                    
                    tgt_ids_translit = tgt_c["entity_id"].tolist()

                    batch_ranges_translit = list(range(0, len(s1_c), self.batch_size))
                    pbar_translit = tqdm(
                        batch_ranges_translit,
                        desc=f"  ⚡ Blocking [{country}|Translit TF-IDF]",
                        unit="batch",
                        leave=False,
                        disable=not self.show_progress,
                    )
                    for start_idx in pbar_translit:
                        end_idx = min(start_idx + self.batch_size, len(s1_c))
                        s1_batch_translit = s1_translit[start_idx:end_idx]
                        s1_batch_ids = s1_c["entity_id"].iloc[start_idx:end_idx].tolist()

                        batch_mat = vec_translit.transform(s1_batch_translit)
                        batch_sims = batch_mat.dot(tgt_translit_mat)

                        for i, s1_id in enumerate(s1_batch_ids):
                            if not s1_batch_translit[i].strip():
                                continue
                            r_start = batch_sims.indptr[i]
                            r_end = batch_sims.indptr[i + 1]
                            if r_start == r_end:
                                continue
                            scores = batch_sims.data[r_start:r_end]
                            col_idx = batch_sims.indices[r_start:r_end]

                            valid_mask = scores >= self.translit_min_sim
                            if not np.any(valid_mask):
                                continue
                            scores = scores[valid_mask]
                            col_idx = col_idx[valid_mask]

                            if len(scores) > self.translit_top_k:
                                top_sub = np.argpartition(scores, -self.translit_top_k)[-self.translit_top_k :]
                                top_cols = col_idx[top_sub]
                            else:
                                top_cols = col_idx

                            for c_idx in top_cols:
                                candidate_pairs[s1_id].add(tgt_ids_translit[c_idx])

            if checkpoint_mgr is not None:
                c_rows_s1: list[str] = []
                c_rows_cand: list[str] = []
                country_s1_ids = set(s1_c["entity_id"])
                for s1_id in country_s1_ids:
                    for cand_id in candidate_pairs.get(s1_id, ()):
                        c_rows_s1.append(s1_id)
                        c_rows_cand.append(cand_id)
                df_country = pd.DataFrame({"s1_id": c_rows_s1, "cand_id": c_rows_cand})
                checkpoint_mgr.save_dataframe(cp_key, df_country)
                checkpoint_mgr.record_stage_completed(
                    f"blocking_{country}",
                    {"country": country, "pairs_count": len(df_country)},
                )

        return candidate_pairs

    def export_candidate_pairs(
        self,
        output_path: str,
        candidates: Mapping[str, set[str]],
    ) -> None:
        """Export candidates dictionary to candidate_pairs.tsv."""
        write_candidate_pairs_tsv(output_path, dict(candidates))
