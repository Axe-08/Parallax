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

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from parallax.data.contracts import write_candidate_pairs_tsv


class DualChannelTFIDFBlocker:
    """
    Generates candidate match pairs using sparse Character 3-Gram TF-IDF
    over both business name and business address fields.
    """

    def __init__(
        self,
        name_top_k: int = 35,
        addr_top_k: int = 25,
        name_min_sim: float = 0.15,
        addr_min_sim: float = 0.20,
        batch_size: int = 500,
    ) -> None:
        self.name_top_k = name_top_k
        self.addr_top_k = addr_top_k
        self.name_min_sim = name_min_sim
        self.addr_min_sim = addr_min_sim
        self.batch_size = batch_size

    def generate_candidates(
        self,
        s1_df: pd.DataFrame,
        target_df: pd.DataFrame,
    ) -> dict[str, set[str]]:
        """
        Generate candidate pairs partitioned by country.
        Returns a dictionary mapping source1_entity_id -> set of candidate entity_ids.
        """
        candidate_pairs: dict[str, set[str]] = {s1_id: set() for s1_id in s1_df["entity_id"]}

        # Process each country partition independently
        countries = s1_df["country"].unique()

        for country in countries:
            s1_c = s1_df[s1_df["country"] == country].reset_index(drop=True)
            tgt_c = target_df[target_df["country"] == country].reset_index(drop=True)

            if len(s1_c) == 0 or len(tgt_c) == 0:
                continue

            # Build dual-representation blocking texts (native + transliterated Latin)
            s1_names: list[str] = []
            for _, r in s1_c.iterrows():
                p = str(r.get("soft_name", r.get("business_name", ""))).strip()
                t = str(r.get("translit_name", "")).strip()
                s1_names.append(f"{p} {t}" if (t and t != p) else p)

            tgt_names: list[str] = []
            for _, r in tgt_c.iterrows():
                p = str(r.get("soft_name", r.get("business_name", ""))).strip()
                t = str(r.get("translit_name", "")).strip()
                tgt_names.append(f"{p} {t}" if (t and t != p) else p)

            s1_addrs: list[str] = []
            for _, r in s1_c.iterrows():
                p = str(r.get("clean_address", r.get("business_address", ""))).strip()
                t = str(r.get("translit_address", "")).strip()
                s1_addrs.append(f"{p} {t}" if (t and t != p) else p)

            tgt_addrs: list[str] = []
            for _, r in tgt_c.iterrows():
                p = str(r.get("clean_address", r.get("business_address", ""))).strip()
                t = str(r.get("translit_address", "")).strip()
                tgt_addrs.append(f"{p} {t}" if (t and t != p) else p)

            # --- Channel A: Name Character 3-Gram TF-IDF ---
            vec_name = TfidfVectorizer(analyzer="char", ngram_range=(3, 3), min_df=1)
            vec_name.fit(s1_names + tgt_names)

            tgt_name_mat = vec_name.transform(tgt_names).T  # shape (vocab, n_tgt)
            tgt_ids = tgt_c["entity_id"].tolist()

            for start_idx in range(0, len(s1_c), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(s1_c))
                s1_batch_names = s1_names[start_idx:end_idx]
                s1_batch_ids = s1_c["entity_id"].iloc[start_idx:end_idx].tolist()

                batch_mat = vec_name.transform(s1_batch_names)
                # Sparse dot product: (batch_size, n_tgt)
                batch_sims = batch_mat.dot(tgt_name_mat).toarray()

                for i, s1_id in enumerate(s1_batch_ids):
                    row_sims = batch_sims[i]
                    top_indices = np.argsort(row_sims)[::-1][: self.name_top_k]
                    for idx in top_indices:
                        if row_sims[idx] >= self.name_min_sim:
                            candidate_pairs[s1_id].add(tgt_ids[idx])

            # --- Channel B: Address Character 3-Gram TF-IDF ---
            # Filter non-empty address vocabulary
            non_empty_addrs = [a for a in (s1_addrs + tgt_addrs) if a.strip()]
            if non_empty_addrs:
                vec_addr = TfidfVectorizer(analyzer="char", ngram_range=(3, 3), min_df=1)
                vec_addr.fit(non_empty_addrs)

                tgt_addr_mat = vec_addr.transform(tgt_addrs).T

                for start_idx in range(0, len(s1_c), self.batch_size):
                    end_idx = min(start_idx + self.batch_size, len(s1_c))
                    s1_batch_addrs = s1_addrs[start_idx:end_idx]
                    s1_batch_ids = s1_c["entity_id"].iloc[start_idx:end_idx].tolist()

                    batch_mat = vec_addr.transform(s1_batch_addrs)
                    batch_sims = batch_mat.dot(tgt_addr_mat).toarray()

                    for i, s1_id in enumerate(s1_batch_ids):
                        if not s1_batch_addrs[i].strip():
                            continue
                        row_sims = batch_sims[i]
                        top_indices = np.argsort(row_sims)[::-1][: self.addr_top_k]
                        for idx in top_indices:
                            if row_sims[idx] >= self.addr_min_sim:
                                candidate_pairs[s1_id].add(tgt_ids[idx])

            # --- Channel C: Building Number Match with Leading Character Prefix ---
            if "numbers" in s1_c and "numbers" in tgt_c:
                num_to_tgt: dict[str, list[str]] = defaultdict(list)
                tgt_prefixes = {
                    row["entity_id"]: str(row.get("soft_name", ""))[:2]
                    for _, row in tgt_c.iterrows()
                }

                for _, row in tgt_c.iterrows():
                    for num in row["numbers"]:
                        num_to_tgt[num].append(row["entity_id"])

                for _, row in s1_c.iterrows():
                    s1_id = row["entity_id"]
                    s1_prefix = str(row.get("soft_name", ""))[:2]
                    if not s1_prefix:
                        continue
                    for num in row["numbers"]:
                        for cand_id in num_to_tgt.get(num, []):
                            if tgt_prefixes.get(cand_id, "") == s1_prefix:
                                candidate_pairs[s1_id].add(cand_id)

        return candidate_pairs

    def export_candidate_pairs(
        self,
        output_path: str,
        candidates: Mapping[str, set[str]],
    ) -> None:
        """Export candidates dictionary to candidate_pairs.tsv."""
        write_candidate_pairs_tsv(output_path, dict(candidates))
