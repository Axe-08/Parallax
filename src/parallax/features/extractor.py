"""
Parallax Pairwise Feature Extractor
===================================
High-throughput lexical, structural, and address alignment feature extraction
using RapidFuzz C++ bindings.
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd
from rapidfuzz import fuzz

FEATURE_COLUMNS = [
    "raw_name_ratio",
    "soft_name_ratio",
    "token_sort_ratio",
    "token_set_ratio",
    "partial_ratio",
    "addr_token_set_ratio",
    "addr_ratio",
    "num_match_score",
    "is_s1_addr_null",
    "is_cand_addr_null",
    "both_addr_present",
    "len_diff_name",
    "len_ratio_name",
]


class PairwiseFeatureExtractor:
    """Extracts comparative similarity features for candidate pairs."""

    def extract_features_df(
        self,
        candidate_pairs: Mapping[str, set[str]],
        s1_df: pd.DataFrame,
        target_df: pd.DataFrame,
        ground_truth: Mapping[str, set[str]] | None = None,
    ) -> pd.DataFrame:
        """
        Build a tabular feature matrix for all candidate pairs.
        If ground_truth is provided, appends the binary target column.
        """
        s1_dict = s1_df.set_index("entity_id").to_dict("index")
        target_dict = target_df.set_index("entity_id").to_dict("index")

        rows: list[dict[str, object]] = []

        for s1_id, cands in candidate_pairs.items():
            s1_row = s1_dict.get(s1_id)
            if not s1_row:
                continue

            s1_raw_name = str(s1_row.get("business_name", ""))
            s1_soft_name = str(s1_row.get("soft_name", s1_raw_name.lower()))
            s1_addr = str(s1_row.get("clean_address", ""))
            s1_nums = set(s1_row.get("numbers", set()))
            s1_null = int(s1_row.get("is_addr_null", 0))

            true_matches = ground_truth.get(s1_id, set()) if ground_truth else None

            for cand_id in cands:
                cand_row = target_dict.get(cand_id)
                if not cand_row:
                    continue

                cand_raw_name = str(cand_row.get("business_name", ""))
                cand_soft_name = str(cand_row.get("soft_name", cand_raw_name.lower()))
                cand_translit_name = str(cand_row.get("translit_name", cand_soft_name))
                cand_addr = str(cand_row.get("clean_address", ""))
                cand_translit_addr = str(cand_row.get("translit_address", cand_addr))
                cand_nums = set(cand_row.get("numbers", set()))
                cand_null = int(cand_row.get("is_addr_null", 0))

                # Lexical Name Features (Max across native and transliterated representations)
                raw_ratio = max(
                    fuzz.ratio(s1_raw_name, cand_raw_name),
                    fuzz.ratio(s1_raw_name, cand_translit_name),
                ) / 100.0
                soft_ratio = max(
                    fuzz.ratio(s1_soft_name, cand_soft_name),
                    fuzz.ratio(s1_soft_name, cand_translit_name),
                ) / 100.0
                token_sort = max(
                    fuzz.token_sort_ratio(s1_soft_name, cand_soft_name),
                    fuzz.token_sort_ratio(s1_soft_name, cand_translit_name),
                ) / 100.0
                token_set = max(
                    fuzz.token_set_ratio(s1_soft_name, cand_soft_name),
                    fuzz.token_set_ratio(s1_soft_name, cand_translit_name),
                ) / 100.0
                partial = max(
                    fuzz.partial_ratio(s1_soft_name, cand_soft_name),
                    fuzz.partial_ratio(s1_soft_name, cand_translit_name),
                ) / 100.0

                # Address Alignment Features
                both_present = 1 if (not s1_null and not cand_null and s1_addr and cand_addr) else 0
                if both_present:
                    addr_token_set = max(
                        fuzz.token_set_ratio(s1_addr, cand_addr),
                        fuzz.token_set_ratio(s1_addr, cand_translit_addr),
                    ) / 100.0
                    addr_ratio = max(
                        fuzz.ratio(s1_addr, cand_addr),
                        fuzz.ratio(s1_addr, cand_translit_addr),
                    ) / 100.0
                else:
                    addr_token_set = 0.0
                    addr_ratio = 0.0

                # Building Number Alignment
                if not s1_nums or not cand_nums:
                    num_match = 0.5  # Neutral / unobserved
                elif len(s1_nums.intersection(cand_nums)) > 0:
                    num_match = 1.0  # Shared building number
                else:
                    num_match = 0.0  # Explicit contradiction

                len_s1 = len(s1_soft_name)
                len_cand = len(cand_soft_name)
                len_diff = abs(len_s1 - len_cand)
                len_ratio = min(len_s1, len_cand) / max(len_s1, len_cand, 1)

                feature_dict: dict[str, object] = {
                    "s1_id": s1_id,
                    "cand_id": cand_id,
                    "raw_name_ratio": raw_ratio,
                    "soft_name_ratio": soft_ratio,
                    "token_sort_ratio": token_sort,
                    "token_set_ratio": token_set,
                    "partial_ratio": partial,
                    "addr_token_set_ratio": addr_token_set,
                    "addr_ratio": addr_ratio,
                    "num_match_score": num_match,
                    "is_s1_addr_null": s1_null,
                    "is_cand_addr_null": cand_null,
                    "both_addr_present": both_present,
                    "len_diff_name": float(len_diff),
                    "len_ratio_name": float(len_ratio),
                }

                if true_matches is not None:
                    feature_dict["target"] = 1 if cand_id in true_matches else 0

                rows.append(feature_dict)

        return pd.DataFrame(rows)
