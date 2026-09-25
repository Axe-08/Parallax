"""
Parallax Pairwise Feature Extractor
===================================
High-throughput lexical, structural, and address alignment feature extraction
using RapidFuzz C++ bindings.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from tqdm import tqdm

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
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """
        Build a tabular feature matrix for all candidate pairs.
        If ground_truth is provided, appends the binary target column.
        """

        def _build_record_lookup(
            df: pd.DataFrame,
        ) -> dict[str, tuple[str, str, str, str, str, set[str], int]]:
            raw_names = (
                df["business_name"].fillna("").astype(str).tolist()
                if "business_name" in df
                else [""] * len(df)
            )
            soft_names = (
                df["soft_name"].fillna("").astype(str).tolist()
                if "soft_name" in df
                else [n.lower() for n in raw_names]
            )
            translit_names = (
                df["translit_name"].fillna("").astype(str).tolist()
                if "translit_name" in df
                else soft_names
            )
            addrs = (
                df["clean_address"].fillna("").astype(str).tolist()
                if "clean_address" in df
                else [""] * len(df)
            )
            translit_addrs = (
                df["translit_address"].fillna("").astype(str).tolist()
                if "translit_address" in df
                else addrs
            )
            nums = df["numbers"].tolist() if "numbers" in df else [set()] * len(df)
            nulls = (
                df["is_addr_null"].astype(int).tolist() if "is_addr_null" in df else [0] * len(df)
            )
            ids = df["entity_id"].astype(str).tolist()

            return {
                eid: (
                    rn,
                    sn or rn.lower(),
                    tn or sn or rn.lower(),
                    ad,
                    tad or ad,
                    n if isinstance(n, set) else set(n or ()),
                    nl,
                )
                for eid, rn, sn, tn, ad, tad, n, nl in zip(
                    ids,
                    raw_names,
                    soft_names,
                    translit_names,
                    addrs,
                    translit_addrs,
                    nums,
                    nulls,
                    strict=False,
                )
            }

        s1_dict = _build_record_lookup(s1_df)
        target_dict = _build_record_lookup(target_df)

        total_pairs = sum(len(cands) for cands in candidate_pairs.values())
        if total_pairs == 0:
            cols = ["s1_id", "cand_id", *FEATURE_COLUMNS]
            if ground_truth is not None:
                cols.append("target")
            return pd.DataFrame(columns=cols)

        s1_id_list: list[str] = []
        cand_id_list: list[str] = []
        raw_ratio_arr = np.empty(total_pairs, dtype=np.float32)
        soft_ratio_arr = np.empty(total_pairs, dtype=np.float32)
        token_sort_arr = np.empty(total_pairs, dtype=np.float32)
        token_set_arr = np.empty(total_pairs, dtype=np.float32)
        partial_arr = np.empty(total_pairs, dtype=np.float32)
        addr_token_set_arr = np.empty(total_pairs, dtype=np.float32)
        addr_ratio_arr = np.empty(total_pairs, dtype=np.float32)
        num_match_arr = np.empty(total_pairs, dtype=np.float32)
        s1_null_arr = np.empty(total_pairs, dtype=np.int8)
        cand_null_arr = np.empty(total_pairs, dtype=np.int8)
        both_present_arr = np.empty(total_pairs, dtype=np.int8)
        len_diff_arr = np.empty(total_pairs, dtype=np.float32)
        len_ratio_arr = np.empty(total_pairs, dtype=np.float32)
        target_arr = np.empty(total_pairs, dtype=np.int8) if ground_truth is not None else None

        idx = 0
        pbar = tqdm(
            candidate_pairs.items(),
            total=len(candidate_pairs),
            desc="  ⚡ Extracting Pairwise Features",
            unit="query",
            leave=False,
            disable=not show_progress,
        )
        for s1_id, cands in pbar:
            s1_row = s1_dict.get(s1_id)

            if not s1_row:
                continue

            (
                s1_raw_name,
                s1_soft_name,
                _,
                s1_addr,
                _,
                s1_nums,
                s1_null,
            ) = s1_row

            true_matches = ground_truth.get(s1_id, set()) if ground_truth else None

            for cand_id in cands:
                cand_row = target_dict.get(cand_id)
                if not cand_row:
                    continue

                (
                    cand_raw_name,
                    cand_soft_name,
                    cand_translit_name,
                    cand_addr,
                    cand_translit_addr,
                    cand_nums,
                    cand_null,
                ) = cand_row

                # Lexical Name Features (Max across native and transliterated representations)
                raw_ratio_arr[idx] = (
                    max(
                        fuzz.ratio(s1_raw_name, cand_raw_name),
                        fuzz.ratio(s1_raw_name, cand_translit_name),
                    )
                    / 100.0
                )
                soft_ratio_arr[idx] = (
                    max(
                        fuzz.ratio(s1_soft_name, cand_soft_name),
                        fuzz.ratio(s1_soft_name, cand_translit_name),
                    )
                    / 100.0
                )
                token_sort_arr[idx] = (
                    max(
                        fuzz.token_sort_ratio(s1_soft_name, cand_soft_name),
                        fuzz.token_sort_ratio(s1_soft_name, cand_translit_name),
                    )
                    / 100.0
                )
                token_set_arr[idx] = (
                    max(
                        fuzz.token_set_ratio(s1_soft_name, cand_soft_name),
                        fuzz.token_set_ratio(s1_soft_name, cand_translit_name),
                    )
                    / 100.0
                )
                partial_arr[idx] = (
                    max(
                        fuzz.partial_ratio(s1_soft_name, cand_soft_name),
                        fuzz.partial_ratio(s1_soft_name, cand_translit_name),
                    )
                    / 100.0
                )

                # Address Alignment Features
                both_present = 1 if (not s1_null and not cand_null and s1_addr and cand_addr) else 0
                both_present_arr[idx] = both_present
                if both_present:
                    addr_token_set_arr[idx] = (
                        max(
                            fuzz.token_set_ratio(s1_addr, cand_addr),
                            fuzz.token_set_ratio(s1_addr, cand_translit_addr),
                        )
                        / 100.0
                    )
                    addr_ratio_arr[idx] = (
                        max(
                            fuzz.ratio(s1_addr, cand_addr),
                            fuzz.ratio(s1_addr, cand_translit_addr),
                        )
                        / 100.0
                    )
                else:
                    addr_token_set_arr[idx] = 0.0
                    addr_ratio_arr[idx] = 0.0

                # Building Number Alignment
                if not s1_nums or not cand_nums:
                    num_match_arr[idx] = 0.5  # Neutral / unobserved
                elif len(s1_nums.intersection(cand_nums)) > 0:
                    num_match_arr[idx] = 1.0  # Shared building number
                else:
                    num_match_arr[idx] = 0.0  # Explicit contradiction

                len_s1 = len(s1_soft_name)
                len_cand = len(cand_soft_name)
                len_diff_arr[idx] = float(abs(len_s1 - len_cand))
                len_ratio_arr[idx] = float(min(len_s1, len_cand) / max(len_s1, len_cand, 1))

                s1_null_arr[idx] = s1_null
                cand_null_arr[idx] = cand_null
                s1_id_list.append(s1_id)
                cand_id_list.append(cand_id)

                if target_arr is not None:
                    target_arr[idx] = 1 if (true_matches and cand_id in true_matches) else 0

                idx += 1

        data_dict: dict[str, object] = {
            "s1_id": s1_id_list,
            "cand_id": cand_id_list,
            "raw_name_ratio": raw_ratio_arr[:idx],
            "soft_name_ratio": soft_ratio_arr[:idx],
            "token_sort_ratio": token_sort_arr[:idx],
            "token_set_ratio": token_set_arr[:idx],
            "partial_ratio": partial_arr[:idx],
            "addr_token_set_ratio": addr_token_set_arr[:idx],
            "addr_ratio": addr_ratio_arr[:idx],
            "num_match_score": num_match_arr[:idx],
            "is_s1_addr_null": s1_null_arr[:idx],
            "is_cand_addr_null": cand_null_arr[:idx],
            "both_addr_present": both_present_arr[:idx],
            "len_diff_name": len_diff_arr[:idx],
            "len_ratio_name": len_ratio_arr[:idx],
        }
        if target_arr is not None:
            data_dict["target"] = target_arr[:idx]

        return pd.DataFrame(data_dict)
