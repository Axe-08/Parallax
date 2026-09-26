"""
Parallax Pairwise Feature Extractor (v3)
========================================
High-throughput lexical, structural, and address alignment feature extraction
using RapidFuzz C++ bindings. Exactly 49 features targeting precision,
fine-grained address / name discrimination, transliteration recovery,
and entity-level context.
"""

from __future__ import annotations

import os
from collections.abc import Collection, Mapping
from typing import Any

import jellyfish
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein
from tqdm import tqdm


def _normalize_num_set(val: Any) -> set[str]:
    """Safely convert sets, lists, tuples, and numpy ndarrays into a set of strings."""
    if isinstance(val, set):
        return {str(x) for x in val}
    if isinstance(val, (list, tuple, np.ndarray)):
        return {str(x) for x in val}
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return set()
    try:
        return {str(x) for x in val}
    except (TypeError, ValueError):
        return set()


_LEGAL_SUFFIXES: dict[str, str] = {
    "pvt": "ltd",
    "private": "ltd",
    "ltd": "ltd",
    "limited": "ltd",
    "llc": "llc",
    "inc": "inc",
    "incorporated": "inc",
    "corp": "corp",
    "corporation": "corp",
    "co": "co",
    "company": "co",
    "llp": "llp",
    "gmbh": "gmbh",
    "sa": "sa",
    "plc": "plc",
    "industries": "industries",
    "services": "services",
    "enterprises": "enterprises",
    "solutions": "solutions",
    "technologies": "technologies",
    "foundation": "foundation",
    "trust": "trust",
    "associates": "associates",
    "group": "group",
}

_PREFIX_TOKENS: set[str] = {
    "dr",
    "smt",
    "ms",
    "mr",
    "mrs",
    "shree",
    "shri",
    "the",
    "www",
    "m/s",
}


def extract_core_and_suffix(soft_name: str) -> tuple[str, str | None]:
    """Extract core business name without prefixes/legal suffixes and determine suffix class."""
    toks = soft_name.split()
    if not toks:
        return "", None

    start_idx = 0
    while start_idx < len(toks) and (
        toks[start_idx] in _PREFIX_TOKENS
        or toks[start_idx].startswith("www.")
        or toks[start_idx].startswith("@")
    ):
        start_idx += 1

    toks = toks[start_idx:]
    if not toks:
        return soft_name, None

    suffix_class: str | None = None
    for _ in range(2):
        if toks and toks[-1] in _LEGAL_SUFFIXES:
            s_val = _LEGAL_SUFFIXES[toks[-1]]
            if suffix_class is None:
                suffix_class = s_val
            toks.pop()

    core = " ".join(toks).strip() or soft_name
    return core, suffix_class


FEATURE_COLUMNS = [
    # --- Retained Baseline Features (1-13) ---
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
    # --- Group 1: Building & Street Number Discrimination (14-18) ---
    "primary_num_match",
    "primary_num_conflict",
    "primary_num_missing",
    "num_jaccard",
    "num_conflict_count",
    # --- Group 2: Postal / PIN Code Alignment (19-21) ---
    "postal_match",
    "postal_conflict",
    "postal_missing",
    # --- Group 3: Lexical Prefix & Token Overlap (22-26) ---
    "jaro_winkler_soft",
    "jaro_winkler_raw",
    "token_jaccard_name",
    "token_overlap_name",
    "first_token_match",
    # --- Group 4: Canonical Address Similarity (27-28) ---
    "canon_addr_ratio",
    "token_jaccard_addr",
    # --- Group 5: Building Number Distance (29) ---
    "primary_num_distance",
    # --- Group 6: City Alignment & Conflict (30-31) ---
    "city_match",
    "city_conflict",
    # --- Group 7: Missing Address Interactions (32-35) ---
    "name_ratio_null_cand_addr",
    "name_ratio_null_either_addr",
    "high_conf_name_no_addr",
    "token_set_null_cand_addr",
    # --- Group 8: Transliteration & Cross-Script Signals (36-38) ---
    "translit_boost_name",
    "is_cross_script",
    "translit_name_ratio",
    # --- Group 9: Name Core & Legal Suffix Analysis (39-41) ---
    "name_core_ratio",
    "suffix_match",
    "name_word_count_diff",
    # --- Group 10: Advanced Edit Geometry & Overlap (42-45) ---
    "name_edit_distance",
    "common_prefix_len",
    "containment_ratio_name",
    "addr_token_overlap",
    # --- Group 11: Phonetic & Context Signals (46-49) ---
    "phonetic_name_match",
    "s1_candidate_count",
    "country_match",
    "blocking_sim_score",
]


class PairwiseFeatureExtractor:
    """Extracts comparative similarity features for candidate pairs."""

    def extract_features_df(
        self,
        candidate_pairs: Mapping[str, Collection[str] | dict[str, float]],
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
            needed_ids: Collection[str] | None = None,
        ) -> dict[
            str,
            tuple[
                str,  # raw_name
                str,  # soft_name
                str,  # translit_name
                str,  # addr
                str,  # translit_addr
                set[str],  # nums
                int,  # is_null
                str | None,  # primary_num
                str | None,  # postal_code
                str,  # canon_addr
                set[str],  # name_toks
                str,  # first_tok
                set[str],  # addr_toks
                str | None,  # city_tok
                str,  # core_name
                str | None,  # suffix_class
                str,  # phonetic_code
                str,  # country
                float,  # is_cross_script
            ],
        ]:
            if needed_ids is not None and "entity_id" in df:
                id_ser = df["entity_id"].astype(str)
                mask = id_ser.isin(needed_ids)
                if not mask.all():
                    df = df[mask].reset_index(drop=True)

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
            primary_nums = (
                df["primary_number"].tolist() if "primary_number" in df else [None] * len(df)
            )
            postal_codes = df["postal_code"].tolist() if "postal_code" in df else [None] * len(df)
            canon_addrs = (
                df["canon_address"].fillna("").astype(str).tolist()
                if "canon_address" in df
                else addrs
            )
            if "city_token" in df:
                city_tokens = df["city_token"].tolist()
            elif "business_address" in df:
                from parallax.preprocessing.normalizer import extract_city_token

                city_tokens = df["business_address"].apply(extract_city_token).tolist()
            else:
                city_tokens = [None] * len(df)

            country_list = (
                df["country"].fillna("").astype(str).tolist() if "country" in df else [""] * len(df)
            )

            ids = df["entity_id"].astype(str).tolist()

            lookup = {}
            for i, eid in enumerate(ids):
                rn = raw_names[i]
                sn = soft_names[i] or rn.lower()
                tn = translit_names[i] or sn
                ad = addrs[i]
                tad = translit_addrs[i] or ad
                n_set = _normalize_num_set(nums[i])
                nl = nulls[i]
                pn = (
                    str(primary_nums[i]).strip()
                    if pd.notna(primary_nums[i])
                    and str(primary_nums[i]).strip().lower() not in ("", "none", "nan")
                    else None
                )
                pc = (
                    str(postal_codes[i]).strip()
                    if pd.notna(postal_codes[i])
                    and str(postal_codes[i]).strip().lower() not in ("", "none", "nan")
                    else None
                )
                ca = canon_addrs[i] or ad
                sn_words = sn.split()
                name_toks = set(sn_words)
                first_tok = sn_words[0] if sn_words else ""
                addr_toks = set(ca.split())
                ct = (
                    str(city_tokens[i]).strip()
                    if pd.notna(city_tokens[i])
                    and str(city_tokens[i]).strip().lower() not in ("", "none", "nan")
                    else None
                )
                core_name, suffix_class = extract_core_and_suffix(sn)
                phonetic_code = jellyfish.metaphone(first_tok) if first_tok else ""
                country = country_list[i]
                is_cross_script = 1.0 if any(ord(c) > 127 for c in rn) else 0.0

                lookup[eid] = (
                    rn,
                    sn,
                    tn,
                    ad,
                    tad,
                    n_set,
                    nl,
                    pn,
                    pc,
                    ca,
                    name_toks,
                    first_tok,
                    addr_toks,
                    ct,
                    core_name,
                    suffix_class,
                    phonetic_code,
                    country,
                    is_cross_script,
                )
            return lookup

        needed_s1 = set(candidate_pairs.keys())
        needed_target = {cand for cands in candidate_pairs.values() for cand in cands}
        s1_dict = _build_record_lookup(s1_df, needed_ids=needed_s1)
        target_dict = _build_record_lookup(target_df, needed_ids=needed_target)

        total_pairs = sum(len(cands) for cands in candidate_pairs.values())
        if total_pairs == 0:
            cols = ["s1_id", "cand_id", *FEATURE_COLUMNS]
            if ground_truth is not None:
                cols.append("target")
            return pd.DataFrame(columns=cols)

        s1_id_list: list[str] = []
        cand_id_list: list[str] = []

        # Baseline Features (1-13)
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

        # Groups 1-4: Building, Postal, Lexical, Canonical Addr (14-28)
        primary_match_arr = np.empty(total_pairs, dtype=np.float32)
        primary_conflict_arr = np.empty(total_pairs, dtype=np.float32)
        primary_missing_arr = np.empty(total_pairs, dtype=np.float32)
        num_jaccard_arr = np.empty(total_pairs, dtype=np.float32)
        num_conflict_count_arr = np.empty(total_pairs, dtype=np.float32)
        postal_match_arr = np.empty(total_pairs, dtype=np.float32)
        postal_conflict_arr = np.empty(total_pairs, dtype=np.float32)
        postal_missing_arr = np.empty(total_pairs, dtype=np.float32)
        jaro_winkler_soft_arr = np.empty(total_pairs, dtype=np.float32)
        jaro_winkler_raw_arr = np.empty(total_pairs, dtype=np.float32)
        token_jaccard_name_arr = np.empty(total_pairs, dtype=np.float32)
        token_overlap_name_arr = np.empty(total_pairs, dtype=np.float32)
        first_token_match_arr = np.empty(total_pairs, dtype=np.float32)
        canon_addr_ratio_arr = np.empty(total_pairs, dtype=np.float32)
        token_jaccard_addr_arr = np.empty(total_pairs, dtype=np.float32)

        # Groups 5-7: Primary Distance, City, Missing Address Interactions (29-35)
        primary_num_distance_arr = np.empty(total_pairs, dtype=np.float32)
        city_match_arr = np.empty(total_pairs, dtype=np.float32)
        city_conflict_arr = np.empty(total_pairs, dtype=np.float32)
        name_ratio_null_cand_arr = np.empty(total_pairs, dtype=np.float32)
        name_ratio_null_either_arr = np.empty(total_pairs, dtype=np.float32)
        high_conf_name_no_addr_arr = np.empty(total_pairs, dtype=np.float32)
        token_set_null_cand_arr = np.empty(total_pairs, dtype=np.float32)

        # Groups 8-11: Transliteration, Core/Suffix, Edit Geometry, Context Signals (36-49)
        translit_boost_name_arr = np.empty(total_pairs, dtype=np.float32)
        is_cross_script_arr = np.empty(total_pairs, dtype=np.float32)
        translit_name_ratio_arr = np.empty(total_pairs, dtype=np.float32)
        name_core_ratio_arr = np.empty(total_pairs, dtype=np.float32)
        suffix_match_arr = np.empty(total_pairs, dtype=np.float32)
        name_word_count_diff_arr = np.empty(total_pairs, dtype=np.float32)
        name_edit_dist_arr = np.empty(total_pairs, dtype=np.float32)
        common_prefix_len_arr = np.empty(total_pairs, dtype=np.float32)
        containment_ratio_arr = np.empty(total_pairs, dtype=np.float32)
        addr_token_overlap_arr = np.empty(total_pairs, dtype=np.float32)
        phonetic_match_arr = np.empty(total_pairs, dtype=np.float32)
        cand_count_arr = np.empty(total_pairs, dtype=np.float32)
        country_match_arr = np.empty(total_pairs, dtype=np.float32)
        blocking_sim_arr = np.empty(total_pairs, dtype=np.float32)

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
                s1_translit_name,
                s1_addr,
                _,
                s1_nums,
                s1_null,
                s1_primary,
                s1_postal,
                s1_canon_addr,
                s1_name_toks,
                s1_first_tok,
                s1_addr_toks,
                s1_city,
                s1_core,
                s1_suffix,
                s1_phonetic,
                s1_country,
                _,
            ) = s1_row

            true_matches = ground_truth.get(s1_id, set()) if ground_truth else None
            num_cands_f = float(len(cands))
            is_cands_dict = isinstance(cands, dict)

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
                    cand_primary,
                    cand_postal,
                    cand_canon_addr,
                    cand_name_toks,
                    cand_first_tok,
                    cand_addr_toks,
                    cand_city,
                    cand_core,
                    cand_suffix,
                    cand_phonetic,
                    cand_country,
                    cand_is_cross_script,
                ) = cand_row

                # --- 1-5: Lexical Name Features ---
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

                # --- 6-7: Baseline Address Alignment ---
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

                # --- 8: Baseline Number Match Score ---
                if not s1_nums or not cand_nums:
                    num_match_arr[idx] = 0.5
                elif len(s1_nums.intersection(cand_nums)) > 0:
                    num_match_arr[idx] = 1.0
                else:
                    num_match_arr[idx] = 0.0

                # --- 9-13: Length and Null Indicators ---
                len_s1 = len(s1_soft_name)
                len_cand = len(cand_soft_name)
                len_diff_arr[idx] = float(abs(len_s1 - len_cand))
                len_ratio_arr[idx] = float(min(len_s1, len_cand) / max(len_s1, len_cand, 1))
                s1_null_arr[idx] = s1_null
                cand_null_arr[idx] = cand_null

                # --- 14-16 & 29: Primary Building Number Discrimination & Distance ---
                if s1_primary and cand_primary:
                    s1_p_str = str(s1_primary)
                    cand_p_str = str(cand_primary)
                    if s1_p_str == cand_p_str:
                        primary_match_arr[idx] = 1.0
                        primary_conflict_arr[idx] = 0.0
                        primary_num_distance_arr[idx] = 0.0
                    else:
                        primary_match_arr[idx] = 0.0
                        primary_conflict_arr[idx] = 1.0
                        str_dist = 1.0 - (fuzz.ratio(s1_p_str, cand_p_str) / 100.0)
                        d1 = "".join(c for c in s1_p_str if c.isdigit())
                        d2 = "".join(c for c in cand_p_str if c.isdigit())
                        if d1 and d2:
                            try:
                                n1, n2 = int(d1), int(d2)
                                num_dist = min(abs(n1 - n2) / max(n1, n2, 1), 1.0)
                                primary_num_distance_arr[idx] = float(max(num_dist, str_dist))
                            except ValueError:
                                primary_num_distance_arr[idx] = float(str_dist)
                        else:
                            primary_num_distance_arr[idx] = float(str_dist)
                    primary_missing_arr[idx] = 0.0
                else:
                    primary_match_arr[idx] = 0.0
                    primary_conflict_arr[idx] = 0.0
                    primary_missing_arr[idx] = 1.0
                    primary_num_distance_arr[idx] = 0.0

                # --- 17-18: Number Jaccard & Conflict Count ---
                if not s1_nums and not cand_nums:
                    num_jaccard_arr[idx] = 0.5
                    num_conflict_count_arr[idx] = 0.0
                elif not s1_nums or not cand_nums:
                    num_jaccard_arr[idx] = 0.5
                    num_conflict_count_arr[idx] = float(len(s1_nums or cand_nums))
                else:
                    u_nums = s1_nums.union(cand_nums)
                    inter_nums = s1_nums.intersection(cand_nums)
                    num_jaccard_arr[idx] = float(len(inter_nums) / len(u_nums)) if u_nums else 0.5
                    num_conflict_count_arr[idx] = float(
                        len(s1_nums.symmetric_difference(cand_nums))
                    )

                # --- 19-21: Postal / PIN Code Alignment ---
                if s1_postal and cand_postal:
                    if s1_postal == cand_postal:
                        postal_match_arr[idx] = 1.0
                        postal_conflict_arr[idx] = 0.0
                    else:
                        postal_match_arr[idx] = 0.0
                        postal_conflict_arr[idx] = 1.0
                    postal_missing_arr[idx] = 0.0
                else:
                    postal_match_arr[idx] = 0.0
                    postal_conflict_arr[idx] = 0.0
                    postal_missing_arr[idx] = 1.0

                # --- 22-23: Jaro-Winkler Similarities ---
                jaro_winkler_soft_arr[idx] = float(
                    max(
                        JaroWinkler.similarity(s1_soft_name, cand_soft_name),
                        JaroWinkler.similarity(s1_soft_name, cand_translit_name),
                    )
                )
                jaro_winkler_raw_arr[idx] = float(
                    max(
                        JaroWinkler.similarity(s1_raw_name, cand_raw_name),
                        JaroWinkler.similarity(s1_raw_name, cand_translit_name),
                    )
                )

                # --- 24-26: Name Token Jaccard, Overlap & First Token Match ---
                if s1_name_toks and cand_name_toks:
                    inter_t = len(s1_name_toks.intersection(cand_name_toks))
                    union_t = len(s1_name_toks.union(cand_name_toks))
                    min_t = min(len(s1_name_toks), len(cand_name_toks))
                    token_jaccard_name_arr[idx] = float(inter_t / union_t) if union_t > 0 else 0.0
                    token_overlap_name_arr[idx] = float(inter_t / min_t) if min_t > 0 else 0.0
                else:
                    token_jaccard_name_arr[idx] = 0.0
                    token_overlap_name_arr[idx] = 0.0

                first_token_match_arr[idx] = (
                    1.0
                    if (s1_first_tok and cand_first_tok and s1_first_tok == cand_first_tok)
                    else 0.0
                )

                # --- 27-28: Canonical Address Ratio & Jaccard ---
                if both_present:
                    canon_addr_ratio_arr[idx] = fuzz.ratio(s1_canon_addr, cand_canon_addr) / 100.0
                    if s1_addr_toks and cand_addr_toks:
                        a_inter = len(s1_addr_toks.intersection(cand_addr_toks))
                        a_union = len(s1_addr_toks.union(cand_addr_toks))
                        token_jaccard_addr_arr[idx] = (
                            float(a_inter / a_union) if a_union > 0 else 0.0
                        )
                    else:
                        token_jaccard_addr_arr[idx] = 0.0
                else:
                    canon_addr_ratio_arr[idx] = 0.0
                    token_jaccard_addr_arr[idx] = 0.0

                # --- 30-31: City Alignment & Conflict ---
                if s1_city and cand_city:
                    if s1_city == cand_city:
                        city_match_arr[idx] = 1.0
                        city_conflict_arr[idx] = 0.0
                    else:
                        s1_c_toks = set(s1_city.split())
                        cand_c_toks = set(cand_city.split())
                        if (
                            s1_c_toks.intersection(cand_c_toks)
                            or fuzz.ratio(s1_city, cand_city) >= 80
                        ):
                            city_match_arr[idx] = 1.0
                            city_conflict_arr[idx] = 0.0
                        else:
                            city_match_arr[idx] = 0.0
                            city_conflict_arr[idx] = 1.0
                else:
                    city_match_arr[idx] = 0.0
                    city_conflict_arr[idx] = 0.0

                # --- 32-35: Missing Address Interactions ---
                s_name_ratio = soft_ratio_arr[idx]
                t_set_ratio = token_set_arr[idx]
                c_null_f = float(cand_null)
                b_pres_f = float(both_present)

                name_ratio_null_cand_arr[idx] = s_name_ratio * c_null_f
                name_ratio_null_either_arr[idx] = s_name_ratio * (1.0 - b_pres_f)
                high_conf_name_no_addr_arr[idx] = (
                    1.0 if (s_name_ratio >= 0.85 and both_present == 0) else 0.0
                )
                token_set_null_cand_arr[idx] = t_set_ratio * c_null_f

                # --- 36-38: Transliteration & Cross-Script Signals ---
                raw_soft_ratio = fuzz.ratio(s1_soft_name, cand_soft_name) / 100.0
                translit_boost_name_arr[idx] = max(0.0, s_name_ratio - raw_soft_ratio)
                is_cross_script_arr[idx] = cand_is_cross_script
                translit_name_ratio_arr[idx] = (
                    fuzz.ratio(s1_translit_name, cand_translit_name) / 100.0
                )

                # --- 39-41: Name Core & Legal Suffix Analysis ---
                name_core_ratio_arr[idx] = fuzz.ratio(s1_core, cand_core) / 100.0
                if s1_suffix and cand_suffix:
                    suffix_match_arr[idx] = 1.0 if s1_suffix == cand_suffix else 0.0
                else:
                    suffix_match_arr[idx] = 0.5
                name_word_count_diff_arr[idx] = float(abs(len(s1_name_toks) - len(cand_name_toks)))

                # --- 42-45: Advanced Edit Geometry & Overlap ---
                name_edit_dist_arr[idx] = float(Levenshtein.distance(s1_soft_name, cand_soft_name))
                common_prefix_len_arr[idx] = float(
                    len(os.path.commonprefix([s1_soft_name, cand_soft_name]))
                )
                containment_ratio_arr[idx] = (
                    1.0
                    if (s1_soft_name in cand_soft_name or cand_soft_name in s1_soft_name)
                    and min(len_s1, len_cand) >= 3
                    else 0.0
                )
                if both_present:
                    min_a = min(len(s1_addr_toks), len(cand_addr_toks))
                    addr_token_overlap_arr[idx] = (
                        float(len(s1_addr_toks.intersection(cand_addr_toks)) / min_a)
                        if min_a > 0
                        else 0.0
                    )
                else:
                    addr_token_overlap_arr[idx] = 0.0

                # --- 46-49: Phonetic & Context Signals ---
                phonetic_match_arr[idx] = (
                    1.0 if (s1_phonetic and cand_phonetic and s1_phonetic == cand_phonetic) else 0.0
                )
                cand_count_arr[idx] = num_cands_f
                country_match_arr[idx] = (
                    1.0 if (s1_country and cand_country and s1_country == cand_country) else 0.0
                )
                blocking_sim_arr[idx] = float(cands[cand_id]) if is_cands_dict else 0.0

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
            "primary_num_match": primary_match_arr[:idx],
            "primary_num_conflict": primary_conflict_arr[:idx],
            "primary_num_missing": primary_missing_arr[:idx],
            "num_jaccard": num_jaccard_arr[:idx],
            "num_conflict_count": num_conflict_count_arr[:idx],
            "postal_match": postal_match_arr[:idx],
            "postal_conflict": postal_conflict_arr[:idx],
            "postal_missing": postal_missing_arr[:idx],
            "jaro_winkler_soft": jaro_winkler_soft_arr[:idx],
            "jaro_winkler_raw": jaro_winkler_raw_arr[:idx],
            "token_jaccard_name": token_jaccard_name_arr[:idx],
            "token_overlap_name": token_overlap_name_arr[:idx],
            "first_token_match": first_token_match_arr[:idx],
            "canon_addr_ratio": canon_addr_ratio_arr[:idx],
            "token_jaccard_addr": token_jaccard_addr_arr[:idx],
            "primary_num_distance": primary_num_distance_arr[:idx],
            "city_match": city_match_arr[:idx],
            "city_conflict": city_conflict_arr[:idx],
            "name_ratio_null_cand_addr": name_ratio_null_cand_arr[:idx],
            "name_ratio_null_either_addr": name_ratio_null_either_arr[:idx],
            "high_conf_name_no_addr": high_conf_name_no_addr_arr[:idx],
            "token_set_null_cand_addr": token_set_null_cand_arr[:idx],
            "translit_boost_name": translit_boost_name_arr[:idx],
            "is_cross_script": is_cross_script_arr[:idx],
            "translit_name_ratio": translit_name_ratio_arr[:idx],
            "name_core_ratio": name_core_ratio_arr[:idx],
            "suffix_match": suffix_match_arr[:idx],
            "name_word_count_diff": name_word_count_diff_arr[:idx],
            "name_edit_distance": name_edit_dist_arr[:idx],
            "common_prefix_len": common_prefix_len_arr[:idx],
            "containment_ratio_name": containment_ratio_arr[:idx],
            "addr_token_overlap": addr_token_overlap_arr[:idx],
            "phonetic_name_match": phonetic_match_arr[:idx],
            "s1_candidate_count": cand_count_arr[:idx],
            "country_match": country_match_arr[:idx],
            "blocking_sim_score": blocking_sim_arr[:idx],
        }
        if target_arr is not None:
            data_dict["target"] = target_arr[:idx]

        return pd.DataFrame(data_dict)
