"""
Parallax Pairwise Feature Extractor (v2)
========================================
High-throughput lexical, structural, and address alignment feature extraction
using RapidFuzz C++ bindings. Exactly 28 features targeting precision and
fine-grained address / name discrimination.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from tqdm import tqdm

from parallax.preprocessing.transliteration import has_brahmic_script

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
]

_HONORIFIC_PREFIX_REGEX = re.compile(
    r"^(?:dr|prof|smt|shri|mr|ms|m\s+s)\b\s*",
    re.IGNORECASE,
)

_SUFFIX_ORDERED_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Multi-token statutory entities (must match before single tokens)
    (re.compile(r"\bprivate\s+limited$", re.IGNORECASE), "pvt_ltd"),
    (re.compile(r"\bpvt\s+ltd$", re.IGNORECASE), "pvt_ltd"),
    (re.compile(r"\blimited\s+liability\s+company$", re.IGNORECASE), "llc"),
    (re.compile(r"\blimited\s+liability\s+partnership$", re.IGNORECASE), "llp"),
    (re.compile(r"\bpublic\s+limited\s+company$", re.IGNORECASE), "plc"),
    # Single-token statutory entities
    (re.compile(r"\bllc$", re.IGNORECASE), "llc"),
    (re.compile(r"\bllp$", re.IGNORECASE), "llp"),
    (re.compile(r"\bpllc$", re.IGNORECASE), "pllc"),
    (re.compile(r"\binc$", re.IGNORECASE), "inc"),
    (re.compile(r"\bincorporated$", re.IGNORECASE), "inc"),
    (re.compile(r"\bcorp$", re.IGNORECASE), "corp"),
    (re.compile(r"\bcorporation$", re.IGNORECASE), "corp"),
    (re.compile(r"\bltd$", re.IGNORECASE), "ltd"),
    (re.compile(r"\blimited$", re.IGNORECASE), "ltd"),
    (re.compile(r"\bcompany$", re.IGNORECASE), "co"),
    (re.compile(r"\bco$", re.IGNORECASE), "co"),
    (re.compile(r"\bpvt$", re.IGNORECASE), "pvt"),
    (re.compile(r"\bprivate$", re.IGNORECASE), "pvt"),
]


def extract_core_and_suffix(soft_name: str) -> tuple[str, str | None]:
    """
    Extract canonical corporate legal suffix and strip prefixes/suffixes
    to yield the core business brand name.

    Safety:
    - Never strips if the resulting core string has length < 2.
    - Word-boundary matching prevents stripping within brand tokens (e.g. Zinc, Incite).
    """
    if not soft_name:
        return "", None

    # Standardize whitespace and remove any remaining punctuation (such as trailing dots in inc. or co.)
    s = re.sub(r"[^\w\s]", " ", str(soft_name).lower())
    s = " ".join(s.split())
    if not s:
        return "", None

    # Strip leading honorifics / trade prefixes (e.g. "Dr", "M/s")
    s = _HONORIFIC_PREFIX_REGEX.sub("", s).strip()

    matched_suffix = None
    for pattern, canon in _SUFFIX_ORDERED_PATTERNS:
        if pattern.search(s):
            matched_suffix = canon
            stripped = pattern.sub("", s).strip()
            # Safety guarantee: ensure we do not remove meaningful brand tokens
            if len(stripped) >= 2:
                s = stripped
            break

    if not s:
        s = soft_name

    return s, matched_suffix


def check_is_cross_script(
    has_brahmic1: bool,
    text1: str,
    has_brahmic2: bool,
    text2: str,
) -> float:
    """Return 1.0 if pair spans different scripts (e.g. Latin vs Brahmic or different Brahmic scripts)."""
    if has_brahmic1 != has_brahmic2:
        return 1.0
    if has_brahmic1 and has_brahmic2:
        b1 = next((ord(c) // 0x80 for c in text1 if 0x0900 <= ord(c) <= 0x0D7F), None)
        b2 = next((ord(c) // 0x80 for c in text2 if 0x0900 <= ord(c) <= 0x0D7F), None)
        if b1 is not None and b2 is not None and b1 != b2:
            return 1.0
    return 0.0


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
                str,  # core_name
                str | None,  # suffix
                bool,  # has_brahmic
            ],
        ]:
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
            ids = df["entity_id"].astype(str).tolist()

            lookup = {}
            for i, eid in enumerate(ids):
                rn = raw_names[i]
                sn = soft_names[i] or rn.lower()
                tn = translit_names[i] or sn
                ad = addrs[i]
                tad = translit_addrs[i] or ad
                n_set = nums[i] if isinstance(nums[i], set) else set(nums[i] or ())
                nl = nulls[i]
                pn = primary_nums[i] if primary_nums[i] and str(primary_nums[i]).strip() else None
                pc = postal_codes[i] if postal_codes[i] and str(postal_codes[i]).strip() else None
                ca = canon_addrs[i] or ad
                sn_words = sn.split()
                name_toks = set(sn_words)
                first_tok = sn_words[0] if sn_words else ""
                addr_toks = set(ca.split())
                core_name, suffix = extract_core_and_suffix(sn)
                has_brahmic = has_brahmic_script(rn)

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
                    core_name,
                    suffix,
                    has_brahmic,
                )
            return lookup

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

        # Preallocate all 34 feature arrays
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

        # Group 1-4 features (14-28)
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

        # Batch 1 features (29-34)
        translit_boost_arr = np.empty(total_pairs, dtype=np.float32)
        is_cross_script_arr = np.empty(total_pairs, dtype=np.float32)
        translit_name_ratio_arr = np.empty(total_pairs, dtype=np.float32)
        name_core_ratio_arr = np.empty(total_pairs, dtype=np.float32)
        suffix_match_arr = np.empty(total_pairs, dtype=np.float32)
        s1_cand_count_arr = np.empty(total_pairs, dtype=np.float32)

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
                s1_core_name,
                s1_suffix,
                s1_has_brahmic,
            ) = s1_row

            true_matches = ground_truth.get(s1_id, set()) if ground_truth else None
            cand_count_val = float(len(cands))

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
                    cand_core_name,
                    cand_suffix,
                    cand_has_brahmic,
                ) = cand_row

                # --- 1-5: Lexical Name Features ---
                raw_ratio_arr[idx] = (
                    max(
                        fuzz.ratio(s1_raw_name, cand_raw_name),
                        fuzz.ratio(s1_raw_name, cand_translit_name),
                    )
                    / 100.0
                )
                soft_ratio_unnorm = fuzz.ratio(s1_soft_name, cand_soft_name)
                cand_translit_unnorm = (
                    fuzz.ratio(s1_soft_name, cand_translit_name)
                    if cand_has_brahmic
                    else soft_ratio_unnorm
                )
                max_soft_unnorm = max(soft_ratio_unnorm, cand_translit_unnorm)
                soft_ratio_arr[idx] = max_soft_unnorm / 100.0
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

                # --- 14-16: Primary Building Number Discrimination ---
                if s1_primary and cand_primary:
                    if s1_primary == cand_primary:
                        primary_match_arr[idx] = 1.0
                        primary_conflict_arr[idx] = 0.0
                    else:
                        primary_match_arr[idx] = 0.0
                        primary_conflict_arr[idx] = 1.0
                    primary_missing_arr[idx] = 0.0
                else:
                    primary_match_arr[idx] = 0.0
                    primary_conflict_arr[idx] = 0.0
                    primary_missing_arr[idx] = 1.0

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

                # --- 29-31: Cross-Script & Transliteration Signals ---
                translit_boost_arr[idx] = max(
                    0.0, float(max_soft_unnorm - soft_ratio_unnorm) / 100.0
                )
                is_cross_script_arr[idx] = check_is_cross_script(
                    s1_has_brahmic, s1_raw_name, cand_has_brahmic, cand_raw_name
                )
                if not s1_has_brahmic and not cand_has_brahmic:
                    translit_name_ratio_arr[idx] = soft_ratio_unnorm / 100.0
                else:
                    translit_name_ratio_arr[idx] = (
                        fuzz.ratio(s1_translit_name, cand_translit_name) / 100.0
                    )

                # --- 32-33: Corporate Entity Designators & Core Name ---
                if s1_core_name == s1_soft_name and cand_core_name == cand_soft_name:
                    name_core_ratio_arr[idx] = soft_ratio_unnorm / 100.0
                else:
                    name_core_ratio_arr[idx] = (
                        fuzz.ratio(s1_core_name, cand_core_name) / 100.0
                    )

                if s1_suffix and cand_suffix:
                    if s1_suffix == cand_suffix:
                        suffix_match_arr[idx] = 1.0
                    elif {s1_suffix, cand_suffix} == {"pvt_ltd", "ltd"}:
                        suffix_match_arr[idx] = 0.5
                    else:
                        suffix_match_arr[idx] = 0.0
                else:
                    suffix_match_arr[idx] = 0.5

                # --- 34: Candidate Pool Context ---
                s1_cand_count_arr[idx] = cand_count_val

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
        }
        if target_arr is not None:
            data_dict["target"] = target_arr[:idx]

        return pd.DataFrame(data_dict)
