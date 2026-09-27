"""
Extract Decoupled Concatenation, Domain, and Handle Features with Concordance Interactions.

Implements rigorous, word-boundary-aware normalizations for:
1. canonical_concat_stem
2. domain_stem
3. handle_stem

Generates continuous interaction features with address ratio and postal match,
strictly avoiding unvalidated hard-coded gates.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

# ==============================================================================
# 1. Exact Normalization Regexes & Rules
# ==============================================================================

# Word-boundary legal corporate suffixes.
# Note: 'co' requires boundary checks to prevent mutilating 'Costco' or 'Co-operative'
LEGAL_SUFFIX_REGEX = re.compile(
    r"(?<![a-zA-Z0-9])(?:pvt|ltd|inc|llc|limited|corp|corporation|gmbh|sa|bv|(?:&|and\s+)?co(?![a-zA-Z0-9\-]))(?=[.\s,]|$)",
    re.IGNORECASE,
)

TLD_PATTERN = re.compile(
    r"(?:\.(?:com|org|net|in|co\.in|fr|io|biz|gov|edu|info|tv|me|us|uk|de|eu|co|ai))(?=[/?#\s]|$)",
    re.IGNORECASE,
)

URL_PREFIX_REGEX = re.compile(r"^(https?://)?(www\.)?", re.IGNORECASE)
HANDLE_PATTERN = re.compile(r"(?:^|\s)@([a-zA-Z0-9_]{2,})")


def extract_canonical_concat_stem(raw_name: str) -> str:
    """
    Extracts canonical concatenated stem.
    - Strips legal corporate suffixes strictly at word boundaries.
    - Strips all non-alphanumeric characters.
    - Lowercases entire token string.
    """
    s = str(raw_name).strip()
    s = LEGAL_SUFFIX_REGEX.sub("", s)
    return re.sub(r"[^a-zA-Z0-9]", "", s).lower()


def extract_domain_stem(raw_name: str) -> tuple[str, bool]:
    """
    Extracts host domain stem if raw_name represents a domain or URL.
    - Detects explicit schemes (http, https, www) or recognized TLDs.
    - Strips protocol, www, trailing URL paths/queries.
    - Strips trailing TLD suffix.
    - Strips corporate legal suffixes.
    - Returns (stem, is_domain).
    """
    s = str(raw_name).strip()
    if not (s.lower().startswith(("http://", "https://", "www.")) or TLD_PATTERN.search(s)):
        return "", False

    s = URL_PREFIX_REGEX.sub("", s)
    s = re.split(r"[/?#\s]", s, maxsplit=1)[0]
    s = re.sub(
        r"(?:\.(?:com|org|net|in|co\.in|fr|io|biz|gov|edu|info|tv|me|us|uk|de|eu|co|ai))+$",
        "",
        s,
        flags=re.IGNORECASE,
    )
    s = LEGAL_SUFFIX_REGEX.sub("", s)
    stem = re.sub(r"[^a-zA-Z0-9]", "", s).lower()
    return stem, bool(stem)


def extract_handle_stem(raw_name: str) -> tuple[str, bool]:
    """
    Extracts social handle stem if raw_name represents an @handle.
    - Detects leading @handle patterns.
    - Strips @ and trailing legal suffixes.
    - Returns (stem, is_handle).
    """
    s = str(raw_name).strip()
    m = HANDLE_PATTERN.search(s)
    if m:
        h = m.group(1)
        h = LEGAL_SUFFIX_REGEX.sub("", h)
        stem = re.sub(r"[^a-zA-Z0-9]", "", h).lower()
        return stem, bool(stem)
    return "", False


# ==============================================================================
# 2. Adversarial Unit Test Suite
# ==============================================================================


def run_unit_tests() -> None:
    """Executes unit tests covering all representative adversarial edge cases."""
    print("Running normalization & stem unit tests...")

    # Case 1: Costco (must NOT strip 'co' from Costco)
    costco_stem = extract_canonical_concat_stem("Costco")
    assert costco_stem == "costco", f"Expected 'costco', got '{costco_stem}'"

    # Case 2: Co-operative (must NOT strip 'Co-' prefix)
    coop_stem = extract_canonical_concat_stem("Co-operative")
    assert coop_stem == "cooperative", f"Expected 'cooperative', got '{coop_stem}'"

    # Case 3: ABC Co (must strip standalone 'Co')
    abc_co_stem = extract_canonical_concat_stem("ABC Co")
    assert abc_co_stem == "abc", f"Expected 'abc', got '{abc_co_stem}'"

    # Case 4: ABC Corporation (must strip 'Corporation')
    abc_corp_stem = extract_canonical_concat_stem("ABC Corporation")
    assert abc_corp_stem == "abc", f"Expected 'abc', got '{abc_corp_stem}'"
    assert fuzz.ratio(abc_co_stem, abc_corp_stem) == 100

    # Case 5: Domain detection on example.com
    ex_stem, is_ex_dom = extract_domain_stem("example.com")
    assert is_ex_dom is True, "Expected example.com to be recognized as domain"
    assert ex_stem == "example", f"Expected 'example', got '{ex_stem}'"

    # Case 6: Handle detection on @example
    h_stem, is_h = extract_handle_stem("@example")
    assert is_h is True, "Expected @example to be recognized as handle"
    assert h_stem == "example", f"Expected 'example', got '{h_stem}'"

    # Case 7: Numeric/address string
    num_stem = extract_canonical_concat_stem("123 Main St #4B")
    assert num_stem == "123mainst4b", f"Expected '123mainst4b', got '{num_stem}'"

    # Case 8: FF Enterprise vs ffenterprise.com
    ff_s1 = extract_canonical_concat_stem("FF Enterprise")
    ff_dom, _ = extract_domain_stem("ffenterprise.com")
    assert ff_s1 == "ffenterprise", f"Expected 'ffenterprise', got '{ff_s1}'"
    assert ff_dom == "ffenterprise", f"Expected 'ffenterprise', got '{ff_dom}'"
    assert fuzz.ratio(ff_s1, ff_dom) == 100

    # Case 9: Truthful scoring on ffenterpriseprivate.com
    ff_priv_dom, _ = extract_domain_stem("ffenterpriseprivate.com")
    assert ff_priv_dom == "ffenterpriseprivate"
    actual_sim = fuzz.ratio(ff_s1, ff_priv_dom) / 100.0
    assert abs(actual_sim - 0.7742) < 0.001, f"Expected 0.7742, got {actual_sim}"

    print("All unit tests PASSED successfully!")


# ==============================================================================
# 3. Pairwise Feature Extraction
# ==============================================================================


def extract_features_for_candidates(
    features_df: pd.DataFrame,
    s1_names: dict[str, str],
    target_names: dict[str, str],
) -> pd.DataFrame:
    """
    Computes decoupled representation features and continuous concordance interactions
    for all candidate pairs.
    """
    n_pairs = len(features_df)
    print(f"Extracting decoupled features for {n_pairs:,} candidate pairs...")

    # Pre-extract unique stems to optimize computation
    unique_s1_ids = features_df["s1_id"].astype(str).unique()
    unique_cand_ids = features_df["cand_id"].astype(str).unique()

    print(f"Precomputing representations for {len(unique_s1_ids):,} S1 entities...")
    s1_canonical = {
        sid: extract_canonical_concat_stem(s1_names.get(sid, "")) for sid in unique_s1_ids
    }

    print(f"Precomputing representations for {len(unique_cand_ids):,} Target entities...")
    cand_canonical: dict[str, str] = {}
    cand_domain: dict[str, tuple[str, bool]] = {}
    cand_handle: dict[str, tuple[str, bool]] = {}

    for cid in unique_cand_ids:
        raw_t = target_names.get(cid, "")
        cand_canonical[cid] = extract_canonical_concat_stem(raw_t)
        cand_domain[cid] = extract_domain_stem(raw_t)
        cand_handle[cid] = extract_handle_stem(raw_t)

    # Compute pairwise similarities
    print("Computing vectorised similarity matrices...")
    concat_sims = np.zeros(n_pairs, dtype=np.float32)
    domain_sims = np.zeros(n_pairs, dtype=np.float32)
    has_domain = np.zeros(n_pairs, dtype=np.float32)
    has_handle = np.zeros(n_pairs, dtype=np.float32)

    s1_id_list = features_df["s1_id"].astype(str).tolist()
    cand_id_list = features_df["cand_id"].astype(str).tolist()

    for idx, (sid, cid) in enumerate(zip(s1_id_list, cand_id_list, strict=False)):
        s1_stem = s1_canonical.get(sid, "")
        t_concat_stem = cand_canonical.get(cid, "")
        t_dom_stem, is_dom = cand_domain.get(cid, ("", False))
        _, is_hnd = cand_handle.get(cid, ("", False))

        # 1. General concatenated token similarity
        if s1_stem and t_concat_stem:
            concat_sims[idx] = fuzz.ratio(s1_stem, t_concat_stem) / 100.0

        # 2. Domain-specific similarity (active only if target has valid domain)
        if is_dom:
            has_domain[idx] = 1.0
            if s1_stem and t_dom_stem:
                domain_sims[idx] = fuzz.ratio(s1_stem, t_dom_stem) / 100.0

        # 3. Handle indicator
        if is_hnd:
            has_handle[idx] = 1.0

    # Retrieve existing address and postal features
    addr_ratio = features_df["addr_ratio"].fillna(0.0).to_numpy(dtype=np.float32)
    postal_match = features_df["postal_match"].fillna(0.0).to_numpy(dtype=np.float32)

    # Construct continuous interaction features (no rigid hard gates)
    concat_addr_product = concat_sims * addr_ratio
    concat_postal_product = concat_sims * postal_match

    domain_addr_concordance = domain_sims * addr_ratio
    domain_postal_concordance = domain_sims * postal_match

    # Build augmented DataFrame
    res_df = features_df.copy()
    res_df["concat_stem_similarity"] = concat_sims
    res_df["concat_addr_product"] = concat_addr_product
    res_df["concat_postal_product"] = concat_postal_product
    res_df["domain_stem_similarity"] = domain_sims
    res_df["domain_addr_concordance"] = domain_addr_concordance
    res_df["domain_postal_concordance"] = domain_postal_concordance
    res_df["has_domain_target"] = has_domain
    res_df["has_handle_target"] = has_handle

    return res_df


# ==============================================================================
# 4. Main CLI
# ==============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Concatenation and Concordance Features")
    parser.add_argument(
        "--features-input",
        type=Path,
        default=Path("baseline_artifacts/features_sample.parquet"),
        help="Path to baseline 213K feature matrix",
    )
    parser.add_argument(
        "--output-parquet",
        type=Path,
        default=Path(
            "experiments/neural_text/caches/augmented_features_with_concordance_5k.parquet"
        ),
        help="Path to save augmented parquet",
    )
    parser.add_argument(
        "--s1-path",
        type=Path,
        default=Path("data/medium_split_200k/train_source1.tsv"),
        help="Path to S1 TSV",
    )
    parser.add_argument(
        "--s2-path",
        type=Path,
        default=Path("data/medium_split_200k/train_source2.tsv"),
        help="Path to S2 TSV",
    )
    parser.add_argument(
        "--s3-path",
        type=Path,
        default=Path("data/medium_split_200k/train_source3.tsv"),
        help="Path to S3 TSV",
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Run unit tests only",
    )
    args = parser.parse_args()

    # Always execute unit tests first
    run_unit_tests()
    if args.test_only:
        return

    print(f"\nLoading baseline features from: {args.features_input}")
    features_df = pd.read_parquet(args.features_input)

    print("Loading entity name dictionaries...")
    s1_df = pd.read_csv(args.s1_path, sep="\t", usecols=["entity_id", "business_name"])
    s2_df = pd.read_csv(args.s2_path, sep="\t", usecols=["entity_id", "business_name"])
    s3_df = pd.read_csv(args.s3_path, sep="\t", usecols=["entity_id", "business_name"])

    s1_names = dict(
        zip(
            s1_df["entity_id"].astype(str),
            s1_df["business_name"].fillna("").astype(str),
            strict=False,
        )
    )
    target_names = dict(
        zip(
            s2_df["entity_id"].astype(str),
            s2_df["business_name"].fillna("").astype(str),
            strict=False,
        )
    )
    target_names.update(
        dict(
            zip(
                s3_df["entity_id"].astype(str),
                s3_df["business_name"].fillna("").astype(str),
                strict=False,
            )
        )
    )

    augmented_df = extract_features_for_candidates(features_df, s1_names, target_names)

    # Verification assertions
    new_cols = [
        "concat_stem_similarity",
        "concat_addr_product",
        "concat_postal_product",
        "domain_stem_similarity",
        "domain_addr_concordance",
        "domain_postal_concordance",
        "has_domain_target",
        "has_handle_target",
    ]
    for col in new_cols:
        assert col in augmented_df.columns, f"Missing feature column: {col}"
        assert augmented_df[col].isna().sum() == 0, f"NaNs detected in column: {col}"
        assert (augmented_df[col] >= 0.0).all() and (augmented_df[col] <= 1.0).all(), (
            f"Values out of bounds [0, 1] in {col}"
        )

    print("\nIntegrity Verification:")
    print(f"Original shape: {features_df.shape} -> Augmented shape: {augmented_df.shape}")
    print(augmented_df[new_cols].describe().T[["mean", "std", "min", "max"]])

    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    augmented_df.to_parquet(args.output_parquet, index=False)
    print(f"\nSuccessfully generated and saved: {args.output_parquet}")


if __name__ == "__main__":
    main()
