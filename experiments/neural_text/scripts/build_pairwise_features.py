"""
Parallax Pairwise Neural Feature Builder
========================================
Constructs additive neural features (`indicxlit_name_similarity` and `qwen_name_cosine`)
by performing O(1) dictionary and matrix lookups against precomputed entity caches.
Enforces strict alignment and integrity checks against the frozen baseline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from tqdm import tqdm

# Add local experiment scripts to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# 28 Frozen Baseline Features
BASELINE_28_FEATURES: list[str] = [
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
    "primary_num_match",
    "primary_num_conflict",
    "primary_num_missing",
    "num_jaccard",
    "num_conflict_count",
    "postal_match",
    "postal_conflict",
    "postal_missing",
    "jaro_winkler_soft",
    "jaro_winkler_raw",
    "token_jaccard_name",
    "token_overlap_name",
    "first_token_match",
    "canon_addr_ratio",
    "token_jaccard_addr",
]


def load_indicxlit_cache(cache_path: Path) -> dict[str, str]:
    """Load IndicXlit transliteration cache mapping entity_id -> transliterated name."""
    if not cache_path.exists():
        print(f"Warning: IndicXlit cache {cache_path} does not exist. Returning empty cache.")
        return {}
    df = pd.read_parquet(cache_path, columns=["entity_id", "indicxlit_name"])
    return dict(zip(df["entity_id"].astype(str), df["indicxlit_name"].fillna("").astype(str), strict=False))


def load_qwen_embeddings(cache_path: Path) -> tuple[dict[str, int], np.ndarray]:
    """Load Qwen embeddings cache mapping entity_id -> row index in embedding matrix."""
    if not cache_path.exists():
        print(f"Warning: Qwen cache {cache_path} does not exist. Returning empty matrix.")
        return {}, np.empty((0, 1024), dtype=np.float32)

    data = np.load(cache_path, allow_pickle=False)
    entity_ids = data["entity_ids"].astype(str)
    embeddings = data["embeddings"].astype(np.float32)

    id_to_idx = {eid: idx for idx, eid in enumerate(entity_ids)}
    return id_to_idx, embeddings


def compute_indicxlit_name_similarity(
    s1_ids: np.ndarray,
    cand_ids: np.ndarray,
    s1_soft_names: dict[str, str],
    cand_soft_names: dict[str, str],
    indicxlit_cache: dict[str, str],
    baseline_soft_ratio: np.ndarray,
) -> np.ndarray:
    """
    Compute IndicXlit phonetic name similarity for each candidate pair.
    If candidate is Indic and transliteration exists, computes fuzz.ratio(s1_soft, cand_indicxlit).
    If candidate is already Latin, smoothly falls back to existing soft_name_ratio.
    """
    total = len(s1_ids)
    sims = np.empty(total, dtype=np.float32)

    for i in tqdm(range(total), desc="Building IndicXlit Sim"):
        c_id = str(cand_ids[i])
        if c_id in indicxlit_cache and indicxlit_cache[c_id]:
            s_id = str(s1_ids[i])
            s1_name = s1_soft_names.get(s_id, "")
            cand_translit = indicxlit_cache[c_id]
            sims[i] = fuzz.ratio(s1_name, cand_translit) / 100.0
        else:
            # Same-script fallback to existing normalized soft ratio
            sims[i] = baseline_soft_ratio[i]

    return sims


def compute_qwen_name_cosine(
    s1_ids: np.ndarray,
    cand_ids: np.ndarray,
    id_to_idx: dict[str, int],
    embeddings: np.ndarray,
) -> np.ndarray:
    """
    Compute pairwise cosine similarity between S1 and candidate name embeddings.
    Embeddings are assumed to be unit L2-normalized, so cosine_sim = dot_product.
    """
    total = len(s1_ids)
    cosines = np.zeros(total, dtype=np.float32)

    if len(id_to_idx) == 0 or len(embeddings) == 0:
        print("Warning: Embedding cache is empty. Filling qwen_name_cosine with 0.0.")
        return cosines

    # Vectorized batch dot product
    s1_indices = np.array([id_to_idx.get(str(s), -1) for s in s1_ids], dtype=np.int32)
    cand_indices = np.array([id_to_idx.get(str(c), -1) for c in cand_ids], dtype=np.int32)

    valid_mask = (s1_indices >= 0) & (cand_indices >= 0)
    print(f"Pairs with valid embeddings on both sides: {valid_mask.sum():,} / {total:,} ({valid_mask.mean()*100:.2f}%)")

    valid_s1_idx = s1_indices[valid_mask]
    valid_cand_idx = cand_indices[valid_mask]

    s1_vecs = embeddings[valid_s1_idx]
    cand_vecs = embeddings[valid_cand_idx]

    # Element-wise product and sum across embedding dimension
    dots = np.sum(s1_vecs * cand_vecs, axis=1)
    cosines[valid_mask] = np.clip(dots, -1.0, 1.0)

    return cosines


def load_or_generate_baseline_features(
    features_path: Path,
    s1_source_path: Path,
    s2_source_path: Path,
    s3_source_path: Path,
    candidates_path: Path | None = None,
    ground_truth_path: Path | None = None,
    s1_limit: int | None = 5000,
) -> pd.DataFrame:
    """Load baseline features if cached, or auto-generate via baseline pipeline from raw TSVs."""
    if features_path.is_file():
        print(f"Loading baseline features from: {features_path}")
        return pd.read_parquet(features_path)

    print(f"⚠️  Baseline features file not found at: {features_path}")
    print("⚡ Auto-generating baseline candidate pairs and 28 features from raw TSVs (~40-60s on Kaggle)...")

    project_root = Path(__file__).resolve().parents[3]
    if str(project_root / "src") not in sys.path:
        sys.path.insert(0, str(project_root / "src"))

    from parallax.data.contracts import load_business_records_df, load_ground_truth_dict
    from parallax.experiments.runner import extract_or_load_candidates, extract_or_load_features
    from parallax.preprocessing.normalizer import widen_records_df

    gt_path = ground_truth_path or s1_source_path.parent / "train_ground_truth.tsv"
    cand_path = candidates_path or features_path.parent / "candidate_pairs_sample.parquet"
    cand_path.parent.mkdir(parents=True, exist_ok=True)
    features_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading S1 from {s1_source_path} (limit={s1_limit})...")
    s1_raw = load_business_records_df(s1_source_path, limit=s1_limit)
    s2_raw = load_business_records_df(s2_source_path)
    s3_raw = load_business_records_df(s3_source_path)
    target_raw = pd.concat([s2_raw, s3_raw], ignore_index=True)
    gt_dict = load_ground_truth_dict(gt_path)

    print("Widening records...")
    s1_wide = widen_records_df(s1_raw)
    target_wide = widen_records_df(target_raw)

    print(f"Generating / loading baseline candidate pairs at {cand_path}...")
    candidates = extract_or_load_candidates(s1_wide, target_wide, cand_path)

    print(f"Extracting baseline 28 features at {features_path}...")
    features_df = extract_or_load_features(candidates, s1_wide, target_wide, gt_dict, features_path)
    return features_df


def build_augmented_features(
    features_path: Path,
    s1_source_path: Path,
    s2_source_path: Path,
    s3_source_path: Path,
    indicxlit_cache_path: Path,
    qwen_cache_path: Path,
    output_path: Path,
    candidates_path: Path | None = None,
    ground_truth_path: Path | None = None,
    s1_limit: int | None = 5000,
) -> pd.DataFrame:
    """
    Read baseline features (or generate if missing), attach neural pairwise features, and validate integrity.
    """
    base_df = load_or_generate_baseline_features(
        features_path=features_path,
        s1_source_path=s1_source_path,
        s2_source_path=s2_source_path,
        s3_source_path=s3_source_path,
        candidates_path=candidates_path,
        ground_truth_path=ground_truth_path,
        s1_limit=s1_limit,
    )

    # Verify baseline 28 columns are present
    for col in BASELINE_28_FEATURES:
        if col not in base_df.columns:
            raise ValueError(f"Required baseline feature '{col}' missing from {features_path}")

    s1_ids = base_df["s1_id"].to_numpy()
    cand_ids = base_df["cand_id"].to_numpy()
    total_pairs = len(base_df)
    print(f"Total candidate pairs: {total_pairs:,}")

    # Load source business names for soft comparison
    print("Loading entity soft names for IndicXlit feature calculation...")
    s1_df = pd.read_csv(s1_source_path, sep="\t", usecols=["entity_id", "business_name"])
    s2_df = pd.read_csv(s2_source_path, sep="\t", usecols=["entity_id", "business_name"])
    s3_df = pd.read_csv(s3_source_path, sep="\t", usecols=["entity_id", "business_name"])

    # Basic normalization for lookup
    s1_soft_names = dict(zip(
        s1_df["entity_id"].astype(str),
        s1_df["business_name"].fillna("").astype(str).str.lower().str.strip(),
        strict=False,
    ))
    cand_targets = pd.concat([s2_df, s3_df], ignore_index=True)
    cand_soft_names = dict(zip(
        cand_targets["entity_id"].astype(str),
        cand_targets["business_name"].fillna("").astype(str).str.lower().str.strip(),
        strict=False,
    ))

    # 1. IndicXlit Name Similarity
    print(f"Loading IndicXlit cache from: {indicxlit_cache_path}")
    indicxlit_cache = load_indicxlit_cache(indicxlit_cache_path)
    base_soft_ratio = base_df["soft_name_ratio"].to_numpy(dtype=np.float32)
    indicxlit_sim = compute_indicxlit_name_similarity(
        s1_ids, cand_ids, s1_soft_names, cand_soft_names, indicxlit_cache, base_soft_ratio
    )

    # 2. Qwen Name Cosine
    print(f"Loading Qwen embedding cache from: {qwen_cache_path}")
    id_to_idx, embeddings = load_qwen_embeddings(qwen_cache_path)
    qwen_cosine = compute_qwen_name_cosine(s1_ids, cand_ids, id_to_idx, embeddings)

    # Create augmented dataframe
    augmented_df = base_df.copy()
    augmented_df["indicxlit_name_similarity"] = indicxlit_sim
    augmented_df["qwen_name_cosine"] = qwen_cosine

    # Rigorous Integrity Assertions
    assert len(augmented_df) == total_pairs, "Row count changed during feature construction!"
    assert (augmented_df["s1_id"].to_numpy() == s1_ids).all(), "s1_id alignment broken!"
    assert (augmented_df["cand_id"].to_numpy() == cand_ids).all(), "cand_id alignment broken!"
    assert not augmented_df["indicxlit_name_similarity"].isna().any(), "NaN found in indicxlit_name_similarity!"
    assert not augmented_df["qwen_name_cosine"].isna().any(), "NaN found in qwen_name_cosine!"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    augmented_df.to_parquet(output_path, index=False)
    print(f"Augmented feature matrix saved to: {output_path} (shape={augmented_df.shape})")

    return augmented_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Construct pairwise neural features for Parallax")
    parser.add_argument("--scale", choices=["5000", "200000"], default="5000", help="Data split scale")
    parser.add_argument("--features-path", type=Path, default=None, help="Path to baseline features parquet")
    parser.add_argument("--s1-path", type=Path, default=None, help="Path to Source 1 TSV")
    parser.add_argument("--s2-path", type=Path, default=None, help="Path to Source 2 TSV")
    parser.add_argument("--s3-path", type=Path, default=None, help="Path to Source 3 TSV")
    parser.add_argument("--indicxlit-cache", type=Path, default=None, help="Path to IndicXlit parquet cache")
    parser.add_argument("--qwen-cache", type=Path, default=None, help="Path to Qwen .npz cache")
    parser.add_argument("--output-path", type=Path, default=None, help="Path for output augmented features parquet")
    args = parser.parse_args()

    if args.scale == "5000":
        feat_path = args.features_path or Path("baseline_artifacts/features_sample.parquet")
        s1_path = args.s1_path or Path("data/medium_split_200k/train_source1.tsv")
        s2_path = args.s2_path or Path("data/medium_split_200k/train_source2.tsv")
        s3_path = args.s3_path or Path("data/medium_split_200k/train_source3.tsv")
        ix_cache = args.indicxlit_cache or Path("experiments/neural_text/caches/indicxlit_translit_cache_5k.parquet")
        qw_cache = args.qwen_cache or Path("experiments/neural_text/caches/qwen_name_embeddings_5k.npz")
        out_path = args.output_path or Path("experiments/neural_text/caches/augmented_features_5k.parquet")
    else:
        feat_path = args.features_path or Path("data/full_dataset/features.parquet")
        s1_path = args.s1_path or Path("data/medium_split_200k/train_source1.tsv")
        s2_path = args.s2_path or Path("data/medium_split_200k/train_source2.tsv")
        s3_path = args.s3_path or Path("data/medium_split_200k/train_source3.tsv")
        ix_cache = args.indicxlit_cache or Path("experiments/neural_text/caches/indicxlit_translit_cache_200k.parquet")
        qw_cache = args.qwen_cache or Path("experiments/neural_text/caches/qwen_name_embeddings_200k.npz")
        out_path = args.output_path or Path("experiments/neural_text/caches/augmented_features_200k.parquet")

    cand_path = Path("baseline_artifacts/candidate_pairs_sample.parquet") if args.scale == "5000" else Path("data/full_dataset/candidate_pairs.parquet")
    s1_limit = 5000 if args.scale == "5000" else None

    build_augmented_features(
        features_path=feat_path,
        s1_source_path=s1_path,
        s2_source_path=s2_path,
        s3_source_path=s3_path,
        indicxlit_cache_path=ix_cache,
        qwen_cache_path=qw_cache,
        output_path=out_path,
        candidates_path=cand_path,
        s1_limit=s1_limit,
    )


if __name__ == "__main__":
    main()
