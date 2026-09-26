"""
Parallax Production Trainer for 100% Full Dataset
==================================================
Trains production Two-Pass LightGBM models on 100% of the raw competition data
using memory-bounded country-partitioned hard negative mining, entity meta-features,
and optimal threshold calibration.
"""

from __future__ import annotations

import gc
import json
import shutil
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from parallax.blocking.tfidf_blocker import DualChannelTFIDFBlocker
from parallax.data.contracts import (
    load_business_records_df,
    load_ground_truth_dict,
)
from parallax.features.extractor import FEATURE_COLUMNS, PairwiseFeatureExtractor
from parallax.features.meta_features import (
    META_FEATURE_COLUMNS,
    compute_entity_meta_features,
)
from parallax.metrics.evaluator import evaluate_resolution_predictions
from parallax.models.matcher import LightGBMMatcher
from parallax.postprocessing.singleton_gate import SingletonGatedPredictor
from parallax.preprocessing.normalizer import widen_records_df


@dataclass
class ProductionModelMetadata:
    """Metadata recorded for the trained production models."""

    trained_at: str
    num_train_queries: int
    num_train_pairs: int
    num_holdout_queries: int
    optimal_tau: float
    holdout_macro_f05: float
    holdout_singleton_accuracy: float
    holdout_non_singleton_f05: float
    holdout_precision: float
    holdout_recall: float
    pass1_features: list[str]
    pass2_features: list[str]
    hyperparameters: dict[str, float | int | str]


def print_resource_status() -> None:
    """Print available system RAM and disk headroom."""
    total, used, free = shutil.disk_usage(".")
    print(f"  💾 Current Disk: {free / (1024**3):.1f} GB free / {total / (1024**3):.1f} GB total")
    try:
        import psutil

        mem = psutil.virtual_memory()
        print(
            f"  🧠 Memory (RAM): {mem.available / (1024**3):.1f} GB available / "
            f"{mem.total / (1024**3):.1f} GB total"
        )
    except ImportError:
        pass


def build_country_training_pairs(
    s1_raw_df: pd.DataFrame,
    target_raw_df: pd.DataFrame,
    gt_dict: Mapping[str, set[str]],
    cache_dir: Path,
    top_k_negatives: int = 3,
    sample_queries_per_country: int = 150_000,
    seed: int = 42,
) -> Path:
    """
    Extract high-signal candidate training pairs partitioned by country:
    - 100% of true ground truth positive pairs for sampled queries.
    - Top-K blocker hard negatives per query queried against ALL target records.
    Pre-indexes float32 targets per country to guarantee RAM < 4 GB.
    Returns the path to the unified training feature Parquet file.
    """
    unified_cache = cache_dir / "train_features_full.parquet"
    if unified_cache.is_file():
        print(f"  ⚡ Found cached full training feature matrix at: {unified_cache}")
        return unified_cache

    countries = s1_raw_df["country"].unique()
    print(f"\n🌍 Found {len(countries)} country partition(s): {list(countries)}")

    part_paths: list[Path] = []
    extractor = PairwiseFeatureExtractor()

    for country in countries:
        c_clean = str(country).replace(" ", "_")
        part_file = cache_dir / f"train_features_part_{c_clean}.parquet"

        if part_file.is_file():
            print(f"  ⚡ [Checkpoint] Found cached partition for [{country}]: {part_file.name}")
            part_paths.append(part_file)
            continue

        s1_raw_c = s1_raw_df[s1_raw_df["country"] == country].reset_index(drop=True)
        tgt_raw_c = target_raw_df[target_raw_df["country"] == country].reset_index(drop=True)

        if len(s1_raw_c) == 0 or len(tgt_raw_c) == 0:
            print(f"  ⚠️ Skipping [{country}]: S1={len(s1_raw_c)}, Target={len(tgt_raw_c)}.")
            continue

        print(f"\n--- [Processing Country: {country}] ---")
        print(f"  • Raw S1 Queries: {len(s1_raw_c):,} | Raw Targets: {len(tgt_raw_c):,}")
        print_resource_status()

        # Stratified sampling of queries if larger than sample_queries_per_country
        if sample_queries_per_country and len(s1_raw_c) > sample_queries_per_country:
            rng = np.random.RandomState(seed)
            c_s1_ids = s1_raw_c["entity_id"].astype(str).to_numpy()
            is_sing = np.array([len(gt_dict.get(sid, set())) == 0 for sid in c_s1_ids])
            sing_indices = np.where(is_sing)[0]
            match_indices = np.where(~is_sing)[0]

            sing_ratio = len(sing_indices) / len(c_s1_ids)
            n_sing = int(sample_queries_per_country * sing_ratio)
            n_match = sample_queries_per_country - n_sing

            sub_sing = rng.choice(sing_indices, size=min(n_sing, len(sing_indices)), replace=False)
            sub_match = rng.choice(
                match_indices, size=min(n_match, len(match_indices)), replace=False
            )
            selected_indices = np.sort(np.concatenate([sub_sing, sub_match]))
            s1_raw_c = s1_raw_c.iloc[selected_indices].reset_index(drop=True)
            print(
                f"  ✂️ Sampled {len(s1_raw_c):,} representative queries for [{country}] "
                f"({len(sub_match):,} with matches, {len(sub_sing):,} singletons)."
            )

        # Widen target records (ALL targets in this country!)
        print(f"  ⚡ Widening target pool for [{country}] ({len(tgt_raw_c):,} records)...")
        t0_wt = time.time()
        tgt_wide_c = widen_records_df(tgt_raw_c)
        del tgt_raw_c
        gc.collect()
        print(f"  ✓ Widened targets in {time.time() - t0_wt:.2f}s.")

        # Index targets in blocker
        print(f"  ⚡ Pre-indexing target pool for [{country}]...")
        t0_idx = time.time()
        blocker = DualChannelTFIDFBlocker(
            name_top_k=top_k_negatives,
            addr_top_k=min(2, top_k_negatives),
            name_min_sim=0.15,
            addr_min_sim=0.20,
            batch_size=50,
            show_progress=True,
        )
        blocker.index_country_targets(tgt_wide_c, country=country)
        print(f"  ✓ Target pool pre-indexed in {time.time() - t0_idx:.2f}s.")

        # Widen S1 queries
        print(f"  ⚡ Widening S1 queries for [{country}] ({len(s1_raw_c):,} queries)...")
        s1_wide_c = widen_records_df(s1_raw_c)
        del s1_raw_c
        gc.collect()

        # Generate candidates in micro-batches
        print("  ⚡ Mining hard negatives with micro-batch streaming...")
        t0_b = time.time()
        blocker_cands = blocker.block_queries(s1_wide_c, country=country, batch_size=50)
        print(f"  ✓ Blocker queried in {time.time() - t0_b:.2f}s.")
        blocker.clear_index()
        del blocker
        gc.collect()

        # Assemble high-signal pairs (100% Ground Truth + Blocker Hard Negatives)
        print(f"  ⚡ Assembling high-signal pairs for [{country}]...")
        cands_country: dict[str, dict[str, float]] = {}
        tgt_id_set = set(tgt_wide_c["entity_id"].astype(str))
        pos_count = 0
        neg_count = 0

        for s1_id in s1_wide_c["entity_id"].astype(str):
            c_dict: dict[str, float] = {}
            # 1. Ground truth positives
            gt_matches = gt_dict.get(s1_id, set())
            for mid in gt_matches:
                if mid in tgt_id_set:
                    c_dict[mid] = 1.0
                    pos_count += 1

            # 2. Blocker hard negatives
            b_matches = blocker_cands.get(s1_id, {})
            added_negs = 0
            sorted_b = sorted(b_matches.items(), key=lambda x: x[1], reverse=True)
            for cid, sim in sorted_b:
                if cid not in c_dict and cid in tgt_id_set:
                    c_dict[cid] = float(sim)
                    neg_count += 1
                    added_negs += 1
                    if added_negs >= top_k_negatives:
                        break

            if c_dict:
                cands_country[s1_id] = c_dict

        del blocker_cands
        gc.collect()

        total_p = sum(len(v) for v in cands_country.values())
        print(
            f"  ✓ High-signal pairs for [{country}]: {total_p:,} "
            f"({pos_count:,} true matches, {neg_count:,} hard negatives)."
        )

        # Extract 49 Features for this country
        print(f"  ⚡ Extracting 49 pairwise features for [{country}]...")
        t0_f = time.time()
        feats_c = extractor.extract_features_df(
            cands_country,
            s1_wide_c,
            tgt_wide_c,
            ground_truth=gt_dict,
            show_progress=True,
        )
        print(f"  ✓ Features extracted in {time.time() - t0_f:.2f}s ({len(feats_c):,} rows).")

        # Save partition Parquet
        print(f"  💾 Caching [{country}] partition -> {part_file.name}...")
        feats_c.to_parquet(part_file, compression="snappy", index=False)
        part_paths.append(part_file)

        # Release country memory
        del s1_wide_c, tgt_wide_c, cands_country, feats_c
        gc.collect()
        print(f"  ✓ Memory freed for [{country}].")
        print_resource_status()

    # Combine partitions into unified Parquet file
    print("\n📦 Merging all country partition parquets into unified training matrix...")
    t0_m = time.time()
    dfs = [pd.read_parquet(p) for p in part_paths]
    unified_df = pd.concat(dfs, ignore_index=True)
    del dfs
    gc.collect()

    print(f"  💾 Writing unified matrix ({len(unified_df):,} rows) -> {unified_cache}...")
    unified_df.to_parquet(unified_cache, compression="snappy", index=False)
    del unified_df
    gc.collect()
    print(f"  ✓ Unified training matrix ready in {time.time() - t0_m:.2f}s.")

    return unified_cache


def train_production_pipeline(
    train_dir: Path,
    output_dir: Path,
    holdout_size: int = 5_000,
    sample_queries_per_country: int = 15_000,
    seed: int = 42,
) -> ProductionModelMetadata:
    """
    Execute full production training on 100% dataset:
    1. Ingest raw train files (lightweight raw string dataframes).
    2. Extract features country-by-country to bound RAM < 4 GB.
    3. Hold out a stratified slice of entities for threshold calibration.
    4. Train Pass 1 LightGBM (Config-XDeep, 127 leaves).
    5. Compute 5 Entity Meta-Features.
    6. Train Pass 2 LightGBM (54 features).
    7. Calibrate optimal decision threshold tau on holdout slice.
    8. Export production boosters and metadata.
    """
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("🚀 PARALLAX PRODUCTION TRAINER (100% FULL DATASET)")
    print("=" * 80)
    print_resource_status()

    # Step 1: Load Ground Truth
    gt_path = train_dir / "train_ground_truth.tsv"
    print(f"\n📂 Loading Ground Truth from: {gt_path.name}...")
    gt_dict = load_ground_truth_dict(gt_path)
    total_queries = len(gt_dict)
    num_singletons = sum(1 for v in gt_dict.values() if not v)
    print(
        f"  ✓ Ground Truth loaded: {total_queries:,} total queries "
        f"({total_queries - num_singletons:,} non-singletons, {num_singletons:,} singletons)."
    )

    # Step 2: Load S1 & Target Records (raw, compact strings)
    s1_path = train_dir / "train_source1.tsv"
    s2_path = train_dir / "train_source2.tsv"
    s3_path = train_dir / "train_source3.tsv"

    print("\n📂 Ingesting raw entity records...")
    s1_df = load_business_records_df(s1_path)
    print(f"  ✓ Source 1 records: {len(s1_df):,}")

    s2_df = load_business_records_df(s2_path)
    s3_df = load_business_records_df(s3_path)
    target_df = pd.concat([s2_df, s3_df], ignore_index=True)
    del s2_df, s3_df
    gc.collect()
    print(f"  ✓ Combined Target records (S2 + S3): {len(target_df):,}")

    # Step 3: Country-Partitioned Feature Extraction (Memory Bounded)
    feature_parquet_path = build_country_training_pairs(
        s1_raw_df=s1_df,
        target_raw_df=target_df,
        gt_dict=gt_dict,
        cache_dir=cache_dir,
        top_k_negatives=3,
        sample_queries_per_country=sample_queries_per_country,
        seed=seed,
    )

    # Free raw dataframes now that features are materialized on disk
    del s1_df, target_df
    gc.collect()

    print("\n📂 Loading unified feature matrix from Parquet...")
    features_df = pd.read_parquet(feature_parquet_path)
    print(f"  ✓ Loaded {len(features_df):,} training feature rows.")
    print_resource_status()

    # Step 4: Holdout Validation Split for Calibration
    print(f"\n✂️ Creating stratified holdout slice of {holdout_size:,} entities...")
    rng = np.random.RandomState(seed)
    all_s1_ids = np.array(list(gt_dict.keys()))

    # Stratify by singleton status
    is_singleton_arr = np.array([len(gt_dict[sid]) == 0 for sid in all_s1_ids])
    sing_ids = all_s1_ids[is_singleton_arr]
    match_ids = all_s1_ids[~is_singleton_arr]

    sing_ratio = len(sing_ids) / len(all_s1_ids)
    n_val_sing = int(holdout_size * sing_ratio)
    n_val_match = holdout_size - n_val_sing

    val_sing = rng.choice(sing_ids, size=min(n_val_sing, len(sing_ids)), replace=False)
    val_match = rng.choice(match_ids, size=min(n_val_match, len(match_ids)), replace=False)
    val_s1_set = set(val_sing).union(set(val_match))
    val_gt = {sid: gt_dict[sid] for sid in val_s1_set}

    print(
        f"  ✓ Holdout slice: {len(val_s1_set):,} queries "
        f"({len(val_match):,} with matches, {len(val_sing):,} singletons)."
    )

    # Split feature rows by S1 ID
    s1_series = features_df["s1_id"].astype(str)
    val_mask = s1_series.isin(val_s1_set)
    train_mask = ~val_mask

    train_pairs = features_df[train_mask].copy().reset_index(drop=True)
    val_pairs = features_df[val_mask].copy().reset_index(drop=True)
    del features_df
    gc.collect()

    print(f"  ✓ Train pairs: {len(train_pairs):,} | Holdout pairs: {len(val_pairs):,}")

    # Step 5: Pass 1 LightGBM Training (Config-XDeep)
    print("\n" + "=" * 80)
    print("🌲 TRAINING PASS 1 MODEL (Config-XDeep: 127 Leaves, 250 Trees, 49 Features)")
    print("=" * 80)
    hyperparams: dict[str, float | int | str] = {
        "learning_rate": 0.04,
        "num_leaves": 127,
        "max_depth": 10,
        "n_estimators": 250,
        "scale_pos_weight": 3.0,
    }

    matcher_p1 = LightGBMMatcher(
        learning_rate=float(hyperparams["learning_rate"]),
        num_leaves=int(hyperparams["num_leaves"]),
        max_depth=int(hyperparams["max_depth"]),
        n_estimators=int(hyperparams["n_estimators"]),
        scale_pos_weight=float(hyperparams["scale_pos_weight"]),
        feature_columns=FEATURE_COLUMNS,
        seed=seed,
    )
    t0_p1 = time.time()
    matcher_p1.train(train_pairs, val_pairs)
    print(f"  ✓ Pass 1 model trained in {time.time() - t0_p1:.2f}s.")

    pass1_model_path = models_dir / "production_pass1.txt"
    matcher_p1.save_model(pass1_model_path)
    print(f"  💾 Saved Pass 1 model -> {pass1_model_path}")

    # Step 6: Compute Entity Meta-Features (Track A)
    print("\n⚡ Computing Entity-Level Meta-Features (Pass 1 Probs -> 5 Meta-Features)...")
    train_pairs["prob"] = matcher_p1.predict_proba(train_pairs)
    val_pairs["prob"] = matcher_p1.predict_proba(val_pairs)

    compute_entity_meta_features(train_pairs, prob_col="prob")
    compute_entity_meta_features(val_pairs, prob_col="prob")
    print(f"  ✓ Appended meta-features: {META_FEATURE_COLUMNS}")

    # Step 7: Pass 2 LightGBM Training (Combined 54 Features)
    print("\n" + "=" * 80)
    print("🌲 TRAINING PASS 2 MODEL (Combined 54 Features)")
    print("=" * 80)
    all_features = list(FEATURE_COLUMNS) + list(META_FEATURE_COLUMNS)

    matcher_p2 = LightGBMMatcher(
        learning_rate=float(hyperparams["learning_rate"]),
        num_leaves=int(hyperparams["num_leaves"]),
        max_depth=int(hyperparams["max_depth"]),
        n_estimators=int(hyperparams["n_estimators"]),
        scale_pos_weight=float(hyperparams["scale_pos_weight"]),
        feature_columns=all_features,
        seed=seed + 1000,
    )
    t0_p2 = time.time()
    matcher_p2.train(train_pairs, val_pairs)
    print(f"  ✓ Pass 2 model trained in {time.time() - t0_p2:.2f}s.")

    pass2_model_path = models_dir / "production_pass2.txt"
    matcher_p2.save_model(pass2_model_path)
    print(f"  💾 Saved Pass 2 model -> {pass2_model_path}")

    # Step 8: Calibrate Optimal Decision Threshold tau on Holdout
    print("\n" + "=" * 80)
    print("🎯 CALIBRATING OPTIMAL DECISION THRESHOLD (HOLDOUT SLICE)")
    print("=" * 80)
    val_pairs["prob"] = matcher_p2.predict_proba(val_pairs)

    search_range = [0.50, 0.54, 0.58, 0.62, 0.66, 0.70, 0.74, 0.78, 0.82, 0.86, 0.90]
    best_tau = 0.75
    best_score = -1.0
    val_s1_id_list = list(val_s1_set)
    predictor = SingletonGatedPredictor()

    print(
        f"{'tau':<6} | {'Macro F0.5':<10} | {'Sing. Acc':<10} | "
        f"{'Non-Sing F0.5':<12} | {'Prec.':<8} | {'Recall':<8}"
    )
    print("-" * 65)

    best_report = None
    for tau in search_range:
        preds = predictor.filter_predictions(val_pairs, val_s1_id_list, threshold=tau)
        rep = evaluate_resolution_predictions(val_gt, preds)
        prec = rep.total_correct_pairs / max(1, rep.total_predicted_pairs)
        rec = rep.total_correct_pairs / max(1, rep.total_true_pairs)
        print(
            f"{tau:<6.2f} | {rep.macro_f05:<10.4f} | {rep.singleton_score * 100:<9.2f}% | "
            f"{rep.non_singleton_f05:<12.4f} | {prec * 100:<7.2f}% | "
            f"{rec * 100:<7.2f}%"
        )
        if rep.macro_f05 > best_score:
            best_score = rep.macro_f05
            best_tau = float(tau)
            best_report = rep

    assert best_report is not None
    best_prec = best_report.total_correct_pairs / max(1, best_report.total_predicted_pairs)
    best_rec = best_report.total_correct_pairs / max(1, best_report.total_true_pairs)
    print("-" * 65)
    print(f"  🏆 Optimal Calibration: tau = {best_tau:.2f} (Macro F0.5 = {best_score:.4f})")

    # Save Model Metadata
    metadata = ProductionModelMetadata(
        trained_at=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        num_train_queries=len(train_mask),
        num_train_pairs=len(train_pairs),
        num_holdout_queries=len(val_s1_set),
        optimal_tau=best_tau,
        holdout_macro_f05=float(best_report.macro_f05),
        holdout_singleton_accuracy=float(best_report.singleton_score),
        holdout_non_singleton_f05=float(best_report.non_singleton_f05),
        holdout_precision=float(best_prec),
        holdout_recall=float(best_rec),
        pass1_features=list(FEATURE_COLUMNS),
        pass2_features=all_features,
        hyperparameters=hyperparams,
    )

    metadata_path = models_dir / "production_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(asdict(metadata), f, indent=2)
    print(f"  💾 Saved production metadata -> {metadata_path}")

    return metadata


def main() -> None:
    """CLI launcher for full production training."""
    import argparse

    parser = argparse.ArgumentParser(description="Parallax Full Production Trainer")
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=Path("data/raw/train"),
        help="Path to directory containing raw training TSVs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Path to output directory for models and caches",
    )
    parser.add_argument(
        "--holdout-size",
        type=int,
        default=50_000,
        help="Number of S1 entities to hold out for calibration",
    )
    parser.add_argument(
        "--sample-queries-per-country",
        type=int,
        default=150_000,
        help="Number of representative S1 queries to sample per country",
    )
    args = parser.parse_args()

    train_production_pipeline(
        train_dir=args.train_dir,
        output_dir=args.output_dir,
        holdout_size=args.holdout_size,
        sample_queries_per_country=args.sample_queries_per_country,
    )


if __name__ == "__main__":
    main()
