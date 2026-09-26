"""
Parallax Streaming Test Inference Engine
=========================================
Executes memory-bounded, country-partitioned streaming inference on the real
competition test dataset (~1.75M queries, 10M targets) using trained Two-Pass
LightGBM models and calibrated singleton gating.
"""

from __future__ import annotations

import gc
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from parallax.blocking.tfidf_blocker import DualChannelTFIDFBlocker
from parallax.data.contracts import (
    load_business_records_df,
)
from parallax.features.extractor import FEATURE_COLUMNS, PairwiseFeatureExtractor
from parallax.features.meta_features import (
    META_FEATURE_COLUMNS,
    compute_entity_meta_features,
)
from parallax.models.matcher import LightGBMMatcher
from parallax.postprocessing.singleton_gate import SingletonGatedPredictor
from parallax.preprocessing.normalizer import widen_records_df


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


def run_streaming_test_inference(
    test_dir: Path,
    models_dir: Path,
    output_tsv_path: Path,
    chunk_size: int = 20_000,
    blocker_top_k: int = 30,
    batch_size: int = 2_000,
) -> Path:
    """
    Execute streaming inference on full test dataset:
    1. Loads production Pass 1 and Pass 2 LightGBM boosters.
    2. Reads metadata for calibrated threshold tau.
    3. Ingests test records (test_source1, test_source2, test_source3).
    4. Streams inference country-by-country in chunks to guarantee high throughput and bounded RAM.
    5. Writes final matching_results.tsv preserving exact test_source1 entity ordering.
    """
    output_tsv_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("🚀 PARALLAX STREAMING TEST INFERENCE ENGINE")
    print("=" * 80)
    print_resource_status()

    # Step 1: Load Models & Calibration Metadata
    metadata_path = models_dir / "production_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Model metadata not found at: {metadata_path}")

    with open(metadata_path, encoding="utf-8") as f:
        meta_dict = json.load(f)

    tau = float(meta_dict.get("optimal_tau", 0.75))
    pass1_features = meta_dict.get("pass1_features", list(FEATURE_COLUMNS))
    pass2_features = meta_dict.get(
        "pass2_features", list(FEATURE_COLUMNS) + list(META_FEATURE_COLUMNS)
    )

    print("\n⚙️ Loaded Production Metadata:")
    print(f"  • Calibrated Threshold (tau): {tau:.2f}")
    print(f"  • Pass 1 Features:            {len(pass1_features)}")
    print(f"  • Pass 2 Features:            {len(pass2_features)}")

    pass1_path = models_dir / "production_pass1.txt"
    pass2_path = models_dir / "production_pass2.txt"

    matcher_p1 = LightGBMMatcher(feature_columns=pass1_features)
    matcher_p1.load_model(pass1_path)

    matcher_p2 = LightGBMMatcher(feature_columns=pass2_features)
    matcher_p2.load_model(pass2_path)
    print("  ✓ Loaded Pass 1 and Pass 2 LightGBM boosters.")

    # Step 2: Ingest Test Records
    s1_path = test_dir / "test_source1.tsv"
    s2_path = test_dir / "test_source2.tsv"
    s3_path = test_dir / "test_source3.tsv"

    print("\n📂 Reading Test Source 1 entities...")
    s1_df = load_business_records_df(s1_path)
    ordered_s1_ids = s1_df["entity_id"].astype(str).tolist()
    total_test_queries = len(ordered_s1_ids)
    print(f"  ✓ Total Test Queries: {total_test_queries:,}")

    print("\n📂 Reading Test Source 2 & 3 targets...")
    s2_df = load_business_records_df(s2_path)
    s3_df = load_business_records_df(s3_path)
    target_df = pd.concat([s2_df, s3_df], ignore_index=True)
    del s2_df, s3_df
    gc.collect()
    print(f"  ✓ Total Test Targets (S2 + S3): {len(target_df):,}")

    # Step 3: Country Partitioned Streaming Inference
    extractor = PairwiseFeatureExtractor()
    predictor = SingletonGatedPredictor(decision_threshold=tau)

    # Master results mapping: s1_id -> set of matched candidate IDs
    all_predictions: dict[str, set[str]] = {s1: set() for s1 in ordered_s1_ids}

    countries = s1_df["country"].unique()
    print(f"\n🌍 Found {len(countries)} country partition(s): {list(countries)}")

    for country in countries:
        s1_raw_c = s1_df[s1_df["country"] == country].reset_index(drop=True)
        tgt_raw_c = target_df[target_df["country"] == country].reset_index(drop=True)

        print(f"\n--- [Country Partition: {country}] ---")
        print(f"  • S1 Queries: {len(s1_raw_c):,} | Target Candidates: {len(tgt_raw_c):,}")

        if len(tgt_raw_c) == 0:
            print(
                f"  ⚠️ No target records for country {country}. "
                f"All {len(s1_raw_c):,} are singletons."
            )
            continue

        print(f"  ⚡ Widening and pre-indexing target pool for country [{country}]...")
        t0_wc = time.time()
        tgt_c = widen_records_df(tgt_raw_c)
        del tgt_raw_c
        gc.collect()

        blocker = DualChannelTFIDFBlocker(
            name_top_k=blocker_top_k,
            addr_top_k=min(20, blocker_top_k),
            name_min_sim=0.15,
            addr_min_sim=0.20,
            batch_size=batch_size,
            show_progress=False,
        )
        blocker.index_country_targets(tgt_c, country=country)
        print(f"  ✓ Indexed TF-IDF target matrices in {time.time() - t0_wc:.2f}s.")

        print(f"  ⚡ Pre-building in-memory record lookup for [{country}]...")
        t0_lk = time.time()
        country_target_lookup = extractor.build_record_lookup(tgt_c)
        print(
            f"  ✓ Target lookup built in {time.time() - t0_lk:.2f}s "
            f"({len(country_target_lookup):,} entities)."
        )
        print_resource_status()

        # Free DataFrame tgt_c now that target_lookup and sparse matrices exist in memory
        del tgt_c
        gc.collect()

        n_chunks = int(np.ceil(len(s1_raw_c) / chunk_size))
        pbar = tqdm(
            range(0, len(s1_raw_c), chunk_size),
            total=n_chunks,
            desc=f"  ⚡ Inferring [{country}]",
            unit="chunk",
        )

        for start_idx in pbar:
            end_idx = min(start_idx + chunk_size, len(s1_raw_c))
            s1_raw_chunk = s1_raw_c.iloc[start_idx:end_idx].reset_index(drop=True)
            s1_chunk = widen_records_df(s1_raw_chunk)
            del s1_raw_chunk
            chunk_s1_ids = s1_chunk["entity_id"].astype(str).tolist()

            # 1. Candidate Generation against pre-indexed targets
            candidates = blocker.block_queries(s1_chunk, country=country, batch_size=batch_size)
            total_pairs = sum(len(v) for v in candidates.values())

            if total_pairs == 0:
                # All queries in this chunk are singletons
                del candidates, s1_chunk
                gc.collect()
                continue

            # 2. Pairwise Feature Extraction
            features_df = extractor.extract_features_df(
                candidates,
                s1_chunk,
                ground_truth=None,
                show_progress=False,
                target_lookup=country_target_lookup,
            )

            if len(features_df) > 0:
                # 3. Pass 1 Prediction
                features_df["prob"] = matcher_p1.predict_proba(features_df)

                # 4. Entity Meta-Features
                compute_entity_meta_features(features_df, prob_col="prob")

                # 5. Pass 2 Prediction
                features_df["prob"] = matcher_p2.predict_proba(features_df)

                # 6. Calibrated Singleton Gating
                chunk_preds = predictor.filter_predictions(features_df, chunk_s1_ids, threshold=tau)

                # Store predictions
                for s1_id, match_set in chunk_preds.items():
                    if match_set:
                        all_predictions[s1_id] = match_set

            del candidates, features_df, s1_chunk
            gc.collect()

        # Free country targets and blocker index
        blocker.clear_index()
        del country_target_lookup, s1_raw_c, blocker
        gc.collect()

    del s1_df, target_df
    gc.collect()

    # Step 5: Serialize Output TSV in Exact S1 Order
    print(f"\n💾 Serializing final submission artifact to: {output_tsv_path}...")
    t0_write = time.time()
    n_singletons = 0
    n_matched = 0

    with open(output_tsv_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in ordered_s1_ids:
            matches = all_predictions.get(s1_id, set())
            if matches:
                n_matched += 1
                sorted_matches = sorted(list(matches))
                f.write(f"{s1_id}\t{','.join(sorted_matches)}\n")
            else:
                n_singletons += 1
                f.write(f"{s1_id}\t\n")

    file_size_mb = output_tsv_path.stat().st_size / (1024**2)
    print(
        f"  ✓ Submission serialized in {time.time() - t0_write:.2f}s "
        f"({total_test_queries:,} rows, {file_size_mb:.1f} MB)."
    )
    print(
        f"  📊 Match Summary: {n_matched:,} entities matched "
        f"({n_matched / total_test_queries * 100:.1f}%), "
        f"{n_singletons:,} singletons ({n_singletons / total_test_queries * 100:.1f}%)."
    )

    return output_tsv_path


def main() -> None:
    """CLI launcher for streaming test inference."""
    import argparse

    parser = argparse.ArgumentParser(description="Parallax Streaming Test Inference")
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=Path("data/raw/test"),
        help="Path to directory containing raw test TSVs",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("output/models"),
        help="Path to directory containing trained models and metadata",
    )
    parser.add_argument(
        "--output-tsv",
        type=Path,
        default=Path("submissions/matching_results.tsv"),
        help="Path to output submission TSV",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=5_000,
        help="Batch size for S1 queries per streaming pass",
    )
    args = parser.parse_args()

    run_streaming_test_inference(
        test_dir=args.test_dir,
        models_dir=args.models_dir,
        output_tsv_path=args.output_tsv,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
