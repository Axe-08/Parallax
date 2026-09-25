"""
Parallax Google Colab Medium-Sized Split Generator (200,000 Records)
=====================================================================
Self-contained script to extract a representative 200,000 S1 benchmark split
from AWS S3 (s3://amazon-ml-challange-2026-parallax/raw/train/), preserve 100%
of true positive S2/S3 matches, sample stratified negative distractors (~500k each),
assign deterministic 5-fold stratified cross-validation, and upload back to S3.

Usage in Google Colab:
---------------------
1. Create a new Google Colab notebook (CPU runtime is sufficient; ~12GB RAM).
2. Paste this entire script into a single cell.
3. (Optional) In Colab left sidebar -> Secrets (key icon), set:
     AWS_ACCESS_KEY_ID
     AWS_SECRET_ACCESS_KEY
   If not set in Secrets, the script will use the team defaults below.
4. Press Run! The split is built and uploaded in ~2-4 minutes.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Ensure boto3 is installed (standard Colab environment usually needs a quick check)
try:
    import boto3
except ImportError:
    import subprocess

    print("Installing boto3...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "boto3"])
    import boto3

import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

# ==============================================================================
# 1. Configuration & Credentials Setup
# ==============================================================================


def get_aws_credentials() -> tuple[str, str, str, str]:
    """Retrieve AWS credentials from Colab userdata, env vars, or team defaults."""
    access_key = ""
    secret_key = ""

    # Attempt 1: Google Colab userdata secrets
    try:
        from google.colab import userdata  # type: ignore

        access_key = userdata.get("AWS_ACCESS_KEY_ID") or ""
        secret_key = userdata.get("AWS_SECRET_ACCESS_KEY") or ""
    except Exception:
        pass

    # Attempt 2: Environment variables
    if not access_key:
        access_key = os.getenv("AWS_ACCESS_KEY_ID", "")
    if not secret_key:
        secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", "")

    region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    bucket = os.getenv("PARALLAX_S3_BUCKET", "amazon-ml-challange-2026-parallax")

    return access_key, secret_key, region, bucket


# ==============================================================================
# 2. S3 Transfer Helpers
# ==============================================================================


def create_s3_client(access_key: str, secret_key: str, region: str) -> Any:
    """Instantiate a configured boto3 S3 client."""
    return boto3.client(
        "s3",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region,
    )


def download_file_with_progress(s3: Any, bucket: str, key: str, local_path: Path) -> None:
    """Download an S3 object to local path with tqdm progress tracking."""
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if local_path.exists() and local_path.stat().st_size > 0:
        mb = local_path.stat().st_size / 1e6
        print(f"  ⚡ Already cached locally: {local_path.name} ({mb:.1f} MB)")
        return

    meta = s3.head_object(Bucket=bucket, Key=key)
    total_bytes = meta["ContentLength"]

    print(f"  📥 Downloading s3://{bucket}/{key} ({total_bytes / 1e6:.1f} MB)...")
    with tqdm(total=total_bytes, unit="B", unit_scale=True, desc=local_path.name) as pbar:

        def callback(bytes_transferred: int) -> None:
            pbar.update(bytes_transferred)

        s3.download_file(bucket, key, str(local_path), Callback=callback)


def upload_file_with_progress(s3: Any, bucket: str, local_path: Path, key: str) -> None:
    """Upload a local file to S3 with tqdm progress tracking."""
    total_bytes = local_path.stat().st_size
    print(
        f"  📤 Uploading {local_path.name} ({total_bytes / 1e6:.1f} MB) -> s3://{bucket}/{key}..."
    )
    with tqdm(total=total_bytes, unit="B", unit_scale=True, desc=local_path.name) as pbar:

        def callback(bytes_transferred: int) -> None:
            pbar.update(bytes_transferred)

        s3.upload_file(str(local_path), bucket, key, Callback=callback)


# ==============================================================================
# 3. Core Splitting & Sampling Engine
# ==============================================================================


def build_medium_split(
    raw_dir: Path,
    output_dir: Path,
    s1_target_size: int = 200_000,
    s2_target_size: int = 500_000,
    s3_target_size: int = 500_000,
    n_splits: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Generate a 200,000 S1 benchmark split with:
    1. Stratified sampling on S1 (country and singleton status).
    2. 100% retention of positive ground-truth S2 and S3 matches.
    3. Stratified negative distractor sampling for S2 and S3 up to target sizes.
    4. Deterministic 5-fold stratified cross-validation on S1.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # --- Step 1: Load Ground Truth and Index Singletons ---
    print("\n[1/5] Loading Ground Truth & indexing singletons...")
    gt_file = raw_dir / "train_ground_truth.tsv"
    gt_df = pd.read_csv(
        gt_file,
        sep="\t",
        dtype={"source1_entity_id": str, "matched_entity_ids": str},
        keep_default_na=False,
    )
    print(f"  Loaded raw ground truth: {len(gt_df):,} records")

    # Determine singleton status (no matches or empty)
    gt_df["matched_entity_ids"] = gt_df["matched_entity_ids"].fillna("").astype(str).str.strip()
    is_singleton_series = (gt_df["matched_entity_ids"] == "") | (
        gt_df["matched_entity_ids"] == "nan"
    )
    singleton_map = dict(zip(gt_df["source1_entity_id"], is_singleton_series, strict=False))

    # --- Step 2: Stratified S1 Sampling ---
    print(f"\n[2/5] Sampling {s1_target_size:,} S1 records stratified by Country & Singletons...")
    s1_file = raw_dir / "train_source1.tsv"
    s1_df = pd.read_csv(
        s1_file,
        sep="\t",
        dtype={"entity_id": str, "business_name": str, "business_address": str, "country": str},
        keep_default_na=False,
    )
    print(f"  Loaded raw S1: {len(s1_df):,} records")

    # Map singleton status to S1
    s1_df["is_singleton"] = s1_df["entity_id"].map(lambda x: singleton_map.get(x, True))
    s1_df["country_clean"] = s1_df["country"].fillna("UNKNOWN").astype(str).str.strip()
    s1_df["strata"] = s1_df["country_clean"] + "__" + s1_df["is_singleton"].astype(str)

    # Perform stratified sampling over strata keys
    if len(s1_df) <= s1_target_size:
        print(f"  ⚠️ Warning: Raw S1 size ({len(s1_df)}) <= target. Using all records.")
        sampled_s1 = s1_df.copy().reset_index(drop=True)
    else:
        sample_ratio = s1_target_size / len(s1_df)
        sampled_parts = []
        for sk in sorted(s1_df["strata"].unique()):
            grp = s1_df[s1_df["strata"] == sk]
            n_sample = max(1, int(round(len(grp) * sample_ratio)))
            sampled_parts.append(grp.sample(n=min(n_sample, len(grp)), random_state=seed))
        sampled_s1 = pd.concat(sampled_parts, ignore_index=True)

        # Adjust rounding discrepancies
        if len(sampled_s1) > s1_target_size:
            sampled_s1 = sampled_s1.sample(n=s1_target_size, random_state=seed)
        elif len(sampled_s1) < s1_target_size:
            missing_count = s1_target_size - len(sampled_s1)
            remaining = s1_df[~s1_df["entity_id"].isin(sampled_s1["entity_id"])]
            sampled_s1 = pd.concat(
                [sampled_s1, remaining.sample(n=missing_count, random_state=seed)],
                ignore_index=True,
            )

    sampled_s1 = sampled_s1.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    sampled_s1["strata"] = (
        sampled_s1["country_clean"] + "__" + sampled_s1["is_singleton"].astype(str)
    )
    sampled_s1_ids = set(sampled_s1["entity_id"])
    s_cnt = int(sampled_s1["is_singleton"].sum())
    s_ratio = float(sampled_s1["is_singleton"].mean())
    print(f"  ✓ Sampled S1: {len(sampled_s1):,} records")
    print(f"    - Countries: {dict(sampled_s1['country_clean'].value_counts())}")
    print(f"    - Singletons: {s_cnt:,} ({s_ratio:.2%})")

    # --- Step 3: Filter Ground Truth & Collect Required Positive IDs ---
    print("\n[3/5] Filtering Ground Truth and extracting required positive matches...")
    sampled_gt = (
        gt_df[gt_df["source1_entity_id"].isin(sampled_s1_ids)].copy().reset_index(drop=True)
    )

    needed_s2_ids: set[str] = set()
    needed_s3_ids: set[str] = set()

    for mids in sampled_gt["matched_entity_ids"]:
        if mids and mids != "nan":
            for mid in mids.split(","):
                clean_mid = mid.strip()
                if clean_mid.startswith("S2-"):
                    needed_s2_ids.add(clean_mid)
                elif clean_mid.startswith("S3-"):
                    needed_s3_ids.add(clean_mid)

    print(f"  ✓ Filtered Ground Truth: {len(sampled_gt):,} records")
    print(f"    - Required S2 positive entities: {len(needed_s2_ids):,}")
    print(f"    - Required S3 positive entities: {len(needed_s3_ids):,}")

    # --- Step 4: Sample S2 & S3 (100% Positives + Stratified Distractors) ---
    def sample_candidate_source(
        source_name: str,
        file_path: Path,
        needed_ids: set[str],
        target_size: int,
    ) -> pd.DataFrame:
        print(f"\n[4/5] Building {source_name} candidate pool (target: {target_size:,})...")
        df = pd.read_csv(
            file_path,
            sep="\t",
            dtype={"entity_id": str, "business_name": str, "business_address": str, "country": str},
            keep_default_na=False,
        )
        print(f"  Loaded raw {source_name}: {len(df):,} records")

        # Partition into positive matches vs potential distractors
        is_pos = df["entity_id"].isin(needed_ids)
        pos_df = df[is_pos].copy()
        distractor_pool = df[~is_pos].copy()

        # Sanity check: Ensure 100% of needed IDs found
        found_pos_ids = set(pos_df["entity_id"])
        missing_pos = needed_ids - found_pos_ids
        if missing_pos:
            print(f"  ⚠️ Warning: {len(missing_pos)} required IDs not found in {source_name}!")

        # Explicitly delete raw DataFrame and trigger garbage collection
        del df
        import gc

        gc.collect()

        n_needed_distractors = max(0, target_size - len(pos_df))
        print(f"  Positive matches to preserve: {len(pos_df):,}")
        print(f"  Negative distractors to sample: {n_needed_distractors:,}")

        if n_needed_distractors > 0:
            distractor_pool["country_clean"] = (
                distractor_pool["country"].fillna("UNKNOWN").astype(str).str.strip()
            )
            distractor_ratio = min(1.0, n_needed_distractors / max(1, len(distractor_pool)))
            distractor_parts = []
            for ctry in sorted(distractor_pool["country_clean"].unique()):
                c_grp = distractor_pool[distractor_pool["country_clean"] == ctry]
                n_c = max(1, int(round(len(c_grp) * distractor_ratio)))
                distractor_parts.append(c_grp.sample(n=min(n_c, len(c_grp)), random_state=seed))
            sampled_distractors = pd.concat(distractor_parts, ignore_index=True)

            if len(sampled_distractors) > n_needed_distractors:
                sampled_distractors = sampled_distractors.sample(
                    n=n_needed_distractors, random_state=seed
                )
            elif len(sampled_distractors) < n_needed_distractors:
                missing_cnt = n_needed_distractors - len(sampled_distractors)
                rem = distractor_pool[
                    ~distractor_pool["entity_id"].isin(sampled_distractors["entity_id"])
                ]
                sampled_distractors = pd.concat(
                    [sampled_distractors, rem.sample(n=missing_cnt, random_state=seed)],
                    ignore_index=True,
                )

            del distractor_pool
            gc.collect()
            combined = pd.concat([pos_df, sampled_distractors], ignore_index=True)
        else:
            del distractor_pool
            gc.collect()
            combined = pos_df

        combined = combined.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        # Drop temporary helper columns
        cols = ["entity_id", "business_name", "business_address", "country"]
        combined = combined[[c for c in cols if c in combined.columns]]
        print(f"  ✓ Final {source_name} pool: {len(combined):,} records")
        return combined

    sampled_s2 = sample_candidate_source(
        "S2", raw_dir / "train_source2.tsv", needed_s2_ids, s2_target_size
    )
    sampled_s3 = sample_candidate_source(
        "S3", raw_dir / "train_source3.tsv", needed_s3_ids, s3_target_size
    )

    # --- Step 5: Assign 5-Fold Stratified Cross-Validation on S1 ---
    print(f"\n[5/5] Generating {n_splits}-fold Stratified CV splits on S1...")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    sampled_s1["fold"] = -1
    for fold_idx, (_, val_idx) in enumerate(skf.split(sampled_s1, sampled_s1["strata"])):
        sampled_s1.iloc[val_idx, sampled_s1.columns.get_loc("fold")] = fold_idx

    # --- Export Files to Local Output Directory ---
    print("\n💾 Writing formatted TSVs...")
    # Clean S1 columns for standard TSV format
    s1_clean = sampled_s1[["entity_id", "business_name", "business_address", "country"]].copy()
    s1_clean.to_csv(output_dir / "train_source1.tsv", sep="\t", index=False)
    sampled_s2.to_csv(output_dir / "train_source2.tsv", sep="\t", index=False)
    sampled_s3.to_csv(output_dir / "train_source3.tsv", sep="\t", index=False)

    # Clean Ground Truth columns (raw format has source1_entity_id and matched_entity_ids)
    valid_cols = ["source1_entity_id", "matched_entity_ids", "match_count"]
    gt_cols = [c for c in valid_cols if c in sampled_gt.columns]
    gt_clean = sampled_gt[gt_cols].copy()
    gt_clean.to_csv(output_dir / "train_ground_truth.tsv", sep="\t", index=False)

    # CV Folds manifest
    folds_df = sampled_s1[["entity_id", "fold", "country", "is_singleton"]].copy()
    folds_df.to_csv(output_dir / "cv_folds_source1.tsv", sep="\t", index=False)

    # Try parquet export if pyarrow is present
    with contextlib.suppress(Exception):
        folds_df.to_parquet(output_dir / "cv_folds_source1.parquet", index=False)

    elapsed_sec = time.time() - t0

    metadata = {
        "dataset_name": "medium_split_200k",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generation_time_sec": round(elapsed_sec, 2),
        "random_seed": seed,
        "n_splits": n_splits,
        "entity_counts": {
            "source1": len(s1_clean),
            "source2": len(sampled_s2),
            "source3": len(sampled_s3),
            "ground_truth_matches": len(gt_clean),
            "s2_positive_matches": len(needed_s2_ids),
            "s3_positive_matches": len(needed_s3_ids),
            "s1_singletons": int(sampled_s1["is_singleton"].sum()),
            "s1_singleton_ratio": round(float(sampled_s1["is_singleton"].mean()), 4),
        },
        "country_distribution_s1": {
            str(k): int(v) for k, v in sampled_s1["country_clean"].value_counts().items()
        },
        "file_sizes_bytes": {
            "train_source1.tsv": (output_dir / "train_source1.tsv").stat().st_size,
            "train_source2.tsv": (output_dir / "train_source2.tsv").stat().st_size,
            "train_source3.tsv": (output_dir / "train_source3.tsv").stat().st_size,
            "train_ground_truth.tsv": (output_dir / "train_ground_truth.tsv").stat().st_size,
            "cv_folds_source1.tsv": (output_dir / "cv_folds_source1.tsv").stat().st_size,
        },
    }

    (output_dir / "split_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


# ==============================================================================
# 4. Main Entrypoint (Orchestrating Download, Build, and S3 Upload)
# ==============================================================================


def run_medium_split_pipeline(
    work_dir: Path | str = "/content/parallax_data",
    s1_size: int = 200_000,
    s2_size: int = 500_000,
    s3_size: int = 500_000,
    s3_target_prefix: str = "splits/medium_split_200k",
    upload_to_s3: bool = True,
    seed: int = 42,
) -> None:
    """End-to-end pipeline: S3 Download -> Stratified Sampling -> S3 Upload."""
    base_dir = Path(work_dir)
    raw_dir = base_dir / "raw_train"
    out_dir = base_dir / "medium_split_200k"
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("🚀 PARALLAX MEDIUM SPLIT GENERATOR (200,000 RECORDS)")
    print("=" * 80)

    # 1. Credentials
    access_key, secret_key, region, bucket = get_aws_credentials()
    print(f"AWS Region: {region}")
    print(f"S3 Bucket:  s3://{bucket}")
    print(f"Target Key: s3://{bucket}/{s3_target_prefix}/")
    print(f"Work Dir:   {base_dir.resolve()}\n")

    s3 = create_s3_client(access_key, secret_key, region)

    # Test S3 Connection
    try:
        s3.head_bucket(Bucket=bucket)
        print("✓ Connected to S3 successfully!\n")
    except Exception as e:
        print(f"❌ Failed to connect to S3: {e}")
        print("Please check your AWS credentials.")
        return

    # 2. Download Raw Train Data
    raw_keys = {
        "train_source1.tsv": "raw/train/train_source1.tsv",
        "train_source2.tsv": "raw/train/train_source2.tsv",
        "train_source3.tsv": "raw/train/train_source3.tsv",
        "train_ground_truth.tsv": "raw/train/train_ground_truth.tsv",
    }

    print("--- [Step 1: Download Raw Train Datasets] ---")
    for fname, s3_key in raw_keys.items():
        download_file_with_progress(s3, bucket, s3_key, raw_dir / fname)

    # 3. Build Stratified Medium Split
    print("\n--- [Step 2: Generate Stratified Split] ---")
    meta = build_medium_split(
        raw_dir=raw_dir,
        output_dir=out_dir,
        s1_target_size=s1_size,
        s2_target_size=s2_size,
        s3_target_size=s3_size,
        n_splits=5,
        seed=seed,
    )

    # 4. Upload to S3
    if upload_to_s3:
        print(f"\n--- [Step 3: Upload Split to s3://{bucket}/{s3_target_prefix}/] ---")
        files_to_upload = [
            "train_source1.tsv",
            "train_source2.tsv",
            "train_source3.tsv",
            "train_ground_truth.tsv",
            "cv_folds_source1.tsv",
            "split_metadata.json",
        ]
        if (out_dir / "cv_folds_source1.parquet").exists():
            files_to_upload.append("cv_folds_source1.parquet")

        for fname in files_to_upload:
            local_p = out_dir / fname
            dest_key = f"{s3_target_prefix}/{fname}"
            upload_file_with_progress(s3, bucket, local_p, dest_key)

    print("\n" + "=" * 80)
    print("🎉 MEDIUM SPLIT GENERATION & UPLOAD COMPLETE!")
    print("=" * 80)
    s1_cnt = meta["entity_counts"]["source1"]
    s2_cnt = meta["entity_counts"]["source2"]
    s2_pos = meta["entity_counts"]["s2_positive_matches"]
    s3_cnt = meta["entity_counts"]["source3"]
    s3_pos = meta["entity_counts"]["s3_positive_matches"]
    gt_cnt = meta["entity_counts"]["ground_truth_matches"]
    sing_cnt = meta["entity_counts"]["s1_singletons"]
    sing_ratio = meta["entity_counts"]["s1_singleton_ratio"]
    gen_time = meta["generation_time_sec"]

    print(f"• S1 Entities:           {s1_cnt:,}")
    print(f"• S2 Candidate Space:    {s2_cnt:,} (includes {s2_pos:,} true positives)")
    print(f"• S3 Candidate Space:    {s3_cnt:,} (includes {s3_pos:,} true positives)")
    print(f"• Ground Truth Rows:     {gt_cnt:,}")
    print(f"• S1 Singletons:         {sing_cnt:,} ({sing_ratio:.2%})")
    print(f"• Total Generation Time: {gen_time:.1f} seconds")
    print(f"• S3 Destination:        s3://{bucket}/{s3_target_prefix}/")
    print("=" * 80)
    print("  uv run python -m parallax.utils.s3_sync pull \\")
    print(f"    {s3_target_prefix} data/medium_split_200k\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallax Medium Split Generator")
    parser.add_argument(
        "--work-dir", default="./parallax_data", help="Local directory for raw and processed data"
    )
    parser.add_argument("--s1-size", type=int, default=200_000, help="S1 sample size")
    parser.add_argument(
        "--s2-size", type=int, default=500_000, help="S2 target candidate pool size"
    )
    parser.add_argument(
        "--s3-size", type=int, default=500_000, help="S3 target candidate pool size"
    )
    parser.add_argument("--s3-prefix", default="splits/medium_split_200k", help="S3 target prefix")
    parser.add_argument("--no-upload", action="store_true", help="Skip uploading to S3")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # In Colab, sys.argv might contain kernel connection args, so parse known args
    args, _ = parser.parse_known_args()

    # Detect if running in Colab
    is_colab = "google.colab" in sys.modules or os.path.exists("/content")
    default_work_dir = Path("/content/parallax_data") if is_colab else Path(args.work_dir)

    run_medium_split_pipeline(
        work_dir=default_work_dir,
        s1_size=args.s1_size,
        s2_size=args.s2_size,
        s3_size=args.s3_size,
        s3_target_prefix=args.s3_prefix,
        upload_to_s3=not args.no_upload,
        seed=args.seed,
    )
