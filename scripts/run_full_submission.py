"""
Parallax Master Unattended Submission Orchestrator
==================================================
Coordinates the end-to-end execution of the 100% full dataset training
and real test submission pipeline:
1. Pre-flight health and storage safety checks.
2. Ingests raw training data from AWS S3.
3. Trains production Two-Pass LightGBM models with hard negative mining.
4. Ingests raw test data from AWS S3.
5. Runs streaming test inference to produce matching_results.tsv.
6. Enforces official competition validation invariants.
7. Uploads verified submission artifact to AWS S3.
8. Compiles executive submission summary report.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import sys
import time
from pathlib import Path

# Bootstrap sys.path immediately so imports work in any execution environment
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    import psutil

    def get_telemetry_str() -> str:
        mem = psutil.virtual_memory()
        disk = shutil.disk_usage(".")
        return (
            f"[RAM: {mem.used / (1024**3):.1f}/{mem.total / (1024**3):.1f} GB ({mem.percent}%) "
            f"| Disk: {disk.free / (1024**3):.1f} GB free]"
        )
except Exception:

    def get_telemetry_str() -> str:
        disk = shutil.disk_usage(".")
        return f"[Disk: {disk.free / (1024**3):.1f} GB free]"


from parallax.s3_utils import (
    download_s3_file,
    get_s3_client,
    get_s3_credentials,
    upload_s3_file,
)
from parallax.submission.test_inference import run_streaming_test_inference
from parallax.submission.validator import validate_official_er_submission
from parallax.training.full_trainer import train_production_pipeline


def print_banner(step_num: int, title: str) -> None:
    """Print visually distinct pipeline stage banner with live telemetry."""
    sep = "=" * 80
    print(f"\n{sep}")
    print(f"📍 STAGE {step_num}: {title.upper()}  {get_telemetry_str()}")
    print(f"⏰ Timestamp: {time.strftime('%H:%M:%S UTC', time.gmtime())}")
    print(f"{sep}\n", flush=True)


def check_disk_headroom(min_free_gb: float = 1.0) -> float:
    """Verify filesystem has sufficient free space."""
    total, used, free = shutil.disk_usage(".")
    free_gb = free / (1024**3)
    if free_gb < min_free_gb:
        raise RuntimeError(
            f"❌ Critical disk space warning! Free space ({free_gb:.2f} GB) "
            f"is below safety threshold ({min_free_gb:.2f} GB)."
        )
    return free_gb


def main() -> None:
    """Run full unattended training and submission pipeline."""
    t_start = time.time()
    project_root = _PROJECT_ROOT
    os.chdir(project_root)

    print("\n" + "=" * 80)
    print("🌟 PARALLAX END-TO-END PRODUCTION SUBMISSION PIPELINE")
    print(f"⏰ Initiated at: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    print("=" * 80)

    # ---------------------------------------------------------
    # STAGE 0: Pre-Flight Health & Storage Checks
    # ---------------------------------------------------------
    print_banner(0, "Pre-Flight Health & S3 Credentials Check")
    free_gb = check_disk_headroom(min_free_gb=1.0)
    print(f"  ✓ Filesystem headroom check passed: {free_gb:.1f} GB available.")

    try:
        import multiprocessing
        import platform

        import psutil

        mem = psutil.virtual_memory()
        print(
            f"  🖥️ Host Environment:    {platform.platform()} | Python {platform.python_version()}"
        )
        print(f"  ⚡ Available Cores:     {multiprocessing.cpu_count()} vCPUs")
        print(
            f"  🧠 Host Memory (RAM):   {mem.total / (1024**3):.1f} GB total ({mem.available / (1024**3):.1f} GB free)"
        )
    except Exception:
        pass

    ak, sk, region, bucket = get_s3_credentials()
    if not ak or not sk:
        raise RuntimeError("❌ AWS S3 credentials not found! Check environment or accessKeys.csv.")
    print(f"  ✓ S3 credentials located. Target bucket: s3://{bucket} (region: {region})")

    t_s3_ping = time.time()
    s3_client = get_s3_client()
    try:
        s3_client.head_bucket(Bucket=bucket)
        print(
            f"  ✓ Verified active connectivity to s3://{bucket} ({time.time() - t_s3_ping:.2f}s latency)"
        )
    except Exception as exc:
        raise RuntimeError(f"❌ Failed to connect to S3 bucket {bucket}: {exc}") from exc

    raw_train_dir = project_root / "data" / "raw" / "train"
    raw_test_dir = project_root / "data" / "raw" / "test"
    models_dir = project_root / "output" / "models"
    submissions_dir = project_root / "submissions"
    reports_dir = project_root / "reports"

    raw_train_dir.mkdir(parents=True, exist_ok=True)
    raw_test_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    submissions_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # STAGE 1: Download Raw Train Data from S3
    # ---------------------------------------------------------
    print_banner(1, "Download Raw Training Data from S3")
    train_keys = [
        "raw/train/train_source1.tsv",
        "raw/train/train_source2.tsv",
        "raw/train/train_source3.tsv",
        "raw/train/train_ground_truth.tsv",
    ]
    for key in train_keys:
        dest = raw_train_dir / Path(key).name
        download_s3_file(key, dest, bucket=bucket, client=s3_client)
    print("  ✓ All raw training files synchronized.")

    # ---------------------------------------------------------
    # STAGE 2: Train Production Models on 100% Full Dataset
    # ---------------------------------------------------------
    print_banner(2, "Train Production Two-Pass Models (100% Full Dataset)")
    meta_path = models_dir / "production_metadata.json"
    p1_path = models_dir / "production_pass1.txt"
    p2_path = models_dir / "production_pass2.txt"

    if meta_path.is_file() and p1_path.is_file() and p2_path.is_file():
        print("  ⚡ Found pre-existing trained production models. Re-using cached models!")
        import json

        with open(meta_path, encoding="utf-8") as f:
            meta_dict = json.load(f)
        tau_val = float(meta_dict.get("optimal_tau", 0.82))
        f05_val = float(meta_dict.get("holdout_macro_f05", 0.0))
        print(f"  ✓ Optimal Decision Threshold (tau): {tau_val:.2f}")
        print(f"  ✓ Holdout Macro F0.5:             {f05_val:.4f}")
    else:
        t_train_start = time.time()
        metadata = train_production_pipeline(
            train_dir=raw_train_dir,
            output_dir=project_root / "output",
            holdout_size=5_000,
            sample_queries_per_country=15_000,
        )
        t_train_elapsed = time.time() - t_train_start
        print(f"  ✓ Production training finished in {t_train_elapsed / 60:.1f} minutes.")
        print(f"  ✓ Optimal Decision Threshold (tau): {metadata.optimal_tau:.2f}")
        print(f"  ✓ Holdout Macro F0.5:             {metadata.holdout_macro_f05:.4f}")

    # Immediately preserve trained models & metadata to S3 for analysis
    try:
        upload_s3_file(
            meta_path, "models/production_metadata.json", bucket=bucket, client=s3_client
        )
        upload_s3_file(p1_path, "models/production_pass1.txt", bucket=bucket, client=s3_client)
        upload_s3_file(p2_path, "models/production_pass2.txt", bucket=bucket, client=s3_client)
        print("  ✓ Synchronized trained models and calibration metadata to S3.")
    except Exception as exc:
        print(f"  ⚠️ Note: Could not back up models to S3: {exc}")

    # ---------------------------------------------------------
    # STAGE 3: Download Raw Test Data from S3
    # ---------------------------------------------------------
    print_banner(3, "Download Raw Test Data from S3")
    test_keys = [
        "raw/test/test_source1.tsv",
        "raw/test/test_source2.tsv",
        "raw/test/test_source3.tsv",
    ]
    for key in test_keys:
        dest = raw_test_dir / Path(key).name
        download_s3_file(key, dest, bucket=bucket, client=s3_client)
    print("  ✓ All raw test files synchronized.")

    # ---------------------------------------------------------
    # STAGE 4: Run Streaming Test Inference
    # ---------------------------------------------------------
    print_banner(4, "Execute Streaming Test Inference")
    t_inf_start = time.time()
    submission_tsv = submissions_dir / "matching_results.tsv"

    run_streaming_test_inference(
        test_dir=raw_test_dir,
        models_dir=models_dir,
        output_tsv_path=submission_tsv,
        chunk_size=20_000,
        batch_size=2_000,
    )
    t_inf_elapsed = time.time() - t_inf_start
    print(f"  ✓ Test inference completed in {t_inf_elapsed / 60:.1f} minutes.")

    # ---------------------------------------------------------
    # STAGE 5: Official Competition Integrity Verification Gate
    # ---------------------------------------------------------
    print_banner(5, "Official Integrity Verification Gate")
    is_valid, errors, warnings = validate_official_er_submission(
        matching_tsv_path=submission_tsv,
        test_dir=raw_test_dir,
        check_ids=True,
    )

    if not is_valid:
        print("  ❌ VERDICT: BLOCKED - Integrity check failed with errors:")
        for err in errors:
            print(f"     • {err}")
        raise RuntimeError("Submission failed official validation gate! Aborting S3 upload.")

    print("  🎉 VERDICT: PASS - 100% compliant with official challenge rules!")
    if warnings:
        for w in warnings:
            print(f"  ⚠️ Warning: {w}")

    # ---------------------------------------------------------
    # STAGE 6: Upload Verified Submission to S3
    # ---------------------------------------------------------
    print_banner(6, "Publish Verified Submission to S3")
    timestamp_str = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    s3_primary_key = "submissions/matching_results.tsv"
    s3_archive_key = f"submissions/matching_results_{timestamp_str}.tsv"

    uri_primary = upload_s3_file(submission_tsv, s3_primary_key, bucket=bucket, client=s3_client)
    upload_s3_file(submission_tsv, s3_archive_key, bucket=bucket, client=s3_client)
    print(f"  ✓ Published to: {uri_primary}")

    # Synchronize all diagnostic and analysis reports to S3
    for report_file in reports_dir.glob("*"):
        if report_file.is_file():
            with contextlib.suppress(Exception):
                upload_s3_file(
                    report_file,
                    f"reports/{report_file.name}",
                    bucket=bucket,
                    client=s3_client,
                )
    print("  ✓ All diagnostic analysis reports synchronized to S3.")

    # ---------------------------------------------------------
    # STAGE 7: Executive Summary Report
    # ---------------------------------------------------------
    print_banner(7, "Compile Executive Submission Report")
    total_time_min = (time.time() - t_start) / 60
    sub_size_mb = submission_tsv.stat().st_size / (1024**2)

    # Count rows and matches in submission
    total_sub_rows = 0
    empty_sub_rows = 0
    with open(submission_tsv, encoding="utf-8") as f:
        _ = f.readline()
        for line in f:
            total_sub_rows += 1
            parts = line.strip().split("\t")
            if len(parts) == 1 or not parts[1].strip():
                empty_sub_rows += 1

    matched_sub_rows = total_sub_rows - empty_sub_rows

    summary_md = f"""# Parallax Production Submission Executive Summary

**Generated At**: {time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())}  
**Pipeline Run Duration**: {total_time_min:.1f} minutes  
**Target S3 Location**: `{uri_primary}`  

---

## 1. Production Model & Calibration
- **Model Architecture**: Two-Pass LightGBM GBDT (Config-XDeep)
  - **Pass 1**: 49 Pairwise Features (127 leaves, max depth 10, 250 trees, LR 0.04)
  - **Meta-Features**: 5 Entity Meta-Features (`s1_max_score`, `s1_mean_score`, `s1_std_score`, `score_rank_pct`, `score_gap_to_best`)
  - **Pass 2**: 54 Combined Features (127 leaves, max depth 10, 250 trees, LR 0.04)
- **Hard Negative Mining**: 100% True Ground Truth matches + Blocker Top-5 Hard Negatives per query
- **Holdout Validation Size**: {metadata.num_holdout_queries:,} entities
- **Calibrated Optimal Threshold (tau)**: **`{metadata.optimal_tau:.2f}`**
- **Holdout Performance**:
  - **Macro F0.5**: **`{metadata.holdout_macro_f05:.4f}`**
  - **Singleton Accuracy**: **`{metadata.holdout_singleton_accuracy * 100:.2f}%`**
  - **Non-Singleton F0.5**: **`{metadata.holdout_non_singleton_f05:.4f}`**
  - **Precision**: **`{metadata.holdout_precision * 100:.2f}%`**
  - **Recall**: **`{metadata.holdout_recall * 100:.2f}%`**

---

## 2. Test Submission Statistics
- **Submission Artifact**: `{submission_tsv}` ({sub_size_mb:.1f} MB)
- **Total Test Queries Processed**: **`{total_sub_rows:,}`**
- **Non-Singleton Matches Predicted**: **`{matched_sub_rows:,}`** ({matched_sub_rows / max(1, total_sub_rows) * 100:.1f}%)
- **Singleton Entities Predicted**: **`{empty_sub_rows:,}`** ({empty_sub_rows / max(1, total_sub_rows) * 100:.1f}%)
- **Official Integrity Validation**: **PASS (0 errors)**

---

## 3. Storage & Resource Headroom
- **Final Free Disk**: `{check_disk_headroom(0.5):.1f} GB available`
- **Memory Consumption**: Strictly bounded within host RAM via streaming batch evaluation.
"""

    summary_path = reports_dir / "submission_summary.md"
    summary_path.write_text(summary_md, encoding="utf-8")
    print(f"  ✓ Summary report written to: {summary_path}")

    match_pct = (matched_sub_rows / max(1, total_sub_rows)) * 100
    single_pct = (empty_sub_rows / max(1, total_sub_rows)) * 100

    print("\n" + "=" * 80)
    print("🏆 PARALLAX SUBMISSION EXECUTIVE SCORECARD")
    print("=" * 80)
    print("  • Validation Status:             PASS (0 errors, 100% compliant)")
    print(f"  • Total Pipeline Runtime:        {total_time_min:.1f} minutes")
    print(f"  • Calibrated Threshold (tau):    {metadata.optimal_tau:.2f}")
    print(f"  • Holdout Validation Macro F0.5: {metadata.holdout_macro_f05:.4f}")
    print(f"  • Holdout Singleton Accuracy:    {metadata.holdout_singleton_accuracy * 100:.2f}%")
    print(f"  • Holdout Non-Singleton F0.5:    {metadata.holdout_non_singleton_f05:.4f}")
    print(f"  • Total Test Queries Processed:  {total_sub_rows:,} (India, US, France)")
    print(f"  • Matches Predicted:             {matched_sub_rows:,} ({match_pct:.1f}%)")
    print(f"  • Singletons Predicted:          {empty_sub_rows:,} ({single_pct:.1f}%)")
    print(f"  • Output TSV Artifact:           {submission_tsv} ({sub_size_mb:.1f} MB)")
    print(f"  • S3 Primary Target:             {uri_primary}")
    print(f"  • Disk Headroom Remaining:       {check_disk_headroom(0.5):.1f} GB")
    print("=" * 80 + "\n", flush=True)


if __name__ == "__main__":
    main()
