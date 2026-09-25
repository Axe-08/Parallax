"""
Parallax V1 End-to-End Pipeline Orchestrator
===========================================
Unified CLI runner for:
- Data widening & preprocessing
- High-recall dual-channel sparse blocking
- Pairwise feature extraction
- LightGBM classification & threshold calibration
- Singleton gating & TSV generation (candidate_pairs.tsv & matching_results.tsv)
- Failure logging & diagnostics generation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from parallax.blocking.tfidf_blocker import DualChannelTFIDFBlocker
from parallax.data.contracts import (
    load_business_records_df,
    load_ground_truth_dict,
    write_candidate_pairs_tsv,
    write_matching_results_tsv,
)
from parallax.diagnostics.failure_logger import FailureDiagnosticsLogger
from parallax.features.extractor import PairwiseFeatureExtractor
from parallax.metrics.evaluator import (
    evaluate_blocking_candidates,
    evaluate_resolution_predictions,
)
from parallax.models.matcher import LightGBMMatcher
from parallax.postprocessing.singleton_gate import SingletonGatedPredictor
from parallax.preprocessing.normalizer import widen_records_df


def run_pipeline(
    data_dir: Path | str,
    output_dir: Path | str = "output",
    reports_dir: Path | str = "reports",
    sample_size: int | None = None,
    eval_mode: bool = True,
) -> dict[str, float]:
    """Execute end-to-end V1 pipeline."""
    data_path = Path(data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    rep_path = Path(reports_dir)
    rep_path.mkdir(parents=True, exist_ok=True)

    print("============================================================")
    print("⚡ PARALLAX V1 ENTITY RESOLUTION PIPELINE")
    print("============================================================")
    print(f"Data Directory:    {data_path}")
    print(f"Output Directory:  {out_path}")
    print(f"Reports Directory: {rep_path}\n")

    # 1. Load Data
    s1_file = data_path / "train_source1.tsv"
    s2_file = data_path / "train_source2.tsv"
    s3_file = data_path / "train_source3.tsv"
    gt_file = data_path / "train_ground_truth.tsv"

    print("==> 1. Loading and Validating Input TSV Records...")
    s1_df = load_business_records_df(s1_file)
    s2_df = load_business_records_df(s2_file)
    s3_df = load_business_records_df(s3_file)
    gt_dict = load_ground_truth_dict(gt_file) if gt_file.is_file() else {}

    if sample_size and sample_size < len(s1_df):
        print(f"   (Sampling {sample_size} Source 1 entities for rapid run)")
        s1_df = s1_df.head(sample_size).copy()
        s1_ids = set(s1_df["entity_id"])
        gt_dict = {k: v for k, v in gt_dict.items() if k in s1_ids}

    print(f"   ✓ Loaded S1: {len(s1_df):,} | S2: {len(s2_df):,} | S3: {len(s3_df):,}")
    print(f"   ✓ Ground truth entries: {len(gt_dict):,}\n")

    # 2. Data Widening & Preprocessing
    print("==> 2. Preprocessing & Widening Records (Unicode NFKC, numbers, domains)...")
    s1_wide = widen_records_df(s1_df)
    s2_wide = widen_records_df(s2_df)
    s3_wide = widen_records_df(s3_df)
    target_wide = pd.concat([s2_wide, s3_wide], ignore_index=True)
    print("   ✓ Preprocessing complete.\n")

    # 3. High-Recall Dual-Channel Blocking
    print("==> 3. Running Dual-Channel Sparse TF-IDF Blocker...")
    blocker = DualChannelTFIDFBlocker(name_top_k=25, addr_top_k=20)
    candidates = blocker.generate_candidates(s1_wide, target_wide)

    cand_file = out_path / "candidate_pairs.tsv"
    write_candidate_pairs_tsv(cand_file, candidates)
    print(f"   ✓ Exported blocking candidates: {cand_file}")

    if gt_dict:
        blocking_report = evaluate_blocking_candidates(gt_dict, candidates, len(target_wide))
        pc_pct = blocking_report.pair_completeness * 100.0
        rr_pct = blocking_report.reduction_ratio * 100.0
        print(f"   📊 Blocking Recall (Pair Completeness): {pc_pct:.2f}%")
        print(f"   📊 Reduction Ratio:                    {rr_pct:.4f}%")
        avg_cands = blocking_report.avg_candidates_per_s1
        print(f"   📊 Avg Candidates per S1:              {avg_cands:.1f}\n")

    # 4. Pairwise Feature Extraction
    print("==> 4. Extracting Pairwise RapidFuzz & Structural Features...")
    extractor = PairwiseFeatureExtractor()
    pairs_df = extractor.extract_features_df(candidates, s1_wide, target_wide, ground_truth=gt_dict)
    print(f"   ✓ Extracted {len(pairs_df):,} candidate pairs.\n")

    # 5. Training / Evaluation
    predictor = SingletonGatedPredictor(decision_threshold=0.75)
    all_s1_ids = s1_df["entity_id"].tolist()

    if eval_mode and gt_dict:
        print("==> 5. Disjoint Train/Validation Split & LightGBM Optimization...")
        unique_s1 = list(s1_df["entity_id"].unique())
        np.random.seed(42)
        np.random.shuffle(unique_s1)
        split_idx = int(0.8 * len(unique_s1))
        train_s1 = set(unique_s1[:split_idx])
        val_s1 = set(unique_s1[split_idx:])

        train_pairs = pairs_df[pairs_df["s1_id"].isin(train_s1)].copy()
        val_pairs = pairs_df[pairs_df["s1_id"].isin(val_s1)].copy()

        matcher = LightGBMMatcher(seed=42)
        matcher.train(train_pairs, val_pairs)

        val_gt = {k: v for k, v in gt_dict.items() if k in val_s1}
        best_tau, best_score = matcher.optimize_threshold(val_pairs, val_gt)
        print(f"   ✓ Optimal Threshold tau = {best_tau:.2f} (Macro F0.5 = {best_score:.4f})\n")

        # Predict on validation holdout
        val_pairs["prob"] = matcher.predict_proba(val_pairs)
        val_preds = predictor.filter_predictions(val_pairs, list(val_s1), threshold=best_tau)
        eval_report = evaluate_resolution_predictions(val_gt, val_preds)

        print("==> 6. Automated Failure Diagnostics & Error Analysis...")
        diag_logger = FailureDiagnosticsLogger(reports_dir=rep_path)
        diag_logger.analyze_and_log_failures(
            ground_truth=val_gt,
            candidates=candidates,
            predictions=val_preds,
            scored_pairs_df=val_pairs,
            s1_df=s1_wide,
            target_df=target_wide,
            report=eval_report,
        )
        print(f"   ✓ Structured error logs written to {rep_path / 'failures_v1.jsonl'}")
        print(f"   ✓ Executive diagnostics report written to {rep_path / 'diagnostics_v1.md'}\n")

        # Predict full dataset for submission file
        pairs_df["prob"] = matcher.predict_proba(pairs_df)
        full_preds = predictor.filter_predictions(pairs_df, all_s1_ids, threshold=best_tau)
        match_file = out_path / "matching_results.tsv"
        write_matching_results_tsv(match_file, full_preds)
        print(f"   ✓ Final predictions saved to: {match_file}")

        print("============================================================")
        print(f"🎯 FINAL HOLDOUT MACRO F0.5: {eval_report.macro_f05:.4f}")
        print(f"🎯 SINGLETON ACCURACY:      {eval_report.singleton_score * 100:.2f}%")
        print(f"🎯 NON-SINGLETON F0.5:       {eval_report.non_singleton_f05:.4f}")
        print("============================================================\n")

        return {
            "macro_f05": eval_report.macro_f05,
            "singleton_score": eval_report.singleton_score,
            "non_singleton_f05": eval_report.non_singleton_f05,
        }

    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallax V1 Entity Resolution Pipeline")
    parser.add_argument("--data-dir", default="data/golden_split", help="Path to dataset directory")
    parser.add_argument("--output-dir", default="output", help="Path to output directory")
    parser.add_argument("--reports-dir", default="reports", help="Path to reports directory")
    parser.add_argument(
        "--sample-size", type=int, default=None, help="Optional sample limit for testing"
    )
    parser.add_argument("--eval", action="store_true", default=True, help="Run in evaluation mode")
    args = parser.parse_args()

    run_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        reports_dir=args.reports_dir,
        sample_size=args.sample_size,
        eval_mode=args.eval,
    )


if __name__ == "__main__":
    main()
