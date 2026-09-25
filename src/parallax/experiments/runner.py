"""
Parallax 200k Medium Benchmark Experiment Engine
================================================
Unified runner for:
- Preprocessing & widening
- Sparse dual-channel candidate blocking with persistent Parquet caching
- RapidFuzz pairwise feature extraction with Parquet caching
- Hyperparameter grid sweep & threshold optimization (Fold 0 holdout)
- Full 5-fold cross-validation with OOF prediction aggregation
- Comprehensive failure logging & root-cause diagnostic reporting.
"""

from __future__ import annotations

import argparse
import os
import shutil
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from parallax.blocking.tfidf_blocker import DualChannelTFIDFBlocker
from parallax.data.contracts import (
    load_business_records_df,
    load_ground_truth_dict,
    write_matching_results_tsv,
)
from parallax.diagnostics.execution_logger import (
    get_global_logger,
    install_global_exception_handler,
    pipeline_stage,
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
from parallax.utils.checkpoint_manager import CheckpointManager


@dataclass
class HyperparamConfig:
    name: str
    learning_rate: float
    num_leaves: int
    max_depth: int
    n_estimators: int


@dataclass
class SweepScore:
    config: HyperparamConfig
    optimal_tau: float
    macro_f05: float
    singleton_acc: float
    non_singleton_f05: float


@dataclass
class FoldMetric:
    fold: int
    train_entities: int
    val_entities: int
    train_pairs: int
    val_pairs: int
    optimal_tau: float
    macro_f05: float
    singleton_acc: float
    non_singleton_f05: float
    precision: float
    recall: float


def print_banner(title: str) -> None:
    sep = "=" * 80
    print(f"\n{sep}")
    print(f"🚀 {title}")
    print(f"{sep}\n")


def print_resource_snapshot() -> None:
    print("--- [Hardware & Resource Status] ---")
    total, used, free = shutil.disk_usage("/")
    print(f"  💾 Root Disk:      {free / (1024**3):.1f} GB free / {total / (1024**3):.1f} GB total")
    try:
        import psutil

        mem = psutil.virtual_memory()
        avail_gb = mem.available / (1024**3)
        tot_gb = mem.total / (1024**3)
        print(f"  🧠 Memory (RAM):   {avail_gb:.1f} GB available / {tot_gb:.1f} GB total")
    except ImportError:
        pass
    print(f"  ⚙️  CPU Threads:    {os.cpu_count()} logical cores\n")


def generate_or_load_candidates(
    s1_wide: pd.DataFrame,
    target_wide: pd.DataFrame,
    cache_path: Path,
    blocker_top_k: int = 35,
    blocker_min_sim: float = 0.15,
    checkpoint_mgr: CheckpointManager | None = None,
) -> dict[str, set[str]]:
    """Generate or retrieve candidate pairs mapping s1_id -> set of candidate entity_ids."""
    if cache_path.is_file():
        print(f"  ⚡ Found cached candidate pairs at: {cache_path}")
        t0 = time.time()
        cands_df = pd.read_parquet(cache_path)
        candidates: dict[str, set[str]] = {s1: set() for s1 in s1_wide["entity_id"]}
        s1_arr = cands_df["s1_id"].to_numpy()
        cand_arr = cands_df["cand_id"].to_numpy()
        for s1, cand in zip(s1_arr, cand_arr, strict=False):
            if s1 in candidates:
                candidates[s1].add(cand)
        print(f"  ✓ Loaded {len(cands_df):,} cached pairs in {time.time() - t0:.2f}s.\n")
        return candidates

    msg = (
        f"  ⚡ Running sparse TF-IDF blocker (top_k={blocker_top_k}, min_sim={blocker_min_sim})..."
    )
    print(msg)
    t0 = time.time()
    blocker = DualChannelTFIDFBlocker(
        name_top_k=blocker_top_k,
        addr_top_k=25,
        name_min_sim=blocker_min_sim,
        addr_min_sim=0.20,
        batch_size=2000,
        show_progress=True,
    )
    candidates = blocker.generate_candidates(s1_wide, target_wide, checkpoint_mgr=checkpoint_mgr)

    elapsed = time.time() - t0
    total_pairs = sum(len(c) for c in candidates.values())
    print(
        f"  ✓ Generated {total_pairs:,} candidate pairs across "
        f"{len(candidates):,} S1 queries in {elapsed:.2f}s."
    )

    # Cache as Parquet
    print(f"  💾 Caching candidate pairs to: {cache_path}...")
    cache_rows_s1: list[str] = []
    cache_rows_cand: list[str] = []
    for s1, cands in candidates.items():
        for cand in cands:
            cache_rows_s1.append(s1)
            cache_rows_cand.append(cand)

    pd.DataFrame({"s1_id": cache_rows_s1, "cand_id": cache_rows_cand}).to_parquet(
        cache_path, compression="snappy", index=False
    )
    print("  ✓ Caching complete.\n")
    return candidates


def extract_or_load_features(
    candidates: Mapping[str, set[str]],
    s1_wide: pd.DataFrame,
    target_wide: pd.DataFrame,
    gt_dict: Mapping[str, set[str]],
    cache_path: Path,
) -> pd.DataFrame:
    """Extract or load pairwise similarity features."""
    if cache_path.is_file():
        print(f"  ⚡ Found cached feature matrix at: {cache_path}")
        t0 = time.time()
        features_df = pd.read_parquet(cache_path)
        print(f"  ✓ Loaded {len(features_df):,} feature rows in {time.time() - t0:.2f}s.\n")
        return features_df

    print("  ⚡ Extracting RapidFuzz and structural features across candidate pairs...")
    t0 = time.time()
    extractor = PairwiseFeatureExtractor()
    features_df = extractor.extract_features_df(
        candidates, s1_wide, target_wide, ground_truth=gt_dict, show_progress=True
    )
    elapsed = time.time() - t0
    print(f"  ✓ Feature extraction finished in {elapsed:.2f}s ({len(features_df):,} rows).")

    print(f"  💾 Caching feature matrix to: {cache_path}...")
    features_df.to_parquet(cache_path, compression="snappy", index=False)
    print("  ✓ Caching complete.\n")
    return features_df


def run_sweep(
    train_pairs: pd.DataFrame,
    val_pairs: pd.DataFrame,
    val_gt: Mapping[str, set[str]],
    checkpoint_mgr: CheckpointManager | None = None,
) -> tuple[HyperparamConfig, float]:
    """Execute hyperparameter sweep on validation holdout to find peak Macro F0.5."""
    print("================================================================================")
    print("🔬 STAGE 4: HYPERPARAMETER GRID SWEEP (VAL HOLDOUT)")
    print("================================================================================")

    if checkpoint_mgr is not None and checkpoint_mgr.has_checkpoint("sweep_results", ext="json"):
        data = checkpoint_mgr.load_json("sweep_results")
        if data and "config" in data and "optimal_tau" in data:
            best_cfg = HyperparamConfig(**data["config"])
            base_tau = float(data["optimal_tau"])
            print(
                f"  ⚡ [Checkpoint] Loaded winning sweep config: "
                f"{best_cfg.name} (tau={base_tau:.2f})\n"
            )
            return best_cfg, base_tau

    configs = [
        HyperparamConfig(
            name="Config-Fast",
            learning_rate=0.08,
            num_leaves=31,
            max_depth=6,
            n_estimators=100,
        ),
        HyperparamConfig(
            name="Config-Balanced",
            learning_rate=0.05,
            num_leaves=31,
            max_depth=6,
            n_estimators=150,
        ),
        HyperparamConfig(
            name="Config-Deep",
            learning_rate=0.05,
            num_leaves=63,
            max_depth=8,
            n_estimators=150,
        ),
        HyperparamConfig(
            name="Config-Conservative",
            learning_rate=0.03,
            num_leaves=31,
            max_depth=6,
            n_estimators=180,
        ),
    ]

    predictor = SingletonGatedPredictor()
    val_s1_ids = list(val_gt.keys())
    scores: list[SweepScore] = []

    print(f"Sweeping {len(configs)} configs on {len(val_s1_ids):,} entities...")
    header = (
        f"{'Config':<18} | {'LR':<5} | {'Leaves':<6} | {'Trees':<5} | "
        f"{'tau':<6} | {'Macro F0.5':<10} | {'Sing. Acc':<10} | {'Non-Sing F0.5':<12}"
    )
    print(header)
    print("-" * len(header))

    for cfg in tqdm(configs, desc="  ⚡ Tuning Fold 0", unit="config", leave=False):
        matcher = LightGBMMatcher(
            learning_rate=cfg.learning_rate,
            num_leaves=cfg.num_leaves,
            max_depth=cfg.max_depth,
            n_estimators=cfg.n_estimators,
            seed=42,
        )
        matcher.train(train_pairs, val_pairs)

        search_range = [0.60, 0.65, 0.70, 0.74, 0.78, 0.82, 0.86, 0.90]
        best_tau, _ = matcher.optimize_threshold(val_pairs, val_gt, search_range=search_range)

        # Evaluate complete metrics at best tau
        val_copy = val_pairs[["s1_id", "cand_id"]].copy()
        val_copy["prob"] = matcher.predict_proba(val_pairs)
        preds = predictor.filter_predictions(val_copy, val_s1_ids, threshold=best_tau)
        report = evaluate_resolution_predictions(val_gt, preds)

        scores.append(
            SweepScore(
                config=cfg,
                optimal_tau=best_tau,
                macro_f05=report.macro_f05,
                singleton_acc=report.singleton_score,
                non_singleton_f05=report.non_singleton_f05,
            )
        )

        row_str = (
            f"{cfg.name:<18} | {cfg.learning_rate:<5.2f} | {cfg.num_leaves:<6} | "
            f"{cfg.n_estimators:<5} | {best_tau:<6.2f} | {report.macro_f05:<10.4f} | "
            f"{report.singleton_score * 100:<9.2f}% | {report.non_singleton_f05:<12.4f}"
        )
        print(row_str)

    best_sweep = max(scores, key=lambda s: s.macro_f05)
    print("-" * len(header))
    print(
        f"🏆 WINNING CONFIG: {best_sweep.config.name} "
        f"(Macro F0.5 = {best_sweep.macro_f05:.4f}, tau = {best_sweep.optimal_tau:.2f})\n"
    )

    if checkpoint_mgr is not None:
        checkpoint_mgr.save_json(
            "sweep_results",
            {
                "config": {
                    "name": best_sweep.config.name,
                    "learning_rate": best_sweep.config.learning_rate,
                    "num_leaves": best_sweep.config.num_leaves,
                    "max_depth": best_sweep.config.max_depth,
                    "n_estimators": best_sweep.config.n_estimators,
                },
                "optimal_tau": best_sweep.optimal_tau,
                "macro_f05": best_sweep.macro_f05,
            },
        )
        checkpoint_mgr.record_stage_completed(
            "sweep",
            {"winning_config": best_sweep.config.name, "tau": best_sweep.optimal_tau},
        )

    return best_sweep.config, best_sweep.optimal_tau


def run_cross_validation(
    features_df: pd.DataFrame,
    cv_folds_df: pd.DataFrame,
    gt_dict: Mapping[str, set[str]],
    best_cfg: HyperparamConfig,
    base_tau: float,
    checkpoint_mgr: CheckpointManager | None = None,
) -> tuple[list[FoldMetric], dict[str, set[str]], pd.DataFrame]:
    """Execute 5-fold cross-validation, return fold metrics and complete OOF predictions."""
    print("================================================================================")
    print("🔁 STAGE 5: FULL 5-FOLD CROSS VALIDATION")
    print("================================================================================")

    # Map entity_id to fold
    fold_map = dict(
        zip(cv_folds_df["entity_id"].astype(str), cv_folds_df["fold"].astype(int), strict=False)
    )
    s1_fold_arr = np.array([fold_map.get(s1, -1) for s1 in features_df["s1_id"].astype(str)])

    predictor = SingletonGatedPredictor()
    fold_results: list[FoldMetric] = []
    oof_predictions: dict[str, set[str]] = {
        s1: set() for s1 in cv_folds_df["entity_id"].astype(str)
    }
    oof_scored_parts: list[pd.DataFrame] = []

    folds = sorted(cv_folds_df["fold"].unique())
    header = (
        f"{'Fold':<5} | {'Train':<10} | {'Val':<8} | {'tau':<5} | "
        f"{'Macro F0.5':<10} | {'Sing. Acc':<10} | "
        f"{'Non-Sing F0.5':<12} | {'Prec.':<8} | {'Recall':<7}"
    )
    print(header)
    print("-" * len(header))

    for k in tqdm(folds, desc="  ⚡ 5-Fold Cross Validation", unit="fold", leave=False):
        val_s1_set = set(cv_folds_df[cv_folds_df["fold"] == k]["entity_id"].astype(str))
        val_gt = {k_id: v for k_id, v in gt_dict.items() if k_id in val_s1_set}

        # Check for completed fold checkpoint
        if (
            checkpoint_mgr is not None
            and checkpoint_mgr.has_checkpoint(f"fold_{k}_metrics", ext="json")
            and checkpoint_mgr.has_checkpoint(f"fold_{k}_scored_pairs")
        ):
            metric_data = checkpoint_mgr.load_json(f"fold_{k}_metrics")
            val_pairs_scored = checkpoint_mgr.load_dataframe(f"fold_{k}_scored_pairs")
            if metric_data is not None and val_pairs_scored is not None:
                cached_metric = FoldMetric(**metric_data)
                fold_results.append(cached_metric)
                oof_scored_parts.append(val_pairs_scored[["s1_id", "cand_id", "prob"]])
                cached_preds = predictor.filter_predictions(
                    val_pairs_scored, list(val_s1_set), threshold=cached_metric.optimal_tau
                )
                for s1, p_set in cached_preds.items():
                    oof_predictions[s1] = p_set

                m_f05 = cached_metric.macro_f05
                s_acc = cached_metric.singleton_acc * 100
                ns_f05 = cached_metric.non_singleton_f05
                prec = cached_metric.precision * 100
                rec = cached_metric.recall * 100
                row_str = (
                    f"{k:<5} | {cached_metric.train_pairs:<10,} | {cached_metric.val_pairs:<8,} | "
                    f"{cached_metric.optimal_tau:<5.2f} | {m_f05:<10.4f} | "
                    f"{s_acc:<9.2f}% | {ns_f05:<12.4f} | "
                    f"{prec:<7.2f}% | {rec:<6.2f}% (cached)"
                )
                print(row_str)
                continue

        val_mask = s1_fold_arr == k
        train_mask = (s1_fold_arr != k) & (s1_fold_arr != -1)

        train_pairs = features_df[train_mask]
        val_pairs = features_df[val_mask].copy()

        matcher = LightGBMMatcher(
            learning_rate=best_cfg.learning_rate,
            num_leaves=best_cfg.num_leaves,
            max_depth=best_cfg.max_depth,
            n_estimators=best_cfg.n_estimators,
            seed=42 + k,
        )
        matcher.train(train_pairs, val_pairs)

        search_range = [
            base_tau - 0.08,
            base_tau - 0.04,
            base_tau,
            base_tau + 0.04,
            base_tau + 0.08,
        ]
        search_range = [t for t in search_range if 0.50 <= t <= 0.95]
        best_tau_k, _ = matcher.optimize_threshold(val_pairs, val_gt, search_range=search_range)

        # Out-of-fold inference
        val_pairs["prob"] = matcher.predict_proba(val_pairs)
        fold_preds = predictor.filter_predictions(val_pairs, list(val_s1_set), threshold=best_tau_k)

        # Merge fold predictions into OOF set
        for s1, p_set in fold_preds.items():
            oof_predictions[s1] = p_set

        oof_scored_parts.append(val_pairs[["s1_id", "cand_id", "prob"]])

        report = evaluate_resolution_predictions(val_gt, fold_preds)
        prec = (
            report.total_correct_pairs / report.total_predicted_pairs
            if report.total_predicted_pairs > 0
            else 0.0
        )
        rec = (
            report.total_correct_pairs / report.total_true_pairs
            if report.total_true_pairs > 0
            else 0.0
        )
        fold_metric = FoldMetric(
            fold=int(k),
            train_entities=len(cv_folds_df[cv_folds_df["fold"] != k]),
            val_entities=len(val_s1_set),
            train_pairs=len(train_pairs),
            val_pairs=len(val_pairs),
            optimal_tau=best_tau_k,
            macro_f05=report.macro_f05,
            singleton_acc=report.singleton_score,
            non_singleton_f05=report.non_singleton_f05,
            precision=prec,
            recall=rec,
        )
        fold_results.append(fold_metric)

        if checkpoint_mgr is not None:
            checkpoint_mgr.save_dataframe(
                f"fold_{k}_scored_pairs", val_pairs[["s1_id", "cand_id", "prob"]]
            )
            checkpoint_mgr.save_json(f"fold_{k}_metrics", asdict(fold_metric))
            checkpoint_mgr.record_stage_completed(
                f"fold_{k}",
                {"optimal_tau": best_tau_k, "macro_f05": report.macro_f05},
            )

        row_str = (
            f"{k:<5} | {len(train_pairs):<10,} | {len(val_pairs):<8,} | {best_tau_k:<5.2f} | "
            f"{report.macro_f05:<10.4f} | {report.singleton_score * 100:<9.2f}% | "
            f"{report.non_singleton_f05:<12.4f} | {prec * 100:<7.2f}% | "
            f"{rec * 100:<6.2f}%"
        )
        print(row_str)

    print("-" * len(header))
    mean_f05 = np.mean([r.macro_f05 for r in fold_results])
    std_f05 = np.std([r.macro_f05 for r in fold_results])
    mean_sing = np.mean([r.singleton_acc for r in fold_results])
    mean_non_sing = np.mean([r.non_singleton_f05 for r in fold_results])

    print(
        f"📊 5-FOLD CV: Macro F0.5 = {mean_f05:.4f} ± {std_f05:.4f} | "
        f"Sing. Acc = {mean_sing * 100:.2f}% | Non-Sing F0.5 = {mean_non_sing:.4f}\n"
    )

    oof_scored_df = pd.concat(oof_scored_parts, ignore_index=True)
    return fold_results, oof_predictions, oof_scored_df


def run_benchmark(
    data_dir: Path | str = "data/medium_split_200k",
    output_dir: Path | str = "output",
    reports_dir: Path | str = "reports",
    checkpoint_dir: Path | str | None = None,
    reset_checkpoints: bool = False,
    skip_sweep: bool = False,
    sample_s1: int | None = None,
) -> None:
    """Execute complete 200k benchmark workflow."""
    data_path = Path(data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    rep_path = Path(reports_dir)
    rep_path.mkdir(parents=True, exist_ok=True)
    chk_path = Path(checkpoint_dir) if checkpoint_dir else out_path / "checkpoints"
    checkpoint_mgr = CheckpointManager(chk_path, reset=reset_checkpoints)

    print_banner("PARALLAX 200K BENCHMARK & ERROR DIAGNOSTICS SUITE")
    print(f"Data Source:       {data_path.resolve()}")
    print(f"Outputs:           {out_path.resolve()}")
    print(f"Diagnostic Logs:   {rep_path.resolve()}")
    print(f"Checkpoints:       {chk_path.resolve()}\n")

    print_resource_snapshot()

    exec_logger = get_global_logger()

    # Step 1: Load Data
    print("--- [Step 1: Loading 200k Benchmark Split] ---")
    with pipeline_stage("Step 1: Loading Data", logger=exec_logger):
        t0 = time.time()
        s1_df = load_business_records_df(data_path / "train_source1.tsv")
        s2_df = load_business_records_df(data_path / "train_source2.tsv")
        s3_df = load_business_records_df(data_path / "train_source3.tsv")
        gt_dict = load_ground_truth_dict(data_path / "train_ground_truth.tsv")
        cv_folds_df = pd.read_csv(data_path / "cv_folds_source1.tsv", sep="\t")

        if sample_s1 and sample_s1 < len(s1_df):
            print(f"  ⚡ Running rapid test sample with {sample_s1:,} S1 entities...")
            s1_df = s1_df.head(sample_s1).copy()
            valid_ids = set(s1_df["entity_id"])
            gt_dict = {k: v for k, v in gt_dict.items() if k in valid_ids}
            cv_folds_df = cv_folds_df[cv_folds_df["entity_id"].isin(valid_ids)].reset_index(
                drop=True
            )

        print(f"  ✓ S1 Queries:        {len(s1_df):,}")
        print(f"  ✓ S2 Candidates:     {len(s2_df):,}")
        print(f"  ✓ S3 Candidates:     {len(s3_df):,}")
        print(f"  ✓ Total Target Pool: {len(s2_df) + len(s3_df):,}")
        print(f"  ✓ Ground Truth Rows: {len(gt_dict):,}")
        n_f = cv_folds_df["fold"].nunique()
        print(f"  ✓ CV Folds defined:  {n_f} folds ({time.time() - t0:.2f}s)\n")

    exec_logger.check_memory_threshold()

    # Step 2: Widening
    print("--- [Step 2: Preprocessing & Unicode / Entity Widening] ---")
    with pipeline_stage("Step 2: Preprocessing & Widening", logger=exec_logger):
        t0 = time.time()
        s1_tag = f"s1_wide_{len(s1_df)}" if sample_s1 else "s1_wide"
        target_tag = "target_wide"
        loaded_s1 = (
            checkpoint_mgr.load_dataframe(s1_tag)
            if not reset_checkpoints and checkpoint_mgr.has_checkpoint(s1_tag)
            else None
        )
        loaded_target = (
            checkpoint_mgr.load_dataframe(target_tag)
            if not reset_checkpoints and checkpoint_mgr.has_checkpoint(target_tag)
            else None
        )
        if loaded_s1 is not None and loaded_target is not None:
            print("  ⚡ [Checkpoint] Loading cached widened tables...")
            s1_wide = loaded_s1
            target_wide = loaded_target
            print(
                f"  ✓ Loaded {len(s1_wide):,} S1 and {len(target_wide):,} "
                f"target records in {time.time() - t0:.2f}s.\n"
            )
        else:
            s1_wide = widen_records_df(s1_df)
            s2_wide = widen_records_df(s2_df)
            s3_wide = widen_records_df(s3_df)
            target_wide = pd.concat([s2_wide, s3_wide], ignore_index=True)
            checkpoint_mgr.save_dataframe(s1_tag, s1_wide)
            checkpoint_mgr.save_dataframe(target_tag, target_wide)
            checkpoint_mgr.record_stage_completed(
                "widening", {"s1_rows": len(s1_wide), "target_rows": len(target_wide)}
            )
            print(f"  ✓ Records widened and transliterated in {time.time() - t0:.2f}s.\n")

    exec_logger.check_memory_threshold()

    # Step 3: Candidate Generation (with Parquet caching)
    print("--- [Step 3: Dual-Channel Sparse Blocking] ---")
    with pipeline_stage("Step 3: Candidate Blocking", logger=exec_logger):
        cand_cache = out_path / (
            "candidate_pairs_sample.parquet" if sample_s1 else "candidate_pairs_medium_200k.parquet"
        )
        candidates = generate_or_load_candidates(
            s1_wide, target_wide, cand_cache, checkpoint_mgr=checkpoint_mgr
        )

        blocking_report = evaluate_blocking_candidates(gt_dict, candidates, len(target_wide))
        pc_val = blocking_report.pair_completeness * 100
        rr_val = blocking_report.reduction_ratio * 100
        print(f"  📊 Blocking Recall (Pair Completeness): {pc_val:.2f}%")
        print(f"  📊 Reduction Ratio:                    {rr_val:.4f}%")
        avg_cands = blocking_report.avg_candidates_per_s1
        print(f"  📊 Average Candidates per S1:          {avg_cands:.1f}\n")

    exec_logger.check_memory_threshold()

    # Step 4: Feature Extraction (with Parquet caching)
    print("--- [Step 4: RapidFuzz Pairwise Feature Extraction] ---")
    with pipeline_stage("Step 4: Feature Extraction", logger=exec_logger):
        feat_cache = out_path / (
            "features_sample.parquet" if sample_s1 else "features_medium_200k.parquet"
        )
        features_df = extract_or_load_features(
            candidates, s1_wide, target_wide, gt_dict, feat_cache
        )

    exec_logger.check_memory_threshold()

    # Step 5: Hyperparameter Sweep (Fold 0 holdout)
    fold_0_s1 = set(cv_folds_df[cv_folds_df["fold"] == 0]["entity_id"].astype(str))
    val_pairs_0 = features_df[features_df["s1_id"].isin(fold_0_s1)].copy()
    train_pairs_0 = features_df[~features_df["s1_id"].isin(fold_0_s1)].copy()
    val_gt_0 = {k: v for k, v in gt_dict.items() if k in fold_0_s1}

    with pipeline_stage("Step 5: Hyperparameter Sweep", logger=exec_logger):
        if not skip_sweep:
            best_cfg, base_tau = run_sweep(
                train_pairs_0, val_pairs_0, val_gt_0, checkpoint_mgr=checkpoint_mgr
            )
        else:
            best_cfg = HyperparamConfig(
                name="Default-Balanced",
                learning_rate=0.05,
                num_leaves=31,
                max_depth=6,
                n_estimators=150,
            )
            base_tau = 0.78

    exec_logger.check_memory_threshold()

    # Step 6: Full 5-Fold Cross Validation
    with pipeline_stage("Step 6: 5-Fold Cross Validation", logger=exec_logger):
        fold_results, oof_predictions, oof_scored_df = run_cross_validation(
            features_df=features_df,
            cv_folds_df=cv_folds_df,
            gt_dict=gt_dict,
            best_cfg=best_cfg,
            base_tau=base_tau,
            checkpoint_mgr=checkpoint_mgr,
        )

        # Save final OOF matching results
        final_tsv = out_path / (
            "matching_results_sample.tsv" if sample_s1 else "matching_results_medium_200k.tsv"
        )
        write_matching_results_tsv(final_tsv, oof_predictions)
        print(f"💾 Full Out-Of-Fold Predictions saved to: {final_tsv}")

    exec_logger.check_memory_threshold()

    # Step 7: Deep Failure Logging & Diagnostics
    print("\n================================================================================")
    print("🔍 STAGE 6: COMPREHENSIVE FAILURE AUDITING & ERROR DIAGNOSTICS")
    print("================================================================================")
    with pipeline_stage("Step 7: Failure Diagnostics", logger=exec_logger):
        oof_report = evaluate_resolution_predictions(gt_dict, oof_predictions)

        logger = FailureDiagnosticsLogger(reports_dir=rep_path)
        log_name = "failures_sample.jsonl" if sample_s1 else "failures_medium_200k.jsonl"
        summary_name = "diagnostics_sample.md" if sample_s1 else "diagnostics_medium_200k.md"
        failures = logger.analyze_and_log_failures(
            ground_truth=gt_dict,
            candidates=candidates,
            predictions=oof_predictions,
            scored_pairs_df=oof_scored_df,
            s1_df=s1_wide,
            target_df=target_wide,
            report=oof_report,
            log_filename=log_name,
            summary_filename=summary_name,
        )

    by_type: dict[str, int] = {}
    for f in failures:
        by_type[f.failure_type.value] = by_type.get(f.failure_type.value, 0) + 1

    fp_cnt = by_type.get("FALSE_MERGE_POSITIVE", 0)
    sing_cnt = by_type.get("SINGLETON_VIOLATION", 0)
    bfn_cnt = by_type.get("BLOCKING_FALSE_NEGATIVE", 0)
    cfn_cnt = by_type.get("CLASSIFICATION_FALSE_NEGATIVE", 0)

    print("Failure Breakdown Across All 200,000 Entities:")
    print(f"  🚨 FALSE MERGE POSITIVES:          {fp_cnt:,}  (Distractors; 2x penalty)")
    print(f"  🚨 SINGLETON VIOLATIONS:           {sing_cnt:,}  (Singletons given matches)")
    print(f"  ⚠️  BLOCKING FALSE NEGATIVES:       {bfn_cnt:,}  (Dropped at blocking)")
    print(f"  ⚠️  CLASSIFICATION FALSE NEGATIVES: {cfn_cnt:,}  (Model score < tau)")
    print(f"\n  ✓ Machine-readable JSONL logs: {rep_path / log_name}")
    print(f"  ✓ Executive Diagnostic Report: {rep_path / summary_name}\n")

    print_banner("EXECUTION COMPLETE: ALL 5 FOLDS AUDITED & LOGGED")


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallax 200k Medium Benchmark Runner")
    parser.add_argument("--data-dir", default="data/medium_split_200k", help="Dataset directory")
    parser.add_argument(
        "--output-dir", default="output", help="Output directory for predictions and cache"
    )
    parser.add_argument(
        "--reports-dir",
        default="reports",
        help="Reports directory for diagnostics and failure logs",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Optional custom directory for stage checkpoints (defaults to output/checkpoints)",
    )
    parser.add_argument(
        "--reset-checkpoints",
        action="store_true",
        help="Reset and recompute all stages from scratch",
    )
    parser.add_argument("--skip-sweep", action="store_true", help="Skip hyperparameter sweep")
    parser.add_argument(
        "--sample-s1", type=int, default=None, help="Optional sample limit for quick dry run"
    )
    args = parser.parse_args()

    rep_dir = Path(args.reports_dir)
    rep_dir.mkdir(parents=True, exist_ok=True)
    install_global_exception_handler(
        log_file=rep_dir / "benchmark_execution.log",
        crash_file=rep_dir / "benchmark_crash.json",
    )

    run_benchmark(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        reports_dir=args.reports_dir,
        checkpoint_dir=args.checkpoint_dir,
        reset_checkpoints=args.reset_checkpoints,
        skip_sweep=args.skip_sweep,
        sample_s1=args.sample_s1,
    )


if __name__ == "__main__":
    main()
