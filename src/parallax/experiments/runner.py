"""
Parallax 200k Medium Benchmark Experiment Engine
================================================
Unified runner for:
- Preprocessing & widening
- Multi-channel sparse & phonetic candidate blocking with persistent Parquet caching
- RapidFuzz 49-feature pairwise extraction with Parquet caching
- Two-Pass Meta-Feature Engineering & Hyperparameter grid sweep (Fold 0 holdout)
- Full 5-fold cross-validation with Two-Pass OOF prediction aggregation
- Comprehensive failure logging & root-cause diagnostic reporting.
"""

from __future__ import annotations

import argparse
import os
import shutil
import time
from collections.abc import Collection, Mapping
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
from parallax.features.extractor import FEATURE_COLUMNS, PairwiseFeatureExtractor
from parallax.features.meta_features import (
    META_FEATURE_COLUMNS,
    compute_entity_meta_features,
)
from parallax.metrics.evaluator import (
    evaluate_blocking_candidates,
    evaluate_resolution_predictions,
)
from parallax.models.matcher import LightGBMMatcher
from parallax.postprocessing.singleton_gate import SingletonGatedPredictor
from parallax.preprocessing.normalizer import widen_many_records_dfs
from parallax.utils.checkpoint_manager import CheckpointManager
from parallax.utils.parallel import resolve_workers, run_pool, total_threads

# Bump when a code change alters stage outputs, to invalidate all config-keyed caches.
PIPELINE_CACHE_VERSION = "v4"


def fingerprint(*parts: object) -> str:
    """Short stable hash of stage inputs, used to key every Parquet/JSON cache."""
    import hashlib

    h = hashlib.sha1(PIPELINE_CACHE_VERSION.encode())
    for p in parts:
        h.update(b"\x1f")
        h.update(repr(p).encode())
    return h.hexdigest()[:10]


def data_fingerprint(data_path: Path, sample_s1: int | None) -> str:
    """Fingerprint input files by name, size and mtime (cheap; no content hashing)."""
    stats = []
    for f in sorted(data_path.glob("*")):
        if f.is_file():
            st = f.stat()
            stats.append((f.name, st.st_size, int(st.st_mtime)))
    return fingerprint("data", str(data_path.resolve()), stats, sample_s1)


def subsample_sweep_pairs(
    features_df: pd.DataFrame,
    cv_folds_df: pd.DataFrame,
    gt_dict: Mapping[str, set[str]],
    sweep_sample_s1: int | None,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, set[str]]]:
    """
    Build the Fold-0 sweep holdout, optionally subsampled by S1 entity:
    up to sweep_sample_s1 training entities (folds != 0) and a quarter as many
    validation entities (fold 0), preserving the 4:1 train/val ratio.
    """
    fold_ids = cv_folds_df["entity_id"].astype(str)
    val_ids = fold_ids[cv_folds_df["fold"].to_numpy() == 0].to_numpy()
    train_ids = fold_ids[cv_folds_df["fold"].to_numpy() != 0].to_numpy()
    if sweep_sample_s1 and sweep_sample_s1 < len(train_ids):
        rng = np.random.default_rng(seed)
        train_ids = rng.choice(train_ids, size=sweep_sample_s1, replace=False)
        n_val = min(len(val_ids), max(1, sweep_sample_s1 // 4))
        val_ids = rng.choice(val_ids, size=n_val, replace=False)
        print(
            f"  ⚡ Sweep subsample: {len(train_ids):,} train S1 / {len(val_ids):,} val S1 "
            f"(of {len(fold_ids):,})"
        )
    s1_col = features_df["s1_id"].astype(str)
    val_set = set(val_ids.tolist())
    val_pairs = features_df[s1_col.isin(val_set)].copy()
    train_pairs = features_df[s1_col.isin(set(train_ids.tolist()))].copy()
    val_gt = {k: v for k, v in gt_dict.items() if k in val_set}
    return train_pairs, val_pairs, val_gt


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
    max_candidates_per_query: int = 35,
    checkpoint_mgr: CheckpointManager | None = None,
    n_jobs: int = 1,
    ckpt_tag: str | None = None,
) -> dict[str, dict[str, float]]:
    """Generate or retrieve candidate pairs mapping s1_id -> candidate dict with sim scores."""
    if cache_path.is_file():
        print(f"  ⚡ Found cached candidate pairs at: {cache_path}")
        t0 = time.time()
        cands_df = pd.read_parquet(cache_path)
        candidates: dict[str, dict[str, float]] = {s1: {} for s1 in s1_wide["entity_id"]}
        s1_arr = cands_df["s1_id"].to_numpy()
        cand_arr = cands_df["cand_id"].to_numpy()
        sim_arr = (
            cands_df["blocking_sim"].to_numpy()
            if "blocking_sim" in cands_df.columns
            else np.zeros(len(cands_df), dtype=np.float32)
        )
        for s1, cand, sim in zip(s1_arr, cand_arr, sim_arr, strict=False):
            if s1 in candidates:
                candidates[s1][cand] = float(sim)
        print(f"  ✓ Loaded {len(cands_df):,} cached pairs in {time.time() - t0:.2f}s.\n")
        return candidates

    msg = (
        f"  ⚡ Running multi-channel blocker (top_k={blocker_top_k}, "
        f"min_sim={blocker_min_sim}, max_cands={max_candidates_per_query}, n_jobs={n_jobs})..."
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
        max_candidates_per_query=max_candidates_per_query,
        n_jobs=n_jobs,
    )
    candidates = blocker.generate_candidates(
        s1_wide, target_wide, checkpoint_mgr=checkpoint_mgr, n_jobs=n_jobs, ckpt_tag=ckpt_tag
    )

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
    cache_rows_sim: list[float] = []
    for s1, cands in candidates.items():
        for cand, sim in cands.items():
            cache_rows_s1.append(s1)
            cache_rows_cand.append(cand)
            cache_rows_sim.append(sim)

    pd.DataFrame(
        {"s1_id": cache_rows_s1, "cand_id": cache_rows_cand, "blocking_sim": cache_rows_sim}
    ).to_parquet(cache_path, compression="snappy", index=False)
    print("  ✓ Caching complete.\n")
    return candidates


def extract_or_load_features(
    candidates: Mapping[str, Collection[str] | dict[str, float]],
    s1_wide: pd.DataFrame,
    target_wide: pd.DataFrame,
    gt_dict: Mapping[str, set[str]],
    cache_path: Path,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Extract or load pairwise similarity features."""
    if cache_path.is_file():
        print(f"  ⚡ Found cached feature matrix at: {cache_path}")
        t0 = time.time()
        features_df = pd.read_parquet(cache_path)

        missing_cols = [c for c in FEATURE_COLUMNS if c not in features_df.columns]
        if not missing_cols:
            print(f"  ✓ Loaded {len(features_df):,} feature rows in {time.time() - t0:.2f}s.\n")
            return features_df
        preview = missing_cols[:3]
        print(
            f"  ⚡ Cached feature matrix missing {len(missing_cols)} features ({preview}...). "
            "Re-extracting..."
        )
        del features_df

    print(
        f"  ⚡ Extracting RapidFuzz and structural features "
        f"across candidate pairs (n_jobs={n_jobs})..."
    )
    t0 = time.time()
    extractor = PairwiseFeatureExtractor()
    features_df = extractor.extract_features_df(
        candidates,
        s1_wide,
        target_wide,
        ground_truth=gt_dict,
        show_progress=True,
        n_jobs=n_jobs,
    )
    elapsed = time.time() - t0
    print(f"  ✓ Feature extraction finished in {elapsed:.2f}s ({len(features_df):,} rows).")

    print(f"  💾 Caching feature matrix to: {cache_path}...")
    features_df.to_parquet(cache_path, compression="snappy", index=False)
    print("  ✓ Caching complete.\n")
    return features_df


_SWEEP_STATE: dict[str, object] = {}
_CV_STATE: dict[str, object] = {}


def _sweep_config_task(task: tuple[HyperparamConfig, int]) -> SweepScore:
    """Two-pass train + tau search for one config on the fork-shared Fold 0 holdout."""
    cfg, threads = task
    train_pairs = _SWEEP_STATE["train"]
    val_pairs = _SWEEP_STATE["val"]
    val_gt = _SWEEP_STATE["val_gt"]
    assert isinstance(train_pairs, pd.DataFrame) and isinstance(val_pairs, pd.DataFrame)
    assert isinstance(val_gt, dict)
    predictor = SingletonGatedPredictor()
    all_features = list(FEATURE_COLUMNS) + list(META_FEATURE_COLUMNS)

    matcher_p1 = LightGBMMatcher(
        learning_rate=cfg.learning_rate,
        num_leaves=cfg.num_leaves,
        max_depth=cfg.max_depth,
        n_estimators=cfg.n_estimators,
        feature_columns=FEATURE_COLUMNS,
        seed=42,
        num_threads=threads,
    )
    matcher_p1.train(train_pairs, val_pairs)

    train_copy = train_pairs.copy()
    val_copy = val_pairs.copy()
    train_copy["prob"] = matcher_p1.predict_proba(train_copy)
    val_copy["prob"] = matcher_p1.predict_proba(val_copy)
    compute_entity_meta_features(train_copy, prob_col="prob")
    compute_entity_meta_features(val_copy, prob_col="prob")

    matcher_p2 = LightGBMMatcher(
        learning_rate=cfg.learning_rate,
        num_leaves=cfg.num_leaves,
        max_depth=cfg.max_depth,
        n_estimators=cfg.n_estimators,
        feature_columns=all_features,
        seed=1042,
        num_threads=threads,
    )
    matcher_p2.train(train_copy, val_copy)
    search_range = [0.50, 0.54, 0.58, 0.62, 0.66, 0.70, 0.74, 0.78, 0.82, 0.86, 0.90]
    best_tau, _ = matcher_p2.optimize_threshold(val_copy, val_gt, search_range=search_range)
    val_copy["prob"] = matcher_p2.predict_proba(val_copy)
    preds = predictor.filter_predictions(val_copy, list(val_gt.keys()), threshold=best_tau)
    report = evaluate_resolution_predictions(val_gt, preds)
    return SweepScore(
        config=cfg,
        optimal_tau=best_tau,
        macro_f05=report.macro_f05,
        singleton_acc=report.singleton_score,
        non_singleton_f05=report.non_singleton_f05,
    )


def _cv_fold_task(
    task: tuple[int, HyperparamConfig, float, int],
) -> tuple[FoldMetric, dict[str, set[str]], pd.DataFrame]:
    """Evaluate one CV fold reading the feature matrix from fork-shared state."""
    k, best_cfg, base_tau, threads = task
    features_df = _CV_STATE["features_df"]
    s1_fold_arr = _CV_STATE["s1_fold_arr"]
    cv_folds_df = _CV_STATE["cv_folds_df"]
    gt_dict = _CV_STATE["gt_dict"]
    assert isinstance(features_df, pd.DataFrame) and isinstance(cv_folds_df, pd.DataFrame)
    assert isinstance(s1_fold_arr, np.ndarray) and isinstance(gt_dict, dict)
    ckpt = _CV_STATE.get("checkpoint_mgr")
    sample = _CV_STATE.get("sample_s1")
    tag = _CV_STATE.get("ckpt_tag")
    return evaluate_single_fold(
        k,
        features_df,
        s1_fold_arr,
        cv_folds_df,
        gt_dict,
        best_cfg,
        base_tau,
        checkpoint_mgr=ckpt if isinstance(ckpt, CheckpointManager) else None,
        sample_s1=sample if isinstance(sample, int) else None,
        threads_per_worker=threads,
        record_manifest=False,
        ckpt_tag=tag if isinstance(tag, str) else None,
    )


def run_sweep(
    train_pairs: pd.DataFrame,
    val_pairs: pd.DataFrame,
    val_gt: Mapping[str, set[str]],
    checkpoint_mgr: CheckpointManager | None = None,
    sample_s1: int | None = None,
    n_jobs: int = 1,
    sweep_worker_gb: float = 7.0,
    ckpt_tag: str | None = None,
) -> tuple[HyperparamConfig, float]:
    """Execute hyperparameter sweep on validation holdout to find peak Macro F0.5."""
    print("================================================================================")
    print("🔬 STAGE 4: HYPERPARAMETER GRID SWEEP (VAL HOLDOUT)")
    print("================================================================================")

    if ckpt_tag:
        sw_key = f"sweep_{ckpt_tag}"
    else:
        sw_key = f"sweep_results_sample_{sample_s1}" if sample_s1 else "sweep_results"
    if checkpoint_mgr is not None and checkpoint_mgr.has_checkpoint(sw_key, ext="json"):
        data = checkpoint_mgr.load_json(sw_key)
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
            name="Config-Dense",
            learning_rate=0.04,
            num_leaves=63,
            max_depth=8,
            n_estimators=250,
        ),
        HyperparamConfig(
            name="Config-XDeep",
            learning_rate=0.04,
            num_leaves=127,
            max_depth=10,
            n_estimators=250,
        ),
        HyperparamConfig(
            name="Config-Conservative",
            learning_rate=0.03,
            num_leaves=31,
            max_depth=6,
            n_estimators=180,
        ),
    ]

    val_s1_ids = list(val_gt.keys())
    scores: list[SweepScore] = []

    print(f"Sweeping {len(configs)} configs on {len(val_s1_ids):,} entities...")
    header = (
        f"{'Config':<18} | {'LR':<5} | {'Leaves':<6} | {'Trees':<5} | "
        f"{'tau':<6} | {'Macro F0.5':<10} | {'Sing. Acc':<10} | {'Non-Sing F0.5':<12}"
    )

    def _cfg_key(cfg: HyperparamConfig) -> str:
        return f"{sw_key}_cfg_{cfg.name}"

    todo: list[HyperparamConfig] = []
    for cfg in configs:
        cached = (
            checkpoint_mgr.load_json(_cfg_key(cfg))
            if checkpoint_mgr is not None and checkpoint_mgr.has_checkpoint(_cfg_key(cfg), "json")
            else None
        )
        if cached is not None:
            scores.append(
                SweepScore(
                    config=cfg,
                    optimal_tau=float(cached["optimal_tau"]),
                    macro_f05=float(cached["macro_f05"]),
                    singleton_acc=float(cached["singleton_acc"]),
                    non_singleton_f05=float(cached["non_singleton_f05"]),
                )
            )
        else:
            todo.append(cfg)

    if todo:
        workers = resolve_workers(n_jobs, per_worker_gb=sweep_worker_gb, n_tasks=len(todo))
        threads = max(1, total_threads() // workers)
        print(
            f"  ⚡ Training {len(todo)} configs concurrently "
            f"({workers} workers × {threads} LightGBM threads)..."
        )
        global _SWEEP_STATE
        _SWEEP_STATE = {"train": train_pairs, "val": val_pairs, "val_gt": val_gt}

        def _on_done(i: int, res: SweepScore) -> None:
            print(
                f"  ✓ [Sweep] {res.config.name}: Macro F0.5={res.macro_f05:.4f} "
                f"tau={res.optimal_tau:.2f}",
                flush=True,
            )
            if checkpoint_mgr is not None:
                checkpoint_mgr.save_json(
                    _cfg_key(res.config),
                    {
                        "optimal_tau": res.optimal_tau,
                        "macro_f05": res.macro_f05,
                        "singleton_acc": res.singleton_acc,
                        "non_singleton_f05": res.non_singleton_f05,
                    },
                )

        try:
            new_scores = run_pool(
                _sweep_config_task,
                [(cfg, threads) for cfg in todo],
                max_workers=workers,
                threads_per_worker=threads,
                label="sweep",
                on_result=_on_done,
            )
        finally:
            _SWEEP_STATE = {}
        scores.extend(new_scores)

    order = {c.name: i for i, c in enumerate(configs)}
    scores.sort(key=lambda sc: order[sc.config.name])
    print(header)
    print("-" * len(header))
    for sc in scores:
        cfg = sc.config
        print(
            f"{cfg.name:<18} | {cfg.learning_rate:<5.2f} | {cfg.num_leaves:<6} | "
            f"{cfg.n_estimators:<5} | {sc.optimal_tau:<6.2f} | {sc.macro_f05:<10.4f} | "
            f"{sc.singleton_acc * 100:<9.2f}% | {sc.non_singleton_f05:<12.4f}"
        )

    best_sweep = max(scores, key=lambda s: s.macro_f05)
    print("-" * len(header))
    print(
        f"🏆 WINNING CONFIG: {best_sweep.config.name} "
        f"(Macro F0.5 = {best_sweep.macro_f05:.4f}, tau = {best_sweep.optimal_tau:.2f})\n"
    )

    if checkpoint_mgr is not None:
        checkpoint_mgr.save_json(
            sw_key,
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
            sw_key,
            {"winning_config": best_sweep.config.name, "tau": best_sweep.optimal_tau},
        )

    return best_sweep.config, best_sweep.optimal_tau


def fold_checkpoint_keys(k: int, sample_s1: int | None, ckpt_tag: str | None) -> tuple[str, str]:
    """(metrics_key, scored_pairs_key) for a fold; ckpt_tag keys them by full stage config."""
    if ckpt_tag:
        return f"fold_{k}_metrics_{ckpt_tag}", f"fold_{k}_scored_pairs_{ckpt_tag}"
    if sample_s1:
        return f"fold_{k}_metrics_sample_{sample_s1}", f"fold_{k}_scored_pairs_sample_{sample_s1}"
    return f"fold_{k}_metrics", f"fold_{k}_scored_pairs"


FOLD_TAU_OFFSETS = [round(-0.12 + 0.02 * i, 2) for i in range(13)]


def evaluate_single_fold(
    k: int,
    features_df: pd.DataFrame,
    s1_fold_arr: np.ndarray,
    cv_folds_df: pd.DataFrame,
    gt_dict: Mapping[str, set[str]],
    best_cfg: HyperparamConfig,
    base_tau: float,
    checkpoint_mgr: CheckpointManager | None = None,
    sample_s1: int | None = None,
    threads_per_worker: int | None = None,
    record_manifest: bool = True,
    ckpt_tag: str | None = None,
) -> tuple[FoldMetric, dict[str, set[str]], pd.DataFrame]:
    """Execute Two-Pass training, meta-feature engineering, and evaluation for a single fold."""
    val_s1_set = set(cv_folds_df[cv_folds_df["fold"] == k]["entity_id"].astype(str))
    val_gt = {k_id: v for k_id, v in gt_dict.items() if k_id in val_s1_set}

    m_key, sp_key = fold_checkpoint_keys(k, sample_s1, ckpt_tag)

    predictor = SingletonGatedPredictor()

    # Check for completed fold checkpoint
    if (
        checkpoint_mgr is not None
        and checkpoint_mgr.has_checkpoint(m_key, ext="json")
        and checkpoint_mgr.has_checkpoint(sp_key)
    ):
        metric_data = checkpoint_mgr.load_json(m_key)
        val_pairs_scored = checkpoint_mgr.load_dataframe(sp_key)
        if metric_data is not None and val_pairs_scored is not None:
            cached_metric = FoldMetric(**metric_data)
            cached_preds = predictor.filter_predictions(
                val_pairs_scored, list(val_s1_set), threshold=cached_metric.optimal_tau
            )
            return cached_metric, cached_preds, val_pairs_scored[["s1_id", "cand_id", "prob"]]

    all_features = list(FEATURE_COLUMNS) + list(META_FEATURE_COLUMNS)
    val_mask = s1_fold_arr == k
    train_mask = (s1_fold_arr != k) & (s1_fold_arr != -1)

    train_pairs = features_df[train_mask].copy()
    val_pairs = features_df[val_mask].copy()

    # Pass 1: Raw 49 Features
    matcher_p1 = LightGBMMatcher(
        learning_rate=best_cfg.learning_rate,
        num_leaves=best_cfg.num_leaves,
        max_depth=best_cfg.max_depth,
        n_estimators=best_cfg.n_estimators,
        feature_columns=FEATURE_COLUMNS,
        seed=42 + k,
        num_threads=threads_per_worker,
    )
    matcher_p1.train(train_pairs, val_df=val_pairs)

    train_pairs["prob"] = matcher_p1.predict_proba(train_pairs)
    val_pairs["prob"] = matcher_p1.predict_proba(val_pairs)

    # Compute Entity Meta-Features (Track A)
    compute_entity_meta_features(train_pairs, prob_col="prob")
    compute_entity_meta_features(val_pairs, prob_col="prob")

    # Pass 2: Combined 54 Features (49 + 5 Meta)
    matcher_p2 = LightGBMMatcher(
        learning_rate=best_cfg.learning_rate,
        num_leaves=best_cfg.num_leaves,
        max_depth=best_cfg.max_depth,
        n_estimators=best_cfg.n_estimators,
        feature_columns=all_features,
        seed=1042 + k,
        num_threads=threads_per_worker,
    )
    matcher_p2.train(train_pairs, val_df=val_pairs)

    # Optimize threshold tau on Pass 2 probabilities
    # Fine grid around the sweep tau (cheap now that the scan is vectorized). The sweep may
    # run on a subsample, so its tau is only a centre point for the full-data fold models.
    search_range = [round(base_tau + d, 2) for d in FOLD_TAU_OFFSETS]
    search_range = [t for t in search_range if 0.30 <= t <= 0.98] or [base_tau]
    best_tau_k, _ = matcher_p2.optimize_threshold(val_pairs, val_gt, search_range=search_range)

    # Out-of-fold inference with Pass 2 probabilities
    val_pairs["prob"] = matcher_p2.predict_proba(val_pairs)
    fold_preds = predictor.filter_predictions(val_pairs, list(val_s1_set), threshold=best_tau_k)

    report = evaluate_resolution_predictions(val_gt, fold_preds)
    prec = (
        report.total_correct_pairs / report.total_predicted_pairs
        if report.total_predicted_pairs > 0
        else 0.0
    )
    rec = (
        report.total_correct_pairs / report.total_true_pairs if report.total_true_pairs > 0 else 0.0
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

    if checkpoint_mgr is not None:
        checkpoint_mgr.save_dataframe(sp_key, val_pairs[["s1_id", "cand_id", "prob"]])
        checkpoint_mgr.save_json(m_key, asdict(fold_metric))
        if record_manifest:
            checkpoint_mgr.record_stage_completed(
                m_key,
                {"optimal_tau": best_tau_k, "macro_f05": report.macro_f05},
            )

    return fold_metric, fold_preds, val_pairs[["s1_id", "cand_id", "prob"]]


def run_cross_validation(
    features_df: pd.DataFrame,
    cv_folds_df: pd.DataFrame,
    gt_dict: Mapping[str, set[str]],
    best_cfg: HyperparamConfig,
    base_tau: float,
    checkpoint_mgr: CheckpointManager | None = None,
    sample_s1: int | None = None,
    n_jobs: int = 1,
    cv_worker_gb: float = 8.0,
    ckpt_tag: str | None = None,
) -> tuple[list[FoldMetric], dict[str, set[str]], pd.DataFrame]:
    """Execute Two-Pass 5-fold CV, return fold metrics and complete OOF predictions."""
    print("================================================================================")
    print("🔁 STAGE 5: FULL 5-FOLD CROSS VALIDATION (TWO-PASS RESOLUTION)")
    print("================================================================================")

    # Map entity_id to fold
    fold_map = dict(
        zip(cv_folds_df["entity_id"].astype(str), cv_folds_df["fold"].astype(int), strict=False)
    )
    s1_fold_arr = np.array([fold_map.get(s1, -1) for s1 in features_df["s1_id"].astype(str)])

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

    if n_jobs > 1 and len(folds) > 1:
        workers = resolve_workers(n_jobs, per_worker_gb=cv_worker_gb, n_tasks=len(folds))
        threads_per_worker = max(1, total_threads() // workers)
        print(
            f"  ⚡ Launching {workers} concurrent fold worker processes "
            f"(threads/worker={threads_per_worker}, shared-memory features)..."
        )
        global _CV_STATE
        _CV_STATE = {
            "features_df": features_df,
            "s1_fold_arr": s1_fold_arr,
            "cv_folds_df": cv_folds_df,
            "gt_dict": dict(gt_dict),
            "checkpoint_mgr": checkpoint_mgr,
            "sample_s1": sample_s1,
            "ckpt_tag": ckpt_tag,
        }

        def _on_fold(i: int, res: tuple[FoldMetric, dict[str, set[str]], pd.DataFrame]) -> None:
            f_m = res[0]
            print(
                f"  ✓ [Parallel Worker] Fold {f_m.fold} completed: "
                f"Macro F0.5={f_m.macro_f05:.4f}, tau={f_m.optimal_tau:.2f}, "
                f"Sing. Acc={f_m.singleton_acc * 100:.2f}%",
                flush=True,
            )

        try:
            fold_outputs = run_pool(
                _cv_fold_task,
                [(int(k), best_cfg, base_tau, threads_per_worker) for k in folds],
                max_workers=workers,
                threads_per_worker=threads_per_worker,
                label="cv",
                on_result=_on_fold,
            )
        finally:
            _CV_STATE = {}
        results_map = {int(k): res for k, res in zip(folds, fold_outputs, strict=True)}
        if checkpoint_mgr is not None:
            for k, res in results_map.items():
                m_key = fold_checkpoint_keys(k, sample_s1, ckpt_tag)[0]
                checkpoint_mgr.record_stage_completed(
                    m_key, {"optimal_tau": res[0].optimal_tau, "macro_f05": res[0].macro_f05}
                )

        for k in folds:
            f_metric, f_preds, f_scored = results_map[k]
            fold_results.append(f_metric)
            oof_predictions.update(f_preds)
            oof_scored_parts.append(f_scored)
            row_str = (
                f"{k:<5} | {f_metric.train_pairs:<10,} | {f_metric.val_pairs:<8,} | "
                f"{f_metric.optimal_tau:<5.2f} | {f_metric.macro_f05:<10.4f} | "
                f"{f_metric.singleton_acc * 100:<9.2f}% | {f_metric.non_singleton_f05:<12.4f} | "
                f"{f_metric.precision * 100:<7.2f}% | {f_metric.recall * 100:<6.2f}%"
            )
            print(row_str)
    else:
        for k in tqdm(folds, desc="  ⚡ 5-Fold Cross Validation", unit="fold", leave=False):
            f_metric, f_preds, f_scored = evaluate_single_fold(
                k=k,
                features_df=features_df,
                s1_fold_arr=s1_fold_arr,
                cv_folds_df=cv_folds_df,
                gt_dict=gt_dict,
                best_cfg=best_cfg,
                base_tau=base_tau,
                checkpoint_mgr=checkpoint_mgr,
                sample_s1=sample_s1,
                ckpt_tag=ckpt_tag,
            )
            fold_results.append(f_metric)
            oof_predictions.update(f_preds)
            oof_scored_parts.append(f_scored)
            row_str = (
                f"{k:<5} | {f_metric.train_pairs:<10,} | {f_metric.val_pairs:<8,} | "
                f"{f_metric.optimal_tau:<5.2f} | {f_metric.macro_f05:<10.4f} | "
                f"{f_metric.singleton_acc * 100:<9.2f}% | {f_metric.non_singleton_f05:<12.4f} | "
                f"{f_metric.precision * 100:<7.2f}% | {f_metric.recall * 100:<6.2f}%"
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


def report_and_apply_target_exclusivity(
    oof_scored_df: pd.DataFrame,
    fold_results: list[FoldMetric],
    cv_folds_df: pd.DataFrame,
    gt_dict: Mapping[str, set[str]],
    oof_predictions: dict[str, set[str]],
    apply: bool = False,
) -> dict[str, set[str]]:
    """
    Print OOF Macro F0.5 with and without the one-owner-per-target rule (keep a pair only if
    its S1 has the highest probability for that target), using each fold's own tau.
    Returns the exclusive predictions if apply=True, else the original predictions.
    """
    if oof_scored_df.empty:
        return oof_predictions
    tau_by_fold = {r.fold: r.optimal_tau for r in fold_results}
    fold_map = dict(
        zip(cv_folds_df["entity_id"].astype(str), cv_folds_df["fold"].astype(int), strict=False)
    )
    df = oof_scored_df[["s1_id", "cand_id", "prob"]].copy()
    df["s1_id"] = df["s1_id"].astype(str)
    df["cand_id"] = df["cand_id"].astype(str)
    taus = df["s1_id"].map(fold_map).map(tau_by_fold).fillna(1.1).to_numpy()
    base_mask = df["prob"].to_numpy() >= taus
    cand_max = df.groupby("cand_id")["prob"].transform("max").to_numpy()
    excl_mask = base_mask & (df["prob"].to_numpy() >= cand_max)

    def _preds(mask: np.ndarray) -> dict[str, set[str]]:
        out: dict[str, set[str]] = {s: set() for s in oof_predictions}
        for s1, cand in zip(
            df["s1_id"].to_numpy()[mask], df["cand_id"].to_numpy()[mask], strict=False
        ):
            if s1 in out:
                out[s1].add(cand)
        return out

    excl_preds = _preds(excl_mask)
    base_score = evaluate_resolution_predictions(gt_dict, oof_predictions).macro_f05
    excl_score = evaluate_resolution_predictions(gt_dict, excl_preds).macro_f05
    contested = int((base_mask & ~excl_mask).sum())
    print(
        f"📊 Target exclusivity what-if: OOF Macro F0.5 {base_score:.4f} -> {excl_score:.4f} "
        f"({contested:,} contested pairs removed) [{'APPLIED' if apply else 'not applied'}]\n"
    )
    return excl_preds if apply else oof_predictions


def run_benchmark(
    data_dir: Path | str = "data/medium_split_200k",
    output_dir: Path | str = "output",
    reports_dir: Path | str = "reports",
    checkpoint_dir: Path | str | None = None,
    reset_checkpoints: bool = False,
    skip_sweep: bool = False,
    sample_s1: int | None = None,
    n_jobs: int = 1,
    max_candidates: int = 35,
    exclusive_targets: bool = False,
    sweep_sample_s1: int | None = 50_000,
) -> None:
    """Execute complete benchmark workflow with two-pass meta-resolution."""
    data_path = Path(data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    rep_path = Path(reports_dir)
    rep_path.mkdir(parents=True, exist_ok=True)
    chk_path = Path(checkpoint_dir) if checkpoint_dir else out_path / "checkpoints"
    checkpoint_mgr = CheckpointManager(chk_path, reset=reset_checkpoints)

    print_banner("PARALLAX 200K BENCHMARK & ERROR DIAGNOSTICS SUITE (V3)")
    print(f"Data Source:       {data_path.resolve()}")
    print(f"Outputs:           {out_path.resolve()}")
    print(f"Diagnostic Logs:   {rep_path.resolve()}")
    print(f"Checkpoints:       {chk_path.resolve()}\n")

    print_resource_snapshot()

    exec_logger = get_global_logger()

    # Step 1: Load Data
    print("--- [Step 1: Loading Benchmark Split] ---")
    with pipeline_stage("Step 1: Loading Data", logger=exec_logger):
        t0 = time.time()
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=4) as tpe:
            f_s1 = tpe.submit(load_business_records_df, data_path / "train_source1.tsv")
            f_s2 = tpe.submit(load_business_records_df, data_path / "train_source2.tsv")
            f_s3 = tpe.submit(load_business_records_df, data_path / "train_source3.tsv")
            f_gt = tpe.submit(load_ground_truth_dict, data_path / "train_ground_truth.tsv")
            s1_df, s2_df, s3_df, gt_dict = (
                f_s1.result(),
                f_s2.result(),
                f_s3.result(),
                f_gt.result(),
            )
        if (data_path / "cv_folds_source1.parquet").is_file():
            cv_folds_df = pd.read_parquet(data_path / "cv_folds_source1.parquet")
        else:
            cv_folds_df = pd.read_csv(data_path / "cv_folds_source1.tsv", sep="\t")

        if sample_s1 and sample_s1 < len(s1_df):
            print(f"  ⚡ Running test sample with {sample_s1:,} S1 entities...")
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
        data_fp = data_fingerprint(data_path, sample_s1)
        s1_tag = f"s1_wide_{data_fp}"
        target_tag = f"target_wide_{data_fp}"
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
        if (
            loaded_s1 is not None
            and loaded_target is not None
            and (
                "primary_number" not in loaded_s1.columns
                or "primary_number" not in loaded_target.columns
                or "city_token" not in loaded_s1.columns
                or "city_token" not in loaded_target.columns
            )
        ):
            print(
                "  ⚡ [Checkpoint] Widened tables lack v2 columns "
                "(primary_number/city_token). Re-widening..."
            )
            loaded_s1 = None
            loaded_target = None

        if loaded_s1 is not None and loaded_target is not None:
            print("  ⚡ [Checkpoint] Loading cached widened tables...")
            s1_wide = loaded_s1
            target_wide = loaded_target
            print(
                f"  ✓ Loaded {len(s1_wide):,} S1 and {len(target_wide):,} "
                f"target records in {time.time() - t0:.2f}s.\n"
            )
        else:
            s1_wide, s2_wide, s3_wide = widen_many_records_dfs([s1_df, s2_df, s3_df], n_jobs=n_jobs)
            target_wide = pd.concat([s2_wide, s3_wide], ignore_index=True)
            checkpoint_mgr.save_dataframe(s1_tag, s1_wide)
            checkpoint_mgr.save_dataframe(target_tag, target_wide)
            checkpoint_mgr.record_stage_completed(
                "widening", {"s1_rows": len(s1_wide), "target_rows": len(target_wide)}
            )
            print(f"  ✓ Records widened and transliterated in {time.time() - t0:.2f}s.\n")

    exec_logger.check_memory_threshold()

    # Step 3: Candidate Generation (with Parquet caching)
    print("--- [Step 3: Multi-Channel Sparse & Phonetic Blocking] ---")
    with pipeline_stage("Step 3: Candidate Blocking", logger=exec_logger):
        # Cache names are keyed by dataset, sample size and candidate cap so that changing
        # the blocking configuration can never silently reuse stale candidates.
        block_fp = fingerprint("blocking", data_fp, max_candidates, 35, 25, 0.15, 0.20, 2000)
        run_tag = f"{data_path.name}_{block_fp}"
        cand_cache = out_path / f"candidate_pairs_{run_tag}.parquet"
        candidates = generate_or_load_candidates(
            s1_wide,
            target_wide,
            cand_cache,
            max_candidates_per_query=max_candidates,
            checkpoint_mgr=checkpoint_mgr,
            n_jobs=n_jobs,
            ckpt_tag=block_fp,
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
    print("--- [Step 4: RapidFuzz Pairwise Feature Extraction (49 Features)] ---")
    with pipeline_stage("Step 4: Feature Extraction", logger=exec_logger):
        feat_fp = fingerprint("features", block_fp, list(FEATURE_COLUMNS))
        feat_cache = out_path / f"features_{data_path.name}_{feat_fp}.parquet"
        features_df = extract_or_load_features(
            candidates, s1_wide, target_wide, gt_dict, feat_cache, n_jobs=n_jobs
        )

    exec_logger.check_memory_threshold()

    # Step 5: Hyperparameter Sweep (Fold 0 holdout)
    sweep_fp = fingerprint("sweep", feat_fp, sweep_sample_s1)
    with pipeline_stage("Step 5: Hyperparameter Sweep", logger=exec_logger):
        if not skip_sweep:
            train_pairs_0, val_pairs_0, val_gt_0 = subsample_sweep_pairs(
                features_df, cv_folds_df, gt_dict, sweep_sample_s1
            )
            best_cfg, base_tau = run_sweep(
                train_pairs_0,
                val_pairs_0,
                val_gt_0,
                checkpoint_mgr=checkpoint_mgr,
                sample_s1=sample_s1,
                n_jobs=n_jobs,
                ckpt_tag=sweep_fp,
            )
            del train_pairs_0, val_pairs_0
        else:
            best_cfg = HyperparamConfig(
                name="Config-Deep",
                learning_rate=0.05,
                num_leaves=63,
                max_depth=8,
                n_estimators=150,
            )
            base_tau = 0.86

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
            sample_s1=sample_s1,
            n_jobs=n_jobs,
            ckpt_tag=fingerprint(
                "cv", feat_fp, asdict(best_cfg), round(base_tau, 4), FOLD_TAU_OFFSETS
            ),
        )

        # One-owner-per-target post-processing (each target matches at most one S1)
        oof_predictions = report_and_apply_target_exclusivity(
            oof_scored_df,
            fold_results,
            cv_folds_df,
            gt_dict,
            oof_predictions,
            apply=exclusive_targets,
        )

        # Save final OOF matching results
        final_tsv = out_path / (
            f"matching_results_sample_{sample_s1}.tsv"
            if sample_s1
            else "matching_results_medium_200k.tsv"
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
        log_name = (
            f"failures_sample_{sample_s1}.jsonl" if sample_s1 else "failures_medium_200k.jsonl"
        )
        summary_name = (
            f"diagnostics_sample_{sample_s1}.md" if sample_s1 else "diagnostics_medium_200k.md"
        )
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

    print(f"Failure Breakdown Across All {len(s1_df):,} Entities:")
    print(f"  🚨 FALSE MERGE POSITIVES:          {fp_cnt:,}  (Distractors; 2x penalty)")
    print(f"  🚨 SINGLETON VIOLATIONS:           {sing_cnt:,}  (Singletons given matches)")
    print(f"  ⚠️  BLOCKING FALSE NEGATIVES:       {bfn_cnt:,}  (Dropped at blocking)")
    print(f"  ⚠️  CLASSIFICATION FALSE NEGATIVES: {cfn_cnt:,}  (Model score < tau)")
    print(f"\n  ✓ Machine-readable JSONL logs: {rep_path / log_name}")
    print(f"  ✓ Executive Diagnostic Report: {rep_path / summary_name}\n")

    print_banner("EXECUTION COMPLETE: ALL 5 FOLDS AUDITED & LOGGED")


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallax Benchmark Runner (v3)")
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
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of concurrent worker processes for end-to-end pipeline stages",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=35,
        help="Maximum candidate matches preserved per query in blocking (default: 35)",
    )
    parser.add_argument(
        "--sweep-sample-s1",
        type=int,
        default=50_000,
        help="Train S1 entities used by the hyperparameter sweep (val uses a quarter as many); "
        "0 = use the full Fold-0 split",
    )
    parser.add_argument(
        "--exclusive-targets",
        action="store_true",
        help="Apply one-owner-per-target post-processing to the final OOF predictions",
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
        n_jobs=args.n_jobs,
        max_candidates=args.max_candidates,
        exclusive_targets=args.exclusive_targets,
        sweep_sample_s1=args.sweep_sample_s1 or None,
    )


if __name__ == "__main__":
    main()
