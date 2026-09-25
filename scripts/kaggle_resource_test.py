#!/usr/bin/env python3
"""
Parallax Kaggle Resource & Benchmark Runner
==========================================
Cross-platform memory profiling, candidate generation, 28-feature extraction,
and LightGBM entity resolution benchmark.

Supports:
- Phase 1: 100K S1 Smoke Test (--sample-s1 100000)
- Phase 2: Full 200K S1 Benchmark (--sample-s1 200000 or omitted)

Compatible with both Kaggle Linux environments and local workstations.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure src is on python path
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

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


# --- Cross-Platform Memory Profiler ---
def get_memory_info() -> tuple[float, float, float]:
    """
    Returns (current_rss_mb, peak_rss_mb, commit_or_vms_mb).
    Supports Linux (Kaggle), macOS, and Windows.
    """
    # 1. Try psutil (Standard in Kaggle environment)
    try:
        import psutil

        proc = psutil.Process()
        mem = proc.memory_info()
        rss_mb = mem.rss / (1024 * 1024)
        vms_mb = mem.vms / (1024 * 1024)

        try:
            import resource

            # On Linux, ru_maxrss is in KiB; on Darwin/macOS it is in Bytes
            ru_max = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if sys.platform == "darwin":
                peak_rss_mb = ru_max / (1024 * 1024)
            else:
                peak_rss_mb = ru_max / 1024
        except ImportError:
            peak_rss_mb = rss_mb

        return rss_mb, max(peak_rss_mb, rss_mb), vms_mb
    except Exception:
        pass

    # 2. Linux /proc/self/status fallback
    if Path("/proc/self/status").exists():
        try:
            with open("/proc/self/status") as f:
                lines = f.readlines()
            stats = {}
            for line in lines:
                parts = line.split(":")
                if len(parts) == 2:
                    stats[parts[0].strip()] = parts[1].strip()
            rss = float(stats.get("VmRSS", "0 kB").split()[0]) / 1024
            peak = float(stats.get("VmPeak", "0 kB").split()[0]) / 1024
            vms = float(stats.get("VmSize", "0 kB").split()[0]) / 1024
            return rss, peak, vms
        except Exception:
            pass

    # 3. Windows ctypes fallback
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                    ("PrivateUsage", ctypes.c_size_t),
                ]

            psapi = ctypes.WinDLL("psapi")
            kernel32 = ctypes.WinDLL("kernel32")
            GetProcessMemoryInfo = psapi.GetProcessMemoryInfo
            GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX),
                wintypes.DWORD,
            ]
            GetProcessMemoryInfo.restype = wintypes.BOOL

            pmc = PROCESS_MEMORY_COUNTERS_EX()
            pmc.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)
            GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
            return (
                pmc.WorkingSetSize / (1024 * 1024),
                pmc.PeakWorkingSetSize / (1024 * 1024),
                pmc.PrivateUsage / (1024 * 1024),
            )
        except Exception:
            pass

    return 0.0, 0.0, 0.0


def log(msg: str = "") -> None:
    """Print with forced immediate flush for real-time notebook streaming."""
    print(msg, flush=True)


def parse_args() -> argparse.Namespace:
    # Auto-detect default data directory
    if Path("/kaggle/input/parallax-200k").is_dir():
        default_data = "/kaggle/input/parallax-200k"
    elif Path("/kaggle/input/medium_split_200k").is_dir():
        default_data = "/kaggle/input/medium_split_200k"
    else:
        default_data = "data/medium_split_200k"

    parser = argparse.ArgumentParser(description="Parallax Kaggle Resource & Benchmark Runner")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=default_data,
        help="Path to medium_split_200k dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/kaggle/working/output" if Path("/kaggle").is_dir() else "output/kaggle_test",
        help="Path to output TSV directory",
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default="/kaggle/working/reports" if Path("/kaggle").is_dir() else "reports/kaggle_test",
        help="Path to reports directory",
    )
    parser.add_argument(
        "--sample-s1",
        type=int,
        default=100000,
        help="Number of S1 entities to evaluate (e.g. 100000 for smoke test, 200000 for full run)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2000,
        help="Batch size for sparse dot product queries",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_dir)
    out_path = Path(args.output_dir)
    rep_path = Path(args.reports_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    rep_path.mkdir(parents=True, exist_ok=True)

    log("=" * 72)
    log("⚡ PARALLAX KAGGLE RESOURCE BENCHMARK RUNNER")
    log("=" * 72)
    log(f"Data Directory:    {data_path}")
    log(f"Output Directory:  {out_path}")
    log(f"Reports Directory: {rep_path}")
    log(f"Target S1 Count:   {args.sample_s1:,}")
    rss, peak, vms = get_memory_info()
    log(f"Process Baseline:  RSS={rss:.1f} MB | Peak={peak:.1f} MB | VMS/Commit={vms:.1f} MB\n")

    start_total = time.perf_counter()

    # --- Step 1: Loading Data ---
    t0 = time.perf_counter()
    log("==> 1. Loading Datasets (Source 1, Source 2, Source 3, Ground Truth)...")
    s1_df = load_business_records_df(data_path / "train_source1.tsv")
    s2_df = load_business_records_df(data_path / "train_source2.tsv")
    s3_df = load_business_records_df(data_path / "train_source3.tsv")
    gt_file = data_path / "train_ground_truth.tsv"
    gt_dict = load_ground_truth_dict(gt_file) if gt_file.is_file() else {}

    total_available_s1 = len(s1_df)
    if args.sample_s1 and args.sample_s1 < total_available_s1:
        log(f"   ⚡ Slicing S1 to {args.sample_s1:,} entities (out of {total_available_s1:,})")
        s1_df = s1_df.head(args.sample_s1).copy()
        s1_ids = set(s1_df["entity_id"])
        gt_dict = {k: v for k, v in gt_dict.items() if k in s1_ids}
    else:
        log(f"   ⚡ Evaluating full S1 dataset: {len(s1_df):,} entities")

    target_df = pd.concat([s2_df, s3_df], ignore_index=True)
    del s2_df, s3_df
    gc.collect()

    t_load = time.perf_counter() - t0
    rss, peak, vms = get_memory_info()
    log(f"   ✓ Loaded S1: {len(s1_df):,} | Target Pool (S2+S3): {len(target_df):,}")
    log(f"   ✓ Load Time: {t_load:.2f}s | Current RAM: {rss:.1f} MB ({rss/1024:.2f} GB) | Peak: {peak:.1f} MB\n")

    # --- Step 2: Widening / Preprocessing ---
    t0 = time.perf_counter()
    log("==> 2. Widening & Preprocessing Records (Unicode NFKC, numbers, postals)...")
    s1_wide = widen_records_df(s1_df)
    target_wide = widen_records_df(target_df)
    del s1_df, target_df
    gc.collect()

    t_widen = time.perf_counter() - t0
    rss, peak, vms = get_memory_info()
    log(f"   ✓ Preprocessing completed in {t_widen:.2f}s")
    log(f"   ✓ Current RAM: {rss:.1f} MB ({rss/1024:.2f} GB) | Peak: {peak:.1f} MB\n")

    # --- Step 3: Country-Partitioned Dual-Channel Blocking ---
    log("==> 3. Running Dual-Channel TF-IDF Blocker...")
    candidate_pairs: dict[str, set[str]] = {s1_id: set() for s1_id in s1_wide["entity_id"]}
    countries = s1_wide["country"].unique()

    # Exact baseline parameters
    name_top_k = 25
    addr_top_k = 20
    name_min_sim = 0.15
    addr_min_sim = 0.20
    batch_size = args.batch_size

    def _build_texts(df: pd.DataFrame, primary_col: str, fallback_col: str, translit_col: str) -> list[str]:
        if primary_col in df:
            p_s = df[primary_col].fillna("").astype(str).str.strip()
            if fallback_col in df:
                empty = p_s == ""
                if empty.any():
                    p_s[empty] = df.loc[empty, fallback_col].fillna("").astype(str).str.strip()
        elif fallback_col in df:
            p_s = df[fallback_col].fillna("").astype(str).str.strip()
        else:
            p_s = pd.Series([""] * len(df), index=df.index)

        if translit_col in df:
            t_s = df[translit_col].fillna("").astype(str).str.strip()
            diff_mask = (t_s != "") & (t_s != p_s)
            res = p_s.copy()
            if diff_mask.any():
                res[diff_mask] = p_s[diff_mask] + " " + t_s[diff_mask]
            return [str(x) for x in res.tolist()]
        return [str(x) for x in p_s.tolist()]

    sparse_matrix_stats = []
    blocking_channel_stats = []

    for country in countries:
        s1_c = s1_wide[s1_wide["country"] == country].reset_index(drop=True)
        tgt_c = target_wide[target_wide["country"] == country].reset_index(drop=True)

        if len(s1_c) == 0 or len(tgt_c) == 0:
            continue

        log(f"   --- Partition [{country}]: S1={len(s1_c):,} | Target Pool={len(tgt_c):,} ---")

        s1_names = _build_texts(s1_c, "soft_name", "business_name", "translit_name")
        tgt_names = _build_texts(tgt_c, "soft_name", "business_name", "translit_name")
        s1_addrs = _build_texts(s1_c, "clean_address", "business_address", "translit_address")
        tgt_addrs = _build_texts(tgt_c, "clean_address", "business_address", "translit_address")

        # --- Channel A: Name TF-IDF ---
        t_ch0 = time.perf_counter()
        vec_name = TfidfVectorizer(analyzer="char", ngram_range=(3, 3), min_df=1, sublinear_tf=True)
        tgt_name_mat = vec_name.fit_transform(tgt_names).T
        tgt_ids = tgt_c["entity_id"].tolist()

        n_vocab_name, n_tgt_name = tgt_name_mat.shape
        nnz_name = tgt_name_mat.nnz
        mat_size_mb_name = (tgt_name_mat.data.nbytes + tgt_name_mat.indices.nbytes + tgt_name_mat.indptr.nbytes) / (1024 * 1024)

        sparse_matrix_stats.append({
            "country": country,
            "channel": "Name TF-IDF",
            "shape": f"({n_vocab_name:,}, {n_tgt_name:,})",
            "nnz": f"{nnz_name:,}",
            "sparsity_pct": f"{(1.0 - nnz_name / (n_vocab_name * n_tgt_name)) * 100:.4f}%",
            "matrix_size_mb": f"{mat_size_mb_name:.1f} MB",
        })

        for start_idx in range(0, len(s1_c), batch_size):
            end_idx = min(start_idx + batch_size, len(s1_c))
            s1_batch_names = s1_names[start_idx:end_idx]
            s1_batch_ids = s1_c["entity_id"].iloc[start_idx:end_idx].tolist()
            batch_mat = vec_name.transform(s1_batch_names)
            batch_sims = batch_mat.dot(tgt_name_mat)

            for i, s1_id in enumerate(s1_batch_ids):
                r_start = batch_sims.indptr[i]
                r_end = batch_sims.indptr[i + 1]
                if r_start == r_end:
                    continue
                scores = batch_sims.data[r_start:r_end]
                col_idx = batch_sims.indices[r_start:r_end]
                valid_mask = scores >= name_min_sim
                if not np.any(valid_mask):
                    continue
                scores = scores[valid_mask]
                col_idx = col_idx[valid_mask]
                if len(scores) > name_top_k:
                    top_sub = np.argpartition(scores, -name_top_k)[-name_top_k:]
                    top_cols = col_idx[top_sub]
                else:
                    top_cols = col_idx
                for c_idx in top_cols:
                    candidate_pairs[s1_id].add(tgt_ids[c_idx])

        t_ch_name = time.perf_counter() - t_ch0
        rss, peak, vms = get_memory_info()
        cands_so_far = sum(len(c) for c in candidate_pairs.values())
        log(f"     ✓ Channel A [Name]: time={t_ch_name:.2f}s | matrix={mat_size_mb_name:.1f}MB | RAM={rss:.1f}MB | pairs={cands_so_far:,}")
        blocking_channel_stats.append({
            "stage": f"[{country}] Name TF-IDF",
            "time_sec": round(t_ch_name, 2),
            "rss_mb": round(rss, 1),
            "peak_rss_mb": round(peak, 1),
            "cumulative_candidates": cands_so_far,
        })

        # Clean intermediate name matrix
        del vec_name, tgt_name_mat
        gc.collect()

        # --- Channel B: Address TF-IDF ---
        t_ch0 = time.perf_counter()
        min_df_addr = 2 if len(tgt_c) > 500 else 1
        max_df_addr = 0.40 if len(tgt_c) > 500 else 1.0
        vec_addr = TfidfVectorizer(analyzer="char", ngram_range=(3, 3), min_df=min_df_addr, max_df=max_df_addr, sublinear_tf=True)
        tgt_addr_mat = vec_addr.fit_transform(tgt_addrs).T

        n_vocab_addr, n_tgt_addr = tgt_addr_mat.shape
        nnz_addr = tgt_addr_mat.nnz
        mat_size_mb_addr = (tgt_addr_mat.data.nbytes + tgt_addr_mat.indices.nbytes + tgt_addr_mat.indptr.nbytes) / (1024 * 1024)

        sparse_matrix_stats.append({
            "country": country,
            "channel": "Addr TF-IDF",
            "shape": f"({n_vocab_addr:,}, {n_tgt_addr:,})",
            "nnz": f"{nnz_addr:,}",
            "sparsity_pct": f"{(1.0 - nnz_addr / (n_vocab_addr * n_tgt_addr)) * 100:.4f}%",
            "matrix_size_mb": f"{mat_size_mb_addr:.1f} MB",
        })

        for start_idx in range(0, len(s1_c), batch_size):
            end_idx = min(start_idx + batch_size, len(s1_c))
            s1_batch_addrs = s1_addrs[start_idx:end_idx]
            s1_batch_ids = s1_c["entity_id"].iloc[start_idx:end_idx].tolist()
            batch_mat = vec_addr.transform(s1_batch_addrs)
            batch_sims = batch_mat.dot(tgt_addr_mat)

            for i, s1_id in enumerate(s1_batch_ids):
                if not s1_batch_addrs[i].strip():
                    continue
                r_start = batch_sims.indptr[i]
                r_end = batch_sims.indptr[i + 1]
                if r_start == r_end:
                    continue
                scores = batch_sims.data[r_start:r_end]
                col_idx = batch_sims.indices[r_start:r_end]
                valid_mask = scores >= addr_min_sim
                if not np.any(valid_mask):
                    continue
                scores = scores[valid_mask]
                col_idx = col_idx[valid_mask]
                if len(scores) > addr_top_k:
                    top_sub = np.argpartition(scores, -addr_top_k)[-addr_top_k:]
                    top_cols = col_idx[top_sub]
                else:
                    top_cols = col_idx
                for c_idx in top_cols:
                    candidate_pairs[s1_id].add(tgt_ids[c_idx])

        t_ch_addr = time.perf_counter() - t_ch0
        rss, peak, vms = get_memory_info()
        cands_so_far = sum(len(c) for c in candidate_pairs.values())
        log(f"     ✓ Channel B [Addr]: time={t_ch_addr:.2f}s | matrix={mat_size_mb_addr:.1f}MB | RAM={rss:.1f}MB | pairs={cands_so_far:,}")
        blocking_channel_stats.append({
            "stage": f"[{country}] Addr TF-IDF",
            "time_sec": round(t_ch_addr, 2),
            "rss_mb": round(rss, 1),
            "peak_rss_mb": round(peak, 1),
            "cumulative_candidates": cands_so_far,
        })

        # Clean intermediate address matrix
        del vec_addr, tgt_addr_mat
        gc.collect()

        # --- Channel C: Building Number Match ---
        t_ch0 = time.perf_counter()
        if "numbers" in s1_c and "numbers" in tgt_c:
            tgt_ids = tgt_c["entity_id"].astype(str).tolist()
            tgt_names_ser = tgt_c["soft_name"].fillna("").astype(str) if "soft_name" in tgt_c else tgt_c["business_name"].fillna("").astype(str)
            tgt_prefixes = dict(zip(tgt_ids, tgt_names_ser.str[:2].tolist(), strict=False))

            num_to_tgt = {}
            for eid, num_set in zip(tgt_ids, tgt_c["numbers"].tolist(), strict=False):
                for num in num_set:
                    num_str = str(num)
                    if len(num_str) >= 2:
                        num_to_tgt.setdefault(num_str, []).append(eid)

            s1_ids = s1_c["entity_id"].astype(str).tolist()
            s1_names_ser = s1_c["soft_name"].fillna("").astype(str) if "soft_name" in s1_c else s1_c["business_name"].fillna("").astype(str)
            s1_prefixes = s1_names_ser.str[:2].tolist()

            for s1_id, s1_pfx, num_set in zip(s1_ids, s1_prefixes, s1_c["numbers"].tolist(), strict=False):
                if not s1_pfx:
                    continue
                for num in num_set:
                    num_str = str(num)
                    if len(num_str) < 2:
                        continue
                    cands = num_to_tgt.get(num_str, [])
                    if len(cands) <= 50:
                        for cand_id in cands:
                            if tgt_prefixes.get(cand_id, "") == s1_pfx:
                                candidate_pairs[s1_id].add(cand_id)

        t_ch_bldg = time.perf_counter() - t_ch0
        rss, peak, vms = get_memory_info()
        cands_so_far = sum(len(c) for c in candidate_pairs.values())
        log(f"     ✓ Channel C [BldgNum]: time={t_ch_bldg:.2f}s | RAM={rss:.1f}MB | pairs={cands_so_far:,}\n")
        blocking_channel_stats.append({
            "stage": f"[{country}] BldgNum",
            "time_sec": round(t_ch_bldg, 2),
            "rss_mb": round(rss, 1),
            "peak_rss_mb": round(peak, 1),
            "cumulative_candidates": cands_so_far,
        })

    # Save candidates TSV
    cand_file = out_path / "candidate_pairs.tsv"
    write_candidate_pairs_tsv(cand_file, candidate_pairs)
    total_candidates = sum(len(c) for c in candidate_pairs.values())
    log(f"   ✓ Exported candidate pairs: {cand_file}")

    if gt_dict:
        blocking_report = evaluate_blocking_candidates(gt_dict, candidate_pairs, len(target_wide))
        log(f"   📊 Blocking Recall (Pair Completeness): {blocking_report.pair_completeness * 100:.2f}%")
        log(f"   📊 Reduction Ratio:                    {blocking_report.reduction_ratio * 100:.4f}%")
        log(f"   📊 Avg Candidates per S1:              {blocking_report.avg_candidates_per_s1:.1f}\n")

    # --- Step 4: Feature Extraction (28 Features) ---
    t0 = time.perf_counter()
    log("==> 4. Extracting 28 RapidFuzz & Canonical Address Features...")
    extractor = PairwiseFeatureExtractor()
    pairs_df = extractor.extract_features_df(candidate_pairs, s1_wide, target_wide, ground_truth=gt_dict)
    t_feat = time.perf_counter() - t0
    rss, peak, vms = get_memory_info()
    pairs_mem_mb = pairs_df.memory_usage(deep=True).sum() / (1024 * 1024)
    log(f"   ✓ Extracted {len(pairs_df):,} feature rows in {t_feat:.2f}s ({len(pairs_df) / max(t_feat, 0.001):,.0f} pairs/sec)")
    log(f"   ✓ pairs_df Memory: {pairs_mem_mb:.1f} MB | Current RAM: {rss:.1f} MB ({rss/1024:.2f} GB) | Peak: {peak:.1f} MB\n")

    # --- Step 5: LightGBM Training & Threshold Calibration ---
    t0 = time.perf_counter()
    log("==> 5. LightGBM Training & Disjoint Holdout Calibration...")
    unique_s1 = list(s1_wide["entity_id"].unique())
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
    log(f"   ✓ Optimal Threshold tau = {best_tau:.2f} (Holdout Macro F0.5 = {best_score:.4f})")

    # Generate predictions on full S1
    predictor = SingletonGatedPredictor(decision_threshold=best_tau)
    pairs_df["prob"] = matcher.predict_proba(pairs_df)
    final_preds = predictor.filter_predictions(pairs_df, list(s1_wide["entity_id"]), threshold=best_tau)

    # Save matching results TSV
    match_file = out_path / "matching_results.tsv"
    write_matching_results_tsv(match_file, final_preds)
    log(f"   ✓ Final predictions exported to: {match_file}\n")

    # Evaluate validation holdout
    val_pairs["prob"] = matcher.predict_proba(val_pairs)
    val_preds = predictor.filter_predictions(val_pairs, list(val_s1), threshold=best_tau)
    eval_report = evaluate_resolution_predictions(val_gt, val_preds)

    # --- Step 6: Failure Diagnostics ---
    log("==> 6. Automated Failure Diagnostics & Error Analysis...")
    diag_logger = FailureDiagnosticsLogger(rep_path)
    val_s1_records = s1_wide[s1_wide["entity_id"].isin(val_s1)]
    diag_logger.log_failures_from_predictions(
        s1_df=val_s1_records,
        target_df=target_wide,
        candidate_pairs=candidate_pairs,
        predictions=val_preds,
        ground_truth=val_gt,
        scored_pairs_df=val_pairs,
    )
    diag_report_path = diag_logger.generate_diagnostics_report(eval_report)
    log(f"   ✓ Executive diagnostics written to: {diag_report_path}\n")

    total_time = time.perf_counter() - start_total
    rss, peak, vms = get_memory_info()

    log("=" * 72)
    log("📊 KAGGLE EXECUTION & MEMORY SCORECARD")
    log("=" * 72)
    log(f"Total Execution Time:        {total_time:.2f}s ({total_time / 60:.2f} min)")
    log(f"Peak Working Set (RAM):       {peak:.1f} MB ({peak / 1024:.2f} GB)")
    log(f"Kaggle 30 GB Headroom:        {30.0 - (peak / 1024):.2f} GB remaining")
    log(f"Total Candidate Pairs:        {total_candidates:,}")
    log(f"Validation Macro F0.5:        {eval_report.macro_f05:.4f}")
    log(f"Validation Precision:         {eval_report.precision * 100:.2f}%")
    log(f"Validation Recall:            {eval_report.recall * 100:.2f}%")
    log(f"Singleton Accuracy:           {eval_report.singleton_accuracy * 100:.2f}%\n")

    log("--- Sparse Matrix Dimensions & Sparsity ---")
    log(pd.DataFrame(sparse_matrix_stats).to_string(index=False))
    log("\n--- Memory Progression by Blocking Stage ---")
    log(pd.DataFrame(blocking_channel_stats).to_string(index=False))
    log("=" * 72)


if __name__ == "__main__":
    main()
