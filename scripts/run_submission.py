#!/usr/bin/env python
"""
Parallax Full-Scale Submission Pipeline
=======================================
Three independent stages so test preparation and final model training can run at the
same time (in separate screen sessions), then join for scoring:

  prepare  test TSVs -> widening -> multi-channel blocking -> pairwise features (parquet)
  train    cached labelled training features -> two-pass LightGBM -> saved boosters + tau
  score    test features + boosters -> two-pass probabilities -> tau -> one-owner-per-target
           -> matching_results.tsv (+ scored pairs parquet for re-thresholding)

Every stage checkpoints its outputs in --work-dir and skips work that is already done.

Examples:
  python scripts/run_submission.py prepare --test-dir data/raw/test --work-dir sub --n-jobs 64
  python scripts/run_submission.py train --train-features output_par/features_x.parquet \
      --work-dir sub --config Config-XDeep --tau 0.88 --threads 16
  python scripts/run_submission.py score --work-dir sub --threads 32
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from parallax.blocking.tfidf_blocker import DualChannelTFIDFBlocker  # noqa: E402
from parallax.data.contracts import (  # noqa: E402
    load_business_records_df,
    write_matching_results_tsv,
)
from parallax.features.extractor import (  # noqa: E402
    FEATURE_COLUMNS,
    PairwiseFeatureExtractor,
)
from parallax.features.meta_features import (  # noqa: E402
    META_FEATURE_COLUMNS,
    compute_entity_meta_features,
)
from parallax.models.matcher import LightGBMMatcher  # noqa: E402
from parallax.preprocessing.normalizer import widen_many_records_dfs  # noqa: E402
from parallax.utils.checkpoint_manager import CheckpointManager  # noqa: E402

CONFIGS = {
    "Config-Fast": (0.08, 31, 6, 100),
    "Config-Balanced": (0.05, 31, 6, 150),
    "Config-Deep": (0.05, 63, 8, 150),
    "Config-Dense": (0.04, 63, 8, 250),
    "Config-XDeep": (0.04, 127, 10, 250),
    "Config-Conservative": (0.03, 31, 6, 180),
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def rss_gb() -> str:
    try:
        import psutil

        vm = psutil.virtual_memory()
        return f"avail {vm.available / 2**30:.0f} GB"
    except Exception:
        return ""


# --------------------------------------------------------------------------- prepare
def stage_prepare(args: argparse.Namespace) -> None:
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    ckpt = CheckpointManager(work / "checkpoints", dataset_name="test")
    feat_path = work / "test_features.parquet"
    cand_path = work / "test_candidates.parquet"
    if feat_path.is_file():
        log(f"Test features already exist: {feat_path}")
        return

    test_dir = Path(args.test_dir)
    t0 = time.time()
    if ckpt.has_checkpoint("test_s1_wide") and ckpt.has_checkpoint("test_target_wide"):
        s1_wide = ckpt.load_dataframe("test_s1_wide")
        target_wide = ckpt.load_dataframe("test_target_wide")
        assert s1_wide is not None and target_wide is not None
        log(f"Loaded widened tables ({len(s1_wide):,} S1 / {len(target_wide):,} targets)")
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(3) as tpe:
            futs = [
                tpe.submit(load_business_records_df, test_dir / f"test_source{i}.tsv")
                for i in (1, 2, 3)
            ]
            s1_df, s2_df, s3_df = (f.result() for f in futs)
        log(
            f"Loaded test: S1 {len(s1_df):,} | S2 {len(s2_df):,} | S3 {len(s3_df):,} "
            f"in {time.time() - t0:.1f}s"
        )
        t0 = time.time()
        s1_wide, s2_wide, s3_wide = widen_many_records_dfs(
            [s1_df, s2_df, s3_df], n_jobs=args.n_jobs
        )
        target_wide = pd.concat([s2_wide, s3_wide], ignore_index=True)
        del s1_df, s2_df, s3_df, s2_wide, s3_wide
        gc.collect()
        ckpt.save_dataframe("test_s1_wide", s1_wide)
        ckpt.save_dataframe("test_target_wide", target_wide)
        log(f"Widened in {time.time() - t0:.1f}s {rss_gb()}")

    # Blocking (per-country checkpoints inside the blocker)
    t0 = time.time()
    if cand_path.is_file():
        cdf = pd.read_parquet(cand_path)
        candidates: dict[str, dict[str, float]] = {s: {} for s in s1_wide["entity_id"]}
        for s, c, sim in zip(
            cdf["s1_id"].to_numpy(),
            cdf["cand_id"].to_numpy(),
            cdf["blocking_sim"].to_numpy(),
            strict=False,
        ):
            candidates[s][c] = float(sim)
        del cdf
        log(f"Loaded cached candidates in {time.time() - t0:.1f}s")
    else:
        blocker = DualChannelTFIDFBlocker(
            name_top_k=35,
            addr_top_k=25,
            name_min_sim=0.15,
            addr_min_sim=0.20,
            batch_size=args.block_batch,
            show_progress=True,
            max_candidates_per_query=args.max_candidates,
            n_jobs=args.n_jobs,
        )
        candidates = blocker.generate_candidates(
            s1_wide, target_wide, checkpoint_mgr=ckpt, n_jobs=args.n_jobs, ckpt_tag="test"
        )
        rows_s1: list[str] = []
        rows_c: list[str] = []
        rows_sim: list[float] = []
        for s, cands in candidates.items():
            for c, sim in cands.items():
                rows_s1.append(s)
                rows_c.append(c)
                rows_sim.append(sim)
        pd.DataFrame({"s1_id": rows_s1, "cand_id": rows_c, "blocking_sim": rows_sim}).to_parquet(
            cand_path, index=False
        )
        del rows_s1, rows_c, rows_sim
        log(
            f"Blocked {len(candidates):,} queries -> "
            f"{sum(len(v) for v in candidates.values()):,} pairs in {time.time() - t0:.1f}s "
            f"{rss_gb()}"
        )
    gc.collect()

    # Features
    t0 = time.time()
    feats = PairwiseFeatureExtractor().extract_features_df(
        candidates, s1_wide, target_wide, ground_truth=None, show_progress=True, n_jobs=args.n_jobs
    )
    log(f"Extracted {len(feats):,} feature rows in {time.time() - t0:.1f}s {rss_gb()}")
    tmp = feat_path.with_suffix(".tmp.parquet")
    feats.to_parquet(tmp, index=False)
    tmp.rename(feat_path)
    log(f"Saved {feat_path}")


# --------------------------------------------------------------------------- train
def _matcher(cfg: str, cols: list[str], seed: int, threads: int) -> LightGBMMatcher:
    lr, leaves, depth, trees = CONFIGS[cfg]
    return LightGBMMatcher(
        learning_rate=lr,
        num_leaves=leaves,
        max_depth=depth,
        n_estimators=trees,
        feature_columns=cols,
        seed=seed,
        num_threads=threads,
    )


def stage_train(args: argparse.Namespace) -> None:
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    meta_path = work / "model_meta.json"
    if meta_path.is_file() and not args.force:
        log(f"Model already trained: {meta_path}")
        return
    t0 = time.time()
    train = pd.read_parquet(args.train_features)
    log(f"Loaded {len(train):,} training pairs in {time.time() - t0:.1f}s")

    t0 = time.time()
    p1 = _matcher(args.config, list(FEATURE_COLUMNS), 42, args.threads)
    p1.train(train)
    p1.save_model(work / "model_pass1.txt")
    log(f"Pass 1 trained in {time.time() - t0:.1f}s")

    t0 = time.time()
    train["prob"] = p1.predict_proba(train)
    compute_entity_meta_features(train, prob_col="prob")
    p2 = _matcher(
        args.config, list(FEATURE_COLUMNS) + list(META_FEATURE_COLUMNS), 1042, args.threads
    )
    p2.train(train)
    p2.save_model(work / "model_pass2.txt")
    log(f"Pass 2 trained in {time.time() - t0:.1f}s")

    meta_path.write_text(
        json.dumps({"config": args.config, "tau": args.tau, "train_pairs": len(train)}, indent=2)
    )
    log(f"Saved models and {meta_path}")


# --------------------------------------------------------------------------- score
def _predict_chunked(m: LightGBMMatcher, df: pd.DataFrame, chunk: int = 4_000_000) -> np.ndarray:
    out = np.empty(len(df), dtype=np.float32)
    for a in range(0, len(df), chunk):
        out[a : a + chunk] = m.predict_proba(df.iloc[a : a + chunk])
    return out


def stage_score(args: argparse.Namespace) -> None:
    work = Path(args.work_dir)
    meta = json.loads((work / "model_meta.json").read_text())
    tau = float(args.tau if args.tau is not None else meta["tau"])
    cfg = meta["config"]

    t0 = time.time()
    feats = pd.read_parquet(work / "test_features.parquet")
    feats["s1_id"] = feats["s1_id"].astype("category")
    log(f"Loaded {len(feats):,} test pairs in {time.time() - t0:.1f}s")

    scored_path = work / "test_scored_pairs.parquet"
    if scored_path.is_file():
        scored = pd.read_parquet(scored_path)
        log("Loaded cached scored pairs")
    else:
        t0 = time.time()
        p1 = _matcher(cfg, list(FEATURE_COLUMNS), 42, args.threads)
        p1.load_model(work / "model_pass1.txt")
        feats["prob"] = _predict_chunked(p1, feats)
        compute_entity_meta_features(feats, prob_col="prob")
        p2 = _matcher(cfg, list(FEATURE_COLUMNS) + list(META_FEATURE_COLUMNS), 1042, args.threads)
        p2.load_model(work / "model_pass2.txt")
        feats["prob"] = _predict_chunked(p2, feats)
        scored = feats[["s1_id", "cand_id", "prob"]].copy()
        scored["s1_id"] = scored["s1_id"].astype(str)
        scored.to_parquet(scored_path, index=False)
        log(f"Scored in {time.time() - t0:.1f}s")
    del feats
    gc.collect()

    probs = scored["prob"].to_numpy()
    mask = probs >= tau
    if not args.no_exclusive:
        cand_max = scored.groupby("cand_id")["prob"].transform("max").to_numpy()
        excl = mask & (probs >= cand_max)
        log(f"Exclusivity removed {int((mask & ~excl).sum()):,} contested pairs")
        mask = excl

    s1_all = pd.read_parquet(work / "checkpoints" / "test_s1_wide.parquet", columns=["entity_id"])
    preds: dict[str, set[str]] = {str(s): set() for s in s1_all["entity_id"]}
    for s, c in zip(
        scored["s1_id"].to_numpy()[mask], scored["cand_id"].to_numpy()[mask], strict=False
    ):
        preds[str(s)].add(str(c))
    out = Path(args.output) if args.output else work / f"matching_results_tau{tau:.2f}.tsv"
    write_matching_results_tsv(out, preds)
    n_pred = sum(len(v) for v in preds.values())
    n_empty = sum(1 for v in preds.values() if not v)
    log(
        f"Wrote {out}: {len(preds):,} S1 rows, {n_pred:,} matches, "
        f"{n_empty:,} predicted singletons (tau={tau}, config={cfg})"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)

    a = sub.add_parser("prepare")
    a.add_argument("--test-dir", default="data/raw/test")
    a.add_argument("--work-dir", default="submission_work")
    a.add_argument("--n-jobs", type=int, default=64)
    a.add_argument("--max-candidates", type=int, default=35)
    a.add_argument("--block-batch", type=int, default=500)

    b = sub.add_parser("train")
    b.add_argument("--train-features", required=True)
    b.add_argument("--work-dir", default="submission_work")
    b.add_argument("--config", default="Config-XDeep", choices=sorted(CONFIGS))
    b.add_argument("--tau", type=float, required=True)
    b.add_argument("--threads", type=int, default=16)
    b.add_argument("--force", action="store_true")

    c = sub.add_parser("score")
    c.add_argument("--work-dir", default="submission_work")
    c.add_argument("--tau", type=float, default=None)
    c.add_argument("--threads", type=int, default=32)
    c.add_argument("--no-exclusive", action="store_true")
    c.add_argument("--output", default=None)

    args = ap.parse_args()
    {"prepare": stage_prepare, "train": stage_train, "score": stage_score}[args.stage](args)


if __name__ == "__main__":
    main()
