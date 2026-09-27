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


# --------------------------------------------------------------------------- features (lean)
# Fork-shared state holds only Arrow tables and numpy arrays (no per-row Python objects),
# so reading it in workers never dirties/copies parent pages (avoids fork+THP COW blowup).
_LEAN: dict[str, object] = {}


def _fixed_bytes_ids(col: object) -> np.ndarray:
    import pyarrow as pa

    assert isinstance(col, pa.ChunkedArray)
    obj = col.to_numpy(zero_copy_only=False)
    width = max(1, max((len(x) for x in obj), default=1))
    out = np.asarray(obj, dtype=f"S{width + 2}")
    del obj
    return out


def _lean_take(prefix: str, ids: np.ndarray) -> pd.DataFrame:
    import pyarrow as pa

    keys = _LEAN[f"{prefix}_keys"]
    order = _LEAN[f"{prefix}_order"]
    table = _LEAN[f"{prefix}_tab"]
    assert isinstance(keys, np.ndarray) and isinstance(order, np.ndarray)
    assert isinstance(table, pa.Table)
    q = ids.astype(keys.dtype)
    pos = np.clip(np.searchsorted(keys, q), 0, len(keys) - 1)
    ok = keys[pos] == q
    rows = np.sort(order[pos[ok]])
    return table.take(pa.array(rows)).to_pandas()


def _lean_feature_task(i: int) -> int:
    import pyarrow as pa

    out_dir = Path(str(_LEAN["out_dir"]))
    out = out_dir / f"part_{i:05d}.parquet"
    if out.is_file():
        return -1
    bounds = _LEAN["bounds"]
    cands = _LEAN["cands"]
    assert isinstance(bounds, list) and isinstance(cands, pa.Table)
    a, b = bounds[i]
    part = cands.slice(a, b - a).to_pandas()
    candidates: dict[str, dict[str, float]] = {}
    for s, c, sim in zip(
        part["s1_id"].to_numpy(),
        part["cand_id"].to_numpy(),
        part["blocking_sim"].to_numpy(),
        strict=False,
    ):
        d = candidates.get(s)
        if d is None:
            d = candidates[s] = {}
        d[c] = float(sim)
    s1_df = _lean_take("s1", np.asarray(list(candidates.keys()), dtype=object))
    tgt_df = _lean_take("tgt", part["cand_id"].unique())
    ex = PairwiseFeatureExtractor()
    feats = ex._extract_features_serial(
        candidates,
        s1_dict=ex.build_record_lookup(s1_df),
        target_dict=ex.build_record_lookup(tgt_df),
        ground_truth=None,
        show_progress=False,
    )
    tmp = out.with_suffix(".tmp")
    feats.to_parquet(tmp, index=False)
    tmp.rename(out)
    return len(feats)


def stage_features(args: argparse.Namespace) -> None:
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    from parallax.utils.parallel import run_pool

    work = Path(args.work_dir)
    out_dir = work / "feat_parts"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cands = pq.read_table(work / "test_candidates.parquet").combine_chunks()
    cands = cands.sort_by("s1_id").combine_chunks()
    s1_col = cands["s1_id"]
    change = pc.not_equal(s1_col.slice(1), s1_col.slice(0, len(s1_col) - 1))
    starts = np.concatenate([[0], np.flatnonzero(change.to_numpy(zero_copy_only=False)) + 1])
    n = len(cands)
    bounds: list[tuple[int, int]] = []
    a = 0
    for cut in range(args.part_pairs, n, args.part_pairs):
        j = int(np.searchsorted(starts, cut))
        b = int(starts[j]) if j < len(starts) else n
        if b > a:
            bounds.append((a, b))
            a = b
    if a < n:
        bounds.append((a, n))
    log(
        f"Candidates: {n:,} pairs, {len(starts):,} S1 -> {len(bounds)} parts ({time.time() - t0:.1f}s)"
    )

    t0 = time.time()
    s1_tab = pq.read_table(work / "checkpoints" / "test_s1_wide.parquet").combine_chunks()
    tgt_tab = pq.read_table(work / "checkpoints" / "test_target_wide.parquet").combine_chunks()
    for prefix, tab in (("s1", s1_tab), ("tgt", tgt_tab)):
        keys = _fixed_bytes_ids(tab["entity_id"])
        order = np.argsort(keys, kind="stable")
        _LEAN[f"{prefix}_keys"] = keys[order]
        _LEAN[f"{prefix}_order"] = order
        _LEAN[f"{prefix}_tab"] = tab
        del keys
    _LEAN["cands"] = cands
    _LEAN["bounds"] = bounds
    _LEAN["out_dir"] = str(out_dir)
    gc.collect()
    log(f"Loaded Arrow tables in {time.time() - t0:.1f}s {rss_gb()}")

    done = [0]
    todo = [i for i in range(len(bounds)) if not (out_dir / f"part_{i:05d}.parquet").is_file()]
    log(
        f"{len(bounds) - len(todo)} parts already done; {len(todo)} to go with {args.n_jobs} workers"
    )
    t0 = time.time()

    def _tick(_i: int, _r: int) -> None:
        done[0] += 1
        if done[0] % 10 == 0 or done[0] == len(todo):
            el = time.time() - t0
            eta = el / done[0] * (len(todo) - done[0])
            log(
                f"features {done[0]}/{len(todo)} parts, {el:.0f}s elapsed, ETA {eta:.0f}s {rss_gb()}"
            )

    run_pool(
        _lean_feature_task,
        todo,
        max_workers=args.n_jobs,
        label="lean-features",
        max_retries=6,
        on_result=_tick,
    )
    missing = [i for i in range(len(bounds)) if not (out_dir / f"part_{i:05d}.parquet").is_file()]
    if missing:
        raise RuntimeError(f"{len(missing)} feature parts missing")
    (work / "feat_parts.done").write_text(str(len(bounds)))
    log(f"All {len(bounds)} feature parts written in {time.time() - t0:.1f}s")


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
    feat_dir = Path(args.features_dir) if args.features_dir else work / "feat_parts"

    scored_path = work / "test_scored_pairs.parquet"
    if scored_path.is_file():
        scored = pd.read_parquet(scored_path)
        log("Loaded cached scored pairs")
    else:
        p1 = _matcher(cfg, list(FEATURE_COLUMNS), 42, args.threads)
        p1.load_model(work / "model_pass1.txt")
        p2 = _matcher(cfg, list(FEATURE_COLUMNS) + list(META_FEATURE_COLUMNS), 1042, args.threads)
        p2.load_model(work / "model_pass2.txt")
        parts = sorted(feat_dir.glob("part_*.parquet"))
        if not parts:
            raise FileNotFoundError(f"No feature parts in {feat_dir}")
        t0 = time.time()
        outs: list[pd.DataFrame] = []
        for k, part in enumerate(parts):
            df = pd.read_parquet(part)
            df["prob"] = p1.predict_proba(df)
            compute_entity_meta_features(df, prob_col="prob")
            df["prob"] = p2.predict_proba(df)
            outs.append(df[["s1_id", "cand_id", "prob"]].copy())
            del df
            if (k + 1) % 20 == 0 or k + 1 == len(parts):
                log(f"scored {k + 1}/{len(parts)} parts in {time.time() - t0:.0f}s")
        scored = pd.concat(outs, ignore_index=True)
        del outs
        scored.to_parquet(scored_path, index=False)
        log(f"Scored {len(scored):,} pairs in {time.time() - t0:.1f}s")
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

    f = sub.add_parser("features")
    f.add_argument("--work-dir", default="submission_work")
    f.add_argument("--n-jobs", type=int, default=24)
    f.add_argument("--part-pairs", type=int, default=250_000)

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
    c.add_argument("--features-dir", default=None)

    args = ap.parse_args()
    {
        "prepare": stage_prepare,
        "features": stage_features,
        "train": stage_train,
        "score": stage_score,
    }[args.stage](args)


if __name__ == "__main__":
    main()
