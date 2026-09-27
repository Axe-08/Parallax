"""
Phase 2: supervised BGE-M3 retriever with multi-channel hard negatives, evaluated on the 5K.

    split (global GT contract)                     eval = the 5K population (never trained on)
        -> S1-grouped train / val draw from NON-eval S1s
        -> zero-shot BGE (full view) encode of every S2/S3 target + S1 queries
        -> mining pool for train/val S1s: zero-shot BGE top-k + lexical name/addr top-k
        -> hard negatives (GT-safe, provenance-tracked)
        -> fine-tune BGE-M3 (masked multi-positive InfoNCE), best checkpoint on val recall@10
        -> encode targets + eval S1s with the fine-tuned model (full / name / address views)
        -> FAISS retrieval for the 5K -> candidate-level metrics vs E0 (recall, BFN, cost, overlap)
        -> matcher arms on the 5K (deterministic grouped outer/inner-OOF harness, paired seeds):
             A0     E0 pool, 28 baseline features
             A1zs   E0 pool, 28 + zero-shot BGE cosine
             A1     E0 pool, 28 + fine-tuned BGE cosines (full/name/address)
             A2@K   E0 ∪ fine-tuned BGE top-K pool, 28 + BGE cosines + provenance/rank

Leakage: the retriever only ever sees GT of non-eval S1s (train for gradients, val for
checkpoint selection). Every target matched to eval S1s is disjoint from training targets
(checked). BGE scores on the 5K are therefore out-of-sample for every eval row, which is a
stricter guarantee than K-fold OOF for the downstream matcher.

Each stage caches its outputs under --work-dir so a crashed/killed run resumes.

Usage (project root, DGX):
    python experiments/hybrid_er/orchestration/run_phase2_bge.py --gpu auto
    python experiments/hybrid_er/orchestration/run_phase2_bge.py --smoke --gpu auto   # plumbing only
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
for p in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from experiments.hybrid_er.core.cv import split_s1_groups  # noqa: E402
from experiments.hybrid_er.core.data import SplitData, load_split  # noqa: E402
from experiments.hybrid_er.core.runtime import RunRecorder, peak_vram_gib, select_device  # noqa: E402
from experiments.hybrid_er.core.serialization import RecordSerializer, SerializerConfig  # noqa: E402
from experiments.hybrid_er.core.validation import scope_ground_truth_to_eval  # noqa: E402
from experiments.hybrid_er.evaluation.matcher_harness import run_grouped_matcher  # noqa: E402
from experiments.hybrid_er.evaluation.metrics import (  # noqa: E402
    evaluate_candidate_recall,
    evaluate_matcher_predictions,
)
from experiments.hybrid_er.orchestration.run_5k import BASELINE_28_FEATURES, load_e0_channel  # noqa: E402
from experiments.hybrid_er.retrieval.dense_index import DenseIndex  # noqa: E402
from experiments.hybrid_er.retrieval.lexical import lexical_retrieve  # noqa: E402
from experiments.hybrid_er.retrieval.union import candidates_to_dict, merge_candidate_tables  # noqa: E402
from experiments.hybrid_er.training.hard_negatives import mine_hard_negatives  # noqa: E402

logger = logging.getLogger("phase2_bge")

VIEWS = ("full", "name", "address")
MAX_LEN = {"full": 128, "name": 48, "address": 96}


# ---------------------------------------------------------------------------- helpers
def _save_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def build_entity_table(data: SplitData, s1_ids: list[str], target_ids: np.ndarray | None) -> pd.DataFrame:
    """S1 subset + targets (all, or a smoke subset) with a `source` column and row index = entity idx."""
    s1 = data.s1[data.s1["entity_id"].isin(set(s1_ids))].assign(source="S1")
    s2 = data.s2.assign(source="S2")
    s3 = data.s3.assign(source="S3")
    tg = pd.concat([s2, s3], ignore_index=True)
    if target_ids is not None:
        tg = tg[tg["entity_id"].isin(set(target_ids))]
    ent = pd.concat([s1, tg], ignore_index=True)
    if ent["entity_id"].duplicated().any():
        raise ValueError("Entity table has duplicate IDs.")
    return ent.reset_index(drop=True)


def serialize_all(ent: pd.DataFrame, tokenizer, work: Path) -> dict[str, np.ndarray]:
    path = work / "texts.parquet"
    if path.exists():
        t = pd.read_parquet(path)
        if len(t) == len(ent) and (t["entity_id"].to_numpy() == ent["entity_id"].to_numpy()).all():
            return {v: t[v].to_numpy(dtype=object) for v in VIEWS}
    ser = RecordSerializer(tokenizer, SerializerConfig())
    texts = {v: np.array(ser.serialize(ent, v), dtype=object) for v in VIEWS}
    logger.info("Serialization truncation counts: %s", ser.truncation_counts)
    pd.DataFrame({"entity_id": ent["entity_id"], **texts}).to_parquet(path)
    _save_json(work / "serialization.json", {"truncation_counts": ser.truncation_counts, "config": vars(ser.config)})
    return texts


def encode_cached(model_path: str | Path, texts: np.ndarray, idx: np.ndarray, view: str, path: Path,
                  devices: list[str], bs: int) -> np.ndarray:
    if path.exists():
        emb = np.load(path, mmap_mode="r")
        if emb.shape[0] == len(idx):
            return emb
    from experiments.hybrid_er.models.bge_encoder import encode_multi_device

    t = time.time()
    emb = encode_multi_device(model_path, [texts[i] for i in idx], devices, batch_size=bs, max_length=MAX_LEN[view])
    np.save(path, emb)
    logger.info("Encoded %d texts (%s) on %s in %.1fs -> %s", len(idx), view, devices, time.time() - t, path.name)
    return emb


def dense_hits(index: DenseIndex, q_emb: np.ndarray, q_ids: np.ndarray, q_countries: np.ndarray, k: int, blocker: str) -> pd.DataFrame:
    hits = index.search(q_emb, q_ids, q_countries, k)
    hits["blocker"] = blocker
    return hits


def pair_cosines(pairs: pd.DataFrame, emb_q: np.ndarray, q_pos: dict[str, int], emb_t: np.ndarray, t_pos: dict[str, int]) -> np.ndarray:
    qi = pairs["s1_id"].map(q_pos)
    ti = pairs["cand_id"].map(t_pos)
    if qi.isna().any() or ti.isna().any():
        raise ValueError("Pair cosine lookup failed: entity missing from embedding table.")
    qi, ti = qi.to_numpy(int), ti.to_numpy(int)
    out = np.empty(len(pairs), dtype=np.float32)
    for s in range(0, len(pairs), 200_000):
        a = np.asarray(emb_q[qi[s : s + 200_000]], dtype=np.float32)
        b = np.asarray(emb_t[ti[s : s + 200_000]], dtype=np.float32)
        out[s : s + 200_000] = np.einsum("ij,ij->i", a, b)
    return out


def compute_baseline_features(
    pairs: pd.DataFrame, split_dir: Path, eval_s1_ids: list[str], legacy_nan_missing: bool = True
) -> pd.DataFrame:
    """28 baseline features exactly as the frozen E0 pipeline computes them (legacy loader + widen)."""
    from parallax.data.contracts import load_business_records_df
    from parallax.features.extractor import PairwiseFeatureExtractor
    from parallax.preprocessing.normalizer import widen_records_df

    need_t = set(pairs["cand_id"])
    s1 = load_business_records_df(split_dir / "train_source1.tsv")
    s1 = s1[s1["entity_id"].isin(set(eval_s1_ids))]
    tg = pd.concat(
        [load_business_records_df(split_dir / "train_source2.tsv"), load_business_records_df(split_dir / "train_source3.tsv")],
        ignore_index=True,
    )
    tg = tg[tg["entity_id"].isin(need_t)]
    cand_dict: dict[str, set[str]] = {}
    for s, c in zip(pairs["s1_id"], pairs["cand_id"]):
        cand_dict.setdefault(s, set()).add(c)
    s1_w, tg_w = widen_records_df(s1), widen_records_df(tg)
    if legacy_nan_missing:
        # The frozen E0 artifact was built from widened records whose missing primary number /
        # postal code were float NaN (not None). NaN is truthy and NaN != NaN, so the frozen
        # features encode "both missing" as a *conflict*. New pairs must carry the same encoding
        # to be comparable with the trusted baseline; check_feature_reproduction verifies it.
        for w in (s1_w, tg_w):
            for col in ("primary_number", "postal_code"):
                w[col] = w[col].map(lambda v: np.nan if v is None else v).astype(object)
    feats = PairwiseFeatureExtractor().extract_features_df(
        cand_dict, s1_w, tg_w, ground_truth=None, show_progress=False
    )
    feats["s1_id"] = feats["s1_id"].astype(str)
    feats["cand_id"] = feats["cand_id"].astype(str)
    if len(feats) != len(pairs):
        raise ValueError(f"Feature extraction returned {len(feats)} rows for {len(pairs)} pairs.")
    return feats[["s1_id", "cand_id", *BASELINE_28_FEATURES]]


def check_feature_reproduction(recomputed: pd.DataFrame, frozen: pd.DataFrame) -> dict[str, float]:
    m = frozen[["s1_id", "cand_id", *BASELINE_28_FEATURES]].merge(
        recomputed, on=["s1_id", "cand_id"], suffixes=("_f", "_r"), how="inner", validate="one_to_one"
    )
    if len(m) != len(frozen):
        raise ValueError(f"Recomputed features cover {len(m)} of {len(frozen)} frozen E0 rows.")
    diffs = {}
    for c in BASELINE_28_FEATURES:
        a, b = m[f"{c}_f"].to_numpy(float), m[f"{c}_r"].to_numpy(float)
        both_nan = np.isnan(a) & np.isnan(b)
        d = np.where(both_nan, 0.0, np.abs(a - b))
        diffs[c] = float(np.nanmax(d)) if len(d) else 0.0
    worst = max(diffs.values())
    if not np.isfinite(worst) or worst > 1e-5:
        raise ValueError(f"Recomputed baseline features disagree with the frozen artifact: {diffs}")
    return diffs


def _matcher_job(job: tuple[str, int, str, list[str], str]) -> tuple[str, dict[str, object]]:
    """
    One (arm, seed) matcher run in a spawned process. Inputs come from files (arm table parquet +
    a pickle of GT/folds/populations), so nothing depends on forked OpenMP/CUDA state.
    """
    import pickle

    arm, seed, rows_path, cols, ctx_path = job
    with open(ctx_path, "rb") as f:
        c = pickle.load(f)
    rows = pd.read_parquet(rows_path)
    t = time.time()
    run = run_grouped_matcher(rows, cols, c["gt_eval"], c["s1_to_fold"], seed=seed)
    cands = candidates_to_dict(rows, c["eval_ids"])
    r = evaluate_matcher_predictions(c["gt_eval"], run.predictions, c["eval_ids"], candidates=cands,
                                     s2_ids=c["s2_ids"], s3_ids=c["s3_ids"])
    return arm, {"seed": seed, "taus": run.taus, "seconds": round(time.time() - t, 1)} | r.to_dict()


# ---------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 2: supervised BGE-M3 + hard negatives (5K evaluation)")
    ap.add_argument("--split-dir", type=Path, default=PROJECT_ROOT / "data/medium_split_200k")
    ap.add_argument("--e0-candidates", type=Path, default=PROJECT_ROOT / "baseline_artifacts/candidate_pairs_sample.parquet")
    ap.add_argument("--e0-features", type=Path,
                    default=PROJECT_ROOT / "experiments/neural_text/caches/augmented_features_with_concordance_5k.parquet")
    ap.add_argument("--cv-folds", type=Path, default=PROJECT_ROOT / "data/medium_split_200k/cv_folds_source1.tsv")
    ap.add_argument("--work-dir", type=Path, default=PROJECT_ROOT / "artifacts/phase2_bge/main")
    ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "artifacts/hybrid_er")
    ap.add_argument("--model", default="BAAI/bge-m3")
    ap.add_argument("--gpu", default="auto", help="training / search device")
    ap.add_argument("--encode-gpus", type=int, nargs="*", default=None,
                    help="extra GPU indices for data-parallel corpus encoding (training stays on --gpu)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-train-s1", type=int, default=40000)
    ap.add_argument("--n-val-s1", type=int, default=2000)
    ap.add_argument("--mine-k", type=int, default=30, help="zero-shot BGE top-k per source for mining")
    ap.add_argument("--neg-per-s1", type=int, default=30)
    ap.add_argument("--val-distractors", type=int, default=50000)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--n-hard", type=int, default=7)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--temperature", type=float, default=0.02)
    ap.add_argument("--precision", default="auto")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--no-freeze-embeddings", action="store_true")
    ap.add_argument("--encode-batch-size", type=int, default=512)
    ap.add_argument("--retrieval-k", type=int, default=50)
    ap.add_argument("--index-backend", default="torch", choices=["torch", "flat", "sq8", "ivf", "ivfpq"])
    ap.add_argument("--pool-ks", type=int, nargs="+", default=[10, 20])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--matcher-workers", type=int, default=6, help="parallel (arm, seed) matcher processes")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny populations + target subsample: plumbing check only")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.smoke:
        args.n_train_s1, args.n_val_s1, args.val_distractors = 400, 100, 2000
        args.max_steps, args.eval_every = args.max_steps or 20, 10
        args.pool_ks, args.seeds = [10], [42]
        if args.work_dir == PROJECT_ROOT / "artifacts/phase2_bge/main":
            args.work_dir = PROJECT_ROOT / "artifacts/phase2_bge/smoke"
    work = args.work_dir
    work.mkdir(parents=True, exist_ok=True)
    device = select_device(args.gpu)
    enc_devices = [f"cuda:{g}" for g in args.encode_gpus] if args.encode_gpus else [device]
    if device not in enc_devices:
        enc_devices = [device, *enc_devices]
    rec = RunRecorder("phase2_bge_smoke" if args.smoke else "phase2_bge", args.out_dir,
                      config={k: str(v) for k, v in vars(args).items()}, device=device)
    rec.put("work_dir", str(work))
    rec.put("encode_devices", enc_devices)

    # ------------------------------------------------------------ 1. populations
    with rec.stage("load_split"):
        data = load_split(args.split_dir)
    rec.put("population_global", data.counts())
    feats_e0 = pd.read_parquet(args.e0_features)
    feats_e0["s1_id"] = feats_e0["s1_id"].astype(str)
    feats_e0["cand_id"] = feats_e0["cand_id"].astype(str)
    eval_ids = sorted(feats_e0["s1_id"].unique())
    gt_eval = scope_ground_truth_to_eval(data.gt, eval_ids)
    train_ids, val_ids = split_s1_groups(data.s1_ids, args.n_train_s1, args.n_val_s1, args.seed, exclude=eval_ids)
    assert not (set(train_ids) & set(val_ids)) and not ((set(train_ids) | set(val_ids)) & set(eval_ids))
    gt_tv = {s: data.gt[s] for s in train_ids + val_ids}
    eval_targets = {t for s in eval_ids for t in gt_eval[s]}
    tv_targets = {t for s in gt_tv for t in gt_tv[s]}
    if eval_targets & tv_targets:
        raise AssertionError("Train/val GT targets overlap eval GT targets.")
    rec.put("population_phase2", {
        "eval_s1": len(eval_ids), "eval_gt_pairs": sum(map(len, gt_eval.values())),
        "train_s1": len(train_ids), "train_s1_with_pos": sum(1 for s in train_ids if data.gt[s]),
        "train_pos_pairs": sum(len(data.gt[s]) for s in train_ids),
        "val_s1": len(val_ids), "val_pos_pairs": sum(len(data.gt[s]) for s in val_ids),
        "train_val_eval_s1_disjoint": True, "gt_target_overlap_eval_vs_trainval": 0,
    })

    target_subset = None
    if args.smoke:
        rng = np.random.default_rng(args.seed)
        all_t = np.array(sorted(data.s2_ids | data.s3_ids), dtype=object)
        target_subset = np.array(sorted(set(rng.choice(all_t, 30000, replace=False)) | eval_targets | tv_targets
                                        | set(feats_e0["cand_id"])), dtype=object)
    ent = build_entity_table(data, train_ids + val_ids + eval_ids, target_subset)
    pos_of = dict(zip(ent["entity_id"], range(len(ent))))
    is_t = ent["source"].isin(["S2", "S3"]).to_numpy()
    t_idx = np.flatnonzero(is_t)
    t_ids = ent["entity_id"].to_numpy(dtype=object)[t_idx]
    t_src = ent["source"].to_numpy(dtype=object)[t_idx]
    t_cty = ent["country"].to_numpy(dtype=object)[t_idx].astype(str)
    t_pos = {e: i for i, e in enumerate(t_ids)}
    rec.put("entity_table", {"rows": len(ent), "targets": int(is_t.sum()), "smoke_target_subset": target_subset is not None})

    with rec.stage("serialize"):
        from transformers import AutoTokenizer

        texts = serialize_all(ent, AutoTokenizer.from_pretrained(args.model), work)

    def q_arrays(ids: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        idx = np.array([pos_of[s] for s in ids])
        return idx, np.array(ids, dtype=object), ent["country"].to_numpy(dtype=object)[idx].astype(str)

    # ------------------------------------------------------------ 2. zero-shot encode
    zs_dir = work / "emb_zeroshot"
    zs_dir.mkdir(exist_ok=True)
    tv_idx, tv_q, tv_cty = q_arrays(train_ids + val_ids)
    ev_idx, ev_q, ev_cty = q_arrays(eval_ids)
    with rec.stage("encode_zeroshot"):
        emb_zs_t = encode_cached(args.model, texts["full"], t_idx, "full", zs_dir / "targets_full.npy", enc_devices, args.encode_batch_size)
        emb_zs_tv = encode_cached(args.model, texts["full"], tv_idx, "full", zs_dir / "trainval_full.npy", enc_devices, args.encode_batch_size)
        emb_zs_ev = encode_cached(args.model, texts["full"], ev_idx, "full", zs_dir / "eval_full.npy", enc_devices, args.encode_batch_size)
        import torch

        torch.cuda.empty_cache()
    rec.put("vram_after_zeroshot_encode_gib", peak_vram_gib(device))

    # ------------------------------------------------------------ 3. mining
    neg_path = work / "hard_negatives.parquet"
    with rec.stage("mine_hard_negatives"):
        if neg_path.exists():
            negs = pd.read_parquet(neg_path)
            mining = json.loads((work / "mining_summary.json").read_text())
        else:
            t0 = time.time()
            zs_index = DenseIndex(args.index_backend, device=device).build(emb_zs_t, t_ids, t_src, t_cty)
            bge_pool = dense_hits(zs_index, emb_zs_tv, tv_q, tv_cty, args.mine_k, "bge0_full")
            t_bge = time.time() - t0
            del zs_index
            torch.cuda.empty_cache()
            t0 = time.time()
            tv_df = ent.iloc[tv_idx]
            lex_pool = lexical_retrieve(tv_df, ent.iloc[t_idx])
            t_lex = time.time() - t0
            pool = pd.concat([bge_pool, lex_pool], ignore_index=True)
            negs, summary = mine_hard_negatives(pool, gt_tv, args.neg_per_s1, ["bge0_full", "lex_name", "lex_addr"])
            pool_recall = {}
            for name, pdf in (("bge0_full", bge_pool), ("lexical", lex_pool), ("union", pool)):
                cd = {}
                for s, c in zip(pdf["s1_id"].astype(str), pdf["cand_id"].astype(str)):
                    cd.setdefault(s, set()).add(c)
                pool_recall[name] = evaluate_candidate_recall(gt_tv, cd, list(gt_tv)).to_dict()
            mining = {"summary": summary.to_dict(), "pool_recall_trainval": pool_recall,
                      "bge_search_seconds": round(t_bge, 1), "lexical_seconds": round(t_lex, 1)}
            negs.to_parquet(neg_path)
            _save_json(work / "mining_summary.json", mining)
    rec.put("mining", mining)

    # ------------------------------------------------------------ 4. training data
    from experiments.hybrid_er.training.bge_trainer import BGETrainer, TrainConfig, TrainingData, ValidationData

    ex = [(pos_of[s], pos_of[t]) for s in train_ids for t in sorted(data.gt[s])]
    anchor_pos = {pos_of[s]: {pos_of[t] for t in data.gt[s]} for s in train_ids + val_ids}
    anchor_negs: dict[int, np.ndarray] = {}
    train_set = set(train_ids)
    for s, grp in negs.sort_values(["s1_id", "neg_order"]).groupby("s1_id", sort=False):
        if s in train_set:
            anchor_negs[pos_of[s]] = np.array([pos_of[c] for c in grp["cand_id"]], dtype=np.int64)
    tdata = TrainingData(examples=np.array(ex, dtype=np.int64), anchor_pos=anchor_pos, anchor_negs=anchor_negs, texts=texts)

    val_q = [s for s in val_ids if data.gt[s]]
    rng = np.random.default_rng(args.seed + 1)
    val_neg_ids = set(negs.loc[negs["s1_id"].isin(set(val_ids)), "cand_id"])
    val_pos_ids = {t for s in val_q for t in data.gt[s]}
    distract = set(rng.choice(t_ids, size=min(args.val_distractors, len(t_ids)), replace=False).tolist())
    corpus = np.array(sorted(val_pos_ids | val_neg_ids | distract), dtype=object)
    vq_idx = np.array([pos_of[s] for s in val_q])
    vc_idx = np.array([pos_of[c] for c in corpus])
    vdata = ValidationData(
        query_idx=vq_idx,
        query_country=ent["country"].to_numpy(dtype=object)[vq_idx].astype(str),
        query_pos=[{pos_of[t] for t in data.gt[s]} for s in val_q],
        corpus_idx=vc_idx,
        corpus_country=ent["country"].to_numpy(dtype=object)[vc_idx].astype(str),
    )
    rec.put("training_data", {"examples": len(ex), "anchors": len({a for a, _ in ex}),
                              "anchors_with_negs": len(anchor_negs), "val_queries": len(val_q), "val_corpus": len(corpus)})

    # ------------------------------------------------------------ 5. training
    cfg = TrainConfig(model_name=args.model, seed=args.seed, epochs=args.epochs, max_steps=args.max_steps,
                      batch_size=args.batch_size, grad_accum=args.grad_accum, n_hard=args.n_hard, lr=args.lr,
                      temperature=args.temperature, precision=args.precision, eval_every=args.eval_every,
                      freeze_word_embeddings=not args.no_freeze_embeddings, encode_batch_size=args.encode_batch_size)
    rec.put("train_config", vars(cfg))
    train_dir = work / "train"
    best_dir = train_dir / "checkpoints" / "best"
    with rec.stage("train_bge"):
        if (train_dir / "train_summary.json").exists() and best_dir.exists():
            tsum = json.loads((train_dir / "train_summary.json").read_text())
        else:
            trainer = BGETrainer(cfg, tdata, vdata, train_dir, device)
            tsum = trainer.train(resume=not args.no_resume)
            del trainer
            torch.cuda.empty_cache()
    rec.put("training", {k: v for k, v in tsum.items() if k != "history"})
    rec.put("training_val_curve", [h for h in tsum["history"] if h.get("type") == "val"])

    # ------------------------------------------------------------ 6. fine-tuned encode
    ft_dir = work / "emb_finetuned"
    ft_dir.mkdir(exist_ok=True)
    emb_ft_t, emb_ft_ev = {}, {}
    with rec.stage("encode_finetuned"):
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(device)
        for v in VIEWS:
            t0 = time.time()
            emb_ft_t[v] = encode_cached(best_dir, texts[v], t_idx, v, ft_dir / f"targets_{v}.npy", enc_devices, args.encode_batch_size)
            emb_ft_ev[v] = encode_cached(best_dir, texts[v], ev_idx, v, ft_dir / f"eval_{v}.npy", enc_devices, args.encode_batch_size)
            rec.record["stages"][f"encode_finetuned_{v}"] = {"seconds": round(time.time() - t0, 1)}
        torch.cuda.empty_cache()
    rec.put("vram_encode_finetuned_gib", peak_vram_gib(device))

    # ------------------------------------------------------------ 7. retrieval + candidate metrics
    e0 = load_e0_channel(args.e0_candidates, data, eval_ids)
    e0_dict = candidates_to_dict(e0, eval_ids)
    channels: dict[str, pd.DataFrame] = {}
    retrieval_stats = {}
    for name, emb_t, emb_q in [("bge0_full", emb_zs_t, emb_zs_ev)] + [(f"bgeft_{v}", emb_ft_t[v], emb_ft_ev[v]) for v in VIEWS]:
        t0 = time.time()
        index = DenseIndex(args.index_backend, device=device).build(emb_t, t_ids, t_src, t_cty)
        t1 = time.time()
        channels[name] = dense_hits(index, emb_q, ev_q, ev_cty, args.retrieval_k, name)
        retrieval_stats[name] = index.stats() | {"search_seconds": round(time.time() - t1, 2),
                                                  "build_plus_search_seconds": round(time.time() - t0, 2)}
        del index
        torch.cuda.empty_cache()
    rec.put("retrieval_index", retrieval_stats)
    for c in channels.values():
        c["s1_id"] = c["s1_id"].astype(str)

    def at_k(ch: pd.DataFrame, k: int) -> pd.DataFrame:
        return ch[ch["rank"] <= k]

    cand_reports = {"E0": evaluate_candidate_recall(gt_eval, e0_dict, eval_ids).to_dict()}
    e0_pairs = {(s, c) for s, cs in e0_dict.items() for c in cs}
    gt_pairs = {(s, t) for s, ts in gt_eval.items() for t in ts}
    for name, ch in channels.items():
        for k in (5, 10, 20, 50):
            if k > args.retrieval_k:
                continue
            sub = at_k(ch, k)
            solo = candidates_to_dict(merge_candidate_tables([sub], s2_ids=data.s2_ids, s3_ids=data.s3_ids), eval_ids)
            uni = merge_candidate_tables([e0, sub], s2_ids=data.s2_ids, s3_ids=data.s3_ids)
            ud = candidates_to_dict(uni, eval_ids)
            bge_pairs = {(s, c) for s, cs in solo.items() for c in cs}
            cand_reports[f"{name}@{k}"] = evaluate_candidate_recall(gt_eval, solo, eval_ids).to_dict() | {
                "overlap_with_e0_pairs": len(bge_pairs & e0_pairs),
                "frac_of_bge_in_e0": len(bge_pairs & e0_pairs) / max(len(bge_pairs), 1),
            }
            cand_reports[f"E0+{name}@{k}"] = evaluate_candidate_recall(gt_eval, ud, eval_ids).to_dict() | {
                "unique_true_pairs_recovered_vs_e0": len((bge_pairs - e0_pairs) & gt_pairs),
                "new_candidates_vs_e0": len(bge_pairs - e0_pairs),
                "e0_true_pairs_also_found": len(bge_pairs & e0_pairs & gt_pairs),
            }
    for k in (10, 20):
        allv = [at_k(channels[f"bgeft_{v}"], k) for v in VIEWS]
        uni = merge_candidate_tables([e0, *allv], s2_ids=data.s2_ids, s3_ids=data.s3_ids)
        rep = evaluate_candidate_recall(gt_eval, candidates_to_dict(uni, eval_ids), eval_ids).to_dict()
        view_pairs = {v: {(s, c) for s, c in zip(a["s1_id"], a["cand_id"])} for v, a in zip(VIEWS, allv)}
        rep["unique_true_pairs_by_view_vs_e0_and_other_views"] = {
            v: len((view_pairs[v] - e0_pairs - set().union(*(view_pairs[o] for o in VIEWS if o != v))) & gt_pairs)
            for v in VIEWS
        }
        cand_reports[f"E0+bgeft_allviews@{k}"] = rep
    rec.put("candidate_generation", cand_reports)
    for name in ("E0", "bge0_full@10", "bgeft_full@10", "E0+bge0_full@10", "E0+bgeft_full@10", "E0+bgeft_full@20"):
        if name in cand_reports:
            r = cand_reports[name]
            logger.info("%-22s recall=%.4f BFN=%d cands=%d mult=%.2f", name, r["candidate_recall"], r["blocking_fn"],
                        r["candidate_count"], r["multiplier"])

    # ------------------------------------------------------------ 8. matcher arms
    folds_df = pd.read_csv(args.cv_folds, sep="\t", dtype={"entity_id": str})
    s1_to_fold = dict(zip(folds_df["entity_id"], folds_df["fold"].astype(int)))
    max_k = max(args.pool_ks)
    ft_k = at_k(channels["bgeft_full"], max_k)
    full_union = merge_candidate_tables([e0, ft_k], s2_ids=data.s2_ids, s3_ids=data.s3_ids)
    pairs_all = full_union[["s1_id", "source", "cand_id"]].copy()

    with rec.stage("baseline_features"):
        feat_path = work / f"baseline28_legacy_union_k{max_k}.parquet"
        if feat_path.exists():
            base = pd.read_parquet(feat_path)
        else:
            base = compute_baseline_features(pairs_all, args.split_dir, eval_ids)
            base.to_parquet(feat_path)
    repro = check_feature_reproduction(base, feats_e0)
    rec.put("baseline_feature_reproduction_max_abs_diff", max(repro.values()))

    ev_pos = {s: i for i, s in enumerate(ev_q)}
    pairs = pairs_all.merge(base, on=["s1_id", "cand_id"], how="left", validate="one_to_one")
    pairs["bge_zs_full_cos"] = pair_cosines(pairs, emb_zs_ev, ev_pos, emb_zs_t, t_pos)
    for v in VIEWS:
        pairs[f"bge_ft_{v}_cos"] = pair_cosines(pairs, emb_ft_ev[v], ev_pos, emb_ft_t[v], t_pos)
    rank_ft = channels["bgeft_full"][["s1_id", "cand_id", "rank"]].rename(columns={"rank": "bge_ft_full_rank"})
    pairs = pairs.merge(rank_ft, on=["s1_id", "cand_id"], how="left")
    pairs["bge_ft_full_rank"] = pairs["bge_ft_full_rank"].fillna(9999).astype(float)
    pairs["found_by_e0"] = [int((s, c) in e0_pairs) for s, c in zip(pairs["s1_id"], pairs["cand_id"])]
    pairs = pairs.sort_values(["s1_id", "source", "cand_id"]).reset_index(drop=True)

    bge_ft_cols = [f"bge_ft_{v}_cos" for v in VIEWS]
    arms: dict[str, tuple[pd.DataFrame, list[str]]] = {}
    e0_rows = pairs[pairs["found_by_e0"] == 1].reset_index(drop=True)
    arms["A0_E0_28"] = (e0_rows, BASELINE_28_FEATURES)
    arms["A1zs_E0_28+bge_zs"] = (e0_rows, BASELINE_28_FEATURES + ["bge_zs_full_cos"])
    arms["A1_E0_28+bge_ft"] = (e0_rows, BASELINE_28_FEATURES + bge_ft_cols)
    for k in args.pool_ks:
        rows = pairs[(pairs["found_by_e0"] == 1) | (pairs["bge_ft_full_rank"] <= k)].reset_index(drop=True)
        arms[f"A2_E0+bgeft@{k}"] = (rows, BASELINE_28_FEATURES + bge_ft_cols + ["found_by_e0", "bge_ft_full_rank"])

    import pickle

    mdir = work / "matcher_inputs"
    mdir.mkdir(exist_ok=True)
    ctx_path = mdir / "context.pkl"
    with open(ctx_path, "wb") as f:
        pickle.dump({"gt_eval": gt_eval, "s1_to_fold": s1_to_fold, "eval_ids": eval_ids,
                     "s2_ids": data.s2_ids, "s3_ids": data.s3_ids}, f)
    jobs = []
    for arm, (rows, cols) in arms.items():
        rp = mdir / f"{arm}.parquet"
        rows[["s1_id", "source", "cand_id", *cols]].to_parquet(rp)
        jobs += [(arm, seed, str(rp), list(cols), str(ctx_path)) for seed in args.seeds]
    t_m = time.time()
    if args.matcher_workers > 1:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=min(args.matcher_workers, len(jobs)), mp_context=mp.get_context("spawn")) as ex:
            results = list(ex.map(_matcher_job, jobs))
    else:
        results = [_matcher_job(j) for j in jobs]
    rec.record["stages"]["matcher_all"] = {"seconds": round(time.time() - t_m, 1), "jobs": len(jobs),
                                           "workers": args.matcher_workers}
    matcher = {}
    for arm, (rows, cols) in arms.items():
        runs = sorted((r for a, r in results if a == arm), key=lambda r: r["seed"])
        for r in runs:
            logger.info("%s seed %d taus %s F0.5=%.4f TP=%d FP=%d CFN=%s BFN=%s", arm, r["seed"], r["taus"],
                        r["macro_f05"], r["tp"], r["fp"], r["classification_fn"], r["blocking_fn"])
        f = np.array([x["macro_f05"] for x in runs])
        matcher[arm] = {"rows": len(rows), "features": cols, "runs": runs,
                        "macro_f05_mean": float(f.mean()), "macro_f05_std": float(f.std())}
    rec.put("matcher", matcher)
    base_f = np.array([x["macro_f05"] for x in matcher["A0_E0_28"]["runs"]])
    rec.put("paired_delta_vs_A0", {
        arm: {"per_seed": (np.array([x["macro_f05"] for x in m["runs"]]) - base_f).round(5).tolist(),
              "mean": float((np.array([x["macro_f05"] for x in m["runs"]]) - base_f).mean())}
        for arm, m in matcher.items()
    })
    path = rec.save()
    logger.info("Run record written to %s", path)
    for arm, m in matcher.items():
        logger.info("%-26s Macro F0.5 = %.4f ± %.4f", arm, m["macro_f05_mean"], m["macro_f05_std"])


if __name__ == "__main__":
    main()
