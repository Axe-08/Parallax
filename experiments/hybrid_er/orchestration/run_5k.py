"""
5K hybrid ER validation run (P0 contract smoke + candidate ablations + E0 matcher).

    global GT (full split)  -> global contract validation
                            -> scope to the 5K evaluation S1s -> GT_eval
    candidate channels      -> strict union boundary (schema, source, population)
                            -> candidate-level metrics only (recall, BFN, cost)
    E0 LightGBM matcher     -> deterministic grouped outer/inner-OOF protocol
                            -> predictions ⊆ E0 candidates -> trusted Macro F0.5

Candidate pools are never scored as if they were matcher predictions.

Usage (from the project root):
    python experiments/hybrid_er/orchestration/run_5k.py
    python experiments/hybrid_er/orchestration/run_5k.py --with-dense --gpu auto   # DGX only
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import logging
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
for p in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from experiments.hybrid_er.core.data import SplitData, load_split  # noqa: E402
from experiments.hybrid_er.core.runtime import RunRecorder, select_device  # noqa: E402
from experiments.hybrid_er.core.validation import (  # noqa: E402
    assign_source_by_membership,
    scope_ground_truth_to_eval,
    validate_evaluation_population,
)
from experiments.hybrid_er.evaluation.matcher_harness import (  # noqa: E402
    BASELINE_LGBM_PARAMS,
    BASELINE_NUM_BOOST_ROUND,
    BASELINE_THRESHOLD_GRID,
    label_pairs,
    run_grouped_matcher,
)
from experiments.hybrid_er.evaluation.metrics import (  # noqa: E402
    evaluate_candidate_recall,
    evaluate_matcher_predictions,
)
from experiments.hybrid_er.retrieval.relational import build_s2_s3_graph, run_relational_expansion  # noqa: E402
from experiments.hybrid_er.retrieval.structural import run_structural_retrieval  # noqa: E402
from experiments.hybrid_er.retrieval.union import candidates_to_dict, merge_candidate_tables  # noqa: E402

logger = logging.getLogger("run_5k")

# Frozen baseline feature list (tag parallax-experimental-baseline-v1).
BASELINE_28_FEATURES: list[str] = [
    "raw_name_ratio", "soft_name_ratio", "token_sort_ratio", "token_set_ratio", "partial_ratio",
    "addr_token_set_ratio", "addr_ratio", "num_match_score", "is_s1_addr_null", "is_cand_addr_null",
    "both_addr_present", "len_diff_name", "len_ratio_name", "primary_num_match", "primary_num_conflict",
    "primary_num_missing", "num_jaccard", "num_conflict_count", "postal_match", "postal_conflict",
    "postal_missing", "jaro_winkler_soft", "jaro_winkler_raw", "token_jaccard_name", "token_overlap_name",
    "first_token_match", "canon_addr_ratio", "token_jaccard_addr",
]


def load_e0_channel(path: Path, data: SplitData, eval_s1_ids: list[str]) -> pd.DataFrame:
    """E0 candidate pool with authoritative (membership-derived) source. E0 rank/score are not
    stored in the frozen artifact, so they are left missing rather than fabricated."""
    e0 = pd.read_parquet(path)[["s1_id", "cand_id"]]
    validate_evaluation_population(e0, eval_s1_ids)
    e0 = assign_source_by_membership(e0, data.s2_ids, data.s3_ids)
    e0["s1_id"] = e0["s1_id"].astype(str)
    e0["blocker"] = "e0"
    e0["rank"] = np.nan
    e0["score"] = np.nan
    return e0


def run_dense_channel(s1_df: pd.DataFrame, data: SplitData, device: str, batch_size: int, k: int) -> pd.DataFrame:
    """Zero-shot BGE-M3 (recall-only reference; the supervised retriever is Phase 2)."""
    from experiments.hybrid_er.core.serialization import serialize_full
    from experiments.hybrid_er.retrieval.neural_bge import DenseRetriever, FaissIndexManager

    retriever = DenseRetriever(device=device, batch_size=batch_size)
    s1_embs = retriever.encode([serialize_full(r) for r in s1_df.to_dict("records")])
    rows = []
    s1_ids = s1_df["entity_id"].tolist()
    for source, tgt_df in (("S2", data.s2), ("S3", data.s3)):
        index = FaissIndexManager(1024, "FlatIP")
        index.add(retriever.encode([serialize_full(r) for r in tgt_df.to_dict("records")]), tgt_df["entity_id"].tolist())
        dist, _, ents = index.search(s1_embs, k=k)
        for i, s1 in enumerate(s1_ids):
            for rank, (score, cand) in enumerate(zip(dist[i], ents[i]), start=1):
                if cand is not None:
                    rows.append((s1, source, cand, "bge_full_zeroshot", rank, float(score)))
    return pd.DataFrame(rows, columns=["s1_id", "source", "cand_id", "blocker", "rank", "score"])


def contract_negative_checks(e0: pd.DataFrame, data: SplitData, gt_eval: dict[str, set[str]], eval_ids: list[str]) -> dict[str, bool]:
    """Proves the contracts fail loudly on corrupted inputs (each must raise ValueError)."""
    s3_example = next(iter(sorted(data.s3_ids)))
    s1_example = next(iter(sorted(data.s1_ids - set(eval_ids))))
    cases = {}

    bad = e0.head(3).copy()
    bad.loc[bad.index[0], "cand_id"] = s3_example  # S3 id labeled S2 (or S2 id relabeled)
    bad.loc[bad.index[0], "source"] = "S2"
    cases["mislabeled_source_rejected_at_union"] = lambda: merge_candidate_tables([bad], s2_ids=data.s2_ids, s3_ids=data.s3_ids)

    unk = e0.head(3).copy()
    unk.loc[unk.index[0], "cand_id"] = "S9-000"
    cases["unknown_candidate_id_rejected"] = lambda: assign_source_by_membership(unk, data.s2_ids, data.s3_ids)

    dup = pd.concat([e0.head(2), e0.head(1)])
    cases["duplicate_within_blocker_rejected"] = lambda: merge_candidate_tables([dup], s2_ids=data.s2_ids, s3_ids=data.s3_ids)

    cases["extra_s1_in_predictions_rejected"] = lambda: evaluate_matcher_predictions(gt_eval, {s1_example: set()}, eval_ids)

    cands = candidates_to_dict(e0, eval_ids)
    s1_first = eval_ids[0]
    cases["prediction_outside_candidates_rejected"] = lambda: evaluate_matcher_predictions(
        gt_eval, {s1_first: {s3_example}}, eval_ids, candidates=cands
    )

    cases["eval_s1_missing_from_gt_rejected"] = lambda: scope_ground_truth_to_eval({}, eval_ids[:1])

    results = {}
    for name, fn in cases.items():
        try:
            fn()
            results[name] = False
        except ValueError:
            results[name] = True
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="5K Hybrid ER validation run")
    parser.add_argument("--split-dir", type=Path, default=PROJECT_ROOT / "data/medium_split_200k")
    parser.add_argument("--e0-candidates", type=Path, default=PROJECT_ROOT / "baseline_artifacts/candidate_pairs_sample.parquet")
    parser.add_argument(
        "--e0-features",
        type=Path,
        default=PROJECT_ROOT / "experiments/neural_text/caches/augmented_features_with_concordance_5k.parquet",
    )
    parser.add_argument("--cv-folds", type=Path, default=PROJECT_ROOT / "data/medium_split_200k/cv_folds_source1.tsv")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42], help="Inner-split/LightGBM seeds for the matcher")
    parser.add_argument("--with-dense", action="store_true", help="Add zero-shot BGE-M3 channel (GPU)")
    parser.add_argument("--dense-k", type=int, default=50)
    parser.add_argument("--gpu", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "artifacts/hybrid_er")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    device = select_device(args.gpu) if args.with_dense else "cpu"
    rec = RunRecorder(
        "run_5k",
        args.out_dir,
        config={k: str(v) for k, v in vars(args).items()}
        | {"lgbm_params": BASELINE_LGBM_PARAMS, "num_boost_round": BASELINE_NUM_BOOST_ROUND,
           "threshold_grid": list(BASELINE_THRESHOLD_GRID), "features": BASELINE_28_FEATURES},
        device=device,
    )

    # 1. Global population + global GT contract.
    with rec.stage("load_split"):
        data = load_split(args.split_dir)
    rec.put("population_global", data.counts())

    # 2. Evaluation population and GT scoping.
    feats = pd.read_parquet(args.e0_features)
    eval_s1_ids = sorted(feats["s1_id"].astype(str).unique())
    not_in_split = set(eval_s1_ids) - data.s1_ids
    if not_in_split:
        raise ValueError(f"{len(not_in_split)} evaluation S1s are not in the split's S1 population.")
    gt_eval = scope_ground_truth_to_eval(data.gt, eval_s1_ids)
    s1_eval_df = data.s1[data.s1["entity_id"].isin(set(eval_s1_ids))].reset_index(drop=True)
    validate_evaluation_population(s1_eval_df, eval_s1_ids)
    rec.put(
        "population_eval",
        {"s1": len(eval_s1_ids), "gt_pairs": sum(len(v) for v in gt_eval.values()),
         "zero_match_s1": sum(1 for v in gt_eval.values() if not v)},
    )

    # 3. Candidate channels (each validated at the union boundary).
    channels: dict[str, pd.DataFrame] = {}
    with rec.stage("channel_e0"):
        channels["E0"] = load_e0_channel(args.e0_candidates, data, eval_s1_ids)
    with rec.stage("channel_f_structural"):
        channels["F"] = run_structural_retrieval(s1_eval_df, data.s2, data.s3)
    with rec.stage("channel_e_relational"):
        graph = build_s2_s3_graph(data.s2, data.s3)
        seeds = merge_candidate_tables([channels["E0"], channels["F"]], s2_ids=data.s2_ids, s3_ids=data.s3_ids)
        channels["E"] = run_relational_expansion(seeds[["s1_id", "source", "cand_id"]], graph)
    if args.with_dense:
        with rec.stage("channel_bge_zeroshot"):
            channels["BGE0"] = run_dense_channel(s1_eval_df, data, device, args.batch_size, args.dense_k)

    # 4. Candidate-level ablations (E0 always included).
    candidate_reports = {}
    extras = [c for c in channels if c != "E0"]
    for r in range(len(extras) + 1):
        for combo in combinations(extras, r):
            name = "+".join(("E0", *combo))
            union = merge_candidate_tables([channels[c] for c in ("E0", *combo)], s2_ids=data.s2_ids, s3_ids=data.s3_ids)
            report = evaluate_candidate_recall(gt_eval, candidates_to_dict(union, eval_s1_ids), eval_s1_ids)
            candidate_reports[name] = report.to_dict()
            logger.info("--- %s ---\n%s", name, report)
    for c in extras:
        solo = merge_candidate_tables([channels[c]], s2_ids=data.s2_ids, s3_ids=data.s3_ids)
        candidate_reports[f"{c}_only"] = evaluate_candidate_recall(
            gt_eval, candidates_to_dict(solo, eval_s1_ids), eval_s1_ids
        ).to_dict()
    rec.put("candidate_generation", candidate_reports)

    # 5. E0 matcher (real predictions, never the candidate pool).
    e0_cands = candidates_to_dict(channels["E0"], eval_s1_ids)
    feat_pairs = set(zip(feats["s1_id"].astype(str), feats["cand_id"].astype(str)))
    pool_pairs = {(s, c) for s, cs in e0_cands.items() for c in cs}
    if feat_pairs != pool_pairs:
        raise ValueError("E0 feature rows do not match the E0 candidate pool exactly.")
    if "target" in feats.columns and not np.array_equal(label_pairs(feats, gt_eval), feats["target"].astype(np.int8).to_numpy()):
        raise ValueError("Stored 'target' column disagrees with GT-derived labels.")

    folds_df = pd.read_csv(args.cv_folds, sep="\t", dtype={"entity_id": str})
    s1_to_fold = dict(zip(folds_df["entity_id"], folds_df["fold"].astype(int)))

    matcher_runs = []
    for seed in args.seeds:
        with rec.stage(f"e0_matcher_seed{seed}"):
            run = run_grouped_matcher(feats, BASELINE_28_FEATURES, gt_eval, s1_to_fold, seed=seed)
        report = evaluate_matcher_predictions(
            gt_eval, run.predictions, eval_s1_ids, candidates=e0_cands, s2_ids=data.s2_ids, s3_ids=data.s3_ids
        )
        logger.info("--- E0 LightGBM matcher (seed %d, taus %s) ---\n%s", seed, run.taus, report)
        matcher_runs.append({"seed": seed, "taus": run.taus, "folds": [vars(f) for f in run.folds]} | report.to_dict())
    f05 = [m["macro_f05"] for m in matcher_runs]
    rec.put(
        "matcher_e0",
        {"runs": matcher_runs, "macro_f05_mean": float(np.mean(f05)), "macro_f05_std": float(np.std(f05)),
         "macro_f05_min": float(np.min(f05)), "macro_f05_max": float(np.max(f05))},
    )

    # 6. Contract negative controls.
    checks = contract_negative_checks(channels["E0"], data, gt_eval, eval_s1_ids)
    rec.put("contract_negative_checks", checks)
    if not all(checks.values()):
        raise AssertionError(f"Contract negative checks failed: {checks}")

    path = rec.save()
    logger.info("Run record written to %s", path)


if __name__ == "__main__":
    main()
