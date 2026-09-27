"""
Parallax Neural Representation Experiment Runner (E0 - E3)
==========================================================
Executes the strictly controlled 5-fold cross-validation harness across:
  E0: 28 Baseline Features
  E1: 28 + IndicXlit Name Similarity
  E2: 28 + Qwen Name Cosine
  E3: 28 + IndicXlit + Qwen
Produces comprehensive metric comparison, error breakdowns, and diagnostic reports.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

# Add src and local scripts to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from build_pairwise_features import BASELINE_28_FEATURES  # noqa: E402

from parallax.data.contracts import load_ground_truth_dict  # noqa: E402
from parallax.metrics.evaluator import evaluate_resolution_predictions  # noqa: E402
from parallax.postprocessing.singleton_gate import SingletonGatedPredictor  # noqa: E402

CONCAT_FEATURES = [
    "concat_stem_similarity",
    "concat_addr_product",
    "concat_postal_product",
]

DOMAIN_FEATURES = [
    "domain_stem_similarity",
    "domain_addr_concordance",
    "domain_postal_concordance",
    "has_domain_target",
]

NEURAL_FEATURES = [
    "indicxlit_name_similarity",
    "has_indicxlit_name",
    "qwen_name_cosine",
]

EXPERIMENT_FEATURE_SETS: dict[str, list[str]] = {
    "E0": BASELINE_28_FEATURES,
    "E1": BASELINE_28_FEATURES + ["indicxlit_name_similarity", "has_indicxlit_name"],
    "E2": BASELINE_28_FEATURES + ["qwen_name_cosine"],
    "E3": BASELINE_28_FEATURES + NEURAL_FEATURES,
    "D1": BASELINE_28_FEATURES + CONCAT_FEATURES,
    "D2": BASELINE_28_FEATURES + DOMAIN_FEATURES,
    "D3": BASELINE_28_FEATURES + CONCAT_FEATURES + DOMAIN_FEATURES + ["has_handle_target"],
    "E4": BASELINE_28_FEATURES
    + NEURAL_FEATURES
    + CONCAT_FEATURES
    + DOMAIN_FEATURES
    + ["has_handle_target"],
}


def verify_feature_integrity() -> None:
    """Rigorous feature integrity assertions and printing for all experimental arms."""
    expected_counts = {
        "E0": 28,
        "D1": 31,
        "D2": 32,
        "D3": 36,
        "E4": 39,
    }

    print("\n========================================================")
    print("FEATURE INTEGRITY VERIFICATION ACROSS ARMS")
    print("========================================================")

    for arm, expected in expected_counts.items():
        actual = len(EXPERIMENT_FEATURE_SETS[arm])
        assert actual == expected, (
            f"Arm {arm} feature count mismatch: expected {expected}, got {actual}!"
        )
        print(f"[{arm}] Total Features: {actual} (matches expected {expected})")
        print(f"      Feature List ({actual}):\n      {EXPERIMENT_FEATURE_SETS[arm]}\n")

    # Assert every E4 feature is strictly from baseline 28, E3 neural, or D3 deterministic
    e4_features = EXPERIMENT_FEATURE_SETS["E4"]
    e4_set = set(e4_features)
    valid_allowed = (
        set(BASELINE_28_FEATURES) | set(NEURAL_FEATURES) | set(EXPERIMENT_FEATURE_SETS["D3"])
    )
    unallowed = e4_set - valid_allowed
    assert len(unallowed) == 0, f"Unallowed features detected in E4: {unallowed}"
    assert len(e4_set) == 39, f"Duplicate features detected in E4: {len(e4_set)} unique vs 39 total"

    # Assert D3 contains all D1 and D2 features plus handle
    d3_set = set(EXPERIMENT_FEATURE_SETS["D3"])
    assert set(EXPERIMENT_FEATURE_SETS["D1"]).issubset(d3_set), "D1 features missing from D3!"
    assert set(EXPERIMENT_FEATURE_SETS["D2"]).issubset(d3_set), "D2 features missing from D3!"
    assert "has_handle_target" in d3_set, "has_handle_target missing from D3!"

    print("All feature integrity assertions PASSED successfully!")
    print(
        "E4 is strictly composed of baseline 28, the 3 tested E3 neural features, and D3 deterministic features."
    )
    print("========================================================\n")


@dataclass
class FoldResult:
    fold: int
    optimal_tau: float
    macro_f05: float
    precision: float
    recall: float
    singleton_acc: float
    non_singleton_f05: float
    train_time: float
    infer_time: float


@dataclass
class ExperimentResult:
    arm: str
    description: str
    feature_count: int
    features: list[str]
    macro_f05_mean: float
    macro_f05_std: float
    precision_mean: float
    recall_mean: float
    singleton_acc_mean: float
    non_singleton_f05_mean: float
    optimal_tau_mean: float
    total_tps: int
    classification_fns: int
    blocking_fns: int
    false_merges: int
    singleton_violations: int
    cross_script_tps_recovered: int
    new_false_merges_vs_e0: int
    recovered_fns_vs_e0: int
    new_singleton_violations_vs_e0: int
    recovered_domain_fns: int
    recovered_concat_fns: int
    recovered_handle_fns: int
    total_train_time: float
    total_infer_time: float
    fold_results: list[FoldResult]


def train_and_eval_fold(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    val_gt: dict[str, set[str]],
    val_s1_ids: list[str],
    seed: int = 42,
    threshold_grid: tuple[float, ...] = (0.60, 0.65, 0.70, 0.74, 0.78, 0.82, 0.86, 0.90),
) -> tuple[FoldResult, np.ndarray, dict[str, set[str]]]:
    """Train LightGBM on fold, locate optimal tau, and evaluate out-of-fold."""
    x_train = train_df[feature_cols]
    y_train = train_df["target"].astype(int)
    x_val = val_df[feature_cols]
    y_val = val_df["target"].astype(int)

    train_data = lgb.Dataset(x_train, label=y_train)
    val_data = lgb.Dataset(x_val, label=y_val, reference=train_data)

    params: dict[str, Any] = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.08,
        "num_leaves": 31,
        "max_depth": 6,
        "verbose": -1,
        "seed": seed,
        "n_jobs": 8,
    }

    t0_train = time.time()
    booster = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        valid_sets=[val_data],
    )
    train_time = time.time() - t0_train

    t0_infer = time.time()
    probs = booster.predict(x_val)
    infer_time = time.time() - t0_infer

    # Threshold optimization on validation set
    predictor = SingletonGatedPredictor()
    val_eval_df = val_df[["s1_id", "cand_id"]].copy()
    val_eval_df["prob"] = probs

    best_tau = 0.74
    best_score = -1.0
    best_preds: dict[str, set[str]] = {}

    for tau in threshold_grid:
        preds = predictor.filter_predictions(val_eval_df, val_s1_ids, threshold=tau)
        rep = evaluate_resolution_predictions(val_gt, preds)
        if rep.macro_f05 > best_score:
            best_score = rep.macro_f05
            best_tau = tau
            best_preds = preds

    final_report = evaluate_resolution_predictions(val_gt, best_preds)
    prec = (
        final_report.total_correct_pairs / final_report.total_predicted_pairs
        if final_report.total_predicted_pairs > 0
        else 0.0
    )
    rec = (
        final_report.total_correct_pairs / final_report.total_true_pairs
        if final_report.total_true_pairs > 0
        else 0.0
    )

    result = FoldResult(
        fold=-1,
        optimal_tau=best_tau,
        macro_f05=final_report.macro_f05,
        precision=prec,
        recall=rec,
        singleton_acc=final_report.singleton_score,
        non_singleton_f05=final_report.non_singleton_f05,
        train_time=train_time,
        infer_time=infer_time,
    )
    return result, probs, best_preds


def run_experiment_arm(
    arm: str,
    augmented_df: pd.DataFrame,
    cv_folds_df: pd.DataFrame,
    ground_truth: dict[str, set[str]],
    e0_oof_preds: dict[str, set[str]] | None = None,
) -> tuple[ExperimentResult, dict[str, set[str]]]:
    """Run full 5-fold cross-validation for a given experimental arm."""
    feature_cols = EXPERIMENT_FEATURE_SETS[arm]
    print("\n========================================================")
    print(
        f"RUNNING ARM {arm}: {len(feature_cols)} features ({', '.join(feature_cols[-2:] if len(feature_cols) > 28 else ['Baseline'])})"
    )
    print("========================================================")

    # Merge fold assignments into features dataframe
    entity_col = "entity_id" if "entity_id" in cv_folds_df.columns else "s1_id"
    s1_to_fold = dict(zip(cv_folds_df[entity_col].astype(str), cv_folds_df["fold"], strict=False))
    s1_fold_arr = np.array(
        [s1_to_fold.get(str(s), -1) for s in augmented_df["s1_id"]], dtype=np.int32
    )

    fold_results: list[FoldResult] = []
    all_oof_preds: dict[str, set[str]] = {}

    for k in range(5):
        val_mask = s1_fold_arr == k
        train_mask = (s1_fold_arr != k) & (s1_fold_arr != -1)

        train_sub = augmented_df[train_mask]
        val_sub = augmented_df[val_mask].copy()

        val_s1_set = set(cv_folds_df[cv_folds_df["fold"] == k][entity_col].astype(str))
        val_gt = {s1: ground_truth.get(s1, set()) for s1 in val_s1_set}

        res, _, fold_preds = train_and_eval_fold(
            train_sub, val_sub, feature_cols, val_gt, list(val_s1_set), seed=42 + k
        )
        res.fold = k
        fold_results.append(res)
        all_oof_preds.update(fold_preds)

        print(
            f"Fold {k}: F0.5={res.macro_f05:.4f} | Prec={res.precision * 100:.2f}% | "
            f"Rec={res.recall * 100:.2f}% | SingAcc={res.singleton_acc * 100:.2f}% | tau={res.optimal_tau:.2f}"
        )

    # Overall OOF Evaluation
    full_report = evaluate_resolution_predictions(ground_truth, all_oof_preds)

    f05_vals = [r.macro_f05 for r in fold_results]
    prec_vals = [r.precision for r in fold_results]
    rec_vals = [r.recall for r in fold_results]
    sing_vals = [r.singleton_acc for r in fold_results]
    ns_vals = [r.non_singleton_f05 for r in fold_results]
    tau_vals = [r.optimal_tau for r in fold_results]

    # Error accounting
    total_true_in_data = augmented_df["target"].sum()
    total_ground_truth_pairs = sum(len(v) for v in ground_truth.values())
    total_tps = full_report.total_correct_pairs
    classification_fns = total_true_in_data - total_tps
    blocking_fns = total_ground_truth_pairs - total_true_in_data
    false_merges = full_report.total_predicted_pairs - total_tps
    singleton_violations = sum(
        1
        for s1, gt_set in ground_truth.items()
        if len(gt_set) == 0 and len(all_oof_preds.get(s1, set())) > 0
    )

    # Cross-script true pair recovery (candidates with Indic characters)
    # Check predictions for cross-script pairs using the boolean indicator
    has_indic_col = "has_indicxlit_name" if "has_indicxlit_name" in augmented_df.columns else None
    if has_indic_col:
        cs_mask = (augmented_df["target"] == 1) & (augmented_df[has_indic_col] == 1.0)
    else:
        cs_mask = (augmented_df["target"] == 1) & (
            augmented_df["indicxlit_name_similarity"].notna()
        )
    cs_pairs = augmented_df[cs_mask][["s1_id", "cand_id"]]
    cs_recovered = 0
    for s1, cand in zip(cs_pairs["s1_id"], cs_pairs["cand_id"], strict=False):
        if str(cand) in all_oof_preds.get(str(s1), set()):
            cs_recovered += 1

    # Comparative deltas vs E0
    new_fm = 0
    recovered_fn = 0
    new_sing_viol = 0
    recov_domain = 0
    recov_concat = 0
    recov_handle = 0

    if e0_oof_preds is not None:
        recov_pair_keys: set[tuple[str, str]] = set()
        for s1, preds in all_oof_preds.items():
            e0_preds = e0_oof_preds.get(s1, set())
            gt_set = ground_truth.get(s1, set())
            # New false merges: predicted in arm, not in e0, and not in gt
            new_fm += len((preds - e0_preds) - gt_set)
            # Recovered FNs: predicted in arm, in gt, but was not in e0
            rec_set = (preds & gt_set) - e0_preds
            recovered_fn += len(rec_set)
            for cand in rec_set:
                recov_pair_keys.add((str(s1), str(cand)))

            # New singleton violations vs E0
            is_e0_viol = len(gt_set) == 0 and len(e0_preds) > 0
            is_arm_viol = len(gt_set) == 0 and len(preds) > 0
            if is_arm_viol and not is_e0_viol:
                new_sing_viol += 1

        # Attribution analysis on recovered positive pairs
        if recov_pair_keys:
            s1_arr = augmented_df["s1_id"].astype(str).tolist()
            cand_arr = augmented_df["cand_id"].astype(str).tolist()
            sub_mask = [(s, c) in recov_pair_keys for s, c in zip(s1_arr, cand_arr, strict=False)]
            recov_df = augmented_df[sub_mask]

            if "has_domain_target" in recov_df.columns:
                dom_sim_col = (
                    recov_df["domain_stem_similarity"]
                    if "domain_stem_similarity" in recov_df.columns
                    else 0.0
                )
                recov_domain = int(
                    ((recov_df["has_domain_target"] == 1.0) & (dom_sim_col >= 0.70)).sum()
                )
            if "concat_stem_similarity" in recov_df.columns:
                raw_ratio_col = (
                    recov_df["raw_name_ratio"] if "raw_name_ratio" in recov_df.columns else 1.0
                )
                recov_concat = int(
                    ((recov_df["concat_stem_similarity"] >= 0.75) & (raw_ratio_col < 0.65)).sum()
                )
            if "has_handle_target" in recov_df.columns:
                recov_handle = int((recov_df["has_handle_target"] == 1.0).sum())

    exp_result = ExperimentResult(
        arm=arm,
        description=f"Arm {arm} ({len(feature_cols)} features)",
        feature_count=len(feature_cols),
        features=feature_cols,
        macro_f05_mean=float(np.mean(f05_vals)),
        macro_f05_std=float(np.std(f05_vals)),
        precision_mean=float(np.mean(prec_vals)),
        recall_mean=float(np.mean(rec_vals)),
        singleton_acc_mean=float(np.mean(sing_vals)),
        non_singleton_f05_mean=float(np.mean(ns_vals)),
        optimal_tau_mean=float(np.mean(tau_vals)),
        total_tps=int(total_tps),
        classification_fns=int(classification_fns),
        blocking_fns=int(blocking_fns),
        false_merges=int(false_merges),
        singleton_violations=int(singleton_violations),
        cross_script_tps_recovered=int(cs_recovered),
        new_false_merges_vs_e0=int(new_fm),
        recovered_fns_vs_e0=int(recovered_fn),
        new_singleton_violations_vs_e0=int(new_sing_viol),
        recovered_domain_fns=int(recov_domain),
        recovered_concat_fns=int(recov_concat),
        recovered_handle_fns=int(recov_handle),
        total_train_time=float(sum(r.train_time for r in fold_results)),
        total_infer_time=float(sum(r.infer_time for r in fold_results)),
        fold_results=fold_results,
    )

    print(
        f"\n>>> ARM {arm} MEAN: F0.5 = {exp_result.macro_f05_mean:.4f} ± {exp_result.macro_f05_std:.4f} | "
        f"Prec = {exp_result.precision_mean * 100:.2f}% | Rec = {exp_result.recall_mean * 100:.2f}% | "
        f"SingAcc = {exp_result.singleton_acc_mean * 100:.2f}%"
    )
    return exp_result, all_oof_preds


def generate_experiment_report(
    results: list[ExperimentResult],
    augmented_df: pd.DataFrame,
    report_path: Path,
) -> None:
    """Generate comprehensive markdown report comparing experimental arms against baseline."""
    report_path.parent.mkdir(parents=True, exist_ok=True)

    e0 = results[0]

    md: list[str] = [
        "# Parallax Representation Experiment Report",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "**Validation:** 5-Fold Stratified CV on Source 1 entities",
        "",
        "## 1. Executive Summary & Primary Metric Comparison",
        "",
        "| Arm | Features | Macro F0.5 (mean ± std) | Precision | Recall | Singleton Acc | Class. FNs | Blocking FNs | False Merges | Sing. Violations |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    for r in results:
        md.append(
            f"| **{r.arm}** | {r.feature_count} | **{r.macro_f05_mean:.4f} ± {r.macro_f05_std:.4f}** | "
            f"{r.precision_mean * 100:.2f}% | {r.recall_mean * 100:.2f}% | {r.singleton_acc_mean * 100:.2f}% | "
            f"{r.classification_fns:,} | {r.blocking_fns:,} | {r.false_merges:,} | {r.singleton_violations:,} |"
        )

    md.extend(
        [
            "",
            "## 2. Direct Comparison Against Baseline E0",
            "",
            "| Arm | Δ Macro F0.5 | Δ Precision | Δ Recall | Δ Sing. Acc | Class. FNs Recov. | Block. FNs Recov. | Addl False Merges | Addl Sing. Violations |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
    )

    for r in results:
        if r.arm == "E0":
            md.append(
                "| **E0** | *BASELINE* | *BASELINE* | *BASELINE* | *BASELINE* | *BASELINE* | *BASELINE* | *BASELINE* | *BASELINE* |"
            )
        else:
            delta_f05 = f"{r.macro_f05_mean - e0.macro_f05_mean:+.4f}"
            delta_prec = f"{(r.precision_mean - e0.precision_mean) * 100:+.2f}%"
            delta_rec = f"{(r.recall_mean - e0.recall_mean) * 100:+.2f}%"
            delta_sing = f"{(r.singleton_acc_mean - e0.singleton_acc_mean) * 100:+.2f}%"
            md.append(
                f"| **{r.arm}** | **{delta_f05}** | {delta_prec} | {delta_rec} | {delta_sing} | "
                f"+{r.recovered_fns_vs_e0} | 0 (Frozen Blocker) | {r.new_false_merges_vs_e0} | {r.new_singleton_violations_vs_e0} |"
            )

    md.extend(
        [
            "",
            "## 3. Error Recovery Attribution Breakdown",
            "",
            "| Arm | Total FNs Recovered | Attributable to Domain Stem | Attributable to Concat Stem | Attributable to Handle | Other/Drift |",
            "|---|---|---|---|---|---|",
        ]
    )

    for r in results:
        if r.arm == "E0":
            continue
        other = max(
            0,
            r.recovered_fns_vs_e0
            - (r.recovered_domain_fns + r.recovered_concat_fns + r.recovered_handle_fns),
        )
        md.append(
            f"| **{r.arm}** | {r.recovered_fns_vs_e0} | {r.recovered_domain_fns} | {r.recovered_concat_fns} | {r.recovered_handle_fns} | {other} |"
        )

    arms = [r.arm for r in results]
    header_f05 = " | ".join(f"{a} F0.5" for a in arms)
    header_tau = " | ".join(f"{a} tau" for a in arms)
    md.extend(
        [
            "",
            "## 4. Fold-by-Fold Stability",
            "",
            f"| Fold | {header_f05} | {header_tau} |",
            f"|{'---|' * (1 + 2 * len(arms))}",
        ]
    )

    for k in range(5):
        f0_vals = " | ".join(f"{r.fold_results[k].macro_f05:.4f}" for r in results)
        t_vals = " | ".join(f"{r.fold_results[k].optimal_tau:.2f}" for r in results)
        md.append(f"| {k} | {f0_vals} | {t_vals} |")

    md.extend(
        [
            "",
            "## 5. Multi-Metric Decision Gate Evaluation",
            "",
        ]
    )

    # Multi-metric promotion evaluation (no arbitrary CFN cutoffs)
    if len(results) > 1:
        best_arm = max(results[1:], key=lambda x: x.macro_f05_mean)
        delta_best = best_arm.macro_f05_mean - e0.macro_f05_mean
        delta_fm = best_arm.new_false_merges_vs_e0
        delta_cfn = e0.classification_fns - best_arm.classification_fns

        if (
            delta_best >= 0.0020
            and best_arm.singleton_acc_mean >= (e0.singleton_acc_mean - 0.005)
            and delta_fm <= 15
        ):
            md.append(
                f"**STATUS: ACCEPT / PROMOTE {best_arm.arm}**\n\n"
                f"Arm {best_arm.arm} achieved Macro F0.5 gain of **{delta_best:+.4f}** (clearing +0.0020 bar), "
                f"recovered {delta_cfn} classification FNs with only {delta_fm} new false merges, "
                f"while preserving singleton accuracy at {best_arm.singleton_acc_mean * 100:.2f}%."
            )
        elif delta_best > 0:
            md.append(
                f"**STATUS: MARGINAL / INCONCLUSIVE**\n\n"
                f"Arm {best_arm.arm} gained {delta_best:+.4f}, within fold noise (std={best_arm.macro_f05_std:.4f}). "
                f"Requires further validation before promotion."
            )
        else:
            md.append(
                f"**STATUS: REJECT**\n\n"
                f"No evaluated arm beat the baseline Macro F0.5 of {e0.macro_f05_mean:.4f}."
            )
    else:
        md.append("**STATUS: BASELINE EVALUATION ONLY**")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print(f"\nExperiment report successfully written to: {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Parallax Representation Experiments (E0, D1-D3, E4)"
    )
    parser.add_argument(
        "--augmented-features",
        type=Path,
        default=Path(
            "experiments/neural_text/caches/augmented_features_with_concordance_5k.parquet"
        ),
    )
    parser.add_argument(
        "--cv-folds", type=Path, default=Path("data/medium_split_200k/cv_folds_source1.tsv")
    )
    parser.add_argument(
        "--ground-truth", type=Path, default=Path("data/medium_split_200k/train_ground_truth.tsv")
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=Path("experiments/neural_text/reports/experiment_ladder_report.md"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("experiments/neural_text/results/experiment_ladder_metrics.json"),
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        default=["E0", "D1", "D2", "D3"],
        help="Experimental arms to run",
    )
    parser.add_argument(
        "--run-e4-if-promoted",
        action="store_true",
        help="Conditionally run E4 only if D3 demonstrates meaningful deterministic lift over E0",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Perform feature integrity verification and print complete lists without running CV",
    )
    args = parser.parse_args()

    # Feature integrity assertions and complete list printing
    verify_feature_integrity()
    if args.verify_only:
        print("Feature verification completed successfully (--verify-only specified). Exiting.")
        return

    print(f"Loading augmented features from: {args.augmented_features}")
    augmented_df = pd.read_parquet(args.augmented_features)

    print(f"Loading CV folds from: {args.cv_folds}")
    cv_folds_df = pd.read_csv(args.cv_folds, sep="\t")

    print(f"Loading ground truth from: {args.ground_truth}")
    gt_dict = load_ground_truth_dict(args.ground_truth)

    # Schema normalization and strict fold verification
    entity_col = "entity_id" if "entity_id" in cv_folds_df.columns else "s1_id"
    present_s1 = set(augmented_df["s1_id"].astype(str).unique())
    cv_folds_df = cv_folds_df[cv_folds_df[entity_col].astype(str).isin(present_s1)].reset_index(
        drop=True
    )
    gt_dict = {k: v for k, v in gt_dict.items() if k in present_s1}

    # Strict fold verification assertions
    assert len(cv_folds_df) == len(present_s1), (
        f"Fold count mismatch: {len(cv_folds_df)} fold assignments vs {len(present_s1)} entities in features!"
    )
    assert cv_folds_df[entity_col].nunique() == len(cv_folds_df), (
        "Duplicate entity IDs found in fold assignments!"
    )
    folds = sorted(cv_folds_df["fold"].unique())
    fold_counts = {f: int((cv_folds_df["fold"] == f).sum()) for f in folds}
    for f in folds:
        cnt = fold_counts[f]
        assert cnt > 0, f"Fold {f} is empty!"
    print(f"Verified CV folds: {len(cv_folds_df):,} entities across 5 folds ({fold_counts}).")

    results: list[ExperimentResult] = []
    e0_preds: dict[str, set[str]] | None = None

    for arm in args.arms:
        res, oof_preds = run_experiment_arm(
            arm=arm,
            augmented_df=augmented_df,
            cv_folds_df=cv_folds_df,
            ground_truth=gt_dict,
            e0_oof_preds=e0_preds,
        )
        if arm == "E0":
            e0_preds = oof_preds
        results.append(res)

    # Conditional E4 execution
    if args.run_e4_if_promoted and "D3" in args.arms and "E4" not in args.arms:
        d3_res = next((r for r in results if r.arm == "D3"), None)
        e0_res = next((r for r in results if r.arm == "E0"), None)
        if d3_res and e0_res and d3_res.macro_f05_mean > e0_res.macro_f05_mean:
            print("\n>>> D3 demonstrated positive deterministic lift! Running conditional E4...")
            res_e4, _ = run_experiment_arm(
                arm="E4",
                augmented_df=augmented_df,
                cv_folds_df=cv_folds_df,
                ground_truth=gt_dict,
                e0_oof_preds=e0_preds,
            )
            results.append(res_e4)
        else:
            print(
                "\n>>> D3 did not beat E0 or was inconclusive; skipping E4 to preserve causal attribution."
            )

    # Save metrics JSON
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"Saved experiment metrics to: {args.output_json}")

    # Generate Markdown Report
    generate_experiment_report(results, augmented_df, args.output_report)


if __name__ == "__main__":
    main()
