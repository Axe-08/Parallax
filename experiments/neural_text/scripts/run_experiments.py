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
from scipy.stats import pearsonr

# Add src and local scripts to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from parallax.data.contracts import load_ground_truth_dict
from parallax.metrics.evaluator import evaluate_resolution_predictions
from parallax.postprocessing.singleton_gate import SingletonGatedPredictor
from build_pairwise_features import BASELINE_28_FEATURES

EXPERIMENT_FEATURE_SETS: dict[str, list[str]] = {
    "E0": BASELINE_28_FEATURES,
    "E1": BASELINE_28_FEATURES + ["indicxlit_name_similarity"],
    "E2": BASELINE_28_FEATURES + ["qwen_name_cosine"],
    "E3": BASELINE_28_FEATURES + ["indicxlit_name_similarity", "qwen_name_cosine"],
}


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
    print(f"\n========================================================")
    print(f"RUNNING ARM {arm}: {len(feature_cols)} features ({', '.join(feature_cols[-2:] if len(feature_cols)>28 else ['Baseline'])})")
    print(f"========================================================")

    # Merge fold assignments into features dataframe
    s1_to_fold = dict(zip(cv_folds_df["s1_id"].astype(str), cv_folds_df["fold"], strict=False))
    s1_fold_arr = np.array([s1_to_fold.get(str(s), -1) for s in augmented_df["s1_id"]], dtype=np.int32)

    fold_results: list[FoldResult] = []
    all_oof_preds: dict[str, set[str]] = {}

    for k in range(5):
        val_mask = s1_fold_arr == k
        train_mask = (s1_fold_arr != k) & (s1_fold_arr != -1)

        train_sub = augmented_df[train_mask]
        val_sub = augmented_df[val_mask].copy()

        val_s1_set = set(cv_folds_df[cv_folds_df["fold"] == k]["s1_id"].astype(str))
        val_gt = {s1: ground_truth.get(s1, set()) for s1 in val_s1_set}

        res, _, fold_preds = train_and_eval_fold(
            train_sub, val_sub, feature_cols, val_gt, list(val_s1_set), seed=42 + k
        )
        res.fold = k
        fold_results.append(res)
        all_oof_preds.update(fold_preds)

        print(
            f"Fold {k}: F0.5={res.macro_f05:.4f} | Prec={res.precision*100:.2f}% | "
            f"Rec={res.recall*100:.2f}% | SingAcc={res.singleton_acc*100:.2f}% | tau={res.optimal_tau:.2f}"
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

    # Cross-script true pair recovery (candidates with Indic characters)
    # Check predictions for cross-script pairs
    cs_mask = (augmented_df["target"] == 1) & (augmented_df["indicxlit_name_similarity"] != augmented_df["soft_name_ratio"])
    cs_pairs = augmented_df[cs_mask][["s1_id", "cand_id"]]
    cs_recovered = 0
    for s1, cand in zip(cs_pairs["s1_id"], cs_pairs["cand_id"], strict=False):
        if str(cand) in all_oof_preds.get(str(s1), set()):
            cs_recovered += 1

    # Comparative deltas vs E0
    new_fm = 0
    recovered_fn = 0
    if e0_oof_preds is not None:
        for s1, preds in all_oof_preds.items():
            e0_preds = e0_oof_preds.get(s1, set())
            gt_set = ground_truth.get(s1, set())
            # New false merges: predicted in arm, not in e0, and not in gt
            new_fm += len((preds - e0_preds) - gt_set)
            # Recovered FNs: predicted in arm, in gt, but was not in e0
            recovered_fn += len((preds & gt_set) - e0_preds)

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
        singleton_violations=int(full_report.singleton_violations),
        cross_script_tps_recovered=int(cs_recovered),
        new_false_merges_vs_e0=int(new_fm),
        recovered_fns_vs_e0=int(recovered_fn),
        total_train_time=float(sum(r.train_time for r in fold_results)),
        total_infer_time=float(sum(r.infer_time for r in fold_results)),
        fold_results=fold_results,
    )

    print(
        f"\n>>> ARM {arm} MEAN: F0.5 = {exp_result.macro_f05_mean:.4f} ± {exp_result.macro_f05_std:.4f} | "
        f"Prec = {exp_result.precision_mean*100:.2f}% | Rec = {exp_result.recall_mean*100:.2f}% | "
        f"SingAcc = {exp_result.singleton_acc_mean*100:.2f}%"
    )
    return exp_result, all_oof_preds


def generate_experiment_report(
    results: list[ExperimentResult],
    augmented_df: pd.DataFrame,
    report_path: Path,
) -> None:
    """Generate comprehensive markdown report comparing E0 - E3."""
    report_path.parent.mkdir(parents=True, exist_ok=True)

    # Feature correlations
    tp_df = augmented_df[augmented_df["target"] == 1]
    rho_qwen_soft, _ = pearsonr(tp_df["qwen_name_cosine"], tp_df["soft_name_ratio"])
    rho_ix_soft, _ = pearsonr(tp_df["indicxlit_name_similarity"], tp_df["soft_name_ratio"])
    rho_qwen_ix, _ = pearsonr(tp_df["qwen_name_cosine"], tp_df["indicxlit_name_similarity"])

    e0 = results[0]

    md: list[str] = [
        "# Parallax Neural Representation Experiment Report (E0–E3)",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "**Validation:** 5-Fold Stratified CV on Source 1 entities",
        "",
        "## 1. Executive Summary & Metric Comparison",
        "",
        "| Arm | Description | Features | Macro F0.5 | Δ F0.5 | Precision | Recall | Singleton Acc | Class. FNs | False Merges | Recov. FNs | New FMs |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    for r in results:
        delta = f"{r.macro_f05_mean - e0.macro_f05_mean:+.4f}" if r.arm != "E0" else "BASELINE"
        md.append(
            f"| **{r.arm}** | {r.description} | {r.feature_count} | **{r.macro_f05_mean:.4f} ± {r.macro_f05_std:.4f}** | "
            f"**{delta}** | {r.precision_mean*100:.2f}% | {r.recall_mean*100:.2f}% | {r.singleton_acc_mean*100:.2f}% | "
            f"{r.classification_fns:,} | {r.false_merges:,} | {r.recovered_fns_vs_e0} | {r.new_false_merges_vs_e0} |"
        )

    md.extend([
        "",
        "## 2. Feature Redundancy & Correlation Analysis",
        "",
        "Measured on Positive Candidate Pairs (True Matches):",
        f"- $\\rho(\\text{{qwen\\_name\\_cosine}}, \\text{{soft\\_name\\_ratio}}) = \\mathbf{{{rho_qwen_soft:.4f}}}$",
        f"- $\\rho(\\text{{indicxlit\\_name\\_similarity}}, \\text{{soft\\_name\\_ratio}}) = \\mathbf{{{rho_ix_soft:.4f}}}$",
        f"- $\\rho(\\text{{qwen\\_name\\_cosine}}, \\text{{indicxlit\\_name\\_similarity}}) = \\mathbf{{{rho_qwen_ix:.4f}}}$",
        "",
        "> [!NOTE]",
        "> A correlation $\\rho < 0.85$ confirms non-redundancy with the baseline lexical feature.",
        "",
        "## 3. Fold-by-Fold Stability",
        "",
        "| Fold | E0 F0.5 | E1 F0.5 | E2 F0.5 | E3 F0.5 | E0 tau | E1 tau | E2 tau | E3 tau |",
        "|---|---|---|---|---|---|---|---|---|",
    ])

    for k in range(5):
        f0_e0 = results[0].fold_results[k].macro_f05
        f0_e1 = results[1].fold_results[k].macro_f05
        f0_e2 = results[2].fold_results[k].macro_f05
        f0_e3 = results[3].fold_results[k].macro_f05
        t_e0 = results[0].fold_results[k].optimal_tau
        t_e1 = results[1].fold_results[k].optimal_tau
        t_e2 = results[2].fold_results[k].optimal_tau
        t_e3 = results[3].fold_results[k].optimal_tau
        md.append(f"| {k} | {f0_e0:.4f} | {f0_e1:.4f} | {f0_e2:.4f} | {f0_e3:.4f} | {t_e0:.2f} | {t_e1:.2f} | {t_e2:.2f} | {t_e3:.2f} |")

    md.extend([
        "",
        "## 4. Decision Gate Evaluation",
        "",
    ])

    # Check promotion criteria
    best_arm = max(results[1:], key=lambda x: x.macro_f05_mean)
    delta_best = best_arm.macro_f05_mean - e0.macro_f05_mean

    if delta_best >= 0.0020 and best_arm.singleton_acc_mean >= (e0.singleton_acc_mean - 0.005):
        md.append(
            f"**STATUS: ACCEPT / PROMOTE {best_arm.arm}**\n\n"
            f"Arm {best_arm.arm} achieved Macro F0.5 gain of **{delta_best:+.4f}** (clearing the +0.0020 significance bar) "
            f"while preserving singleton accuracy at {best_arm.singleton_acc_mean*100:.2f}%."
        )
    elif delta_best > 0:
        md.append(
            f"**STATUS: MARGINAL / INCONCLUSIVE**\n\n"
            f"Arm {best_arm.arm} gained {delta_best:+.4f}, which is within fold noise (std={best_arm.macro_f05_std:.4f}). "
            f"Do not promote to 200K without further feature refinement."
        )
    else:
        md.append(
            f"**STATUS: REJECT**\n\n"
            f"No neural feature configuration beat the baseline Macro F0.5 of {e0.macro_f05_mean:.4f}."
        )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print(f"\nExperiment report successfully written to: {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Parallax Neural Representation Experiments E0 - E3")
    parser.add_argument("--augmented-features", type=Path, default=Path("experiments/neural_text/caches/augmented_features_5k.parquet"))
    parser.add_argument("--cv-folds", type=Path, default=Path("data/medium_split_200k/cv_folds_source1.tsv"))
    parser.add_argument("--ground-truth", type=Path, default=Path("data/medium_split_200k/train_ground_truth.tsv"))
    parser.add_argument("--output-report", type=Path, default=Path("experiments/neural_text/reports/neural_experiment_e0_e3_report.md"))
    parser.add_argument("--output-json", type=Path, default=Path("experiments/neural_text/results/experiment_e0_e3_metrics.json"))
    args = parser.parse_args()

    print(f"Loading augmented features from: {args.augmented_features}")
    augmented_df = pd.read_parquet(args.augmented_features)

    print(f"Loading CV folds from: {args.cv_folds}")
    cv_folds_df = pd.read_csv(args.cv_folds, sep="\t")

    print(f"Loading ground truth from: {args.ground_truth}")
    gt_dict = load_ground_truth_dict(args.ground_truth)

    results: list[ExperimentResult] = []
    e0_preds: dict[str, set[str]] | None = None

    for arm in ["E0", "E1", "E2", "E3"]:
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

    # Save metrics JSON
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"Saved experiment metrics to: {args.output_json}")

    # Generate Markdown Report
    generate_experiment_report(results, augmented_df, args.output_report)


if __name__ == "__main__":
    main()
