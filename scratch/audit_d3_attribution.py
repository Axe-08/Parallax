"""
Detailed audit of E0 vs D3 predictions, probabilities, and error attribution.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

# Add local path and src
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "neural_text" / "scripts"))

from build_pairwise_features import BASELINE_28_FEATURES
from parallax.data.contracts import load_ground_truth_dict
from parallax.metrics.evaluator import evaluate_resolution_predictions
from parallax.postprocessing.singleton_gate import SingletonGatedPredictor
from run_experiments import CONCAT_FEATURES, DOMAIN_FEATURES

# Load data
print("Loading augmented features...")
feats_path = PROJECT_ROOT / "experiments" / "neural_text" / "caches" / "augmented_features_with_concordance_5k.parquet"
df = pd.read_parquet(feats_path)

cv_folds_df = pd.read_csv(PROJECT_ROOT / "data" / "medium_split_200k" / "cv_folds_source1.tsv", sep="\t")
gt_dict = load_ground_truth_dict(PROJECT_ROOT / "data" / "medium_split_200k" / "train_ground_truth.tsv")

present_s1 = set(df["s1_id"].astype(str).unique())
cv_folds_df = cv_folds_df[cv_folds_df["entity_id"].astype(str).isin(present_s1)].reset_index(drop=True)
gt_dict = {k: v for k, v in gt_dict.items() if k in present_s1}

s1_to_fold = dict(zip(cv_folds_df["entity_id"].astype(str), cv_folds_df["fold"], strict=False))
s1_fold_arr = np.array([s1_to_fold.get(str(s), -1) for s in df["s1_id"]], dtype=np.int32)

feature_sets = {
    "E0": BASELINE_28_FEATURES,
    "D3": BASELINE_28_FEATURES + CONCAT_FEATURES + DOMAIN_FEATURES + ["has_handle_target"],
}

threshold_grid = (0.60, 0.65, 0.70, 0.74, 0.78, 0.82, 0.86, 0.90)

probs_dict: dict[str, np.ndarray] = {}
optimal_taus: dict[str, dict[int, float]] = {"E0": {}, "D3": {}}
oof_predictions: dict[str, dict[str, set[str]]] = {"E0": {}, "D3": {}}

for arm, fcols in feature_sets.items():
    print(f"\nTraining and evaluating Arm {arm} ({len(fcols)} features)...")
    oof_probs = np.zeros(len(df), dtype=np.float32)
    
    for k in range(5):
        val_mask = s1_fold_arr == k
        train_mask = (s1_fold_arr != k) & (s1_fold_arr != -1)
        
        x_train, y_train = df.loc[train_mask, fcols], df.loc[train_mask, "target"].astype(int)
        x_val, y_val = df.loc[val_mask, fcols], df.loc[val_mask, "target"].astype(int)
        
        train_data = lgb.Dataset(x_train, label=y_train)
        val_data = lgb.Dataset(x_val, label=y_val, reference=train_data)
        
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "learning_rate": 0.08,
            "num_leaves": 31,
            "max_depth": 6,
            "verbose": -1,
            "seed": 42 + k,
            "n_jobs": 8,
        }
        
        booster = lgb.train(params, train_data, num_boost_round=100, valid_sets=[val_data])
        val_p = booster.predict(x_val)
        oof_probs[val_mask] = val_p
        
        val_s1_set = set(cv_folds_df[cv_folds_df["fold"] == k]["entity_id"].astype(str))
        val_gt = {s1: gt_dict.get(s1, set()) for s1 in val_s1_set}
        val_eval_df = df.loc[val_mask, ["s1_id", "cand_id"]].copy()
        val_eval_df["prob"] = val_p
        
        predictor = SingletonGatedPredictor()
        best_tau = 0.74
        best_score = -1.0
        best_preds = {}
        for tau in threshold_grid:
            preds = predictor.filter_predictions(val_eval_df, list(val_s1_set), threshold=tau)
            rep = evaluate_resolution_predictions(val_gt, preds)
            if rep.macro_f05 > best_score:
                best_score = rep.macro_f05
                best_tau = tau
                best_preds = preds
                
        optimal_taus[arm][k] = best_tau
        oof_predictions[arm].update(best_preds)
        print(f"Fold {k}: best tau={best_tau:.2f}, F0.5={best_score:.4f}")
        
    probs_dict[arm] = oof_probs

# Add columns to DataFrame for pair-level analysis
df["prob_e0"] = probs_dict["E0"]
df["prob_d3"] = probs_dict["D3"]

# Compute binary predictions per pair
df["pred_e0"] = [str(c) in oof_predictions["E0"].get(str(s), set()) for s, c in zip(df["s1_id"], df["cand_id"], strict=False)]
df["pred_d3"] = [str(c) in oof_predictions["D3"].get(str(s), set()) for s, c in zip(df["s1_id"], df["cand_id"], strict=False)]

print("\n========================================================")
print("1. AGGREGATE METRIC RECONCILIATION")
print("========================================================")

rep_e0 = evaluate_resolution_predictions(gt_dict, oof_predictions["E0"])
rep_d3 = evaluate_resolution_predictions(gt_dict, oof_predictions["D3"])

print(f"E0: Macro F0.5 = {rep_e0.macro_f05:.4f} | Prec = {rep_e0.total_correct_pairs/rep_e0.total_predicted_pairs*100:.2f}% | Rec = {rep_e0.total_correct_pairs/rep_e0.total_true_pairs*100:.2f}%")
print(f"D3: Macro F0.5 = {rep_d3.macro_f05:.4f} | Prec = {rep_d3.total_correct_pairs/rep_d3.total_predicted_pairs*100:.2f}% | Rec = {rep_d3.total_correct_pairs/rep_d3.total_true_pairs*100:.2f}%")

total_true_in_cands = int(df["target"].sum())
total_gt_pairs = sum(len(v) for v in gt_dict.values())

e0_tps = rep_e0.total_correct_pairs
e0_fps = rep_e0.total_predicted_pairs - e0_tps
e0_cfns = total_true_in_cands - e0_tps

d3_tps = rep_d3.total_correct_pairs
d3_fps = rep_d3.total_predicted_pairs - d3_tps
d3_cfns = total_true_in_cands - d3_tps

print(f"\nE0 Aggregate: TPs={e0_tps:,} | False Merges (FPs)={e0_fps} | Class. FNs={e0_cfns:,}")
print(f"D3 Aggregate: TPs={d3_tps:,} | False Merges (FPs)={d3_fps} | Class. FNs={d3_cfns:,}")
print(f"Net Changes (D3 - E0): Delta_TPs = {d3_tps - e0_tps:+d} | Delta_FPs = {d3_fps - e0_fps:+d} | Delta_CFNs = {d3_cfns - e0_cfns:+d}")

print("\n========================================================")
print("2. PAIR-LEVEL PREDICTION TRANSITIONS (TPs and FPs)")
print("========================================================")

pos_df = df[df["target"] == 1].copy()
neg_df = df[df["target"] == 0].copy()

# Positive transitions
recov_mask = (~pos_df["pred_e0"]) & (pos_df["pred_d3"])
dropped_mask = (pos_df["pred_e0"]) & (~pos_df["pred_d3"])
both_tp_mask = (pos_df["pred_e0"]) & (pos_df["pred_d3"])
neither_tp_mask = (~pos_df["pred_e0"]) & (~pos_df["pred_d3"])

n_recov = int(recov_mask.sum())
n_dropped = int(dropped_mask.sum())
print(f"True Positive Transitions:")
print(f"  - Recovered FNs (Predicted in D3, NOT in E0): +{n_recov}")
print(f"  - Dropped TPs   (Predicted in E0, NOT in D3): -{n_dropped}")
print(f"  - Net True Positive change: {n_recov - n_dropped:+d} (matches Delta_TPs: {d3_tps - e0_tps})")
print(f"  - Consistently True in both: {both_tp_mask.sum():,}")
print(f"  - Unresolved FNs in both:    {neither_tp_mask.sum():,}")

# Negative transitions (False Merges)
new_fp_mask = (~neg_df["pred_e0"]) & (neg_df["pred_d3"])
fixed_fp_mask = (neg_df["pred_e0"]) & (~neg_df["pred_d3"])
both_fp_mask = (neg_df["pred_e0"]) & (neg_df["pred_d3"])

n_new_fp = int(new_fp_mask.sum())
n_fixed_fp = int(fixed_fp_mask.sum())
print(f"\nFalse Merge (Distractor) Transitions:")
print(f"  - Additional False Merges (New in D3, NOT in E0): +{n_new_fp}")
print(f"  - Eliminated False Merges (Present in E0, FIXED in D3): -{n_fixed_fp}")
print(f"  - Net False Merge change: {n_new_fp - n_fixed_fp:+d} (matches Delta_FPs: {d3_fps - e0_fps})")
print(f"  - Persisting False Merges in both: {both_fp_mask.sum()}")

print("\n========================================================")
print("3. DEEP ATTRIBUTION OF THE 148 RECOVERED FNs")
print("========================================================")

recov_pairs = pos_df[recov_mask].copy()

# Categorize attribution
def categorize_recovery(row):
    prob_delta = row["prob_d3"] - row["prob_e0"]
    has_dom = row["has_domain_target"] == 1.0 and row["domain_stem_similarity"] >= 0.70
    has_hnd = row["has_handle_target"] == 1.0
    has_concat = row["concat_stem_similarity"] >= 0.75 and row["raw_name_ratio"] < 0.65
    
    if has_dom:
        return "Domain Concordance"
    if has_hnd:
        return "Handle Stem"
    if has_concat:
        return "Concat Token Recovery"
    if row["concat_stem_similarity"] >= 0.70 and prob_delta >= 0.08:
        return "Concat Probability Boost"
    if prob_delta >= 0.05:
        return "General Tree Synergy (+prob)"
    elif abs(prob_delta) < 0.05:
        return "Tau Threshold Shift (prob unchanged)"
    else:
        return "Threshold Shift (-prob, lower tau)"

recov_pairs["category"] = recov_pairs.apply(categorize_recovery, axis=1)
cat_counts = recov_pairs["category"].value_counts()
print("Attribution Breakdown of the 148 Recovered FNs:")
for cat, cnt in cat_counts.items():
    print(f"  - {cat:35s}: {cnt:3d} ({cnt/len(recov_pairs)*100:.1f}%)")

print("\nSummary of Probability Shifts on Recovered FNs:")
print(f"  Mean prob in E0: {recov_pairs['prob_e0'].mean():.4f}")
print(f"  Mean prob in D3: {recov_pairs['prob_d3'].mean():.4f}")
print(f"  Mean prob increase: {(recov_pairs['prob_d3'] - recov_pairs['prob_e0']).mean():+.4f}")
print(f"  Pairs with prob_d3 > prob_e0: {(recov_pairs['prob_d3'] > recov_pairs['prob_e0']).sum()} / {len(recov_pairs)} ({(recov_pairs['prob_d3'] > recov_pairs['prob_e0']).mean()*100:.1f}%)")

# Save detailed attribution table
out_table_path = PROJECT_ROOT / "scratch" / "d3_recovered_attribution_table.csv"
export_cols = [
    "s1_id", "cand_id", "target", "prob_e0", "prob_d3", "pred_e0", "pred_d3",
    "raw_name_ratio", "addr_ratio", "concat_stem_similarity", "concat_addr_product",
    "domain_stem_similarity", "domain_addr_concordance", "has_domain_target", "has_handle_target",
    "category"
]
recov_pairs[export_cols].to_csv(out_table_path, index=False)
print(f"\nDetailed pair-level attribution table saved to: {out_table_path}")

print("\nSample of Recovered Pairs (First 5):")
print(recov_pairs[["s1_id", "cand_id", "prob_e0", "prob_d3", "concat_stem_similarity", "domain_stem_similarity", "category"]].head(5))

print("\n========================================================")
print("4. DEEP ATTRIBUTION OF THE 162 DROPPED TPs (WHY DID THEY DROP?)")
print("========================================================")
dropped_pairs = pos_df[dropped_mask].copy()
print(f"Total Dropped TPs: {len(dropped_pairs)}")
print(f"Mean prob in E0: {dropped_pairs['prob_e0'].mean():.4f}")
print(f"Mean prob in D3: {dropped_pairs['prob_d3'].mean():.4f}")
print(f"Mean prob change: {(dropped_pairs['prob_d3'] - dropped_pairs['prob_e0']).mean():+.4f}")

# Threshold changes across folds
print("\nOptimal Threshold Tau changes per fold:")
for k in range(5):
    t_e0 = optimal_taus["E0"][k]
    t_d3 = optimal_taus["D3"][k]
    print(f"  Fold {k}: tau_E0 = {t_e0:.2f} -> tau_D3 = {t_d3:.2f} (Delta = {t_d3 - t_e0:+.2f})")

# 5. Overlap analysis: Neural (Cross-Script) vs Deterministic (Domain/Concat)
print("\n========================================================")
print("5. OVERLAP ANALYSIS: NEURAL (CROSS-SCRIPT) VS DETERMINISTIC (DOMAIN/CONCAT)")
print("========================================================")
from detect_script import has_indic_script
s1_df = pd.read_csv(PROJECT_ROOT / "data" / "medium_split_200k" / "train_source1.tsv", sep="\t", usecols=["entity_id", "business_name"])
s2_df = pd.read_csv(PROJECT_ROOT / "data" / "medium_split_200k" / "train_source2.tsv", sep="\t", usecols=["entity_id", "business_name"])
s3_df = pd.read_csv(PROJECT_ROOT / "data" / "medium_split_200k" / "train_source3.tsv", sep="\t", usecols=["entity_id", "business_name"])
s1_names = dict(zip(s1_df["entity_id"].astype(str), s1_df["business_name"].fillna("").astype(str), strict=False))
target_names = dict(zip(s2_df["entity_id"].astype(str), s2_df["business_name"].fillna("").astype(str), strict=False))
target_names.update(dict(zip(s3_df["entity_id"].astype(str), s3_df["business_name"].fillna("").astype(str), strict=False)))

pos_df["is_cross_script"] = [has_indic_script(str(s1_names.get(s, ""))) or has_indic_script(str(target_names.get(c, ""))) for s, c in zip(pos_df["s1_id"], pos_df["cand_id"], strict=False)]
pos_df["is_domain"] = pos_df["has_domain_target"] == 1.0
pos_df["is_concat"] = (pos_df["concat_stem_similarity"] >= 0.85) & (pos_df["raw_name_ratio"] < 0.70)

# Check baseline classification FNs
baseline_cfn_df = pos_df[~pos_df["pred_e0"]].copy()
cs_cfns = set(zip(baseline_cfn_df[baseline_cfn_df["is_cross_script"]]["s1_id"], baseline_cfn_df[baseline_cfn_df["is_cross_script"]]["cand_id"], strict=False))
dom_cfns = set(zip(baseline_cfn_df[baseline_cfn_df["is_domain"]]["s1_id"], baseline_cfn_df[baseline_cfn_df["is_domain"]]["cand_id"], strict=False))
concat_cfns = set(zip(baseline_cfn_df[baseline_cfn_df["is_concat"]]["s1_id"], baseline_cfn_df[baseline_cfn_df["is_concat"]]["cand_id"], strict=False))
det_cfns = dom_cfns | concat_cfns

print(f"Total Baseline Classification FNs: {len(baseline_cfn_df):,}")
print(f"  - Cross-Script (Indic) CFNs:     {len(cs_cfns):,}")
print(f"  - Domain Target CFNs:             {len(dom_cfns):,}")
print(f"  - Concatenation Stem CFNs:        {len(concat_cfns):,}")
print(f"  - Combined Deterministic CFNs:    {len(det_cfns):,}")
print(f"\nExact Overlap:")
print(f"  - Cross-Script AND Domain CFNs:   {len(cs_cfns & dom_cfns)}")
print(f"  - Cross-Script AND Concat CFNs:   {len(cs_cfns & concat_cfns)}")
print(f"  - Cross-Script AND (Domain|Concat): {len(cs_cfns & det_cfns)}")
if len(cs_cfns | det_cfns) > 0:
    print(f"  - Intersection / Union:           {len(cs_cfns & det_cfns)} / {len(cs_cfns | det_cfns)} ({len(cs_cfns & det_cfns) / len(cs_cfns | det_cfns) * 100:.2f}%)")
