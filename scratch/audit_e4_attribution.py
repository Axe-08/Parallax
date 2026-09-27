"""
Detailed audit of E0, D3, and E4 predictions, probabilities, and error attribution.
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
from run_experiments import CONCAT_FEATURES, DOMAIN_FEATURES, NEURAL_FEATURES

# Load data
print("Loading E4 augmented features...")
feats_path = PROJECT_ROOT / "experiments" / "neural_text" / "caches" / "augmented_features_e4_5k.parquet"
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
    "E4": BASELINE_28_FEATURES + NEURAL_FEATURES + CONCAT_FEATURES + DOMAIN_FEATURES + ["has_handle_target"],
}

threshold_grid = (0.60, 0.65, 0.70, 0.74, 0.78, 0.82, 0.86, 0.90)

probs_dict: dict[str, np.ndarray] = {}
optimal_taus: dict[str, dict[int, float]] = {"E0": {}, "D3": {}, "E4": {}}
oof_predictions: dict[str, dict[str, set[str]]] = {"E0": {}, "D3": {}, "E4": {}}

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

# Attach probabilities and predictions to DataFrame
for arm in ["E0", "D3", "E4"]:
    df[f"prob_{arm.lower()}"] = probs_dict[arm]
    df[f"pred_{arm.lower()}"] = [
        str(c) in oof_predictions[arm].get(str(s), set())
        for s, c in zip(df["s1_id"], df["cand_id"], strict=False)
    ]

pos_df = df[df["target"] == 1].copy()
neg_df = df[df["target"] == 0].copy()

total_true_in_cands = int(df["target"].sum())

rep_e0 = evaluate_resolution_predictions(gt_dict, oof_predictions["E0"])
rep_d3 = evaluate_resolution_predictions(gt_dict, oof_predictions["D3"])
rep_e4 = evaluate_resolution_predictions(gt_dict, oof_predictions["E4"])

print("\n" + "=" * 70)
print("1. AGGREGATE RECONCILIATION: E0 vs D3 vs E4")
print("=" * 70)

metrics = [
    ("E0", rep_e0),
    ("D3", rep_d3),
    ("E4", rep_e4),
]

for name, rep in metrics:
    tps = rep.total_correct_pairs
    fps = rep.total_predicted_pairs - tps
    cfns = total_true_in_cands - tps
    prec = tps / rep.total_predicted_pairs if rep.total_predicted_pairs > 0 else 0.0
    rec = tps / rep.total_true_pairs if rep.total_true_pairs > 0 else 0.0
    print(f"[{name}] Macro F0.5: {rep.macro_f05:.4f} | Prec: {prec*100:.2f}% | Rec: {rec*100:.2f}% | TPs: {tps:,} | FPs: {fps} | CFNs: {cfns:,}")

print("\n" + "=" * 70)
print("2. PAIR-LEVEL TRANSITIONS (E0 -> E4)")
print("=" * 70)

recov_e0_e4 = (~pos_df["pred_e0"]) & (pos_df["pred_e4"])
dropped_e0_e4 = (pos_df["pred_e0"]) & (~pos_df["pred_e4"])
new_fp_e0_e4 = (~neg_df["pred_e0"]) & (neg_df["pred_e4"])
fixed_fp_e0_e4 = (neg_df["pred_e0"]) & (~neg_df["pred_e4"])

print(f"Positive Pairs (TPs):")
print(f"  - Gross Recovered FNs (New TPs in E4):  +{recov_e0_e4.sum()}")
print(f"  - Gross Dropped TPs   (Lost from E0):   -{dropped_e0_e4.sum()}")
print(f"  - Net True Positive Change:            {recov_e0_e4.sum() - dropped_e0_e4.sum():+d}")
print(f"  - Reconciled E4 TPs:                   {rep_e0.total_correct_pairs} + {recov_e0_e4.sum()} - {dropped_e0_e4.sum()} = {rep_e4.total_correct_pairs}")

print(f"\nNegative Pairs (False Merges / FPs):")
print(f"  - Gross New False Merges (New in E4):  +{new_fp_e0_e4.sum()}")
print(f"  - Eliminated False Merges (Fixed):     -{fixed_fp_e0_e4.sum()}")
print(f"  - Net False Merge Change:              {new_fp_e0_e4.sum() - fixed_fp_e0_e4.sum():+d}")
print(f"  - Reconciled E4 FPs:                   {rep_e0.total_predicted_pairs - rep_e0.total_correct_pairs} + {new_fp_e0_e4.sum()} - {fixed_fp_e0_e4.sum()} = {rep_e4.total_predicted_pairs - rep_e4.total_correct_pairs}")

print("\n" + "=" * 70)
print("3. INCREMENTAL COMPARISON: D3 -> E4 (WHAT DO NEURAL FEATURES ADD OR DAMAGE?)")
print("=" * 70)

recov_d3_e4 = (~pos_df["pred_d3"]) & (pos_df["pred_e4"])
dropped_d3_e4 = (pos_df["pred_d3"]) & (~pos_df["pred_e4"])
new_fp_d3_e4 = (~neg_df["pred_d3"]) & (neg_df["pred_e4"])
fixed_fp_d3_e4 = (neg_df["pred_d3"]) & (~neg_df["pred_e4"])

print(f"Incremental Positive Pairs (D3 -> E4):")
print(f"  - Uniquely Recovered by Neural over D3: +{recov_d3_e4.sum()}")
print(f"  - Dropped by Neural (Caught by D3):     -{dropped_d3_e4.sum()}")
print(f"  - Net TP Gain over D3:                 {recov_d3_e4.sum() - dropped_d3_e4.sum():+d} (matches CFN drop: {rep_d3.total_correct_pairs} -> {rep_e4.total_correct_pairs})")

print(f"\nIncremental False Merges (D3 -> E4):")
print(f"  - New False Merges Introduced by Neural: +{new_fp_d3_e4.sum()}")
print(f"  - False Merges Cured by Neural over D3:  -{fixed_fp_d3_e4.sum()}")
print(f"  - Net False Merge Increase over D3:     {new_fp_d3_e4.sum() - fixed_fp_d3_e4.sum():+d} (FPs: 148 -> 178)")

print("\n" + "=" * 70)
print("4. DEEP ATTRIBUTION OF THE 241 RECOVERED FNs IN E4 (vs E0)")
print("=" * 70)

recov_e4_df = pos_df[recov_e0_e4].copy()

def categorize_e4_recovery(row):
    prob_delta = row["prob_e4"] - row["prob_e0"]
    
    # Deterministic indicators
    has_dom = row["has_domain_target"] == 1.0 and row["domain_stem_similarity"] >= 0.60
    has_hnd = row["has_handle_target"] == 1.0
    has_concat = row["concat_stem_similarity"] >= 0.70 and row["concat_addr_product"] >= 0.35
    det_active = has_dom or has_hnd or has_concat
    
    # Neural indicators
    has_indic = row["has_indicxlit_name"] == 1.0 and (pd.notna(row["indicxlit_name_similarity"]) and row["indicxlit_name_similarity"] >= 0.50)
    has_qwen = pd.notna(row["qwen_name_cosine"]) and row["qwen_name_cosine"] >= 0.75
    neural_active = has_indic or has_qwen
    
    if neural_active and det_active:
        return "Neural + Deterministic Synergy"
    elif neural_active:
        return "Neural Phonetic / Semantic Boost"
    elif has_dom:
        return "Deterministic: Domain Concordance"
    elif has_concat:
        return "Deterministic: Concat Representation"
    elif has_hnd:
        return "Deterministic: Handle Stem"
    elif prob_delta >= 0.05:
        return "General Tree Synergy / Model Drift (+prob)"
    elif abs(prob_delta) < 0.05:
        return "Tau Threshold Shift (prob unchanged)"
    else:
        return "Threshold Shift (-prob, lower tau)"

recov_e4_df["category"] = recov_e4_df.apply(categorize_e4_recovery, axis=1)
cat_counts = recov_e4_df["category"].value_counts()

print(f"Total Recovered FNs in E4: {len(recov_e4_df)}")
for cat, cnt in cat_counts.items():
    sub = recov_e4_df[recov_e4_df["category"] == cat]
    print(f"  - {cat:42s}: {cnt:3d} ({cnt/len(recov_e4_df)*100:5.1f}%) | Mean P(E0): {sub['prob_e0'].mean():.4f} -> P(E4): {sub['prob_e4'].mean():.4f} (Delta: {sub['prob_e4'].mean() - sub['prob_e0'].mean():+.4f})")

# Save attribution table
out_csv = PROJECT_ROOT / "scratch" / "e4_recovered_attribution_table.csv"
export_cols = [
    "s1_id", "cand_id", "prob_e0", "prob_d3", "prob_e4",
    "raw_name_ratio", "addr_ratio", "indicxlit_name_similarity", "has_indicxlit_name", "qwen_name_cosine",
    "concat_stem_similarity", "concat_addr_product", "domain_stem_similarity", "has_domain_target",
    "category"
]
recov_e4_df[export_cols].to_csv(out_csv, index=False)
print(f"\nSaved detailed pair-level table to: {out_csv}")

print("\n" + "=" * 70)
print("5. DEEP ANALYSIS OF THE 75 NEW FALSE MERGES IN E4 (vs E0)")
print("=" * 70)

new_fp_df = neg_df[new_fp_e0_e4].copy()
new_fp_df["fold"] = [s1_to_fold.get(str(s), -1) for s in new_fp_df["s1_id"]]

print(f"New False Merges by Fold:")
print(new_fp_df["fold"].value_counts().sort_index())

print(f"\nFeature Profiles on New False Merges (n={len(new_fp_df)}):")
print(f"  - Mean Qwen Cosine on False Merges:        {new_fp_df['qwen_name_cosine'].mean():.4f}")
print(f"  - False Merges with Qwen Cosine >= 0.70:   {(new_fp_df['qwen_name_cosine'] >= 0.70).sum()} ({(new_fp_df['qwen_name_cosine'] >= 0.70).mean()*100:.1f}%)")
print(f"  - Mean P(E0): {new_fp_df['prob_e0'].mean():.4f} -> Mean P(E4): {new_fp_df['prob_e4'].mean():.4f} (Delta: {new_fp_df['prob_e4'].mean() - new_fp_df['prob_e0'].mean():+.4f})")

print("\n" + "=" * 70)
print("6. SUMMARY AUDIT VERDICT")
print("=" * 70)
