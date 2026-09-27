import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

def get_rank_stats(scores, sim):
    strictly_greater = np.sum(scores > sim)
    equal = np.sum(scores == sim)
    best_rank = strictly_greater + 1
    worst_rank = strictly_greater + equal
    return best_rank, worst_rank

def audit_rank_logic():
    print("--- 2. AUDITING RANK CALCULATION ---")
    scores = np.array([0.0, 0.0, 0.0, 0.5])
    true_score = 0.0
    best, worst = get_rank_stats(scores, true_score)
    print(f"Test case: scores={scores}, true_score={true_score}")
    print(f"Best Rank: {best} | Worst Rank: {worst}")
    assert best == 2
    assert worst == 4
    
    scores2 = np.array([0.9, 0.9, 0.5, 0.1])
    best2, worst2 = get_rank_stats(scores2, 0.9)
    assert best2 == 1
    assert worst2 == 2
    print("Rank tie-handling tests passed.\n")

def manual_macro_f05(gt_dict, pred_dict):
    scores = []
    tp_total = 0
    fp_total = 0
    fn_total = 0
    
    singleton_acc_list = []
    
    for s1_id in gt_dict.keys():
        true_set = gt_dict[s1_id]
        pred_set = pred_dict.get(s1_id, set())
        
        tp = len(true_set.intersection(pred_set))
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)
        
        tp_total += tp
        fp_total += fp
        fn_total += fn
        
        if len(true_set) == 0:
            score = 1.0 if len(pred_set) == 0 else 0.0
            singleton_acc_list.append(score)
        else:
            if len(pred_set) == 0 or tp == 0:
                score = 0.0
            else:
                precision = tp / len(pred_set)
                recall = tp / len(true_set)
                denom = (0.25 * precision) + recall
                score = (1.25 * precision * recall) / denom if denom > 0 else 0.0
                
        scores.append(score)
        
    return np.mean(scores), np.mean(singleton_acc_list), tp_total, fp_total, fn_total

def main():
    print("========================================")
    print("    PARALLAX 5K FORENSIC AUDIT")
    print("========================================\n")
    
    audit_rank_logic()
    
    print("\n--- 1. DEFINING AUTHORITATIVE 5K POPULATION ---")
    feat_df = pd.read_parquet("baseline_artifacts/features_sample.parquet")
    eval_s1_ids = set(feat_df['s1_id'].astype(str).unique())
    print(f"Authoritative 5K Eval S1 Count: {len(eval_s1_ids)}")
    assert len(eval_s1_ids) == 5000, f"Expected 5000 S1s, got {len(eval_s1_ids)}"
    
    print("\n--- 5. AUDIT CANDIDATE PAIR ACCOUNTING ---")
    cands_df = pd.read_parquet("baseline_artifacts/candidate_pairs_sample.parquet")
    
    cand_s1s = set(cands_df['s1_id'].astype(str).unique())
    assert cand_s1s <= eval_s1_ids, "Candidate pairs contain out-of-scope S1s!"

    print(f"Loaded {len(cands_df):,} total candidates.")
    
    dup_count = cands_df.duplicated(subset=['s1_id', 'cand_id']).sum()
    print(f"Duplicate (s1_id, cand_id) pairs: {dup_count}")
    
    print("\n--- 7. AUDIT GROUND TRUTH CONSTRUCTION ---")
    gt = pd.read_csv("data/medium_split_200k/train_ground_truth.tsv", sep="\t")
    gt['source1_entity_id'] = gt['source1_entity_id'].astype(str)
    
    cv_folds = pd.read_csv("data/medium_split_200k/cv_folds_source1.tsv", sep="\t")
    cv_folds['entity_col'] = cv_folds['s1_id' if 's1_id' in cv_folds.columns else 'entity_id'].astype(str)
    cv_folds = cv_folds[cv_folds['entity_col'].isin(eval_s1_ids)]
    
    valid_s1_ids = eval_s1_ids
    
    gt_dict = {}
    gt_pairs = set()
    total_matches = 0
    empty_matches = 0
    
    for _, row in gt.iterrows():
        s1 = str(row['source1_entity_id'])
        if s1 not in valid_s1_ids:
            continue
            
        matches_str = str(row['matched_entity_ids'])
        if pd.isna(row['matched_entity_ids']) or not matches_str.strip():
            gt_dict[s1] = set()
            empty_matches += 1
            continue
            
        m_list = [m.strip() for m in matches_str.split(',') if m.strip()]
        gt_dict[s1] = set(m_list)
        if len(m_list) == 0:
            empty_matches += 1
        else:
            total_matches += len(m_list)
            for m in m_list:
                gt_pairs.add((s1, m))
                
    print(f"Valid S1 in 5K: {len(valid_s1_ids)}")
    print(f"S1 in GT dict: {len(gt_dict)}")
    print(f"S1 with zero matches (Singletons): {empty_matches}")
    print(f"S1 with matches: {len(gt_dict) - empty_matches}")
    print(f"Total canonical positive pairs: {total_matches}")
    
    print("\n--- 4. AUDIT TP/FP/FN RECONCILIATION ---")
    cand_pairs_set = set(zip(cands_df['s1_id'].astype(str), cands_df['cand_id'].astype(str)))
    
    candidate_positives = gt_pairs.intersection(cand_pairs_set)
    blocking_fns = gt_pairs - cand_pairs_set
    
    print(f"Canonical Positives: {len(gt_pairs)}")
    print(f"Candidate Positives (Captured by blockers): {len(candidate_positives)}")
    print(f"Blocking False Negatives (Lost by blockers): {len(blocking_fns)}")
    
    print("\n--- 14. APPLES-TO-APPLES REPRODUCTION (E0) ---")
    feat_df = pd.read_parquet("baseline_artifacts/features_sample.parquet")
    feat_df = feat_df.merge(cv_folds, left_on='s1_id', right_on='entity_id' if 'entity_id' in cv_folds.columns else 's1_id', how='inner')
    
    BASELINE_28_FEATURES = ['raw_name_ratio', 'soft_name_ratio', 'token_sort_ratio', 'token_set_ratio', 'partial_ratio', 'addr_token_set_ratio', 'addr_ratio', 'num_match_score', 'is_s1_addr_null', 'is_cand_addr_null', 'both_addr_present', 'len_diff_name', 'len_ratio_name', 'primary_num_match', 'primary_num_conflict', 'primary_num_missing', 'num_jaccard', 'num_conflict_count', 'postal_match', 'postal_conflict', 'postal_missing', 'jaro_winkler_soft', 'jaro_winkler_raw', 'token_jaccard_name', 'token_overlap_name', 'first_token_match', 'canon_addr_ratio', 'token_jaccard_addr']
    
    e0_preds = {}
    fold_f05s = []
    fold_tps, fold_fps, fold_c_fns = 0, 0, 0
    false_merges = 0
    singleton_violations = 0
    
    thresholds = [0.60, 0.65, 0.70, 0.74, 0.78, 0.82, 0.86, 0.90]
    
    for fold in range(5):
        train_df = feat_df[feat_df['fold'] != fold]
        val_df = feat_df[feat_df['fold'] == fold]
        
        train_data = lgb.Dataset(train_df[BASELINE_28_FEATURES], label=train_df['target'].astype(int))
        val_data = lgb.Dataset(val_df[BASELINE_28_FEATURES], label=val_df['target'].astype(int), reference=train_data)
        
        params = {"objective": "binary", "metric": "binary_logloss", "learning_rate": 0.08, "num_leaves": 31, "max_depth": 6, "verbose": -1, "seed": 42, "n_jobs": 8}
        
        booster = lgb.train(params, train_data, num_boost_round=100, valid_sets=[val_data])
        probs = booster.predict(val_df[BASELINE_28_FEATURES])
        
        # Threshold search over fold validation (Note: this is threshold leakage! We are searching tau on the validation set itself)
        best_tau, best_f05 = 0.60, -1
        fold_gt = {s1: gt_dict[s1] for s1 in val_df['s1_id'].unique() if s1 in gt_dict}
        
        for tau in thresholds:
            p_dict = {}
            for idx, (_, row) in enumerate(val_df.iterrows()):
                s1, c_id = str(row['s1_id']), str(row['cand_id'])
                if probs[idx] >= tau:
                    p_dict.setdefault(s1, set()).add(c_id)
            f05, _, _, _, _ = manual_macro_f05(fold_gt, p_dict)
            if f05 > best_f05:
                best_f05 = f05
                best_tau = tau
                
        # Use optimal tau
        for idx, (_, row) in enumerate(val_df.iterrows()):
            if probs[idx] >= best_tau:
                s1, c_id = str(row['s1_id']), str(row['cand_id'])
                e0_preds.setdefault(s1, set()).add(c_id)
                
        fold_f05s.append(best_f05)
        
    print("\n--- 9. THRESHOLD LEAKAGE AUDIT ---")
    print("WARNING: The evaluation explicitly searches for the optimal threshold (tau) on the VALIDATION FOLD and uses it to report performance.")
    print("This is a direct form of threshold leakage. The validation set is used for tuning, invalidating the out-of-fold generalization.")
    
    print("\n--- 3. AUDIT MACRO F0.5 FROM FIRST PRINCIPLES ---")
    macro_f05, sing_acc, tp, fp, fn = manual_macro_f05(gt_dict, e0_preds)
    
    classification_fns = 0
    for pair in candidate_positives:
        s1, c_id = pair
        if c_id not in e0_preds.get(s1, set()):
            classification_fns += 1
            
    for s1 in gt_dict:
        if len(gt_dict[s1]) == 0:
            if len(e0_preds.get(s1, set())) > 0:
                singleton_violations += 1
                false_merges += len(e0_preds[s1])
                
    print(f"Macro F0.5: {macro_f05:.4f}")
    print(f"Singleton Accuracy: {sing_acc:.4f}")
    print(f"Predicted TPs: {tp}")
    print(f"Predicted FPs: {fp}")
    print(f"Total FNs (CFN + BFN): {fn}")
    print(f"Classification FNs (In candidates, missed by model): {classification_fns}")
    print(f"Singleton Violations (Zero-match S1 with predictions): {singleton_violations}")
    print(f"False Merges (FPs on zero-match S1s): {false_merges}")
    
    assert tp + classification_fns == len(candidate_positives), "TP + CFN must equal Candidate Positives!"
    assert fn == classification_fns + len(blocking_fns), "Total FN must equal CFN + BFN!"
    print("\nRECONCILIATION SUCCESSFUL: TP + CFN == Candidate Positives && CFN + BFN == Total FNs.")

if __name__ == "__main__":
    main()
